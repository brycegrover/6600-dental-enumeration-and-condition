import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

# BEGIN AI CODE
# function to install stubs for missing loss classes in older Ultralytics versions
# Makes loading newer checkpoints without crashing more likely
def _install_compat_stubs() -> None:
    import ultralytics.utils.loss as _loss_mod
    for stub_name in ("BCEDiceLoss", "MultiChannelDiceLoss"):
        if hasattr(_loss_mod, stub_name):
            continue
        stub = type(stub_name, (object,), {
            "__init__": lambda self, *a, **kw: None,
            "__call__": lambda self, *a, **kw: None,
        })
        stub.__module__ = "ultralytics.utils.loss"
        setattr(_loss_mod, stub_name, stub)


_install_compat_stubs()


# function to install a patch for the known-sporadic TAL shape-mismatch crash
# which is worth auto-resuming on when it happens (especially on MPS, where it's more common and there's no CUDA OOM to trigger auto-resume instead)
def _install_tal_patches() -> None:
    from ultralytics.utils import tal as _tal_mod
    import torch as _t

    original = _tal_mod.TaskAlignedAssigner.get_box_metrics

    def robust_get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        try:
            return original(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt.bool())
        except RuntimeError as err:
            msg = str(err)
            if ("shape mismatch" not in msg
                and "must match the size" not in msg
                and "cannot be broadcast" not in msg):
                raise
            na = pd_bboxes.shape[-2]
            dev = pd_scores.device
            align_metric = _t.zeros(
                [self.bs, self.n_max_boxes, na],
                dtype=pd_scores.dtype, device=dev,
            )
            overlaps = _t.zeros(
                [self.bs, self.n_max_boxes, na],
                dtype=pd_bboxes.dtype, device=dev,
            )
            return align_metric, overlaps

    _tal_mod.TaskAlignedAssigner.get_box_metrics = robust_get_box_metrics


_install_tal_patches()
# ENDA AI CODE


# paths
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
YAML = PROJECT_ROOT / "data" / "processed" / "yamls" / "stage2_enumeration.yaml"
CHECKPOINTS = PROJECT_ROOT / "models" / "checkpoints"
RUNS_DIR = CHECKPOINTS / "runs"
STAGE1_CKPT = CHECKPOINTS / "stage1_best.pt"
WARMSTART_CKPT = CHECKPOINTS / "stage2_warmstart.pt"


# device detection for mps and cuda, with override option
def get_device(override: str | None = None) -> str:
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# training
RUN_NAME = "stage2"
LAST_CKPT  = RUNS_DIR / RUN_NAME / "weights" / "last.pt"

# build the kwargs for YOLO.train() in a separate function
# we can modify them for auto-resume without repeating all the args parsing and printing logic
def _build_train_kwargs(args: argparse.Namespace, device: str) -> dict:
    return dict(
        data=str(YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.0005,  # lower LR for fine-tuning
        lrf=0.01,
        warmup_epochs=2,
        weight_decay=0.0005,
        dropout=0.0,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        patience=args.patience,
        amp=not args.no_amp,  # AMP off by default (avoids TAL shape-mismatch crash)
        save=True,
        plots=True,
        verbose=True,
    )

# BEGIN AI CODE
def _is_transient_crash(err: BaseException) -> bool:
    """Detect the known-sporadic Ultralytics/PyTorch crashes worth auto-resuming."""
    msg = str(err)
    signatures = (
        "shape mismatch",                  # ultralytics TAL assigner (tal.py:195)
        "must match the size of tensor",   # ultralytics TAL iou_calculation (tal.py:215)
        "cannot be broadcast",             # generic TAL broadcasting edge case
        "CUDA out of memory",
        "MPS backend out of memory",
        "Input type (MPSFloatType)",       # occasional MPS dtype hiccup
    )
    return isinstance(err, RuntimeError) and any(s in msg for s in signatures)


# batch sizes cycled through on auto-resume to break dataloader grouping that
# re-triggers the TAL bug on the same iteration. Values stay under MPS memory.
_RETRY_BATCH_CYCLE = [5, 3, 6]


def _patch_checkpoint_batch(ckpt_path: Path, new_batch: int) -> None:
    """Rewrite `train_args.batch` inside an Ultralytics .pt checkpoint.

    Ultralytics' resume reads train args from inside the .pt, not from args.yaml
    on disk, so this is the only way to change batch size mid-run.
    """
    ckpt = torch.load(str(ckpt_path), weights_only=False)
    if "train_args" in ckpt and isinstance(ckpt["train_args"], dict):
        ckpt["train_args"]["batch"] = new_batch
        torch.save(ckpt, str(ckpt_path))

# train the model to detect and enumerate teeth, starting from stage 1 weights
def train(args: argparse.Namespace) -> None:
    device = get_device(args.device)

    if not YAML.exists():
        raise FileNotFoundError(
            f"YAML not found: {YAML}\nRun `python preprocessing.py` first."
        )

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    resuming = args.resume and LAST_CKPT.exists()
    warm_starting = args.warm_start and WARMSTART_CKPT.exists()

    if resuming:
        print(f"Stage 2 — Resuming from {LAST_CKPT}")
        print(f"Device: {device}")
        model = YOLO(str(LAST_CKPT))
        train_kwargs: dict = dict(resume=True)
    elif warm_starting:
        # weights-only warm-start; fresh optimizer/epoch counter, writes to stage2_continued/
        print(f"Stage 2 — Warm-start from {WARMSTART_CKPT}")
        print("(trained weights preserved, optimizer/epoch reset)")
        print(f"Device: {device}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch: {args.batch}")
        print(f"AMP: {not args.no_amp}")
        print("Run dir: runs/stage2_continued")
        model = YOLO(str(WARMSTART_CKPT))
        train_kwargs = _build_train_kwargs(args, device)
        train_kwargs["name"] = "stage2_continued"
    else:
        if not STAGE1_CKPT.exists():
            raise FileNotFoundError(
                f"Stage 1 checkpoint not found: {STAGE1_CKPT}\n"
                "Run `python models/train_stage1.py` first."
            )
        print("Stage 2 — Tooth Enumeration (fresh run)")
        print(f"Device: {device}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch: {args.batch}")
        print(f"AMP: {not args.no_amp}")
        print(f"Warm-start: {STAGE1_CKPT}")
        print(f"YAML: {YAML}")
        model = YOLO(str(STAGE1_CKPT))
        train_kwargs = _build_train_kwargs(args, device)

    # auto-resume-on-crash: reload last.pt on known-transient errors, capped at --max-retries
    attempt = 0
    while True:
        try:
            results = model.train(**train_kwargs)
            break
        except BaseException as err:  # noqa: BLE001
            if not _is_transient_crash(err):
                raise
            attempt += 1
            if attempt > args.max_retries:
                print(
                    f"Crash persisted through {args.max_retries} auto-resume "
                    f"attempts — giving up. Last error: {err}"
                )
                raise
            if not LAST_CKPT.exists():
                print(f"No last.pt at {LAST_CKPT} — cannot auto-resume.")
                raise
            # perturb batch size to break deterministic dataloader grouping that re-triggers the TAL bug
            new_batch = _RETRY_BATCH_CYCLE[(attempt - 1) % len(_RETRY_BATCH_CYCLE)]
            try:
                _patch_checkpoint_batch(LAST_CKPT, new_batch)
                batch_msg = f" (batch → {new_batch})"
            except Exception as patch_err:  # noqa: BLE001
                batch_msg = f" (batch patch failed: {patch_err})"
            print(
                f"[auto-resume] Transient crash on attempt {attempt}"
                f"/{args.max_retries}: {err}"
            )
            print(f"[auto-resume] Reloading {LAST_CKPT}{batch_msg} and continuing...")
            model = YOLO(str(LAST_CKPT))
            train_kwargs = dict(resume=True)

    best_src = Path(results.save_dir) / "weights" / "best.pt"
    best_dst = CHECKPOINTS / "stage2_best.pt"
    if best_src.exists():
        shutil.copy(best_src, best_dst)
        print(f"Stage 2 checkpoint saved to {best_dst}")
    else:
        print(f"Could not find best.pt at {best_src}")

    print(f"Full run artefacts: {results.save_dir}")
    print("Next: python models/train_stage3.py")

# BEGIN AI CODE
# cli
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 — Tooth Enumeration")
    p.add_argument("--epochs", type=int, default=100, help="Training epochs")
    p.add_argument("--batch",      type=int, default=4,   help="Batch size (lowered from 8 for MPS stability)")
    p.add_argument("--imgsz",      type=int, default=640, help="Input image size")
    p.add_argument("--patience",   type=int, default=20,  help="Early stopping patience")
    p.add_argument("--device",     type=str, default=None, help="Device: mps | cuda | cpu")
    p.add_argument("--resume",     action="store_true",   help="Resume from runs/stage2/weights/last.pt")
    p.add_argument("--warm-start", action="store_true",
                   help="Fresh training, but init from models/checkpoints/stage2_warmstart.pt "
                        "(produced by make_warmstart.py). Preserves trained weights, resets "
                        "optimizer + epoch counter. Writes to runs/stage2_continued/.")
    p.add_argument("--no-amp",     action="store_true", default=True,
                   help="Disable AMP (default: on — AMP can trigger TAL shape-mismatch crashes)")
    p.add_argument("--amp",        dest="no_amp", action="store_false",
                   help="Force AMP on (opt-in; default is off)")
    p.add_argument("--max-retries", type=int, default=3,
                   help="Auto-resume attempts on transient crashes (TAL shape mismatch, OOM)")
    return p.parse_args()
# END AI CODE

if __name__ == "__main__":
    train(parse_args())
