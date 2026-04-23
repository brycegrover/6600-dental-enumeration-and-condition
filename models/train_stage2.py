"""
train_stage2.py — Stage 2: Tooth Enumeration
=============================================
Fine-tunes from the Stage 1 checkpoint to predict individual tooth positions
within each quadrant (8 classes: tooth positions 1–8).

Training data  : 634 enumeration images + 705 disease images = 1,339 images
Labels used    : category_id_2 (tooth number 1–8 within quadrant)
Warm-start     : models/checkpoints/stage1_best.pt
Output         : models/checkpoints/stage2_best.pt

Usage:
    python models/train_stage2.py
    python models/train_stage2.py --epochs 80 --batch 16 --device cuda
"""

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


# ── 8.4→8.3 compat shims ─────────────────────────────────────────────────────
# Our stage-2 checkpoint was produced by ultralytics 8.4.41, which pickled
# references to loss classes that don't exist in 8.3.40 (BCEDiceLoss,
# MultiChannelDiceLoss). We inject inert stubs under the expected module path
# so unpickling succeeds; 8.3.40 rebuilds its own criterion at train time so
# these stubs are never actually called.
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


# ── TAL shape-mismatch crash mitigation ──────────────────────────────────────
# Ultralytics' Task-Aligned Learning assigner occasionally crashes with:
#   RuntimeError: shape mismatch: value tensor of shape [N] cannot be broadcast
#   to indexing result of shape [M]
# at ultralytics/utils/tal.py:143, line:
#   bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]
# Root cause: mask_gt arrives here as a float tensor (from mask_in_gts * mask_gt
# upstream). MPS's float-mask advanced-indexing semantics occasionally produce
# different element counts on LHS vs RHS. Bool masks don't exhibit this.
#
# Two-layer fix:
#   1. Cast mask_gt to bool before delegating to the original implementation.
#   2. If any batch still manages to crash inside the assigner, return zero
#      align_metric / overlaps for that batch → TAL assigns no positives → the
#      batch contributes ~0 loss → training proceeds instead of dying.
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

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
YAML          = PROJECT_ROOT / "data" / "processed" / "yamls" / "stage2_enumeration.yaml"
CHECKPOINTS   = PROJECT_ROOT / "models" / "checkpoints"
RUNS_DIR      = CHECKPOINTS / "runs"
STAGE1_CKPT   = CHECKPOINTS / "stage1_best.pt"
WARMSTART_CKPT = CHECKPOINTS / "stage2_warmstart.pt"


# ── Device detection ─────────────────────────────────────────────────────────
def get_device(override: str | None = None) -> str:
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Training ─────────────────────────────────────────────────────────────────
RUN_NAME   = "stage2"
LAST_CKPT  = RUNS_DIR / RUN_NAME / "weights" / "last.pt"


def _build_train_kwargs(args: argparse.Namespace, device: str) -> dict:
    """Full training kwargs used for a fresh run (ignored when resuming)."""
    return dict(
        data=str(YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        # ── Optimizer ──────────────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.0005,               # lower LR for fine-tuning
        lrf=0.01,
        warmup_epochs=2,
        # ── Regularisation ─────────────────────────────────────────────
        weight_decay=0.0005,
        dropout=0.0,
        # ── Augmentation ───────────────────────────────────────────────
        augment=True,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        # ── Early stopping ─────────────────────────────────────────────
        patience=args.patience,
        # ── Stability ──────────────────────────────────────────────────
        amp=not args.no_amp,      # AMP off by default (avoids TAL shape-mismatch crash)
        # ── Output ─────────────────────────────────────────────────────
        save=True,
        plots=True,
        verbose=True,
    )


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


# Batch sizes to cycle through on successive auto-resume retries. Perturbing
# batch breaks the deterministic dataloader grouping that re-triggers the TAL
# bug on the exact same iteration. Values chosen to stay well under MPS memory.
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
        print(f"\n{'='*60}")
        print(f"  Stage 2 — Resuming from {LAST_CKPT}")
        print(f"  Device      : {device}")
        print(f"{'='*60}\n")
        model = YOLO(str(LAST_CKPT))
        train_kwargs: dict = dict(resume=True)
    elif warm_starting:
        # Weights-only warm-start from stage2_warmstart.pt. Fresh optimizer,
        # fresh epoch counter, writes to runs/stage2_continued/ to preserve
        # the original runs/stage2/ history.
        print(f"\n{'='*60}")
        print(f"  Stage 2 — Warm-start from {WARMSTART_CKPT}")
        print(f"  (trained weights preserved, optimizer/epoch reset)")
        print(f"  Device      : {device}")
        print(f"  Epochs      : {args.epochs}")
        print(f"  Batch       : {args.batch}")
        print(f"  AMP         : {not args.no_amp}")
        print(f"  Run dir     : runs/stage2_continued")
        print(f"{'='*60}\n")
        model = YOLO(str(WARMSTART_CKPT))
        train_kwargs = _build_train_kwargs(args, device)
        train_kwargs["name"] = "stage2_continued"
    else:
        if not STAGE1_CKPT.exists():
            raise FileNotFoundError(
                f"Stage 1 checkpoint not found: {STAGE1_CKPT}\n"
                "Run `python models/train_stage1.py` first."
            )
        print(f"\n{'='*60}")
        print(f"  Stage 2 — Tooth Enumeration (fresh run)")
        print(f"  Device      : {device}")
        print(f"  Epochs      : {args.epochs}")
        print(f"  Batch       : {args.batch}")
        print(f"  AMP         : {not args.no_amp}")
        print(f"  Warm-start  : {STAGE1_CKPT}")
        print(f"  YAML        : {YAML}")
        print(f"{'='*60}\n")
        model = YOLO(str(STAGE1_CKPT))
        train_kwargs = _build_train_kwargs(args, device)

    # Auto-resume-on-crash: if training blows up with a known-transient error,
    # reload last.pt and continue. Caps at --max-retries to avoid infinite loops.
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
                    f"\n⚠ Crash persisted through {args.max_retries} auto-resume "
                    f"attempts — giving up.\n  Last error: {err}"
                )
                raise
            if not LAST_CKPT.exists():
                print(f"\n⚠ No last.pt at {LAST_CKPT} — cannot auto-resume.")
                raise
            # Perturb the batch size before each retry to break the deterministic
            # dataloader grouping that otherwise triggers the TAL bug on the
            # same iteration of the same epoch on every re-resume.
            new_batch = _RETRY_BATCH_CYCLE[(attempt - 1) % len(_RETRY_BATCH_CYCLE)]
            try:
                _patch_checkpoint_batch(LAST_CKPT, new_batch)
                batch_msg = f" (batch → {new_batch})"
            except Exception as patch_err:  # noqa: BLE001
                batch_msg = f" (batch patch failed: {patch_err})"
            print(
                f"\n[auto-resume] Transient crash on attempt {attempt}"
                f"/{args.max_retries}: {err}\n"
                f"[auto-resume] Reloading {LAST_CKPT}{batch_msg} and continuing…\n"
            )
            model = YOLO(str(LAST_CKPT))
            train_kwargs = dict(resume=True)

    best_src = Path(results.save_dir) / "weights" / "best.pt"
    best_dst = CHECKPOINTS / "stage2_best.pt"
    if best_src.exists():
        shutil.copy(best_src, best_dst)
        print(f"\n✓ Stage 2 checkpoint saved → {best_dst}")
    else:
        print(f"\n⚠ Could not find best.pt at {best_src}")

    print(f"  Full run artefacts  → {results.save_dir}")
    print(f"\nNext: python models/train_stage3.py\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 — Tooth Enumeration")
    p.add_argument("--epochs",     type=int, default=100, help="Training epochs")
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


if __name__ == "__main__":
    train(parse_args())
