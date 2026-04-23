"""
train_stage3.py — Stage 3: Diagnosis Detection (Curriculum)
============================================================
Fine-tunes from the Stage 2 checkpoint to detect dental diagnoses with
polygon segmentation masks (Caries, Deep Caries, Periapical Lesion, Impacted).

This is the final curriculum stage. Its performance is compared against the
baseline (train_baseline.py) to answer RQ1.

Training data  : 705 fully-labeled images
Labels used    : category_id_3 + polygon segmentation masks
Warm-start     : models/checkpoints/stage2_best.pt
Output         : models/checkpoints/stage3_best.pt

Usage:
    python models/train_stage3.py
    python models/train_stage3.py --epochs 80 --batch 8 --device cuda
"""

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
YAML          = PROJECT_ROOT / "data" / "processed" / "yamls" / "stage3_disease.yaml"
CHECKPOINTS   = PROJECT_ROOT / "models" / "checkpoints"
RUNS_DIR      = CHECKPOINTS / "runs"
STAGE2_CKPT   = CHECKPOINTS / "stage2_best.pt"


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
def train(args: argparse.Namespace) -> None:
    device = get_device(args.device)

    if not STAGE2_CKPT.exists():
        raise FileNotFoundError(
            f"Stage 2 checkpoint not found: {STAGE2_CKPT}\n"
            "Run `python models/train_stage2.py` first."
        )
    if not YAML.exists():
        raise FileNotFoundError(
            f"YAML not found: {YAML}\n"
            "Run `python preprocessing.py` first."
        )

    print(f"\n{'='*60}")
    print(f"  Stage 3 — Diagnosis Detection (Curriculum)")
    print(f"  Device      : {device}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch       : {args.batch}")
    print(f"  Warm-start  : {STAGE2_CKPT}")
    print(f"  YAML        : {YAML}")
    print(f"{'='*60}\n")

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Load from Stage 2 checkpoint (warm-start — full curriculum)
    model = YOLO(str(STAGE2_CKPT))

    results = model.train(
        data=str(YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(RUNS_DIR),
        name="stage3_curriculum",
        exist_ok=True,
        # ── Optimizer ──────────────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.0003,               # lower LR — close to final curriculum stage
        lrf=0.01,
        warmup_epochs=2,
        # ── Regularisation ─────────────────────────────────────────────
        weight_decay=0.0005,
        dropout=0.0,
        # ── Augmentation ───────────────────────────────────────────────
        # Slightly stronger augmentation since training set is only 705 images
        augment=True,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        copy_paste=0.1,           # copy-paste augmentation for small dataset
        # ── Early stopping ─────────────────────────────────────────────
        patience=args.patience,
        # ── Output ─────────────────────────────────────────────────────
        save=True,
        plots=True,
        verbose=True,
    )

    best_src = Path(results.save_dir) / "weights" / "best.pt"
    best_dst = CHECKPOINTS / "stage3_best.pt"
    if best_src.exists():
        shutil.copy(best_src, best_dst)
        print(f"\n✓ Stage 3 (curriculum) checkpoint saved → {best_dst}")
    else:
        print(f"\n⚠ Could not find best.pt at {best_src}")

    print(f"  Full run artefacts  → {results.save_dir}")
    print(f"\nCurriculum training complete!")
    print(f"Run `python models/train_baseline.py` for the comparison baseline.")
    print(f"Then open `models/02_evaluation.ipynb` to compare results.\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3 — Diagnosis Detection (Curriculum)")
    p.add_argument("--epochs",   type=int,   default=100,  help="Training epochs")
    p.add_argument("--batch",    type=int,   default=8,    help="Batch size")
    p.add_argument("--imgsz",    type=int,   default=640,  help="Input image size")
    p.add_argument("--patience", type=int,   default=25,   help="Early stopping patience (higher: small dataset)")
    p.add_argument("--device",   type=str,   default=None, help="Device: mps | cuda | cpu")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
