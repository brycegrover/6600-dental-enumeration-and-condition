"""
train_stage1.py — Stage 1: Quadrant Detection
==============================================
Trains a YOLOv8m-seg model to detect the four dental quadrants.
Initializes from ImageNet-pretrained weights (no curriculum warm-start at this stage).

Training data  : 693 quadrant images + 705 disease images = 1,398 images
Labels used    : category_id_1 (quadrant 1–4)
Output         : models/checkpoints/stage1_best.pt

Usage:
    python models/train_stage1.py
    python models/train_stage1.py --epochs 80 --batch 16 --device cuda
"""

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
YAML         = PROJECT_ROOT / "data" / "processed" / "yamls" / "stage1_quadrant.yaml"
CHECKPOINTS  = PROJECT_ROOT / "models" / "checkpoints"
RUNS_DIR     = CHECKPOINTS / "runs"


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
    print(f"\n{'='*60}")
    print(f"  Stage 1 — Quadrant Detection")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  YAML    : {YAML}")
    print(f"{'='*60}\n")

    if not YAML.exists():
        raise FileNotFoundError(
            f"YAML not found: {YAML}\n"
            "Run `python preprocessing.py` first to generate processed data."
        )

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Load pretrained YOLOv8m segmentation model
    model = YOLO("yolov8m-seg.pt")

    results = model.train(
        data=str(YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(RUNS_DIR),
        name="stage1",
        exist_ok=True,            # overwrite previous run of same name
        # ── Optimizer ──────────────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,                 # final LR = lr0 * lrf
        warmup_epochs=3,
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
        # ── Output ─────────────────────────────────────────────────────
        save=True,
        plots=True,
        verbose=True,
    )

    # Copy best checkpoint to a predictable location
    best_src = Path(results.save_dir) / "weights" / "best.pt"
    best_dst = CHECKPOINTS / "stage1_best.pt"
    if best_src.exists():
        shutil.copy(best_src, best_dst)
        print(f"\n✓ Stage 1 checkpoint saved → {best_dst}")
    else:
        print(f"\n⚠ Could not find best.pt at {best_src}")

    print(f"  Full run artefacts  → {results.save_dir}")
    print(f"\nNext: python models/train_stage2.py\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1 — Quadrant Detection")
    p.add_argument("--epochs",   type=int,   default=100,  help="Training epochs")
    p.add_argument("--batch",    type=int,   default=8,    help="Batch size (try 16 on CUDA)")
    p.add_argument("--imgsz",    type=int,   default=640,  help="Input image size")
    p.add_argument("--patience", type=int,   default=20,   help="Early stopping patience")
    p.add_argument("--device",   type=str,   default=None, help="Device: mps | cuda | cpu (auto-detected if omitted)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
