import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
YAML = PROJECT_ROOT / "data" / "processed" / "yamls" / "stage1_quadrant.yaml"
CHECKPOINTS = PROJECT_ROOT / "models" / "checkpoints"
RUNS_DIR = CHECKPOINTS / "runs"


# device detection
def get_device(override: str | None = None) -> str:
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# training
def train(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    print("Stage 1 — Quadrant Detection")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch: {args.batch}")
    print(f"YAML: {YAML}")

    if not YAML.exists():
        raise FileNotFoundError(
            f"YAML not found: {YAML}\n"
            "Run `python preprocessing.py` first to generate processed data."
        )

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8m-seg.pt")

    results = model.train(
        data=str(YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(RUNS_DIR),
        name="stage1",
        exist_ok=True,  # overwrite previous run of same name
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,  # final LR = lr0 * lrf
        warmup_epochs=3,
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
        save=True,
        plots=True,
        verbose=True,
    )

    # copy best checkpoint to a predictable location
    best_src = Path(results.save_dir) / "weights" / "best.pt"
    best_dst = CHECKPOINTS / "stage1_best.pt"
    if best_src.exists():
        shutil.copy(best_src, best_dst)
        print(f"Stage 1 checkpoint saved to {best_dst}")
    else:
        print(f"Could not find best.pt at {best_src}")

    print(f"Full run artefacts: {results.save_dir}")
    print("Next: python models/train_stage2.py")


# cli
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
