"""
01_convert_training_to_yolo.py
-------------------------------
Converts the DENTEX COCO-format training and validation JSONs into
YOLO segmentation .txt label files, organized per curriculum stage.

Run from the project root:
    python scripts/01_convert_training_to_yolo.py
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"

QUADRANT_IMGS = RAW / "training" / "training_data" / "quadrant" / "xrays"
ENUM_IMGS     = RAW / "training" / "training_data" / "quadrant_enumeration" / "xrays"
DISEASE_IMGS  = RAW / "training" / "training_data" / "quadrant-enumeration-disease" / "xrays"
VAL_IMGS      = RAW / "validation" / "validation_data" / "quadrant_enumeration_disease" / "xrays"
TEST_IMGS     = RAW / "test" / "disease" / "input"

QUADRANT_JSON = RAW / "training" / "training_data" / "quadrant" / "train_quadrant.json"
ENUM_JSON     = RAW / "training" / "training_data" / "quadrant_enumeration" / "train_quadrant_enumeration.json"
DISEASE_JSON  = RAW / "training" / "training_data" / "quadrant-enumeration-disease" / "train_quadrant_enumeration_disease.json"
VAL_JSON      = RAW / "validation_triple.json"


def force_symlink(target, link_path):
    """Create a symlink, replacing any existing file/symlink at link_path."""
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(target)


def coco_seg_to_yolo(segmentation, img_w, img_h):
    if not segmentation:
        return None
    poly = max(segmentation, key=len)
    coords = []
    for i in range(0, len(poly), 2):
        x = max(0.0, min(1.0, poly[i] / img_w))
        y = max(0.0, min(1.0, poly[i + 1] / img_h))
        coords.append(f"{x:.6f}")
        coords.append(f"{y:.6f}")
    return " ".join(coords)


def build_index(data):
    img_index = {img["id"]: img for img in data["images"]}
    ann_index = defaultdict(list)
    for ann in data["annotations"]:
        ann_index[ann["image_id"]].append(ann)
    return img_index, ann_index


def write_yolo_labels(data, label_field, out_label_dir, img_src_dir, out_img_dir,
                      img_prefix=""):
    out_label_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir.mkdir(parents=True, exist_ok=True)

    img_index, ann_index = build_index(data)
    written = 0

    for img_id, img_meta in img_index.items():
        fname = img_meta["file_name"]
        img_w = img_meta["width"]
        img_h = img_meta["height"]

        stem = Path(fname).stem
        out_stem = f"{img_prefix}{stem}" if img_prefix else stem
        label_path = out_label_dir / f"{out_stem}.txt"
        img_dst = out_img_dir / f"{out_stem}.png"

        anns = ann_index.get(img_id, [])
        lines = []

        for ann in anns:
            class_id = ann.get(label_field)
            if class_id is None:
                continue
            seg = ann.get("segmentation")
            if not seg:
                x, y, w, h = ann["bbox"]
                seg = [[x, y, x + w, y, x + w, y + h, x, y + h]]
            poly_str = coco_seg_to_yolo(seg, img_w, img_h)
            if poly_str:
                lines.append(f"{class_id} {poly_str}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        src_img = img_src_dir / fname
        if src_img.exists():
            force_symlink(src_img.resolve(), img_dst)
        else:
            print(f"  WARNING: image not found: {src_img}")

        written += 1

    print(f"  Wrote {written} label files")
    return written


def build_stage1():
    print("\n── Stage 1: Quadrant Detection ──────────────────────────────────")
    stage = PROCESSED / "stage1_quadrant"

    print("  [train] Quadrant tier (693 images)...")
    with open(QUADRANT_JSON) as f:
        q_data = json.load(f)
    write_yolo_labels(q_data, "category_id",
                      stage / "labels" / "train", QUADRANT_IMGS,
                      stage / "images" / "train", img_prefix="q_")

    print("  [train] Disease tier (705 images)...")
    with open(DISEASE_JSON) as f:
        d_data = json.load(f)
    write_yolo_labels(d_data, "category_id_1",
                      stage / "labels" / "train", DISEASE_IMGS,
                      stage / "images" / "train", img_prefix="d_")

    print("  [val] Validation (50 images)...")
    with open(VAL_JSON) as f:
        val_data = json.load(f)
    write_yolo_labels(val_data, "category_id_1",
                      stage / "labels" / "val", VAL_IMGS,
                      stage / "images" / "val")

    print(f"  Stage 1 complete -> {stage}")


def build_stage2():
    print("\n── Stage 2: Tooth Enumeration ───────────────────────────────────")
    stage = PROCESSED / "stage2_enumeration"

    print("  [train] Enumeration tier (634 images)...")
    with open(ENUM_JSON) as f:
        qe_data = json.load(f)
    write_yolo_labels(qe_data, "category_id_2",
                      stage / "labels" / "train", ENUM_IMGS,
                      stage / "images" / "train", img_prefix="e_")

    print("  [train] Disease tier (705 images)...")
    with open(DISEASE_JSON) as f:
        d_data = json.load(f)
    write_yolo_labels(d_data, "category_id_2",
                      stage / "labels" / "train", DISEASE_IMGS,
                      stage / "images" / "train", img_prefix="d_")

    print("  [val] Validation (50 images)...")
    with open(VAL_JSON) as f:
        val_data = json.load(f)
    write_yolo_labels(val_data, "category_id_2",
                      stage / "labels" / "val", VAL_IMGS,
                      stage / "images" / "val")

    print(f"  Stage 2 complete -> {stage}")


def build_stage3():
    print("\n── Stage 3: Diagnosis Detection ─────────────────────────────────")
    stage = PROCESSED / "stage3_disease"

    print("  [train] Disease tier (705 images)...")
    with open(DISEASE_JSON) as f:
        d_data = json.load(f)
    write_yolo_labels(d_data, "category_id_3",
                      stage / "labels" / "train", DISEASE_IMGS,
                      stage / "images" / "train")

    print("  [val] Validation (50 images)...")
    with open(VAL_JSON) as f:
        val_data = json.load(f)
    write_yolo_labels(val_data, "category_id_3",
                      stage / "labels" / "val", VAL_IMGS,
                      stage / "images" / "val")

    print("  [test] Symlinking test images (250)...")
    test_img_out = stage / "images" / "test"
    test_img_out.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_file in sorted(TEST_IMGS.glob("*.png")):
        dst = test_img_out / img_file.name
        force_symlink(img_file.resolve(), dst)
        count += 1
    print(f"  Symlinked {count} test images")
    (stage / "labels" / "test").mkdir(parents=True, exist_ok=True)

    print(f"  Stage 3 complete -> {stage}")


if __name__ == "__main__":
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output root  : {PROCESSED}")
    build_stage1()
    build_stage2()
    build_stage3()
    print("\nDone. Next: run 02_convert_test_to_yolo.py")
