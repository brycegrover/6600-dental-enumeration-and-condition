#!/usr/bin/env python3
"""
Reorganizes the raw DENTEX HuggingFace download into the directory structure
expected by preprocessing.py and the conversion scripts.

Usage:
    1. Download the DENTEX dataset from HuggingFace:
       https://huggingface.co/datasets/ibrahimhamamci/DENTEX
    2. Unzip everything into a single folder (e.g. raw_DENTEX_data/)
    3. Run this script from the project root:
           python setup_raw_data.py <path_to_download>
       Example:
           python setup_raw_data.py raw_DENTEX_data

The script will create data/raw/ with the following structure:

    data/raw/
    ├── training/
    │   └── training_data/
    │       ├── quadrant/                          (xrays/ + train_quadrant.json)
    │       ├── quadrant_enumeration/              (xrays/ + train_quadrant_enumeration.json)
    │       ├── quadrant-enumeration-disease/       (xrays/ + train_quadrant_enumeration_disease.json)
    │       └── unlabelled/                         (xrays/)
    ├── validation/
    │   └── validation_data/
    │       └── quadrant_enumeration_disease/       (xrays/)
    ├── test/
    │   └── disease/
    │       ├── input/                              (test images)
    │       └── label/                              (test label JSONs)
    └── validation_triple.json

After running this script, run:
    python preprocessing.py
"""

import shutil
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python setup_raw_data.py <path_to_huggingface_download>")
        print("Example: python setup_raw_data.py raw_DENTEX_data")
        sys.exit(1)

    src = Path(sys.argv[1]).resolve()
    if not src.exists():
        print(f"ERROR: Source directory not found: {src}")
        sys.exit(1)

    project_root = Path(__file__).resolve().parent
    dst = project_root / "data" / "raw"

    if dst.exists():
        print(f"WARNING: {dst} already exists.")
        response = input("Overwrite? (y/n): ").strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(0)
        shutil.rmtree(dst)

    print(f"Source: {src}")
    print(f"Destination: {dst}")

    # (src relative path) -> (dst relative path)
    dir_moves = [
        ("training_data/quadrant",                       "training/training_data/quadrant"),
        ("training_data/quadrant_enumeration",           "training/training_data/quadrant_enumeration"),
        ("training_data/quadrant-enumeration-disease",   "training/training_data/quadrant-enumeration-disease"),
        ("training_data/unlabelled",                     "training/training_data/unlabelled"),
        ("validation_data/quadrant_enumeration_disease", "validation/validation_data/quadrant_enumeration_disease"),
        ("disease",                                      "test/disease"),
    ]

    file_moves = [
        ("validation_triple.json", "validation_triple.json"),
    ]

    # copy directories
    for src_rel, dst_rel in dir_moves:
        s = src / src_rel
        d = dst / dst_rel
        if not s.exists():
            print(f"  WARNING: source not found, skipping: {s}")
            continue
        print(f"  Copying {src_rel}  ->  {dst_rel}")
        shutil.copytree(s, d)

    # copy files
    for src_rel, dst_rel in file_moves:
        s = src / src_rel
        d = dst / dst_rel
        if not s.exists():
            print(f"  WARNING: source not found, skipping: {s}")
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Copying {src_rel}  ->  {dst_rel}")
        shutil.copy2(s, d)

    # verify
    print("Verification:")

    expected = [
        dst / "training" / "training_data" / "quadrant" / "train_quadrant.json",
        dst / "training" / "training_data" / "quadrant" / "xrays",
        dst / "training" / "training_data" / "quadrant_enumeration" / "train_quadrant_enumeration.json",
        dst / "training" / "training_data" / "quadrant_enumeration" / "xrays",
        dst / "training" / "training_data" / "quadrant-enumeration-disease" / "train_quadrant_enumeration_disease.json",
        dst / "training" / "training_data" / "quadrant-enumeration-disease" / "xrays",
        dst / "validation" / "validation_data" / "quadrant_enumeration_disease" / "xrays",
        dst / "validation_triple.json",
        dst / "test" / "disease" / "input",
        dst / "test" / "disease" / "label",
    ]

    all_ok = True
    for p in expected:
        exists = p.exists()
        if not exists:
            all_ok = False
        print(f"  {'OK' if exists else 'MISSING'}  {p.relative_to(dst)}")

    if all_ok:
        print("All OK — data is ready.")
        print("Next: python preprocessing.py")
    else:
        print("Some files missing — check the warnings above.")


if __name__ == "__main__":
    main()
