 #!/usr/bin/env python3

import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_LABEL_DIR = PROJECT_ROOT / "data" / "raw" / "test" / "disease" / "label"
OUT_LABEL_DIR = PROJECT_ROOT / "data" / "processed" / "stage3_disease" / "labels" / "test"

# Turkish term -> disease class ID
TURKISH_TO_CLASS = {
    "\u00e7\u00fcr\u00fck": 1, # çürük:Caries
    "k\u00fcretaj": 3, # küretaj:Deep Caries
    "kanal": 2, # kanal:Periapical Lesion
    "lezyon": 2, # lezyon:Periapical Lesion
    "g\u00f6m\u00fcl\u00fc": 0, # gömülü:Impacted
}

# Terms with no training class equivalent — skipped
SKIP_TERMS = {
    "saglam", # healthy
    "\u00e7ekim", # çekim:extraction
    "k\u0131r\u0131k", # kırık:broken
}

# function to parse a label string like "quadrant-enumeration-disease-16-çürük" into (class_id, fdi)
def parse_label(label_str):
    parts = label_str.split("-")
    if len(parts) < 3:
        return None
    fdi_str = parts[-1]
    term = parts[-2]
    if term in SKIP_TERMS:
        return None
    class_id = TURKISH_TO_CLASS.get(term)
    if class_id is None:
        print(f"  WARNING: unknown term '{term}' in '{label_str}'")
        return None
    try:
        fdi = int(fdi_str)
    except ValueError:
        print(f"  WARNING: bad FDI '{fdi_str}' in '{label_str}'")
        return None
    return class_id, fdi

# function to convert LabelMe polygon points to YOLO format
def poly_to_yolo(points, img_w, img_h):
    """Convert LabelMe [[x,y],...] to normalized YOLO string."""
    coords = []
    for x, y in points:
        nx = max(0.0, min(1.0, x / img_w))
        ny = max(0.0, min(1.0, y / img_h))
        coords.append(f"{nx:.6f}")
        coords.append(f"{ny:.6f}")
    return " ".join(coords)

# function to read each test label JSON, convert its annotations to YOLO format, and write a .txt file
def convert_test_labels():
    OUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    json_files = sorted(TEST_LABEL_DIR.glob("*.json"))
    print(f"Found {len(json_files)} test label files")

    total_anns = skipped_anns = written_files = 0

    # for each JSON file, read the annotations, convert to YOLO format, and write to a .txt file
    for json_path in json_files:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")
        if not img_w or not img_h:
            print(f"  WARNING: missing dims in {json_path.name}")
            continue

        lines = []
        # for each annotation, parse the label and convert the polygon to YOLO format
        for shape in data.get("shapes", []):
            total_anns += 1
            if shape.get("shape_type") != "polygon":
                skipped_anns += 1
                continue
            parsed = parse_label(shape["label"])
            if parsed is None:
                skipped_anns += 1
                continue
            class_id, fdi = parsed
            poly_str = poly_to_yolo(shape["points"], img_w, img_h)
            lines.append(f"{class_id} {poly_str}")

        out_path = OUT_LABEL_DIR / (json_path.stem + ".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        written_files += 1
    # summary
    kept = total_anns - skipped_anns

# function to count class distribution in the converted test labels and print it
def print_class_distribution():
    if not OUT_LABEL_DIR.exists():
        return
    class_names = {0: "Impacted", 1: "Caries", 2: "Periapical Lesion", 3: "Deep Caries"}
    counts = Counter()
    for txt in OUT_LABEL_DIR.glob("*.txt"):
        for line in txt.read_text().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    print("Test class distribution:")
    for cid in sorted(counts):
        print(f"  [{cid}] {class_names.get(cid, '?'):<20}: {counts[cid]}")

# main function to run the conversion and print the class distribution
if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    convert_test_labels()
    print_class_distribution()
