# src/prepare_nitk_dataset.py

import os
import csv

from .config import PROJECT_ROOT, RAW_DIR, PROCESSED_DIR

# folder where you unzipped the dataset
NITK_ROOT = os.path.join(RAW_DIR, "nitk_eye_tracker","data")

# subfolder under processed where we'll copy images
OUTPUT_DIR = os.path.join(PROCESSED_DIR, "nitk")

os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_LABELS = {"left", "center", "right"}

def normalize_label(raw_label: str):
    """Map raw folder names to our coarse labels."""
    raw = raw_label.lower()

    if "left" in raw:
        return "left"
    if "right" in raw:
        return "right"
    if "straight" in raw:
        return "center"

    # up/down/other -> skip by returning None
    return None

def main():
    rows = []
    count_total = 0
    count_used = 0

    for root, dirs, files in os.walk(NITK_ROOT):
        # assume label is the immediate folder name under NITK_ROOT
        rel = os.path.relpath(root, NITK_ROOT)
        parts = rel.split(os.sep)
        if len(parts) == 1 and parts[0] == ".":
            # this is NITK_ROOT itself
            label_folder = None
        else:
            label_folder = parts[0]

        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            count_total += 1

            if label_folder is None:
                # if images are directly under root without label folder
                # we can't infer label, so skip
                continue

            norm_label = normalize_label(label_folder)
            if norm_label is None or norm_label not in ALLOWED_LABELS:
                continue

            # build source path
            src_path = os.path.join(root, fname)

            # build destination path inside data/processed/nitk
            # we keep label in the filename for clarity
            out_fname = f"{norm_label}_{fname}"
            dst_path = os.path.join(OUTPUT_DIR, out_fname)

            # copy file
            if not os.path.exists(dst_path):
                with open(src_path, "rb") as fsrc, open(dst_path, "wb") as fdst:
                    fdst.write(fsrc.read())

            # store relative path from project root
            rel_path = os.path.relpath(dst_path, PROJECT_ROOT)

            # subject_id is unknown here; we can set a placeholder or try to parse from filename
            subject_id = "nitk"

            rows.append((rel_path.replace("\\", "/"), norm_label, subject_id))
            count_used += 1

    # write metadata.csv at data/metadata.csv
    metadata_path = os.path.join(os.path.dirname(RAW_DIR), "metadata.csv")
    with open(metadata_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "subject_id"])
        for r in rows:
            writer.writerow(r)

    print(f"Total image files found: {count_total}")
    print(f"Images used (left/center/right): {count_used}")
    print(f"Metadata written to: {metadata_path}")

if __name__ == "__main__":
    main()
