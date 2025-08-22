# convert_to_textract.py
"""
Convert Label Studio export JSON back into our model training schema (textract schema).

"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

def pct_to_norm(rect):
    """Convert LS percent coords [0..100] back to normalized [0..1]."""
    return {
        "left": rect["x"] / 100.0,
        "top": rect["y"] / 100.0,
        "width": rect["width"] / 100.0,
        "height": rect["height"] / 100.0,
    }

def extract_annotations(ls_export):
    """
    Convert LS tasks → training schema.
    """
    results = []

    for task in ls_export:
        page = 1  # default, can extend by parsing filename if multipage
        annos = task.get("annotations", [])
        if not annos:
            continue

        for anno in annos:
            # Group results by region id
            region_map = defaultdict(dict)
            for r in anno.get("result", []):
                rid = r.get("id")
                if not rid:
                    continue
                region_map[rid].setdefault("raw", []).append(r)

            for rid, bundle in region_map.items():
                choice = None
                value = None
                boxes = []

                for r in bundle["raw"]:
                    if r["type"] == "choices" and r["from_name"] == "field":
                        choice = r["value"]["choices"][0]
                    elif r["type"] == "textarea" and r["from_name"] == "value":
                        if r["value"].get("text"):
                            value = r["value"]["text"][0]
                    elif r["type"] == "rectanglelabels":
                        boxes.append(pct_to_norm(r["value"]))

                if choice and value and boxes:
                    results.append({
                        "key": choice,
                        "value": value,
                        "page": page,
                        "valueCoordinates": boxes,
                        "valueConfidence": 100.0,
                        "keyConfidence": 100,
                        "keyCoordinates": []
                    })

    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ls_export", type=Path, help="Label Studio export JSON")
    ap.add_argument("--out", type=Path, default=Path("retrain.json"))
    args = ap.parse_args()

    with open(args.ls_export, "r") as f:
        ls_export = json.load(f)

    retrain_data = extract_annotations(ls_export)

    with open(args.out, "w") as f:
        json.dump(retrain_data, f, indent=2)

    print(f"Wrote {len(retrain_data)} items to {args.out}")

if __name__ == "__main__":
    main()
