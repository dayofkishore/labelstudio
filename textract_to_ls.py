"""
textract_to_ls.py
-----------------
Convert your model JSON (Textract-style) into LS-compatible JSON tasks.
Usage:
    python textract_to_ls.py <model_json_file>
"""

import sys
import os
import json


def textractToLS(json_file: str) -> str:
    """
    Convert model JSON with textract-style coordinates into LS-style tasks.

    Args:
        json_file (str): Path to model JSON file.

    Returns:
        str: Path to the created LS JSON file.
    """
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        model_data = json.load(f)

    ls_tasks = []

    # Expect model_data to be a list of items with {page, key, value, valueCoordinates}
    for item in model_data:
        page_num = item.get("page", 1)
        image_file = f"{os.path.splitext(os.path.basename(json_file))[0]}_{page_num}.png"

        # Collect predicted bounding boxes
        results = []
        for coord in item.get("valueCoordinates", []):
            results.append({
                "from_name": "bbox",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": coord["left"] * 100,
                    "y": coord["top"] * 100,
                    "width": coord["width"] * 100,
                    "height": coord["height"] * 100,
                    "rotation": 0,
                    "labels": ["value"]
                }
            })

        # Prepare table rows for LS <Table>
        pairs = []
        for kv in item.get("pairs", []):
            pairs.append({
                "key": kv.get("key", ""),
                "value": kv.get("value", "")
            })

        # If this item itself is just one pair, fall back to key/value
        if not pairs and "key" in item:
            pairs.append({
                "key": item.get("key", ""),
                "value": item.get("value", "")
            })

        task = {
            "data": {
                "image": image_file,
                "pairs": pairs
            },
            "predictions": [
                {
                    "result": results
                }
            ]
        }
        ls_tasks.append(task)

    out_file = os.path.splitext(json_file)[0] + "_textract.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ls_tasks, f, indent=2)

    return out_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python textract_to_ls.py <model_json_file>")
        sys.exit(1)

    model_json = sys.argv[1]
    try:
        out_file = textractToLS(model_json)
        print(f"✅ Success. LS JSON saved at: {out_file}")
    except Exception as e:
        print(f"❌ Failed: {e}")
