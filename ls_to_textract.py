"""
ls_to_textract.py
-----------------
Convert LS output JSON back into Textract-style JSON (original ML schema).
Usage:
    python ls_to_textract.py <ls_output_json>
"""

import sys
import os
import json


def LSToTextract(ls_json_file: str, original_json_file: str) -> str:
    """
    Convert LS export JSON back to Textract-style JSON.

    Args:
        ls_json_file (str): Path to LS output JSON.
        original_json_file (str): Path to the original model JSON file.

    Returns:
        str: Path to the created Textract-style JSON file.
    """
    if not os.path.exists(ls_json_file):
        raise FileNotFoundError(f"LS JSON not found: {ls_json_file}")
    if not os.path.exists(original_json_file):
        raise FileNotFoundError(f"Original model JSON not found: {original_json_file}")

    with open(ls_json_file, "r", encoding="utf-8") as f:
        ls_data = json.load(f)
    with open(original_json_file, "r", encoding="utf-8") as f:
        model_data = json.load(f)

    # Map back using keys
    key_to_item = {item["key"]: item for item in model_data}

    for task in ls_data:
        key = task.get("data", {}).get("key")
        value = task.get("data", {}).get("value", "")
        results = task.get("annotations", [{}])[0].get("result", [])

        if key in key_to_item:
            key_to_item[key]["value"] = value
            key_to_item[key]["valueCoordinates"] = []

            for res in results:
                val = res.get("value", {})
                coord = {
                    "left": val["x"] / 100,
                    "top": val["y"] / 100,
                    "width": val["width"] / 100,
                    "height": val["height"] / 100,
                }
                key_to_item[key]["valueCoordinates"].append(coord)

    out_file = os.path.splitext(ls_json_file)[0] + "_back.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(list(key_to_item.values()), f, indent=2)

    return out_file


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ls_to_textract.py <ls_output_json> <original_model_json>")
        sys.exit(1)

    ls_file = sys.argv[1]
    original_json = sys.argv[2]

    try:
        out_file = LSToTextract(ls_file, original_json)
        print(f"✅ Success. Textract-style JSON saved at: {out_file}")
    except Exception as e:
        print(f"❌ Failed: {e}")
