#convert_to_ls.py
"""
convert_to_ls.py

Builds Label Studio (LS) task JSON that:
- Uses AWS Textract WORD boxes as the sole source of geometry (atomic, consistent).
- Maps our existing model predictions onto those WORD boxes (pre-annotations).
- Produces one LS task per page image.

Inputs:
  1) textract_json: AWS Textract output (full or reduced) containing WORD blocks with normalized bounding boxes.
  2) model_json: our model predictions .
  3) image_source:
       - Either a single image path (for single-page), or
       - A template string containing "{page}" placeholder, e.g. "/path/doc_page_{page}.png"
         (the script will substitute page numbers from Textract).

Usage:
  python convert_to_ls.py <textract.json> <model.json> <image_path_or_template> [--out ls_tasks.json]

Notes:
- Coordinates are expected normalized [0..1] like Textract. LS expects percentages [0..100].
- Model predictions are aligned to Textract WORDs using: (a) center-in-box OR (b) IoU threshold.
- For each WORD, we create:
    - a rectangle region (label "_token")  -> from_name="word_boxes"
    - a per-region Choices control          -> from_name="field"
    - a per-region TextArea                 -> from_name="value"
  All three share the SAME region id so they are bound together in LS.

- If a model item spans multiple boxes, each matched word token gets the same field/value prefill;
  annotators can merge/adjust by selecting the right tokens.

- If a model item has no coordinates, we skip spatial prefill (still possible to add task-level meta if needed).
"""

import json
import sys
import uuid
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# ----------------------------
# Helpers: geometry & matching
# ----------------------------

def iou(a, b) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

def center_inside(child, parent) -> bool:
    cx = child[0] + child[2] / 2.0
    cy = child[1] + child[3] / 2.0
    return (parent[0] <= cx <= parent[0] + parent[2]) and (parent[1] <= cy <= parent[1] + parent[3])

def norm_to_pct(bbox):
    # Textract normalized [0..1] -> LS wants percent [0..100]
    x, y, w, h = bbox
    return {
        "x": x * 100.0,
        "y": y * 100.0,
        "width": w * 100.0,
        "height": h * 100.0,
        "rotation": 0
    }

# ----------------------------
# Textract parsing
# ----------------------------

def extract_words_from_textract(textract: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Returns a dict: page_number -> list of words
    Each word: {
      "text": str,
      "bbox": (left, top, width, height),  # normalized 0..1
      "page": int
    }
    Supports:
      - Full Textract JSON with Blocks
      - Reduced custom JSON with items containing "text"/"word" + coords
    """
    pages: Dict[int, List[Dict[str, Any]]] = {}

    if "Blocks" in textract:
        # Full AWS Textract structure
        for b in textract["Blocks"]:
            if b.get("BlockType") == "WORD" and "Geometry" in b and "BoundingBox" in b["Geometry"]:
                bb = b["Geometry"]["BoundingBox"]
                page = b.get("Page", 1)
                wobj = {
                    "text": b.get("Text", ""),
                    "bbox": (bb.get("Left", 0.0), bb.get("Top", 0.0), bb.get("Width", 0.0), bb.get("Height", 0.0)),
                    "page": page
                }
                pages.setdefault(page, []).append(wobj)
        return pages

    # Fallback: try to infer from simplified structures
    # Expect a list of dicts with "page", "text"/"word", and normalized coords {left, top, width, height}
    if isinstance(textract, list):
        for item in textract:
            page = int(item.get("page", 1) or 1)
            text = item.get("text") or item.get("word") or item.get("value") or ""
            coords = item.get("bbox") or item.get("coordinates") or item.get("Geometry", {}).get("BoundingBox")
            if not coords:
                # try flattened keys
                coords = {
                    "Left": item.get("left", item.get("Left", 0.0)),
                    "Top": item.get("top", item.get("Top", 0.0)),
                    "Width": item.get("width", item.get("Width", 0.0)),
                    "Height": item.get("height", item.get("Height", 0.0)),
                }
            left = coords.get("Left", coords.get("left", 0.0))
            top = coords.get("Top", coords.get("top", 0.0))
            width = coords.get("Width", coords.get("width", 0.0))
            height = coords.get("Height", coords.get("height", 0.0))
            wobj = {"text": text, "bbox": (left, top, width, height), "page": page}
            pages.setdefault(page, []).append(wobj)
        return pages

    raise ValueError("Unsupported Textract JSON shape. Provide full Textract with Blocks or a list of word items.")

# ----------------------------
# Model parsing
# ----------------------------

def extract_predictions(model_json: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalizes your model predictions into:
    {
      "key": str,                   # field label, e.g. "bank"
      "value": str,                 # predicted text value
      "page": int,                  # page number (if any; default 1)
      "boxes": [(l,t,w,h), ...],    # normalized boxes (valueCoordinates)
      "score": float or None
    }
    """
    preds = []
    for item in model_json:
        key = item.get("key", "unknown")
        value = item.get("value", "")
        page = int(item.get("page", 1) or 1)
        score = item.get("valueConfidence", None)
        boxes = []
        for c in item.get("valueCoordinates") or []:
            boxes.append((c.get("left", 0.0), c.get("top", 0.0), c.get("width", 0.0), c.get("height", 0.0)))
        preds.append({"key": key, "value": value, "page": page, "boxes": boxes, "score": score})
    return preds

def union_box(boxes: List[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    return (x1, y1, x2 - x1, y2 - y1)

# ----------------------------
# Alignment: model → words
# ----------------------------

def align_predictions_to_words(
    pages_words: Dict[int, List[Dict[str, Any]]],
    preds: List[Dict[str, Any]],
    iou_thresh: float = 0.2
) -> Dict[int, Dict[int, List[Dict[str, Any]]]]:
    """
    For each page, for each prediction, find matching Textract words.
    Returns: page -> { pred_index: [word_indices...] }
    """
    aligned: Dict[int, Dict[int, List[int]]] = {}
    for p_i, page in enumerate(sorted(pages_words.keys())):
        aligned.setdefault(page, {})

    for pi, pred in enumerate(preds):
        page = int(pred["page"] or 1)
        words = pages_words.get(page, [])
        if not words:
            continue

        if pred["boxes"]:
            ubox = union_box(pred["boxes"])
            matches = []
            for wi, w in enumerate(words):
                wb = w["bbox"]
                # match policy: center-inside OR IoU >= threshold
                if center_inside(wb, ubox) or iou(wb, ubox) >= iou_thresh:
                    matches.append(wi)
            if matches:
                aligned.setdefault(page, {})[pi] = matches
        else:
            # No coords — cannot spatially align; skip (or handle task-level)
            pass

    return aligned

# ----------------------------
# LS task building
# ----------------------------

def make_ls_tasks(
    pages_words: Dict[int, List[Dict[str, Any]]],
    preds: List[Dict[str, Any]],
    image_source: str
) -> List[Dict[str, Any]]:
    """
    Builds one LS task per page.
    Controls used in LS config (must match names):
      - Image name="document"
      - RectangleLabels name="word_boxes" (token rectangles)
      - Choices name="field" perRegion=true   (semantic label)
      - TextArea name="value" perRegion=true  (corrected text)
      - (Optional) TextArea name="ocr" perRegion=true (raw OCR word for reference)
    """
    tasks = []
    aligned = align_predictions_to_words(pages_words, preds)

    # Build a set of known field labels from model keys (for validation / completeness)
    field_labels = sorted({p["key"] for p in preds if p.get("key")})

    for page in sorted(pages_words.keys()):
        words = pages_words[page]
        # Image path resolution
        if "{page}" in image_source:
            image_path = image_source.format(page=page)
        else:
            image_path = image_source

        # Prepare predictions/result arrays
        result_items: List[Dict[str, Any]] = []

        # Create a stable UUID for each word region
        word_region_ids: List[str] = [str(uuid.uuid4()) for _ in words]

        # 1) Emit one rectangle per WORD (label="_token"), plus per-region OCR text
        for wi, w in enumerate(words):
            rid = word_region_ids[wi]
            bbox_pct = norm_to_pct(w["bbox"])

            rect = {
                "id": rid,
                "type": "rectanglelabels",
                "value": {**bbox_pct, "rectanglelabels": ["_token"]},
                "to_name": "document",
                "from_name": "word_boxes",
                "score": None,
                "origin": "manual",  # still shows as prediction; LS treats as suggestion
            }
            result_items.append(rect)

            # Attach raw OCR text for reference (locked by default; annotators edit "value" instead)
            ocr_text = {
                "id": rid,
                "type": "textarea",
                "value": {"text": [w.get("text", "")]},
                "to_name": "document",
                "from_name": "ocr",
                "readonly": True
            }
            result_items.append(ocr_text)

        # 2) Pre-fill model predictions onto matched words
        page_align = aligned.get(page, {})
        for pi, word_idxs in page_align.items():
            pred = preds[pi]
            # (Annotators can prune/merge by selecting only the intended tokens)
            for wi in word_idxs:
                rid = word_region_ids[wi]

                # Per-region semantic label (Choices)
                field_choice = {
                    "id": rid,
                    "type": "choices",
                    "value": {"choices": [pred["key"]]},
                    "to_name": "document",
                    "from_name": "field",
                    "score": pred.get("score")
                }
                result_items.append(field_choice)

                # Per-region value textarea (prefill model text for quick correction)
                val_text = {
                    "id": rid,
                    "type": "textarea",
                    "value": {"text": [pred.get("value", "")]},
                    "to_name": "document",
                    "from_name": "value",
                    "score": pred.get("score")
                }
                result_items.append(val_text)

        task = {
            "data": {
                "image": image_path,
                # Optional: keep field catalog to drive dynamic UIs / validators
                "field_labels": field_labels
            },
            "predictions": [
                {
                    "model_version": "v1",
                    "result": result_items
                }
            ]
        }
        tasks.append(task)

    return tasks

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("textract_json", type=Path)
    ap.add_argument("model_json", type=Path)
    ap.add_argument("image", type=str,
                    help="Single image path or a template with {page}, e.g. '/images/doc_page_{page}.png'")
    ap.add_argument("--out", type=Path, default=Path("ls_tasks.json"))
    ap.add_argument("--iou", type=float, default=0.20, help="IoU threshold for aligning predictions to words")
    args = ap.parse_args()

    with open(args.textract_json, "r") as f:
        textract = json.load(f)
    with open(args.model_json, "r") as f:
        model = json.load(f)

    pages_words = extract_words_from_textract(textract)
    preds = extract_predictions(model)

    tasks = make_ls_tasks(pages_words, preds, args.image)

    with open(args.out, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"Wrote {len(tasks)} LS task(s) to {args.out}")

if __name__ == "__main__":
    main()
