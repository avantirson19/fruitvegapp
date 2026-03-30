import os
from functools import lru_cache

import numpy as np
from PIL import Image


YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "fruit_detector.pt")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.25"))


@lru_cache(maxsize=1)
def get_detector_status():
    try:
        from ultralytics import YOLO
    except Exception as e:
        return {
            "available": False,
            "error": f"ultralytics import failed: {e}",
            "model_path": YOLO_MODEL_PATH,
        }

    if not os.path.exists(YOLO_MODEL_PATH):
        return {
            "available": False,
            "error": f"YOLO model file not found: {YOLO_MODEL_PATH}",
            "model_path": YOLO_MODEL_PATH,
        }

    try:
        model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        return {
            "available": False,
            "error": f"Failed to load YOLO model: {e}",
            "model_path": YOLO_MODEL_PATH,
        }

    return {
        "available": True,
        "error": None,
        "model_path": YOLO_MODEL_PATH,
        "model": model,
    }


def detect_items(image: Image.Image, conf_threshold: float = YOLO_CONFIDENCE):
    status = get_detector_status()
    if not status["available"]:
        return [], status

    rgb = image.convert("RGB")
    image_np = np.asarray(rgb)
    results = status["model"].predict(image_np, conf=conf_threshold, verbose=False)
    if not results:
        return [], status

    result = results[0]
    names = getattr(result, "names", {})
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return [], status

    detections = []
    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        left, top, right, bottom = [int(v) for v in xyxy]
        confidence = float(box.conf[0].item())
        class_id = int(box.cls[0].item())
        label = names.get(class_id, str(class_id))

        crop = rgb.crop((left, top, right, bottom))
        detections.append(
            {
                "bbox": (left, top, right, bottom),
                "confidence": confidence,
                "label": str(label),
                "crop": crop,
            }
        )

    detections.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return detections, status
