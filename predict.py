import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image
import numpy as np
import json
import os
import urllib.request

BASE_DIR = os.path.dirname(__file__)
HF_MODEL_URL = os.getenv(
    "HF_MODEL_URL",
    "https://huggingface.co/avantison19/fruitvegmodel/resolve/main/fruitvegappoptimizedmodels.keras",
)
FORCE_MODEL_REFRESH = os.getenv("FORCE_MODEL_REFRESH", "1") == "1"


def _ensure_model_file(local_path: str, download_url: str):
    if os.path.exists(local_path) and not FORCE_MODEL_REFRESH:
        return
    try:
        urllib.request.urlretrieve(download_url, local_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download model from {download_url}: {e}")


def _load_model_compat(path, custom_objects=None):
    # Try permissive load mode first for cross-version compatibility on cloud.
    try:
        return load_model(
            path,
            compile=False,
            custom_objects=custom_objects,
            safe_mode=False,
        )
    except TypeError:
        # Older loaders may not support safe_mode.
        return load_model(path, compile=False, custom_objects=custom_objects)
    except Exception as e:
        message = str(e)
        if "expects 1 input(s), but it received 2 input tensors" in message:
            raise RuntimeError(
                "This saved model still cannot be deserialized under the deployed "
                f"TensorFlow/Keras runtime ({tf.__version__}). The dependency install is "
                "now correct; the next fix is to re-export the model from the original "
                "training environment or provide a cloud-compatible model file."
            ) from e
        raise


MODEL_CANDIDATES = [
    {
        "name": "optimized_plain",
        "path": os.path.join(BASE_DIR, "fruitvegappoptimizedmodels.keras"),
        "input_mode": "inception_external",
        "custom_objects": None,
        "download_url": HF_MODEL_URL,
        "required": True,
    },
    {
        "name": "optimized_with_lambda",
        "path": os.path.join(BASE_DIR, "fruitvegapp__optimizedmodels.keras"),
        "input_mode": "raw_0_255",
        "custom_objects": {"preprocess_input": preprocess_input},
        "download_url": None,
        "required": False,
    },
]
LOW_CONFIDENCE_THRESHOLD = 18.0

loaded_models = []
load_errors = []
for candidate in MODEL_CANDIDATES:
    try:
        if candidate["download_url"]:
            _ensure_model_file(candidate["path"], candidate["download_url"])
        elif not os.path.exists(candidate["path"]) and not candidate["required"]:
            continue
        loaded_model = _load_model_compat(
            candidate["path"], custom_objects=candidate["custom_objects"]
        )
        loaded_models.append(
            {
                "name": candidate["name"],
                "model": loaded_model,
                "input_mode": candidate["input_mode"],
            }
        )
    except Exception as e:
        load_errors.append(f"{candidate['name']}: {e}")


def get_model_status():
    return {"loaded_count": len(loaded_models), "errors": load_errors}

DEFAULT_CLASS_NAMES = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
    'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]


def _load_class_names():
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "class_indices.json"),
        os.path.join(os.path.dirname(__file__), "labels.json"),
    ]
    for path in possible_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Supports {"apple": 0, ...} or {"0": "apple", ...}
            if data and isinstance(next(iter(data.values())), int):
                ordered = [name for name, _ in sorted(data.items(), key=lambda x: x[1])]
            else:
                ordered = [data[str(i)] for i in range(len(data))]

            if ordered:
                return ordered
        except Exception:
            pass
    return DEFAULT_CLASS_NAMES


class_names = _load_class_names()


def _predict_with_inception_preprocess(image: Image.Image):
    img_array = np.expand_dims(np.array(image, dtype=np.float32), axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def _prepare_raw_0_255(image: Image.Image):
    return np.expand_dims(np.array(image, dtype=np.float32), axis=0)


def _build_tta_views(image: Image.Image):
    w, h = image.size
    crop_margin = int(min(w, h) * 0.06)
    crop_margin = max(crop_margin, 8)

    views = [
        image,
        image.transpose(Image.FLIP_LEFT_RIGHT),
    ]

    if w > 2 * crop_margin and h > 2 * crop_margin:
        center_crop = image.crop((crop_margin, crop_margin, w - crop_margin, h - crop_margin))
        views.append(center_crop.resize((299, 299)))
        views.append(center_crop.transpose(Image.FLIP_LEFT_RIGHT).resize((299, 299)))

    return [view.resize((299, 299)).convert("RGB") for view in views]


def _predict_model_with_tta(model_entry, views):
    preds = []
    for view in views:
        if model_entry["input_mode"] == "inception_external":
            x = _predict_with_inception_preprocess(view)
            mode = "inception_preprocess_input"
        else:
            x = _prepare_raw_0_255(view)
            mode = "raw_0_255 (lambda preprocess inside model)"
        preds.append(model_entry["model"].predict(x, verbose=0)[0])
    prediction = np.mean(preds, axis=0)
    return prediction, mode


def predict_image(image: Image.Image):
    if not loaded_models:
        detail = "; ".join(load_errors) if load_errors else "unknown load failure"
        raise RuntimeError(f"Model not loaded. Details: {detail}")

    for model_entry in loaded_models:
        if model_entry["model"].output_shape[-1] != len(class_names):
            raise RuntimeError(
                f"Class count mismatch in {model_entry['name']}: "
                f"model outputs {model_entry['model'].output_shape[-1]} classes, "
                f"but class_names has {len(class_names)}."
            )

    image = image.convert("RGB")
    views = _build_tta_views(image)
    best = None
    for model_entry in loaded_models:
        prediction, preprocessing_mode = _predict_model_with_tta(model_entry, views)
        confidence = float(np.max(prediction)) * 100.0
        pred_class = class_names[int(np.argmax(prediction))]
        if best is None or confidence > best["confidence"]:
            best = {
                "pred_class": pred_class,
                "confidence": confidence,
                "preprocessing_mode": preprocessing_mode,
                "model_name": model_entry["name"],
            }

    pred_class = best["pred_class"]
    confidence = best["confidence"]
    is_low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD
    return pred_class, confidence, is_low_confidence, best["preprocessing_mode"], best["model_name"]
