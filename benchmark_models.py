import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from keras.applications.inception_v3 import preprocess_input


MODEL_CANDIDATES = [
    {
        "name": "optimized_plain",
        "path": r"D:\PROJECT\fruitvegapp_\fruitvegappoptimizedmodels.keras",
        "input_mode": "inception_external",
        "custom_objects": None,
    },
    {
        "name": "optimized_with_lambda",
        "path": r"D:\PROJECT\fruitvegapp_\fruitvegapp__optimizedmodels.keras",
        "input_mode": "raw_0_255",
        "custom_objects": {"preprocess_input": preprocess_input},
    },
]

DEFAULT_CLASS_NAMES = [
    "apple",
    "banana",
    "beetroot",
    "bell pepper",
    "cabbage",
    "capsicum",
    "carrot",
    "cauliflower",
    "chilli pepper",
    "corn",
    "cucumber",
    "eggplant",
    "garlic",
    "ginger",
    "grapes",
    "jalepeno",
    "kiwi",
    "lemon",
    "lettuce",
    "mango",
    "onion",
    "orange",
    "paprika",
    "pear",
    "peas",
    "pineapple",
    "pomegranate",
    "potato",
    "raddish",
    "soy beans",
    "spinach",
    "sweetcorn",
    "sweetpotato",
    "tomato",
    "turnip",
    "watermelon",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_class_names(dataset_root: Path):
    json_candidates = [
        dataset_root / "class_indices.json",
        Path(__file__).resolve().parent / "class_indices.json",
        Path(__file__).resolve().parent / "labels.json",
    ]
    for p in json_candidates:
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data and isinstance(next(iter(data.values())), int):
                ordered = [k for k, _ in sorted(data.items(), key=lambda x: x[1])]
            else:
                ordered = [data[str(i)] for i in range(len(data))]
            if ordered:
                return ordered, str(p)
        except Exception:
            pass
    return DEFAULT_CLASS_NAMES, "default_hardcoded"


def gather_samples(dataset_root: Path, class_to_idx: dict):
    samples = []
    for class_dir in sorted(dataset_root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name.strip().lower()
        if class_name not in class_to_idx:
            continue
        y = class_to_idx[class_name]
        for fp in class_dir.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in IMAGE_EXTS:
                samples.append((fp, y))
    return samples


def build_tta_views(image: Image.Image):
    image = image.convert("RGB")
    w, h = image.size
    crop_margin = max(int(min(w, h) * 0.06), 8)
    views = [image, image.transpose(Image.FLIP_LEFT_RIGHT)]
    if w > 2 * crop_margin and h > 2 * crop_margin:
        center_crop = image.crop((crop_margin, crop_margin, w - crop_margin, h - crop_margin))
        views.append(center_crop)
        views.append(center_crop.transpose(Image.FLIP_LEFT_RIGHT))
    return [v.resize((299, 299)).convert("RGB") for v in views]


def prepare_input(view: Image.Image, input_mode: str):
    arr = np.array(view, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    if input_mode == "inception_external":
        return preprocess_input(arr)
    return arr


def macro_f1(y_true, y_pred, n_classes: int):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    f1s = []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def evaluate_model(model_entry, samples, n_classes):
    model = model_entry["model"]
    y_true = []
    y_pred = []
    confidences = []
    latencies = []

    for fp, y in samples:
        try:
            image = Image.open(fp)
            views = build_tta_views(image)
            preds = []
            start = time.perf_counter()
            for view in views:
                x = prepare_input(view, model_entry["input_mode"])
                preds.append(model.predict(x, verbose=0)[0])
            pred = np.mean(preds, axis=0)
            elapsed = (time.perf_counter() - start) * 1000.0

            y_true.append(y)
            y_pred.append(int(np.argmax(pred)))
            confidences.append(float(np.max(pred)) * 100.0)
            latencies.append(elapsed)
        except Exception:
            continue

    if not y_true:
        return {
            "name": model_entry["name"],
            "num_samples": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "avg_confidence": 0.0,
            "avg_latency_ms": 0.0,
        }

    y_true_np = np.array(y_true, dtype=np.int32)
    y_pred_np = np.array(y_pred, dtype=np.int32)
    accuracy = float((y_true_np == y_pred_np).mean())
    f1 = macro_f1(y_true_np, y_pred_np, n_classes=n_classes)
    return {
        "name": model_entry["name"],
        "num_samples": int(len(y_true)),
        "accuracy": accuracy,
        "macro_f1": f1,
        "avg_confidence": float(np.mean(confidences)),
        "avg_latency_ms": float(np.mean(latencies)),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark fruit/veg models on folder dataset.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset root with class subfolders. Example: test/apple/*.jpg",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset).resolve()
    if not dataset_root.exists():
        raise SystemExit(f"Dataset path not found: {dataset_root}")

    class_names, class_source = load_class_names(dataset_root)
    class_to_idx = {c.strip().lower(): i for i, c in enumerate(class_names)}
    samples = gather_samples(dataset_root, class_to_idx)
    if not samples:
        raise SystemExit(
            "No images found in class subfolders. Expected: dataset/class_name/image.jpg"
        )

    print(f"Dataset: {dataset_root}")
    print(f"Label source: {class_source}")
    print(f"Classes: {len(class_names)}")
    print(f"Samples found: {len(samples)}")

    loaded = []
    for c in MODEL_CANDIDATES:
        if not os.path.exists(c["path"]):
            print(f"SKIP {c['name']}: file not found -> {c['path']}")
            continue
        try:
            model = load_model(c["path"], compile=False, custom_objects=c["custom_objects"])
            loaded.append(
                {
                    "name": c["name"],
                    "model": model,
                    "input_mode": c["input_mode"],
                    "path": c["path"],
                }
            )
            print(f"Loaded: {c['name']}")
        except Exception as e:
            print(f"SKIP {c['name']}: failed to load -> {e}")

    if not loaded:
        raise SystemExit("No model could be loaded.")

    print("\nResults:")
    print("name,num_samples,accuracy,macro_f1,avg_confidence,avg_latency_ms")
    for model_entry in loaded:
        out_dim = int(model_entry["model"].output_shape[-1])
        if out_dim != len(class_names):
            print(
                f"{model_entry['name']},0,0,0,0,0  # class mismatch: model={out_dim}, labels={len(class_names)}"
            )
            continue
        result = evaluate_model(model_entry, samples, n_classes=len(class_names))
        print(
            f"{result['name']},{result['num_samples']},"
            f"{result['accuracy']:.4f},{result['macro_f1']:.4f},"
            f"{result['avg_confidence']:.2f},{result['avg_latency_ms']:.2f}"
        )


if __name__ == "__main__":
    main()
