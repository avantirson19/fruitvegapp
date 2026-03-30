import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train a YOLO fruit detector.")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to YOLO data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model checkpoint")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cpu or 0")
    parser.add_argument("--project", default="runs/detect", help="Output project directory")
    parser.add_argument("--name", default="fruit_detector", help="Run name")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise SystemExit(f"Failed to import ultralytics: {e}")

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset config not found: {data_path}")

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
