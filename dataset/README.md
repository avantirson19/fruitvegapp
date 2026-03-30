# YOLO Dataset Layout

Put your labeled detection dataset here in standard YOLO format:

```text
dataset/
  data.yaml
  images/
    train/
    val/
  labels/
    train/
    val/
```

Each image in `images/train` or `images/val` must have a matching label file in
`labels/train` or `labels/val` with the same base filename.

Example:

```text
images/train/apple_scene_01.jpg
labels/train/apple_scene_01.txt
```

Each line in a YOLO label file is:

```text
class_id x_center y_center width height
```

All box values must be normalized to the range `0..1`.

Example label file for two apples:

```text
0 0.31 0.47 0.22 0.30
0 0.69 0.46 0.24 0.31
```

Train the detector with:

```bash
python train_detector.py --data dataset/data.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 8
```

After training, copy the best weights to the project root as:

```text
fruit_detector.pt
```
