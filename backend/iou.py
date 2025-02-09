import os
import numpy as np
import yaml
from pathlib import Path
from iou_utils import calculate_batch_iou, evaluate_model_iou

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

# Get project root directory and data.yaml path
project_root = Path(__file__).parent.parent
data_yaml_path = project_root / "dataset-detect" / "data.yaml"
data_config = load_yaml(data_yaml_path)

# Construct path to labels directory based on data.yaml
train_path = data_config['train']  # Gets "../train/images"
label_dir = (project_root / "dataset-detect" / "train" / "labels").resolve()
boxes = []

try:
    print(f"Looking for label files in: {label_dir}")
    # Read all .txt files in the labels directory
    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            with open(os.path.join(label_dir, file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # Convert YOLO format (class x y w h) to [x1 y1 x2 y2]
                    values = list(map(float, line.strip().split()))
                    if len(values) == 5:  # class, x, y, w, h format
                        x, y, w, h = values[1:]
                        # Convert from relative coordinates to absolute coordinates
                        x1 = x - w/2
                        y1 = y - h/2
                        x2 = x + w/2
                        y2 = y + h/2
                        boxes.append([x1, y1, x2, y2])
                        print(f"Processed box: [{x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}]")

    if len(boxes) > 1:
        # Calculate IOU between all boxes
        evaluation = evaluate_model_iou(boxes[::2], boxes[1::2])
        print(f"\nIOU Evaluation Results:")
        print(f"Total boxes processed: {len(boxes)}")
        print(f"Average IOU: {evaluation['average_iou']:.4f}")
        print(f"Precision: {evaluation['precision']:.4f}")
        print(f"Recall: {evaluation['recall']:.4f}")
    else:
        print("Not enough boxes found for IOU calculation")

except Exception as e:
    print(f"Error processing label files: {str(e)}")
    print(f"Attempted to read from: {label_dir}")