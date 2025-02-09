import numpy as np

def calculate_iou(box1, box2):
    """Calculate IOU between two boxes"""
    # Get coordinates of intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IOU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def calculate_batch_iou(predictions, ground_truth):
    """Calculate IOU for batch of predictions"""
    ious = []
    for pred_box in predictions:
        box_ious = [calculate_iou(pred_box, gt_box) for gt_box in ground_truth]
        ious.append(max(box_ious) if box_ious else 0)
    return ious

def evaluate_model_iou(predictions, ground_truth, iou_threshold=0.5):
    """Evaluate model performance using IOU"""
    ious = calculate_batch_iou(predictions, ground_truth)
    
    correct_detections = sum(1 for iou in ious if iou >= iou_threshold)
    total_predictions = len(predictions)
    total_ground_truth = len(ground_truth)
    
    precision = correct_detections / total_predictions if total_predictions > 0 else 0
    recall = correct_detections / total_ground_truth if total_ground_truth > 0 else 0
    average_iou = sum(ious) / len(ious) if ious else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "average_iou": average_iou,
        "ious": ious
    }