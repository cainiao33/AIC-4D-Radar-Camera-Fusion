"""
计算验证集预测的mAP和AP指标。
此脚本计算实际的mAP@0.5和mAP@0.7指标。
"""

import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import time

# KITTI evaluation tools
try:
    from evaluate_training import evaluate as kitti_evaluate
except ImportError:
    print("Warning: KITTI evaluation tools not available, using simplified evaluation")
    kitti_evaluate = None

# Path setup
src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from config import kitti_config as cnf
from data_process.kitti_data_utils import Calibration
from data_process.transformation import lidar_to_camera_box
from utils.visualization_utils import compute_box_3d


CLASS_NAMES = ['Car', 'Cyclist', 'Truck']
CLASS_ID_TO_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate mAP and AP metrics')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Directory containing prediction files (.txt)')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Directory containing ground truth files (.txt)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file for results (JSON format)')
    parser.add_argument('--iou_thresholds', type=float, nargs='+', default=[0.5, 0.7],
                        help='IoU thresholds for evaluation')
    parser.add_argument('--min_distance', type=float, default=0,
                        help='Minimum distance for range-based evaluation')
    parser.add_argument('--max_distance', type=float, default=50,
                        help='Maximum distance for range-based evaluation')
    return parser.parse_args()


def load_predictions(pred_dir: str) -> Dict[str, List[Dict]]:
    """Load prediction files"""
    predictions = {}
    pred_path = Path(pred_dir)

    for pred_file in pred_path.glob('*.txt'):
        sample_id = pred_file.stem
        predictions[sample_id] = []

        with open(pred_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 15:
                        pred = {
                            'type': parts[0],
                            'truncated': float(parts[1]),
                            'occluded': int(parts[2]),
                            'alpha': float(parts[3]),
                            'bbox': [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
                            'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],
                            'location': [float(parts[11]), float(parts[12]), float(parts[13])],
                            'rotation_y': float(parts[14]),
                            'score': float(parts[15]) if len(parts) > 15 else 1.0
                        }
                        predictions[sample_id].append(pred)

    return predictions


def load_ground_truth(gt_dir: str) -> Dict[str, List[Dict]]:
    """Load ground truth files"""
    ground_truth = {}
    gt_path = Path(gt_dir)

    for gt_file in gt_path.glob('*.txt'):
        sample_id = gt_file.stem
        ground_truth[sample_id] = []

        with open(gt_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 15:
                        gt = {
                            'type': parts[0],
                            'truncated': float(parts[1]),
                            'occluded': int(parts[2]),
                            'alpha': float(parts[3]),
                            'bbox': [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
                            'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],
                            'location': [float(parts[11]), float(parts[12]), float(parts[13])],
                            'rotation_y': float(parts[14]),
                            'score': 1.0  # GT objects have score 1.0
                        }
                        ground_truth[sample_id].append(gt)

    return ground_truth


def calculate_bev_iou(box1: Dict, box2: Dict) -> float:
    """Calculate BEV IoU between two 3D boxes"""
    # Extract center and dimensions
    center1 = np.array([box1['location'][0], box1['location'][1]])
    center2 = np.array([box2['location'][0], box2['location'][1]])

    # Extract dimensions (length and width for BEV)
    dims1 = np.array([box1['dimensions'][0], box1['dimensions'][1]])  # l, w
    dims2 = np.array([box2['dimensions'][0], box2['dimensions'][1]])  # l, w

    # Extract rotation angles
    rot1 = box1['rotation_y']
    rot2 = box2['rotation_y']

    # Create corner points for each box
    def get_corners(center, dims, rot):
        l, w = dims
        # Box corners in local coordinates
        corners_local = np.array([
            [-l/2, -w/2],
            [l/2, -w/2],
            [l/2, w/2],
            [-l/2, w/2]
        ])

        # Rotation matrix
        rot_matrix = np.array([
            [np.cos(rot), -np.sin(rot)],
            [np.sin(rot), np.cos(rot)]
        ])

        # Rotate and translate corners
        corners_world = corners_local @ rot_matrix.T + center
        return corners_world

    corners1 = get_corners(center1, dims1, rot1)
    corners2 = get_corners(center2, dims2, rot2)

    # Calculate intersection area using shapely-like approach
    def polygon_area(corners):
        x = corners[:, 0]
        y = corners[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def polygon_intersection(corners1, corners2):
        # Simple intersection approximation using axis-aligned bounding boxes
        # This is an approximation - for accurate results, use shapely
        min_x1, max_x1 = corners1[:, 0].min(), corners1[:, 0].max()
        min_y1, max_y1 = corners1[:, 1].min(), corners1[:, 1].max()
        min_x2, max_x2 = corners2[:, 0].min(), corners2[:, 0].max()
        min_y2, max_y2 = corners2[:, 1].min(), corners2[:, 1].max()

        inter_min_x = max(min_x1, min_x2)
        inter_max_x = min(max_x1, max_x2)
        inter_min_y = max(min_y1, min_y2)
        inter_max_y = min(max_y1, max_y2)

        if inter_min_x < inter_max_x and inter_min_y < inter_max_y:
            return (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
        return 0.0

    area1 = polygon_area(corners1)
    area2 = polygon_area(corners2)
    inter_area = polygon_intersection(corners1, corners2)

    union_area = area1 + area2 - inter_area
    return inter_area / max(union_area, 1e-10)


def calculate_distance(location: List[float]) -> float:
    """Calculate distance from origin"""
    return np.sqrt(location[0]**2 + location[1]**2)


def evaluate_class(predictions: Dict, ground_truth: Dict, class_name: str,
                  iou_threshold: float = 0.5, min_dist: float = 0, max_dist: float = 50) -> Dict:
    """Evaluate predictions for a specific class"""

    # Filter by class and distance
    pred_boxes = []
    gt_boxes = []

    for sample_id in ground_truth:
        # Get ground truth boxes for this class and distance range
        gt_sample = []
        for gt in ground_truth.get(sample_id, []):
            if gt['type'] == class_name:
                distance = calculate_distance(gt['location'])
                if min_dist <= distance <= max_dist:
                    gt_sample.append(gt)

        if gt_sample:
            gt_boxes.extend([(sample_id, gt) for gt in gt_sample])

            # Get predictions for this sample
            pred_sample = []
            for pred in predictions.get(sample_id, []):
                if pred['type'] == class_name:
                    distance = calculate_distance(pred['location'])
                    if min_dist <= distance <= max_dist:
                        pred_sample.append(pred)

            pred_boxes.extend([(sample_id, pred) for pred in pred_sample])

    if not gt_boxes:
        return {'ap': 0.0, 'precision': 0.0, 'recall': 0.0, 'tp': 0, 'fp': 0, 'fn': len(gt_boxes)}

    # Sort predictions by confidence
    pred_boxes.sort(key=lambda x: x[1]['score'], reverse=True)

    # Initialize
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    gt_matched = set()

    # Evaluate each prediction
    for i, (sample_id, pred) in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1

        # Find best matching ground truth
        for j, (gt_sample_id, gt) in enumerate(gt_boxes):
            if gt_sample_id != sample_id:
                continue
            if j in gt_matched:
                continue

            iou = calculate_bev_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        # Check if match is good enough
        if best_iou >= iou_threshold:
            tp[i] = 1
            gt_matched.add(best_gt_idx)
        else:
            fp[i] = 1

    # Calculate precision and recall
    fp_cumsum = np.cumsum(fp)
    tp_cumsum = np.cumsum(tp)

    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-10)
    recalls = tp_cumsum / max(len(gt_boxes), 1)

    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        mask = recalls >= t
        if np.any(mask):
            ap += precisions[mask].max()
    ap /= 11.0

    # Final precision and recall
    final_precision = precisions[-1] if len(precisions) > 0 else 0.0
    final_recall = recalls[-1] if len(recalls) > 0 else 0.0

    return {
        'ap': ap,
        'precision': final_precision,
        'recall': final_recall,
        'tp': int(tp.sum()),
        'fp': int(fp.sum()),
        'fn': len(gt_boxes) - len(gt_matched)
    }


def main():
    args = parse_args()

    print(f"Loading predictions from: {args.pred_dir}")
    print(f"Loading ground truth from: {args.gt_dir}")
    print(f"IoU thresholds: {args.iou_thresholds}")

    # Load data
    predictions = load_predictions(args.pred_dir)
    ground_truth = load_ground_truth(args.gt_dir)

    print(f"Found {len(predictions)} prediction files")
    print(f"Found {len(ground_truth)} ground truth files")

    # Evaluate for each IoU threshold
    all_results = {}

    for iou_thresh in args.iou_thresholds:
        print(f"\nEvaluating at IoU threshold {iou_thresh}:")
        print("-" * 50)

        results = {
            'iou_threshold': iou_thresh,
            'classes': {},
            'overall': {}
        }

        # Overall evaluation
        overall_result = evaluate_class(predictions, ground_truth, 'all', iou_thresh,
                                      args.min_distance, args.max_distance)
        results['overall'] = overall_result

        # Class-wise evaluation
        class_aps = []
        for class_name in CLASS_NAMES:
            class_result = evaluate_class(predictions, ground_truth, class_name, iou_thresh,
                                        args.min_distance, args.max_distance)
            results['classes'][class_name] = class_result
            class_aps.append(class_result['ap'])

            print(f"{class_name:10s} AP: {class_result['ap']:.3f} | "
                  f"P: {class_result['precision']:.3f} | "
                  f"R: {class_result['recall']:.3f} | "
                  f"TP: {class_result['tp']:3d} | "
                  f"FP: {class_result['fp']:3d} | "
                  f"FN: {class_result['fn']:3d}")

        # Calculate mAP
        map_score = np.mean(class_aps)
        results['overall']['map'] = map_score

        print(f"{'Overall':10s} mAP: {map_score:.3f}")

        all_results[f'iou_{iou_thresh}'] = results

    # Distance-based evaluation (0-20m, 20-35m, 35-50m)
    distance_ranges = [(0, 20, 'close'), (20, 35, 'mid'), (35, 50, 'far')]

    print(f"\nDistance-based evaluation at IoU 0.5:")
    print("-" * 50)

    distance_results = {}
    for min_dist, max_dist, range_name in distance_ranges:
        print(f"\n{range_name.capitalize()} range ({min_dist}-{max_dist}m):")

        range_result = {}
        class_aps = []

        for class_name in CLASS_NAMES:
            class_result = evaluate_class(predictions, ground_truth, class_name, 0.5,
                                        min_dist, max_dist)
            range_result[class_name] = class_result
            class_aps.append(class_result['ap'])

            print(f"  {class_name:10s} AP: {class_result['ap']:.3f} | "
                  f"P: {class_result['precision']:.3f} | "
                  f"R: {class_result['recall']:.3f}")

        range_result['map'] = np.mean(class_aps)
        distance_results[range_name] = range_result

        print(f"  {'Overall':10s} mAP: {range_result['map']:.3f}")

    all_results['distance_analysis'] = distance_results

    # Save results
    output_file = args.output_file or f'map_results_{int(time.time())}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print('='*60)

    for iou_thresh in args.iou_thresholds:
        result = all_results[f'iou_{iou_thresh}']
        print(f"mAP@{iou_thresh}: {result['overall']['map']:.4f}")

        for class_name in CLASS_NAMES:
            print(f"  {class_name} AP@{iou_thresh}: {result['classes'][class_name]['ap']:.4f}")

    print(f"\nDistance-based performance (IoU=0.5):")
    for range_name, result in distance_results.items():
        print(f"  {range_name.capitalize()} range: {result['map']:.4f}")

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == '__main__':
    main()