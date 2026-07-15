"""
KITTI标准评估脚本 - 163轮模型评测
参照KITTI 3D Object Detection评测标准
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

# KITTI标准类别 - 确保与技术方案一致
KITTI_CLASSES = ['Car', 'Cyclist', 'Truck']
KITTI_CLASS_TO_ID = {
    'Car': 0,
    'Cyclist': 1,
    'Truck': 2
}

# KITTI评测难度等级
KITTI_DIFFICULTY = {
    'easy': 0,
    'moderate': 1,
    'hard': 2
}

def parse_args():
    parser = argparse.ArgumentParser(description='KITTI标准评估 - 163轮模型')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='预测结果目录')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='真值标签目录')
    parser.add_argument('--iou_thresh', type=float, nargs='+', default=[0.5, 0.7],
                        help='IoU阈值列表')
    parser.add_argument('--output_file', type=str, default='kitti_eval_results.json',
                        help='结果输出文件')
    return parser.parse_args()

def load_kitti_predictions(pred_dir: str) -> Dict[str, List[Dict]]:
    """加载KITTI格式预测结果"""
    predictions = {}
    pred_path = Path(pred_dir)

    print(f"加载预测结果从: {pred_path}")

    for pred_file in pred_path.glob('*.txt'):
        sample_id = pred_file.stem
        predictions[sample_id] = []

        with open(pred_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 15:
                        class_name = parts[0]
                        if class_name in KITTI_CLASSES:
                            pred = {
                                'type': class_name,
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

    print(f"加载了 {len(predictions)} 个预测文件")
    return predictions

def load_kitti_ground_truth(gt_dir: str) -> Dict[str, List[Dict]]:
    """加载KITTI格式真值标签"""
    ground_truth = {}
    gt_path = Path(gt_dir)

    print(f"加载真值标签从: {gt_path}")

    for gt_file in gt_path.glob('*.txt'):
        sample_id = gt_file.stem
        ground_truth[sample_id] = []

        with open(gt_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 15:
                        class_name = parts[0]
                        if class_name in KITTI_CLASSES:
                            gt = {
                                'type': class_name,
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

    print(f"加载了 {len(ground_truth)} 个真值文件")
    return ground_truth

def compute_iou_3d_kitti(box1: Dict, box2: Dict) -> float:
    """KITTI标准3D IoU计算"""
    # 使用BEV投影的简化IoU计算
    center1 = np.array([box1['location'][0], box1['location'][1]])
    center2 = np.array([box2['location'][0], box2['location'][1]])

    # 尺寸 (h, w, l) -> (l, w) for BEV
    dims1 = np.array([box1['dimensions'][0], box1['dimensions'][1]])  # l, w
    dims2 = np.array([box2['dimensions'][0], box2['dimensions'][1]])  # l, w

    # 角度
    yaw1 = box1['rotation_y']
    yaw2 = box2['rotation_y']

    # 创建矩形角点
    def get_corners(center, dims, angle):
        l, w = dims
        corners = np.array([
            [-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]
        ])
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                              [np.sin(angle), np.cos(angle)]])
        return corners @ rot_matrix.T + center

    corners1 = get_corners(center1, dims1, yaw1)
    corners2 = get_corners(center2, dims2, yaw2)

    # 计算多边形面积
    def polygon_area(corners):
        x, y = corners[:, 0], corners[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # 计算交集面积（简化版本）
    def intersection_area(corners1, corners2):
        # 使用轴对齐边界框的交集作为近似
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
    inter_area = intersection_area(corners1, corners2)

    union_area = area1 + area2 - inter_area
    return inter_area / max(union_area, 1e-10)

def evaluate_kitti_class(predictions: Dict, ground_truth: Dict,
                        class_name: str, iou_thresh: float = 0.5) -> Dict:
    """KITTI标准单类别评估"""

    pred_boxes = []
    gt_boxes = []

    # 收集该类别的预测和真值
    for sample_id in ground_truth:
        gt_sample = []
        for gt in ground_truth.get(sample_id, []):
            if gt['type'] == class_name:
                gt_sample.append(gt)

        if gt_sample:
            gt_boxes.extend([(sample_id, gt) for gt in gt_sample])

            pred_sample = []
            for pred in predictions.get(sample_id, []):
                if pred['type'] == class_name:
                    pred_sample.append(pred)

            pred_boxes.extend([(sample_id, pred) for pred in pred_sample])

    if not gt_boxes:
        return {
            'class': class_name,
            'ap': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'tp': 0, 'fp': 0, 'fn': 0,
            'total_gt': 0, 'total_pred': 0
        }

    # 按置信度排序预测
    pred_boxes.sort(key=lambda x: x[1]['score'], reverse=True)

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    gt_matched = set()

    # 匹配预测到真值
    for i, (sample_id, pred) in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1

        for j, (gt_sample_id, gt) in enumerate(gt_boxes):
            if gt_sample_id != sample_id or j in gt_matched:
                continue

            iou = compute_iou_3d_kitti(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_thresh:
            tp[i] = 1
            gt_matched.add(best_gt_idx)
        else:
            fp[i] = 1

    # 计算精度和召回率
    fp_cumsum = np.cumsum(fp)
    tp_cumsum = np.cumsum(tp)

    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-10)
    recalls = tp_cumsum / max(len(gt_boxes), 1)

    # 计算AP (11-point插值)
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        mask = recalls >= t
        if np.any(mask):
            ap += precisions[mask].max()
    ap /= 11.0

    final_precision = precisions[-1] if len(precisions) > 0 else 0.0
    final_recall = recalls[-1] if len(recalls) > 0 else 0.0
    final_f1 = 2 * final_precision * final_recall / max(final_precision + final_recall, 1e-10)

    return {
        'class': class_name,
        'ap': ap,
        'precision': final_precision,
        'recall': final_recall,
        'f1': final_f1,
        'tp': int(tp.sum()),
        'fp': int(fp.sum()),
        'fn': len(gt_boxes) - len(gt_matched),
        'total_gt': len(gt_boxes),
        'total_pred': len(pred_boxes)
    }

def main():
    args = parse_args()

    print("="*80)
    print("KITTI标准评估 - SFA3D-Modified 163轮模型")
    print("="*80)
    print(f"预测目录: {args.pred_dir}")
    print(f"真值目录: {args.gt_dir}")
    print(f"IoU阈值: {args.iou_thresh}")
    print(f"评估类别: {KITTI_CLASSES}")
    print("="*80)

    # 加载数据
    predictions = load_kitti_predictions(args.pred_dir)
    ground_truth = load_kitti_ground_truth(args.gt_dir)

    # 评估每个IoU阈值
    all_results = {}

    for iou_thresh in args.iou_thresh:
        print(f"\n评估 IoU阈值 {iou_thresh}:")
        print("-" * 60)

        results = {
            'iou_threshold': iou_thresh,
            'classes': {},
            'overall': {}
        }

        class_aps = []

        for class_name in KITTI_CLASSES:
            class_result = evaluate_kitti_class(predictions, ground_truth, class_name, iou_thresh)
            results['classes'][class_name] = class_result
            class_aps.append(class_result['ap'])

            print(f"{class_name:12s} AP: {class_result['ap']:.4f} | "
                  f"P: {class_result['precision']:.4f} | "
                  f"R: {class_result['recall']:.4f} | "
                  f"F1: {class_result['f1']:.4f} | "
                  f"TP: {class_result['tp']:4d} | "
                  f"FP: {class_result['fp']:4d} | "
                  f"FN: {class_result['fn']:4d}")

        # 计算整体mAP
        map_score = np.mean(class_aps) if class_aps else 0.0

        # 整体统计
        total_tp = sum(r['tp'] for r in results['classes'].values())
        total_fp = sum(r['fp'] for r in results['classes'].values())
        total_fn = sum(r['fn'] for r in results['classes'].values())

        overall_precision = total_tp / max(total_tp + total_fp, 1e-10)
        overall_recall = total_tp / max(total_tp + total_fn, 1e-10)
        overall_f1 = 2 * overall_precision * overall_recall / max(overall_precision + overall_recall, 1e-10)

        results['overall'] = {
            'map': map_score,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn
        }

        print(f"{'Overall':12s} mAP: {map_score:.4f} | "
              f"P: {overall_precision:.4f} | "
              f"R: {overall_recall:.4f} | "
              f"F1: {overall_f1:.4f}")

        all_results[f'iou_{iou_thresh}'] = results

    # 保存结果
    import json
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print("KITTI评估总结")
    print("="*80)

    for iou_thresh in args.iou_thresh:
        result = all_results[f'iou_{iou_thresh}']
        print(f"mAP@{iou_thresh}: {result['overall']['map']:.4f}")

        for class_name in KITTI_CLASSES:
            if class_name in result['classes']:
                class_result = result['classes'][class_name]
                print(f"  {class_name} AP@{iou_thresh}: {class_result['ap']:.4f}")

    print(f"\n详细结果已保存到: {args.output_file}")
    print("="*80)

if __name__ == '__main__':
    main()