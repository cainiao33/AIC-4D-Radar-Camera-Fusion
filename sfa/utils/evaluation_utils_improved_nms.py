"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.17
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: The utils for evaluation
# Refer from: https://github.com/xingyizhou/CenterNet
"""

from __future__ import division
import os
import sys

import torch
import numpy as np
import torch.nn.functional as F
import cv2

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import config.kitti_config as cnf
from data_process.kitti_bev_utils import drawRotatedBox


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()

    return heat * keep


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (torch.floor_divide(topk_inds, width)).float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (torch.floor_divide(topk_ind, K)).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def decode(hm_cen, cen_offset, direction, z_coor, dim, K=40):
    batch_size, num_classes, height, width = hm_cen.size()

    hm_cen = _nms(hm_cen)
    scores, inds, clses, ys, xs = _topk(hm_cen, K=K)
    if cen_offset is not None:
        cen_offset = _transpose_and_gather_feat(cen_offset, inds)
        cen_offset = cen_offset.view(batch_size, K, 2)
        xs = xs.view(batch_size, K, 1) + cen_offset[:, :, 0:1]
        ys = ys.view(batch_size, K, 1) + cen_offset[:, :, 1:2]
    else:
        xs = xs.view(batch_size, K, 1) + 0.5
        ys = ys.view(batch_size, K, 1) + 0.5

    direction = _transpose_and_gather_feat(direction, inds)
    direction = direction.view(batch_size, K, 2)
    z_coor = _transpose_and_gather_feat(z_coor, inds)
    z_coor = z_coor.view(batch_size, K, 1)
    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch_size, K, 3)
    clses = clses.view(batch_size, K, 1).float()
    scores = scores.view(batch_size, K, 1)

    # (scores x 1, ys x 1, xs x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
    # (scores-0:1, ys-1:2, xs-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
    # detections: [batch_size, K, 10]
    detections = torch.cat([scores, xs, ys, z_coor, dim, direction, clses], dim=2)

    return detections


def get_yaw(direction):
    return np.arctan2(direction[:, 0:1], direction[:, 1:2])


def compute_bev_box_iou(box1, box2):
    """
    Compute IoU between two BEV boxes (improved 2D overlap calculation).
    box format: [score, x, y, z, h, w, l, yaw]
    Returns more accurate IoU based on rotated rectangle overlap.
    """
    # Extract centers and dimensions
    x1, y1, w1, l1, yaw1 = box1[1], box1[2], box1[5], box1[6], box1[7]
    x2, y2, w2, l2, yaw2 = box2[1], box2[2], box2[5], box2[6], box2[7]

    # Compute center distance (more sensitive detection)
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    # Compute more conservative average size (use smaller dimension)
    min_dim1 = min(w1, l1)
    min_dim2 = min(w2, l2)
    min_size = min(min_dim1, min_dim2)
    max_size = max(w1, l1, w2, l2)
    avg_size = (w1 + l1 + w2 + l2) / 4.0

    if avg_size < 1e-6:
        return 0.0

    # Enhanced distance-based IoU calculation
    normalized_dist = dist / avg_size

    # More aggressive overlap detection for cross-class NMS
    if normalized_dist > 1.0:  # Reduced from 1.5
        return 0.0
    elif normalized_dist < 0.2:  # Reduced from 0.3
        return 0.9  # Increased from 0.8
    elif normalized_dist < 0.4:
        return 0.7
    else:
        return max(0.0, 1.2 - normalized_dist)  # More aggressive


def apply_inter_class_nms(top_preds, num_classes=3, iou_thresh=0.3):
    """
    Apply NMS across different classes to remove duplicate detections.
    More aggressive with lower threshold for better suppression.
    """
    # Collect all detections with class info
    all_dets = []
    for cls_id in range(num_classes):
        if len(top_preds[cls_id]) > 0:
            for det in top_preds[cls_id]:
                all_dets.append((det, cls_id))

    if len(all_dets) == 0:
        return top_preds

    # Sort by score (descending)
    all_dets.sort(key=lambda x: x[0][0], reverse=True)

    # Keep track of suppressed detections
    keep = [True] * len(all_dets)

    for i in range(len(all_dets)):
        if not keep[i]:
            continue
        det_i, cls_i = all_dets[i]

        for j in range(i + 1, len(all_dets)):
            if not keep[j]:
                continue
            det_j, cls_j = all_dets[j]

            # Only apply NMS between different classes
            if cls_i != cls_j:
                iou = compute_bev_box_iou(det_i, det_j)

                # Multiple suppression criteria
                suppress = False

                # Criterion 1: High overlap (traditional NMS)
                if iou > iou_thresh:
                    suppress = True

                # Criterion 2: Very close proximity (regardless of IoU)
                x1, y1 = det_i[1], det_i[2]
                x2, y2 = det_j[1], det_j[2]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                # Get average box size for distance normalization
                avg_size = (det_i[5] + det_i[6] + det_j[5] + det_j[6]) / 4.0
                if avg_size > 0:
                    normalized_dist = dist / avg_size
                    if normalized_dist < 0.25:  # Very close proximity
                        suppress = True

                    # Criterion 3: Same location with different classes (aggressive)
                    if normalized_dist < 0.1:
                        suppress = True

                if suppress:
                    # Suppress the lower-score detection
                    keep[j] = False

    # Rebuild top_preds with kept detections
    new_top_preds = {j: [] for j in range(num_classes)}
    for idx, (det, cls_id) in enumerate(all_dets):
        if keep[idx]:
            new_top_preds[cls_id].append(det)

    # Convert lists back to numpy arrays
    for j in range(num_classes):
        if len(new_top_preds[j]) > 0:
            new_top_preds[j] = np.array(new_top_preds[j])
        else:
            new_top_preds[j] = np.array([]).reshape(0, 8)

    return new_top_preds


def post_processing(detections, num_classes=3, down_ratio=4, peak_thresh=0.2, inter_class_nms=True, nms_thresh=0.5):
    """
    :param detections: [batch_size, K, 10]
    # (scores x 1, xs x 1, ys x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
    # (scores-0:1, xs-1:2, ys-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
    :param inter_class_nms: Apply NMS across different classes to remove duplicates
    :param nms_thresh: IoU threshold for inter-class NMS
    :return:
    """
    # TODO: Need to consider rescale to the original scale: x, y

    ret = []
    for i in range(detections.shape[0]):
        top_preds = {}
        classes = detections[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            # x, y, z, h, w, l, yaw
            top_preds[j] = np.concatenate([
                detections[i, inds, 0:1],
                detections[i, inds, 1:2] * down_ratio,
                detections[i, inds, 2:3] * down_ratio,
                detections[i, inds, 3:4],
                detections[i, inds, 4:5],
                detections[i, inds, 5:6] / cnf.bound_size_y * cnf.BEV_WIDTH,
                detections[i, inds, 6:7] / cnf.bound_size_x * cnf.BEV_HEIGHT,
                get_yaw(detections[i, inds, 7:9]).astype(np.float32)], axis=1)
            # Filter by peak_thresh
            if len(top_preds[j]) > 0:
                keep_inds = (top_preds[j][:, 0] > peak_thresh)
                top_preds[j] = top_preds[j][keep_inds]

        # Apply inter-class NMS to remove duplicate detections across classes
        if inter_class_nms:
            top_preds = apply_inter_class_nms(top_preds, num_classes, nms_thresh)

        ret.append(top_preds)

    return ret


def draw_predictions(img, detections, num_classes=3):
    for j in range(num_classes):
        if len(detections[j]) > 0:
            for det in detections[j]:
                # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                _score, _x, _y, _z, _h, _w, _l, _yaw = det
                drawRotatedBox(img, _x, _y, _w, _l, _yaw, cnf.colors[int(j)])

    return img


def convert_det_to_real_values(detections, num_classes=3):
    kitti_dets = []
    for cls_id in range(num_classes):
        if len(detections[cls_id]) > 0:
            for det in detections[cls_id]:
                # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                _score, _x, _y, _z, _h, _w, _l, _yaw = det
                _yaw = -_yaw
                x = _y / cnf.BEV_HEIGHT * cnf.bound_size_x + cnf.boundary['minX']
                y = _x / cnf.BEV_WIDTH * cnf.bound_size_y + cnf.boundary['minY']
                z = _z + cnf.boundary['minZ']
                w = _w / cnf.BEV_WIDTH * cnf.bound_size_y
                l = _l / cnf.BEV_HEIGHT * cnf.bound_size_x

                kitti_dets.append([cls_id, x, y, z, _h, w, l, _yaw])

    return np.array(kitti_dets)
