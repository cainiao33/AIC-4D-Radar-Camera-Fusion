"""
修正了BEV纵横比并进行了正确坐标对齐的最终版本。
BEV默认会旋转以匹配相机视角。
"""
import argparse
import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from config import kitti_config as cnf
from data_process.lidar_mapping import read_lidar_file_with_fallback
from data_process.kitti_data_utils import Calibration
from utils.visualization_utils import compute_box_3d, project_to_image


def draw_3d_boxes_on_image(img, detections, calib):
    """Draw 3D boxes projected onto 2D image."""
    for line in detections:
        parts = line.strip().split()
        if len(parts) < 15:
            continue

        cls_name = parts[0]
        score = float(parts[-1])

        # Parse 3D box in camera coordinates (KITTI format)
        h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
        x_cam, y_cam, z_cam = float(parts[11]), float(parts[12]), float(parts[13])
        ry = float(parts[14])

        if z_cam <= 0:
            continue

        # Compute 3D box corners in camera coordinates
        location = np.array([x_cam, y_cam, z_cam])
        dims = np.array([h, w, l])

        corners_3d = compute_box_3d(dims, location, ry)
        corners_2d = project_to_image(corners_3d, calib.P2)

        # Check if box is within image bounds
        x_min, y_min = corners_2d[:, 0].min(), corners_2d[:, 1].min()
        x_max, y_max = corners_2d[:, 0].max(), corners_2d[:, 1].max()

        img_h, img_w = img.shape[:2]
        if x_max < 0 or y_max < 0 or x_min >= img_w or y_min >= img_h:
            continue

        # Clip to image bounds
        x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
        x_max, y_max = min(img_w-1, int(x_max)), min(img_h-1, int(y_max))

        # Color based on class
        if cls_name == 'Car':
            color = (0, 255, 0)  # Green
        elif cls_name == 'Pedestrian':
            color = (255, 0, 0)  # Blue
        elif cls_name == 'Cyclist':
            color = (0, 0, 255)  # Red
        else:
            color = (255, 255, 0)  # Cyan

        # Draw 2D bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

        # Draw class label and score
        label = f'{cls_name}: {score:.2f}'
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(img, (x_min, y_min - 25), (x_min + label_size[0], y_min), color, -1)
        cv2.putText(img, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw 3D box projection
        corners_2d = corners_2d.astype(np.int32)

        # Draw the 8 corners
        for corner in corners_2d:
            if 0 <= corner[0] < img_w and 0 <= corner[1] < img_h:
                cv2.circle(img, tuple(corner), 2, color, -1)

        # Draw edges of the 3D box
        # Bottom face
        for i in range(4):
            pt1 = tuple(corners_2d[i])
            pt2 = tuple(corners_2d[(i+1)%4])
            if all(0 <= p < img_w for p in [pt1[0], pt2[0]]) and all(0 <= p < img_h for p in [pt1[1], pt2[1]]):
                cv2.line(img, pt1, pt2, color, 2)

        # Top face
        for i in range(4, 8):
            pt1 = tuple(corners_2d[i])
            pt2 = tuple(corners_2d[4 + (i-4+1)%4])
            if all(0 <= p < img_w for p in [pt1[0], pt2[0]]) and all(0 <= p < img_h for p in [pt1[1], pt2[1]]):
                cv2.line(img, pt1, pt2, color, 2)

        # Vertical edges
        for i in range(4):
            pt1 = tuple(corners_2d[i])
            pt2 = tuple(corners_2d[i+4])
            if all(0 <= p < img_w for p in [pt1[0], pt2[0]]) and all(0 <= p < img_h for p in [pt1[1], pt2[1]]):
                cv2.line(img, pt1, pt2, color, 2)

    return img


def create_colored_bev_from_points(points, cnf, rotate_180=True):
    """Create colored BEV image with proper aspect ratio and rotation."""
    xyz_i = points[:, :4].copy()

    # Filter points within boundaries
    x_lim = (cnf.boundary['minX'], cnf.boundary['maxX'])
    y_lim = (cnf.boundary['minY'], cnf.boundary['maxY'])
    z_lim = (cnf.boundary['minZ'], cnf.boundary['maxZ'])

    mask = (xyz_i[:, 0] >= x_lim[0]) & (xyz_i[:, 0] < x_lim[1]) & \
           (xyz_i[:, 1] >= y_lim[0]) & (xyz_i[:, 1] < y_lim[1]) & \
           (xyz_i[:, 2] >= z_lim[0]) & (xyz_i[:, 2] < z_lim[1])

    xyz_i = xyz_i[mask]

    # Use original BEV dimensions for compatibility
    bev_height = cnf.BEV_HEIGHT
    bev_width = cnf.BEV_WIDTH

    if len(xyz_i) == 0:
        return np.full((bev_height, bev_width, 3), 255, dtype=np.uint8)

    x = xyz_i[:, 0]
    y = xyz_i[:, 1]
    z = xyz_i[:, 2]
    intensity = xyz_i[:, 3]

    # Don't rotate by default - use standard coordinate system
    # Y+ = Forward, X+ = Right, X- = Left (matches camera perspective)
    pass

    # Normalize to BEV grid (using original dimensions)
    x_img = np.floor((y - y_lim[0]) / (y_lim[1] - y_lim[0]) * bev_width).astype(np.int32)
    y_img = np.floor((x - x_lim[0]) / (x_lim[1] - x_lim[0]) * bev_height).astype(np.int32)

    x_img = np.clip(x_img, 0, bev_width - 1)
    y_img = np.clip(y_img, 0, bev_height - 1)

    # Create white background
    bev_map = np.full((bev_height, bev_width, 3), 255, dtype=np.uint8)

    # Color points based on height (z coordinate)
    z_min, z_max = z.min(), z.max()
    if z_max > z_min:
        z_normalized = (z - z_min) / (z_max - z_min)

        # Create color gradient: blue (low) -> green (mid) -> red (high)
        for i in range(len(x_img)):
            z_norm = z_normalized[i]
            if z_norm < 0.5:
                # Blue to Green
                ratio = z_norm * 2
                color = (0, int(255 * ratio), int(255 * (1 - ratio)))
            else:
                # Green to Red
                ratio = (z_norm - 0.5) * 2
                color = (int(255 * ratio), int(255 * (1 - ratio)), 0)

            # Add some intensity variation
            intensity_factor = max(0.3, min(1.0, intensity[i]))
            color = tuple(int(c * intensity_factor) for c in color)

            bev_map[y_img[i], x_img[i]] = color

    # Don't add coordinate system labels (removed per user request)

    return bev_map, (bev_width, bev_height)


def draw_boxes_on_bev(bev_img, detections, cnf, bev_size, rotate_180=True):
    """Draw detection boxes on BEV image (without text labels)."""
    if bev_img is None:
        return None, []

    bev_width, bev_height = bev_size
    detection_info = []  # Store detection info for later text drawing

    for line in detections:
        parts = line.strip().split()
        if len(parts) < 15:
            continue

        cls_name = parts[0]
        score = float(parts[-1])

        # Parse 3D box in camera coordinates
        h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
        x_cam, y_cam, z_cam = float(parts[11]), float(parts[12]), float(parts[13])

        # Convert camera to lidar coordinates
        x_lidar = z_cam
        y_lidar = -x_cam

        # No rotation needed - use standard coordinate system
        # Y+ = Forward, X+ = Right, X- = Left

        # Convert to BEV pixel coordinates
        x_bev = int((y_lidar - cnf.boundary['minY']) / cnf.bound_size_y * bev_width)
        y_bev = int((x_lidar - cnf.boundary['minX']) / cnf.bound_size_x * bev_height)

        # Box dimensions in BEV with correct aspect ratio
        l_bev = int(l / cnf.bound_size_x * bev_height)
        w_bev = int(w / cnf.bound_size_y * bev_width)

        # Draw box
        if 0 <= x_bev < bev_width and 0 <= y_bev < bev_height:
            # Color based on class (darker for better visibility on white background)
            if cls_name == 'Car':
                color = (0, 100, 0)  # Dark Green
            elif cls_name == 'Pedestrian':
                color = (100, 0, 0)  # Dark Blue
            elif cls_name == 'Cyclist':
                color = (0, 0, 100)  # Dark Red
            else:
                color = (100, 100, 0)  # Dark Cyan

            # Draw center point
            cv2.circle(bev_img, (x_bev, y_bev), 4, color, -1)

            # Draw box (thicker lines)
            pt1 = (x_bev - w_bev // 2, y_bev - l_bev // 2)
            pt2 = (x_bev + w_bev // 2, y_bev + l_bev // 2)
            cv2.rectangle(bev_img, pt1, pt2, color, 3)

            # Draw orientation line (showing which way the car is facing)
            ry = float(parts[14])

            # Calculate orientation vector (no rotation needed)
            orient_length = 20
            orient_x = int(x_bev + orient_length * np.cos(ry))
            orient_y = int(y_bev + orient_length * np.sin(ry))
            cv2.line(bev_img, (x_bev, y_bev), (orient_x, orient_y), color, 2)

            # Store detection info for text drawing after rotation
            detection_info.append({
                'x': x_bev,
                'y': y_bev,
                'cls_name': cls_name,
                'score': score,
                'color': color
            })

    return bev_img, detection_info


def draw_text_labels_after_rotation(bev_img, detection_info):
    """Draw text labels on BEV after 180° rotation."""
    if bev_img is None or len(detection_info) == 0:
        return bev_img

    bev_height, bev_width = bev_img.shape[:2]

    for det in detection_info:
        # Calculate rotated position (180° rotation)
        x_rot = bev_width - 1 - det['x']
        y_rot = bev_height - 1 - det['y']

        cls_name = det['cls_name']
        score = det['score']
        color = det['color']

        # Draw score with background
        text = f'{cls_name[:3]}:{score:.2f}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_bg_pt1 = (x_rot + 5, y_rot - 5 - text_size[1])
        text_bg_pt2 = (x_rot + 5 + text_size[0], y_rot - 5)

        # White background for text
        cv2.rectangle(bev_img, text_bg_pt1, text_bg_pt2, (255, 255, 255), -1)
        cv2.putText(bev_img, text, (x_rot + 5, y_rot - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return bev_img


def process_sample(sample_id, dataset_dir, result_dir, output_dir):
    """Process a single sample and save visualization."""
    # Paths
    bin_path = os.path.join(dataset_dir, 'testing', 'velodyne', f'{sample_id:06d}.bin')
    img_path = os.path.join(dataset_dir, 'testing', 'image_2', f'{sample_id:06d}.png')
    result_path = os.path.join(result_dir, f'{sample_id:06d}.txt')
    calib_path = os.path.join(dataset_dir, 'testing', 'calib', f'{sample_id:06d}.txt')

    # Check if files exist
    if not all(os.path.exists(path) for path in [bin_path, img_path, calib_path]):
        return None

    # Load data
    try:
        points = read_lidar_file_with_fallback(bin_path)
        calib = Calibration(calib_path)
    except Exception as e:
        return None

    # Load detections
    detections = []
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            detections = f.readlines()

    # Load camera image and draw 3D boxes
    img_rgb = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_with_boxes = draw_3d_boxes_on_image(img_rgb, detections, calib)

    # Create BEV map with colored points (standard coordinate system)
    bev_map, bev_size = create_colored_bev_from_points(points, cnf, rotate_180=False)
    bev_with_boxes, detection_info = draw_boxes_on_bev(bev_map, detections, cnf, bev_size, rotate_180=False)

    # Rotate BEV 180° to match test.py behavior (line 173)
    if bev_with_boxes is not None:
        bev_with_boxes = cv2.rotate(bev_with_boxes, cv2.ROTATE_180)
        # Draw text labels after rotation (so they appear correctly oriented)
        bev_with_boxes = draw_text_labels_after_rotation(bev_with_boxes, detection_info)

    # Create combined visualization
    if img_with_boxes is not None and bev_with_boxes is not None:
        # Resize images to same height
        h_img = img_with_boxes.shape[0]
        h_bev = bev_with_boxes.shape[0]

        # Scale BEV to match image height
        if h_img != h_bev:
            scale = h_img / h_bev
            new_w = int(bev_with_boxes.shape[1] * scale)
            bev_with_boxes = cv2.resize(bev_with_boxes, (new_w, h_img))

        # Combine horizontally
        combined = np.hstack([img_with_boxes, bev_with_boxes])

        # Add title
        combined = cv2.copyMakeBorder(combined, 80, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # Count detections by class
        class_counts = {}
        for det in detections:
            parts = det.strip().split()
            if len(parts) >= 1:
                cls_name = parts[0]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        det_str = ', '.join([f'{cls}:{cnt}' for cls, cnt in sorted(class_counts.items())])
        title = f'Sample {sample_id:06d} - BEV (Rotated 180° to match test.py) - Total: {len(detections)} ({det_str})'

        cv2.putText(combined, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Add coordinate system labels for BEV (after 180° rotation)
        cv2.putText(combined, "BEV: Top=Back(Y-), Bottom=Forward(Y+), Left=Right(X+), Right=Left(X-)", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(combined, "Front vehicle -> Bottom of BEV (after rotation)", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # Add color legend for BEV points
        legend_y = 70
        cv2.putText(combined, "Height:", (400, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.rectangle(combined, (450, legend_y-8), (460, legend_y), (0, 0, 255), -1)  # Blue (low)
        cv2.putText(combined, "Low", (465, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.rectangle(combined, (500, legend_y-8), (510, legend_y), (0, 255, 0), -1)  # Green (mid)
        cv2.putText(combined, "Mid", (515, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.rectangle(combined, (550, legend_y-8), (560, legend_y), (255, 0, 0), -1)  # Red (high)
        cv2.putText(combined, "High", (565, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Save
        output_path = os.path.join(output_dir, f'{sample_id:06d}_final.png')
        cv2.imwrite(output_path, combined)

        return len(detections)

    return None


def main():
    parser = argparse.ArgumentParser(description='Final batch visualization with corrected BEV')
    parser.add_argument('--dataset-dir', type=str,
                       default=r'D:\bev\SFA3D-modified\DRadDataset',
                       help='Dataset root directory')
    parser.add_argument('--result-dir', type=str,
                       default=r'D:\bev\SFA3D-modified\ultra_aggressive_results',
                       help='Inference results directory')
    parser.add_argument('--output-dir', type=str,
                       default=r'D:\bev\SFA3D-modified\visualizations_final',
                       help='Output directory for visualizations')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Process only first N samples (default: all)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("FINAL BATCH VISUALIZATION - STANDARD COORDINATE SYSTEM")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Results: {args.result_dir}")
    print(f"Output: {args.output_dir}")
    print(f"BEV Coordinate System:")
    print(f"  - Top of BEV = Forward (Y+)")
    print(f"  - Left of BEV = Left (X-)")
    print(f"  - Right of BEV = Right (X+)")
    print(f"  - Front vehicle appears at top of BEV")
    print(f"  - Left-front vehicle appears at top-left of BEV")
    print(f"{'='*80}\n")

    # Get all velodyne files
    velodyne_dir = os.path.join(args.dataset_dir, 'testing', 'velodyne')
    bin_files = sorted(Path(velodyne_dir).glob('*.bin'))

    if args.num_samples:
        bin_files = bin_files[:args.num_samples]

    print(f"Processing {len(bin_files)} samples...\n")

    # Process all samples
    t_start = time.time()
    total_detections = 0
    processed = 0

    for bin_file in tqdm(bin_files, desc="Generating final visualizations"):
        sample_id = int(bin_file.stem)
        num_det = process_sample(sample_id, args.dataset_dir, args.result_dir, args.output_dir)

        if num_det is not None:
            total_detections += num_det
            processed += 1

    elapsed = time.time() - t_start

    print(f"\n{'='*80}")
    print(f"COMPLETED")
    print(f"{'='*80}")
    print(f"Processed samples: {processed}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per sample: {total_detections / max(processed, 1):.2f}")
    print(f"Time elapsed: {elapsed:.2f}s ({processed / max(elapsed, 1e-9):.2f} samples/s)")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
