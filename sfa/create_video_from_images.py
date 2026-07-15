"""
用可视化图像以24帧每秒的速度制作视频-
"""
import cv2
import os
from pathlib import Path
from tqdm import tqdm

def create_video(image_dir, output_video, fps=24):
    """Create video from images in directory."""
    # Get all PNG files sorted by name
    image_files = sorted(Path(image_dir).glob('*_final.png'))

    if len(image_files) == 0:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images")

    # Read first image to get dimensions
    first_img = cv2.imread(str(image_files[0]))
    height, width, _ = first_img.shape

    print(f"Video dimensions: {width}x{height}")
    print(f"Frame rate: {fps} fps")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Write all images to video
    for img_path in tqdm(image_files, desc="Creating video"):
        img = cv2.imread(str(img_path))
        video_writer.write(img)

    video_writer.release()

    # Calculate video duration
    duration = len(image_files) / fps
    print(f"\nVideo created successfully!")
    print(f"Output: {output_video}")
    print(f"Total frames: {len(image_files)}")
    print(f"Duration: {duration:.2f} seconds")


if __name__ == '__main__':
    image_dir = r'D:\bev\SFA3D-modified\ultra_aggressive_all_visualizations_fixed'
    output_video = r'D:\bev\SFA3D-modified\ultra_aggressive_visualization_12fps_fixed.mp4'

    create_video(image_dir, output_video, fps=12)
