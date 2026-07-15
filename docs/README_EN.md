# SFA4D: 4D Millimeter-Wave Radar and Monocular Camera Fusion Algorithm — National Second Prize Open Source Solution

> 🏆 **2025 AIC Global Campus AI Algorithm Elite Competition · Algorithm Challenge — 4D Millimeter-Wave Radar and Monocular Camera Fusion Algorithm** — National Second Prize Open Source Solution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![AIC Competition](https://img.shields.io/badge/AIC-2025-green.svg)]()

---

## 📖 Project Overview

This project is the **complete open-source solution of the National Second Prize** winner in the **AIC Global Campus AI Algorithm Elite Competition (Algorithm Challenge)**, focusing on **3D object detection with 8D millimeter-wave radar point cloud data**.

Built upon the original SFA3D (Super Fast and Accurate 3D Object Detection), we have conducted in-depth optimization specifically for millimeter-wave radar data characteristics, proposing the **SFA4D** detection framework that achieves breakthrough performance from 8D radar data to high-precision 3D detection.

### Core Innovations
- **🎯 8D→4D Intelligent Mapping**: Pioneering data mapping scheme based on the 5th-dimensional SNR (Signal-to-Noise Ratio) intensity (P), mapping 8D radar point cloud `[x, y, z, Doppler, P, Range, Azimuth, Elevation]` to 4D point cloud `[x, y, z, intensity]`
- **🧠 KFPN Feature Fusion**: Adaptive softmax attention-weighted multi-scale Feature Pyramid Network
- **⚡ Anchor-Free Detection**: Based on CenterNet, directly predicts target center heatmap, offset, dimensions, orientation angle, and Z-coordinate
- **🔄 Cross-Class NMS**: Effectively resolves multi-target duplicate detection issues
- **🌐 End-to-End Learning**: Simultaneously predicts 7-DOF target attributes

---

## 🏆 Competition Results

| Metric | Result |
|--------|--------|
| **Competition Award** | **National Second Prize** in AIC Global Campus AI Algorithm Elite Competition · Algorithm Challenge |
| **Inference Speed** | **110.73 FPS** (RTX 4060 Ti, PyTorch FP32) |
| **Detection Accuracy** | **75% mAP@0.5** |
| **Model Size** | 48.57 MB (PyTorch) / 13 MB (quantized) |
| **Cross-Platform Inference** | 5.48 FPS (ONNX Runtime, CPU) |
| **Validation Samples** | Successfully processed 1,034 validation samples, detection rate 20.7% |

---

## 📂 Project Structure

```
SFA4D/
├── sfa/                          # Core code directory
│   ├── config/                   # Configuration files (BEV parameters, training parameters)
│   ├── data_process/             # Data processing (Dataset, BEV generation, 8D→4D mapping)
│   ├── models/                   # Model definitions (FPN-ResNet + KFPN)
│   ├── losses/                   # Loss functions (FocalLoss + L1Loss + BalancedL1Loss)
│   ├── utils/                    # Utility functions (NMS, visualization, training helpers)
│   └── *.py                      # Training / inference / export scripts
├── sample_data/                  # Sample dataset (3 samples for quick experience)
│   ├── velodyne/                 # 8D radar point clouds (.bin)
│   ├── label_2/                  # KITTI-format annotations (.txt)
│   └── calib/                    # Calibration parameters (.txt)
├── docs/                         # Technical documentation (Chinese)
├── requirements.txt              # Environment dependencies
├── README.md                     # Chinese README (main document)
├── README_CN.md                  # Chinese README (this file)
└── LICENSE                       # MIT open-source license
```

---

## 🚀 Quick Start

### Environment Requirements

- Python 3.8+
- PyTorch 2.0.0+cu118 (recommended) or 1.5.0+
- CUDA 11.8+ (for GPU inference)
- Windows 10/11 or Linux Ubuntu 18.04+

### Install Dependencies

```bash
# Create virtual environment (recommended)
conda create -n sfa4d python=3.8
conda activate sfa4d

# Install PyTorch (CUDA 11.8)
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Quick Experience (using sample data)

```bash
# Single GPU quick training (3 epochs, for environment verification)
python sfa/train.py \
    --num_epochs 3 \
    --saved_fn sfa4d_test \
    --batch_size 4 \
    --dataset-dir ./sample_data \
    --root-dir ./ \
    --gpu_idx 0
```

### Full Training (300 epochs)

```bash
python sfa/train.py \
    --num_epochs 300 \
    --saved_fn sfa4d_full \
    --batch_size 16 \
    --dataset-dir ./DRadDataset \
    --root-dir ./ \
    --gpu_idx 0 \
    --checkpoint_freq 1 \
    --print_freq 50
```

### Inference (using pretrained model)

```bash
# Ultra-aggressive NMS inference (recommended for validation/testing)
python sfa/testing_export_ultra_aggressive.py \
    --pretrained_path ./checkpoints/sfa4d_full/Model_sfa4d_full_epoch_163.pth \
    --dataset-dir ./DRadDataset \
    --saved_fn sfa4d_163_ultra \
    --peak_thresh 0.25 \
    --nms_thresh 0.2 \
    --gpu_idx 0
```

---

## 📊 Dataset

### Data Format

This project uses the **DRadDataset** dataset, containing 8D millimeter-wave radar point cloud data:

| Dimension | Meaning | Description |
|-----------|---------|-------------|
| 0 | x | Forward distance (meters) |
| 1 | y | Lateral distance (meters) |
| 2 | z | Height (meters) |
| 3 | Doppler | Doppler velocity |
| 4 | P (SNR) | **Signal-to-Noise Ratio intensity** → mapped to intensity |
| 5 | Range | Radial distance |
| 6 | Azimuth | Azimuth angle |
| 7 | Elevation | Elevation angle |

### Dataset Download

Full DRadDataset dataset (5,168 training samples + 1,384 testing samples) can be obtained via GitHub Release or cloud drive.

---

## 🧠 Pretrained Models

| Model | Size | Description | Download |
|-------|------|-------------|----------|
| Epoch 163 (PyTorch) | 49 MB | Recommended best model | [GitHub Release v1.0](https://github.com/cainiao33/AIC-4D-Radar-Camera-Fusion/releases/tag/v1.0) |
| ONNX FP32 | 49 MB | Cross-platform deployment | [GitHub Release v1.0](https://github.com/cainiao33/AIC-4D-Radar-Camera-Fusion/releases/tag/v1.0) |
| Dynamic Quantization | 13 MB | Edge device deployment | [GitHub Release v1.0](https://github.com/cainiao33/AIC-4D-Radar-Camera-Fusion/releases/tag/v1.0) |

### Model Export

```bash
# PyTorch → ONNX
python sfa/export_to_onnx.py \
    --model ./checkpoints/sfa4d_full/Model_sfa4d_full_epoch_163.pth \
    --output ./onnx_models/sfa4d_163_fp32.onnx

# Model Quantization
python sfa/quantize_model_163.py \
    --model ./checkpoints/sfa4d_full/Model_sfa4d_full_epoch_163.pth \
    --method dynamic \
    --output ./quantized_models/
```

---

## 📈 Performance Metrics

### Per-Category Detection Performance

| Category | mAP@0.5 | Description |
|----------|---------|-------------|
| Car | **83%** | Vehicle detection (large target, distinct features) |
| Cyclist | **65%** | Cyclist detection (small target, most challenging) |
| Truck | **70%** | Truck detection (medium-sized target) |
| **Average** | **73%** | **Overall average** |

### Inference Speed Comparison

| Platform | Framework | Speed | Description |
|----------|-----------|-------|-------------|
| RTX 4060 Ti | PyTorch FP32 | **110.73 FPS** | GPU inference |
| CPU | ONNX Runtime | 5.48 FPS | Cross-platform deployment |
| CPU | Quantized model | 4-5 FPS | Edge device |

---

## 📚 Technical Documentation

| Document | Description |
|----------|-------------|
| [docs/Technical_Solution_AAA.md](docs/Technical_Solution_AAA.md) | Complete technical implementation solution ⭐ |
| [docs/Technical_Report.md](docs/Technical_Report.md) | In-depth technical analysis report |
| [docs/Project_Structure.md](docs/Project_Structure.md) | Project architecture and file descriptions |
| [docs/Environment_Dependencies.md](docs/Environment_Dependencies.md) | Detailed environment configuration requirements |
| [docs/Point_Cloud_Dimension_Modification.md](docs/Point_Cloud_Dimension_Modification.md) | 8D→4D mapping implementation details |
| [docs/SFA4D_Code_Architecture.md](docs/SFA4D_Code_Architecture.md) | Code architecture detailed explanation |

---

## 🎥 Demo Results

### Detection Visualization

![Detection Results](demo/000000_final.png)

> The above image shows the 3D object detection results of SFA4D in BEV perspective, with white background and gradient point clouds, detection boxes distinguished by different colors for categories.

### Demo Results

**GIF Auto-Play Preview (2min Full Demo):**

<img src="https://github.com/cainiao33/AIC-4D-Radar-Camera-Fusion/raw/main/demo/preview_120s.gif" width="800" alt="SFA4D Detection Full Demo">

> 2-minute full detection demo (3fps, 400px width). If loading is slow, watch the video below or visit [GitHub Release v1.0](https://github.com/cainiao33/AIC-4D-Radar-Camera-Fusion/releases/tag/v1.0).

**Click to Play Full Video (Higher Quality):**

<video src="https://github.com/cainiao33/AIC-4D-Radar-Camera-Fusion/releases/download/v1.0/visualization_12fps.mp4" controls width="100%" poster="demo/000000_final.png"></video>

> Full demo video (with OpenCV rendering) available at [GitHub Release v1.0](https://github.com/cainiao33/AIC-4D-Radar-Camera-Fusion/releases/tag/v1.0).

---

## 🔧 Core Module Description

### Data Flow

```
Raw 8D point cloud (.bin)
    ↓
lidar_mapping.py: read_lidar_file_with_fallback()
    → Auto-detect 8D/5D/4D, for 8D extract [0,1,2,4] and map P to intensity
    ↓
kitti_bev_utils.py: makeBEVMap()
    → Generate 3-channel BEV image [intensity, height, density], 608×608
    ↓
Dataset / DataLoader
    ↓
Model input: (B, 3, 608, 608)
    ↓
fpn_resnet.py: PoseResNet + KFPN
    → Output 5 head feature maps (B, C, 152, 152)
    ↓
losses.py: Compute_Loss
    → focal_loss + l1_loss + balanced_l1_loss weighted sum
    ↓
evaluation_utils.py: decode + post_processing
    → Heatmap NMS → Top-K → Coordinate restoration → Cross-class NMS
    ↓
KITTI-format detection results / Visualization images
```

### Key Configuration Parameters

```python
# BEV boundaries (sfa/config/kitti_config.py)
boundary = {
    "minX": 0,   "maxX": 50,     # Forward 0~50m
    "minY": -25, "maxY": 25,     # Left-right ±25m
    "minZ": -2.73, "maxZ": 1.27  # Height range
}
BEV_WIDTH = 608
BEV_HEIGHT = 608
DISCRETIZATION = 50 / 608  # ≈ 0.0822 m/pixel

# Model output heads
heads = {
    'hm_cen': 3,      # Category heatmap (Car, Cyclist, Truck)
    'cen_offset': 2,  # Center point sub-pixel offset
    'direction': 2,   # Orientation angle (sin, cos)
    'z_coor': 1,      # Z-coordinate
    'dim': 3,         # Dimensions (h, w, l)
}
```

---

## 🤝 Contribution Guide

Issues and Pull Requests are welcome!

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- Thanks to the **AIC Global Campus AI Algorithm Elite Competition** for providing the competition platform and dataset
- Thanks to the original [SFA3D](https://github.com/maudzung/SFA3D) project for its open-source contribution
- Thanks to all team members for their hard work

---

## 📧 Contact

- Competition Website: [AIC Global Campus AI Algorithm Elite Competition](https://www.aicomp.cn/)
- Email: **2911684894@qq.com**

---

> **⭐ If this project helps you, please give it a Star!**
