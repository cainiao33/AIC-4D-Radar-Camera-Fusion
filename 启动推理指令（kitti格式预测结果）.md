# SFA4D 推理指令文档

## 概述

本文档包含SFA4D项目的推理指令，包括超激进NMS策略的推理命令、模型配置和参数说明。

## 环境要求

- Python 3.8+
- PyTorch 2.0.0+cu118
- CUDA 11.8+
- conda环境：sfa3d

## 激活环境

```bash
source activate sfa3d
```

## 超激进NMS推理（推荐）

### 核心推理指令

```bash
source activate sfa3d && python sfa/testing_export_ultra_aggressive.py --pretrained_path ./checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth --dataset-dir ./DRadDataset --saved_fn sfa4d_163_ultra_aggressive --peak_thresh 0.25 --nms_thresh 0.2 --gpu_idx 0
```

### 测试推理指令（10个样本）

```bash
source activate sfa3d && python sfa/testing_export_ultra_aggressive.py --pretrained_path ./checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth --dataset-dir ./DRadDataset --saved_fn sfa4d_163_test --num_samples 10 --peak_thresh 0.25 --nms_thresh 0.2 --gpu_idx 0
```

### 完整数据集推理

```bash
source activate sfa3d && python sfa/testing_export_ultra_aggressive.py --pretrained_path ./checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth --dataset-dir ./DRadDataset --saved_fn sfa4d_163_full --peak_thresh 0.25 --nms_thresh 0.2 --gpu_idx 0 --batch_size 1
```

## 参数说明

### 核心参数

- `--pretrained_path`: 模型权重路径（推荐使用163轮模型）
- `--dataset-dir`: 数据集根目录
- `--saved_fn`: 输出文件名前缀
- `--gpu_idx`: GPU设备索引

### 超激进NMS参数

- `--peak_thresh 0.25`: 峰值阈值（0.25），更高阈值减少误检
- `--nms_thresh 0.2`: NMS IoU阈值（0.2），更严格去除重复检测

### 其他常用参数

- `--num_samples N`: 限制推理样本数量（用于快速测试）
- `--batch_size N`: 批处理大小（推荐1）
- `--output-dir PATH`: 自定义输出目录
- `--no_cuda`: 强制使用CPU推理

## 推理脚本对比

### 1. 超激进NMS推理（推荐）
```bash
python sfa/testing_export_ultra_aggressive.py
```
**特点**：
- `peak_thresh=0.25`（高精度）
- `nms_thresh=0.2`（去重严格）
- 适用于高精度要求的场景

### 2. 标准推理
```bash
python sfa/testing.py
```
**特点**：
- 默认参数设置
- 平衡精度和召回率

### 3. 验证集推理
```bash
python sfa/validation_ultra_aggressive_163.py
```
**特点**：
- 专门用于验证集评估
- 支持指标计算

## 性能基准

### 推理速度

- **初始化时间**: ~500ms（首次加载模型）
- **单样本推理**: 8-15ms
- **平均FPS**: ~2.8（包含I/O）

### 检测效果

- **平均检测数**: 1.0个/样本
- **超激进策略**: 显著减少误检
- **精度优化**: 适合自动驾驶场景

## 输出结果

### 目录结构

```
results/
├── sfa4d_163_ultra_aggressive/
│   ├── data/          # KITTI格式检测结果
│   └── images/        # 可视化结果（可选）
└── logs/              # 推理日志
```

### 输出格式

- **检测结果**: KITTI格式 `.txt` 文件
- **格式**: `[类别] [截断] [遮挡] [角度] [边界框] [维度] [位置] [旋转] [得分]`

## 模型信息

### 163轮模型（推荐）

**路径**: `./checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth`

**特点**：
- 经过完整300轮训练
- 在第163轮达到最佳性能
- 针对8D毫米波雷达数据优化
- 支持Car、Cyclist、Truck三类检测

### 自训练模型

如果使用自己的训练模型：

```bash
source activate sfa3d && python sfa/testing_export_ultra_aggressive.py --pretrained_path ./checkpoints/your_model/Model_your_model_best.pth --dataset-dir ./DRadDataset --saved_fn your_inference --peak_thresh 0.25 --nms_thresh 0.2 --gpu_idx 0
```

## 数据处理说明

### 8D数据处理流程

```
8D数据 [x,y,z,D,P,R,A,E]
→ 读取 [0,1,2,4] 维度
→ P映射为intensity
→ 输出4D [x,y,z,intensity]
```

### 维度对应关系

- **0**: X坐标
- **1**: Y坐标
- **2**: Z坐标
- **4**: SNR强度值

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少批处理大小
   --batch_size 1
   ```

2. **模型路径错误**
   ```bash
   # 检查模型文件是否存在
   ls -la ./checkpoints/sfa3d_8d_full_300epochs/
   ```

3. **数据集路径错误**
   ```bash
   # 确认数据集目录结构
   ls -la ./DRadDataset/testing/velodyne/
   ```

### 性能优化

1. **提高推理速度**
   ```bash
   # 增加批处理大小
   --batch_size 2
   ```

2. **减少内存使用**
   ```bash
   # 限制样本数量
   --num_samples 100
   ```

## 日志和监控

### 推理日志示例

```
================================================================================
ULTRA-AGGRESSIVE NMS INFERENCE MODE
================================================================================
NMS Threshold: 0.2 (lower = more aggressive)
Peak Threshold: 0.25 (higher = fewer detections)
Inter-class NMS: Enabled
================================================================================

Processed batch 0 (1 samples) in 492.7 ms
...
Finished 10 samples in 3.54s (2.82 FPS)
Total detections: 10
Average detections per sample: 1.00
```

## 版本信息

- **创建日期**: 2025年11月5日
- **项目版本**: SFA4D v1.0
- **推理引擎**: PyTorch 2.0.0+cu118
- **推荐模型**: 163轮训练模型

---

**注意**:
1. 推理前确保已激活sfa3d虚拟环境
2. 推荐使用163轮模型以获得最佳性能
3. 超激进NMS参数（0.25, 0.2）已针对毫米波雷达数据优化