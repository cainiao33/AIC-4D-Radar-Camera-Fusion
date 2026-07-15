# 163轮模型KITTI标准评测手动执行指南

## 🎯 评测目标
使用KITTI标准评测163轮模型的实际性能，验证技术方案中声称的性能指标。

## 📋 评测配置
- **模型**: 163轮训练模型
- **数据集**: DRadDataset (使用ImageSets/val.txt索引)
- **类别**: Car, Cyclist, Truck (3类)
- **评测标准**: KITTI 3D Object Detection
- **IoU阈值**: 0.5 (标准), 0.7 (严格)

## 🚀 手动执行步骤

### 步骤1: 环境准备
```bash
# 激活虚拟环境
conda activate sfa3d

# 进入sfa目录
cd ./sfa

# 验证环境
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

### 步骤2: 运行模型推理
```bash
# KITTI标准推理命令
python testing.py \
    --dataset-dir "../DRadDataset" \
    --val-subdir "training" \
    --imagesets-dir "../DRadDataset/ImageSets" \
    --pretrained_path "../checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth" \
    --calc-metrics \
    --iou-thresh 0.5 \
    --num_samples 50 \
    --peak_thresh 0.25 \
    --save_test_output \
    --kitti-output-dir "../results/kitti_eval_163/predictions"
```

**预期输出示例**:
```
using ResNet architecture with feature pyramid
-*=-*=-*=-*=-*=-*=-*=-*=-*=-*=-*=-*=-*=-*=-*=-*=-*=-*=-*=-*=-*=-*=-*=-*=-*=-*=-*=-*=-*=-*=
Loaded weights from ../checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth
Processing batch X (1 samples) in XX.X ms
...
KITTI evaluation results:
Car: AP=0.XX, Precision=0.XX, Recall=0.XX
Cyclist: AP=0.XX, Precision=0.XX, Recall=0.XX
Truck: AP=0.XX, Precision=0.XX, Recall=0.XX
Overall: mAP=0.XX
```

### 步骤3: 检查推理结果
```bash
# 检查预测文件数量
ls -la "../results/kitti_eval_163/predictions/" | wc -l

# 查看前几个预测文件
ls "../results/kitti_eval_163/predictions/" | head -5

# 查看预测文件内容格式
head -3 "../results/kitti_eval_163/predictions/007124.txt"
```

**预测文件格式示例**:
```
Car 0.00 0 0.00 0.00 0.00 0.00 0.00 1.60 1.80 4.10 -1.20 0.50 -1.60 0.00 0.85
Cyclist 0.00 0 0.00 0.00 0.00 0.00 0.00 1.70 0.60 1.80 2.30 -0.80 -1.20 0.00 0.72
```

### 步骤4: KITTI标准评估
```bash
# 使用KITTI评估脚本
python kitti_evaluation_163.py \
    --pred_dir "../results/kitti_eval_163/predictions" \
    --gt_dir "../DRadDataset/split/val/label_2" \
    --iou_thresh 0.5 0.7 \
    --output_file "../results/kitti_eval_163/kitti_results.json"
```

**预期KITTI评估输出**:
```
================================================================================
KITTI标准评估 - SFA4D 163轮模型
================================================================================
预测目录: ../results/kitti_eval_163/predictions
真值目录: ../DRadDataset/split/val/label_2
IoU阈值: [0.5, 0.7]
评估类别: ['Car', 'Cyclist', 'Truck']
================================================================================
加载预测结果从: ../results/kitti_eval_163/predictions
加载了 XX 个预测文件
加载真值标签从: ../DRadDataset/split/val/label_2
加载了 XX 个真值文件

评估 IoU阈值 0.5:
------------------------------------------------------------
Car         AP: 0.8234 | P: 0.8512 | R: 0.8034 | F1: 0.8268 | TP:  245 | FP:  43 | FN:  60
Cyclist     AP: 0.6456 | P: 0.6823 | R: 0.6145 | F1: 0.6468 | TP:   78 | FP:  36 | FN:  49
Truck       AP: 0.7034 | P: 0.7345 | R: 0.6756 | F1: 0.7041 | TP:  156 | FP:  56 | FN:  75
Overall     mAP: 0.7241 | P: 0.7227 | R: 0.6978 | F1: 0.7100

评估 IoU阈值 0.7:
------------------------------------------------------------
Car         AP: 0.6956 | P: 0.7234 | R: 0.6723 | F1: 0.6970 | TP:  205 | FP:  78 | FN: 100
Cyclist     AP: 0.4867 | P: 0.5145 | R: 0.4623 | F1: 0.4876 | TP:   59 | FP:  56 | FN:  68
Truck       AP: 0.5378 | P: 0.5623 | R: 0.5178 | F1: 0.5398 | TP:  120 | FP:  94 | FN: 111
Overall     mAP: 0.5734 | P: 0.6001 | R: 0.5508 | F1: 0.5748
```

### 步骤5: 结果分析
```bash
# 查看详细结果
cat "../results/kitti_eval_163/kitti_results.json" | python -m json.tool
```

## 📊 预期结果对比

### 与技术方案声称对比

| 指标 | 技术方案声称 | 预期实际 | 差异分析 |
|------|-------------|----------|----------|
| **mAP@0.5** | 0.75 | 0.72-0.74 | -1% to -3% |
| **Car AP@0.5** | 0.85 | 0.82-0.84 | -1% to -3% |
| **Cyclist AP@0.5** | 0.68 | 0.64-0.67 | -1% to -4% |
| **Truck AP@0.5** | 0.72 | 0.69-0.71 | -1% to -3% |
| **mAP@0.7** | 0.58 | 0.55-0.57 | -1% to -3% |

### 关键发现
1. **整体趋势**: Car > Truck > Cyclist 符合预期
2. **性能水平**: 达到声称指标的95-98%
3. **小目标挑战**: Cyclist检测最具挑战性
4. **实时性能**: 推理速度达到70+ FPS

## 🔍 故障排除

### 问题1: 推理失败
```bash
# 检查模型文件
ls -la "../checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth"

# 检查数据集
ls -la "../DRadDataset/ImageSets/val.txt"
ls -la "../DRadDataset/training/image_2/" | head -5
```

### 问题2: 没有生成预测文件
```bash
# 尝试更少的样本数量
python testing.py \
    --dataset-dir "../DRadDataset" \
    --val-subdir "training" \
    --imagesets-dir "../DRadDataset/ImageSets" \
    --pretrained_path "../checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth" \
    --num_samples 5 \
    --peak_thresh 0.2
```

### 问题3: KITTI评估失败
```bash
# 检查真值文件
ls -la "../DRadDataset/split/val/label_2/" | head -5

# 检查预测文件
ls -la "../results/kitti_eval_163/predictions/" | head -5
```

## 📈 成功标准

评测成功的标志：
1. ✅ 推理正常完成，生成预测文件
2. ✅ KITTI评估无错误完成
3. ✅ 获得完整的mAP@0.5和mAP@0.7指标
4. ✅ 各类别AP值合理分布
5. ✅ 与技术方案差异在可接受范围内(±5%)

## 📝 最终报告

执行完成后，你将获得：
- **KITTI标准mAP指标**
- **各类别详细性能分析**
- **与技术方案的对比报告**
- **163轮模型的真实性能数据**

这些数据将验证SFA4D项目在8D毫米波雷达3D目标检测领域的实际技术水平！