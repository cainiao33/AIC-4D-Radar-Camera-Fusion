# ONNX模型导出和推理指南

## 一、概述

本指南介绍如何将163轮训练的SFA3D-Modified模型导出为ONNX格式，并使用ONNX Runtime进行高效推理(实际推理使用SFA4D\checkpoints\sfa3d_8d_full_300epochs\Model_sfa3d_8d_full_300epochs_epoch_163.pth模型)。

**ONNX的优势**：
- ✅ **跨平台**：支持Windows、Linux、macOS
- ✅ **跨框架**：不依赖PyTorch，可在任何支持ONNX的环境运行
- ✅ **高效推理**：ONNX Runtime经过高度优化
- ✅ **边缘部署**：适合嵌入式设备和移动端
- ✅ **模型压缩**：支持多种量化和优化技术

---

## 二、环境准备

### 2.1 安装依赖包

```bash
# 激活sfa3d环境
conda activate sfa3d

# 安装ONNX相关包
pip install onnx onnxruntime

# 可选：安装GPU版本的ONNX Runtime（如果有CUDA GPU）
pip install onnxruntime-gpu
```

### 2.2 验证安装

```python
import onnx
import onnxruntime as ort
print(f"ONNX版本: {onnx.__version__}")
print(f"ONNX Runtime版本: {ort.__version__}")
print(f"可用Providers: {ort.get_available_providers()}")
```

---

## 三、模型导出

### 3.1 导出FP32模型

**基本导出**：
```bash
cd D:/bev/SFA3D-modified

python sfa/export_to_onnx.py \
  --model checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth \
  --output onnx_models/sfa3d_163_fp32.onnx \
  --opset 11
```

**带测试的导出**：
```bash
python sfa/export_to_onnx.py \
  --model checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth \
  --output onnx_models/sfa3d_163_fp32.onnx \
  --opset 11 \
  --test
```

**参数说明**：
- `--model`: 输入的PyTorch模型路径
- `--output`: 输出的ONNX文件路径
- `--opset`: ONNX opset版本（推荐11或更高）
- `--test`: 导出后立即测试推理

### 3.2 导出配置

**模型输入**：
- **名称**: `bev_input`
- **形状**: `[batch_size, 3, 608, 608]`
- **类型**: `float32`
- **说明**: 3通道BEV图像 (intensity, height, density)

**模型输出**：
1. **hm_cen**: `[batch_size, 3, 152, 152]` - 中心点热力图
2. **cen_offset**: `[batch_size, 2, 152, 152]` - 中心点偏移
3. **direction**: `[batch_size, 2, 152, 152]` - 方向向量
4. **z_coor**: `[batch_size, 1, 152, 152]` - Z坐标
5. **dim**: `[batch_size, 3, 152, 152]` - 3D尺寸

### 3.3 验证ONNX模型

```python
import onnx

# 加载模型
model = onnx.load("onnx_models/sfa3d_163_fp32.onnx")

# 验证模型
onnx.checker.check_model(model)

# 打印模型信息
print(onnx.helper.printable_graph(model.graph))
```

---

## 四、ONNX Runtime推理

### 4.1 基本推理

```python
import onnxruntime as ort
import numpy as np

# 创建推理会话
session = ort.InferenceSession("onnx_models/sfa3d_163_fp32.onnx")

# 准备输入数据
input_data = np.random.randn(1, 3, 608, 608).astype(np.float32)

# 推理
outputs = session.run(None, {'bev_input': input_data})

# 解析输出
hm_cen, cen_offset, direction, z_coor, dim = outputs
```

### 4.2 在验证集上推理

```bash
python sfa/run_onnx_inference.py \
  --onnx_model onnx_models/sfa3d_163_fp32.onnx \
  --dataset-dir DRadDataset \
  --imagesets-dir DRadDataset/ImageSets \
  --num_samples 1034
```

### 4.3 性能优化

**启用并行计算**：
```python
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4  # 设置线程数
sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

session = ort.InferenceSession(
    "onnx_models/sfa3d_163_fp32.onnx",
    sess_options,
    providers=['CPUExecutionProvider']
)
```

**使用GPU加速**（需安装onnxruntime-gpu）：
```python
session = ort.InferenceSession(
    "onnx_models/sfa3d_163_fp32.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

---

## 五、模型优化

### 5.1 ONNX模型量化

**动态量化**：
```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input='onnx_models/sfa3d_163_fp32.onnx',
    model_output='onnx_models/sfa3d_163_int8_dynamic.onnx',
    weight_type=QuantType.QInt8
)
```

**静态量化**（需要校准数据）：
```python
from onnxruntime.quantization import quantize_static, CalibrationDataReader

class BEVDataReader(CalibrationDataReader):
    def __init__(self, calibration_dataset):
        self.dataset = calibration_dataset
        self.iterator = iter(self.dataset)

    def get_next(self):
        try:
            return {'bev_input': next(self.iterator)}
        except StopIteration:
            return None

# 执行静态量化
quantize_static(
    model_input='onnx_models/sfa3d_163_fp32.onnx',
    model_output='onnx_models/sfa3d_163_int8_static.onnx',
    calibration_data_reader=BEVDataReader(calibration_data)
)
```

### 5.2 模型图优化

```python
from onnxruntime.transformers.optimizer import optimize_model

# 优化ONNX图
optimized_model = optimize_model(
    'onnx_models/sfa3d_163_fp32.onnx',
    model_type='bert',  # 或其他类型
    num_heads=0,
    hidden_size=0
)

optimized_model.save_model_to_file('onnx_models/sfa3d_163_optimized.onnx')
```

---

## 六、性能对比

### 6.1 预期性能

| 模型格式 | 文件大小 | 推理速度 | 精度 | 部署难度 |
|---------|---------|---------|------|---------|
| PyTorch FP32 | 49 MB | 110 FPS | 基准 | 高（需要PyTorch） |
| ONNX FP32 | ~49 MB | 120-150 FPS | ≈100% | 低（仅需ONNX Runtime） |
| ONNX INT8 (动态) | ~13 MB | 150-200 FPS | ~98% | 低 |
| ONNX INT8 (静态) | ~13 MB | 200-250 FPS | ~97% | 低 |

*注：实际性能取决于硬件平台和优化设置*

### 6.2 测试方法

```bash
# 测试PyTorch模型
python sfa/run_163_working.py \
  --dataset-dir DRadDataset \
  --imagesets-dir DRadDataset/ImageSets \
  --num_samples 1034

# 测试ONNX模型
python sfa/run_onnx_inference.py \
  --onnx_model onnx_models/sfa3d_163_fp32.onnx \
  --dataset-dir DRadDataset \
  --imagesets-dir DRadDataset/ImageSets \
  --num_samples 1034
```

---

## 七、常见问题

### 7.1 导出时出现TracerWarning

**问题**：
```
TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect
```

**解决方案**：
这是正常的警告，不影响模型功能。可以通过修改模型代码避免动态控制流。

### 7.2 ONNX模型验证失败

**问题**：
```
onnx.checker.ValidationError: ...
```

**解决方案**：
1. 检查opset版本是否兼容
2. 尝试更低的opset版本（如opset=11）
3. 更新onnx包：`pip install --upgrade onnx`

### 7.3 推理速度不如预期

**解决方案**：
1. 使用GPU Provider（如果有GPU）
2. 增加线程数：`sess_options.intra_op_num_threads`
3. 启用图优化：`sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL`
4. 使用量化模型

---

## 八、部署示例

### 8.1 Python部署

```python
import onnxruntime as ort
import numpy as np

class SFA3DDetector:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, bev_image):
        """
        Args:
            bev_image: numpy array, shape (3, 608, 608)
        Returns:
            detections: list of detection results
        """
        # 添加batch维度
        input_data = np.expand_dims(bev_image, axis=0).astype(np.float32)

        # 推理
        outputs = self.session.run(None, {self.input_name: input_data})

        # 后处理
        # ... (根据需要实现)

        return outputs

# 使用
detector = SFA3DDetector('onnx_models/sfa3d_163_fp32.onnx')
detections = detector.detect(bev_image)
```

### 8.2 C++部署

```cpp
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

class SFA3DDetector {
private:
    Ort::Env env;
    Ort::Session session;

public:
    SFA3DDetector(const char* model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "SFA3D"),
          session(env, model_path, Ort::SessionOptions()) {}

    std::vector<float> detect(const std::vector<float>& input_data) {
        // 创建输入tensor
        std::vector<int64_t> input_shape = {1, 3, 608, 608};
        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(input_data.data()),
            input_data.size(), input_shape.data(), input_shape.size());

        // 推理
        const char* input_names[] = {"bev_input"};
        const char* output_names[] = {"hm_cen", "cen_offset",
                                      "direction", "z_coor", "dim"};

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
            output_names, 5);

        // 处理输出
        // ...

        return results;
    }
};
```

---

## 九、文件清单

### 9.1 脚本文件

| 文件名 | 说明 |
|--------|------|
| sfa/export_to_onnx.py | ONNX导出脚本 |
| sfa/run_onnx_inference.py | ONNX Runtime推理脚本 |
| onnx_models/ONNX_EXPORT_GUIDE.md | 本指南文档 |

### 9.2 模型文件（待生成）

| 文件名 | 大小 | 说明 |
|--------|------|------|
| onnx_models/sfa3d_163_fp32.onnx | ~49 MB | FP32 ONNX模型 |
| onnx_models/sfa3d_163_int8_dynamic.onnx | ~13 MB | INT8动态量化模型 |
| onnx_models/sfa3d_163_int8_static.onnx | ~13 MB | INT8静态量化模型 |

---

## 十、后续工作

### 10.1 待完成任务

- [ ] 安装ONNX和ONNX Runtime
- [ ] 导出FP32 ONNX模型
- [ ] 验证ONNX模型推理正确性
- [ ] 在验证集上测试ONNX模型性能
- [ ] 导出INT8量化ONNX模型
- [ ] 对比PyTorch vs ONNX性能
- [ ] 在边缘设备上部署测试

### 10.2 优化方向

1. **性能优化**：
   - 图优化（fusion、constant folding等）
   - 混合精度推理（FP16 + INT8）
   - TensorRT加速（NVIDIA GPU）

2. **精度优化**：
   - 量化敏感层分析
   - QAT（Quantization-Aware Training）
   - 校准数据集优化

3. **部署优化**：
   - 模型压缩（pruning + quantization）
   - 动态shape支持
   - 批处理优化

---

**文档版本**: 1.0
**创建时间**: 2025-11-02
**更新时间**: 2025-11-02

**参考资源**：
- [ONNX官方文档](https://onnx.ai/)
- [ONNX Runtime文档](https://onnxruntime.ai/)
- [PyTorch ONNX导出](https://pytorch.org/docs/stable/onnx.html)
