"""
PyTorch Official Quantization Tool for 163-Epoch Model
使用PyTorch官方量化工具将FP32模型转换为INT8

支持的量化方法：
1. Dynamic Quantization (动态量化) - 适用于LSTM/Transformer
2. Static Quantization (静态量化) - 适用于CNN，需要校准数据
3. Quantization Aware Training (量化感知训练) - 训练时量化
"""

import torch
import torch.nn as nn
import torch.quantization
from torch.quantization import get_default_qconfig, quantize_jit
import sys
import os
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from models.fpn_resnet import get_pose_net
from quantize_config import get_quantization_config


class ModelQuantizer:
    """模型量化器 - 使用PyTorch官方量化API"""

    def __init__(self, model_path, output_dir='quantized_models'):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        print("="*80)
        print("PyTorch Official Model Quantization Tool")
        print("="*80)
        print(f"Model Path: {model_path}")
        print(f"Output Directory: {output_dir}")
        print()

    def load_fp32_model(self):
        """加载FP32原始模型"""
        print("[1/5] Loading FP32 Model...")

        # 使用独立的量化配置
        cfg = get_quantization_config()

        model = get_pose_net(
            num_layers=cfg.num_layers,
            heads=cfg.heads,
            head_conv=cfg.head_conv,
            imagenet_pretrained=cfg.imagenet_pretrained
        )

        checkpoint = torch.load(self.model_path, map_location='cpu')

        # 兼容不同的checkpoint格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # 直接是state_dict
            model.load_state_dict(checkpoint)

        model.eval()

        # 计算模型大小
        model_size = os.path.getsize(self.model_path) / (1024 * 1024)
        print(f"   Model loaded successfully")
        print(f"   Model size: {model_size:.2f} MB")

        return model

    def get_model_size(self, model):
        """获取模型大小（内存占用）"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb

    def dynamic_quantization(self, model):
        """动态量化 - 最简单，适用于权重量化"""
        print("\n[2/5] Applying Dynamic Quantization...")
        print("   Method: torch.quantization.quantize_dynamic")
        print("   Target layers: nn.Linear, nn.Conv2d")

        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},  # 量化的层类型
            dtype=torch.qint8
        )

        print("   Dynamic quantization completed")
        return quantized_model

    def static_quantization(self, model, calibration_data=None):
        """静态量化 - 需要校准数据，精度更高"""
        print("\n[2/5] Applying Static Quantization...")
        print("   Method: torch.quantization (with calibration)")

        # 设置量化配置
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        # 准备模型
        print("   Preparing model for quantization...")
        torch.quantization.prepare(model, inplace=True)

        # 校准（如果提供校准数据）
        if calibration_data is not None:
            print("   Calibrating with representative data...")
            model.eval()
            with torch.no_grad():
                for data in calibration_data:
                    model(data)
        else:
            print("   WARNING: No calibration data provided, using dummy data")
            # 使用虚拟数据进行校准
            cfg = get_quantization_config()
            dummy_input = torch.randn(1, cfg.num_input_channels, cfg.input_size[0], cfg.input_size[1])
            with torch.no_grad():
                model(dummy_input)

        # 转换为量化模型
        print("   Converting to quantized model...")
        torch.quantization.convert(model, inplace=True)

        print("   Static quantization completed")
        return model

    def save_quantized_model(self, model, method_name):
        """保存量化模型"""
        print(f"\n[3/5] Saving Quantized Model ({method_name})...")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"Model_163_quantized_{method_name}_{timestamp}.pth"

        # 保存完整模型（包括量化参数）
        torch.save({
            'model': model,
            'quantization_method': method_name,
            'timestamp': timestamp
        }, output_path)

        # 获取文件大小
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   Saved to: {output_path}")
        print(f"   File size: {file_size:.2f} MB")

        return output_path, file_size

    def benchmark_inference(self, model, num_iterations=100):
        """基准测试推理速度"""
        print(f"\n[4/5] Benchmarking Inference Speed ({num_iterations} iterations)...")

        cfg = get_quantization_config()
        dummy_input = torch.randn(1, cfg.num_input_channels, cfg.input_size[0], cfg.input_size[1])
        model.eval()

        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # 计时
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        end_time = time.time()

        avg_time = (end_time - start_time) / num_iterations
        fps = 1.0 / avg_time

        print(f"   Average inference time: {avg_time*1000:.2f} ms")
        print(f"   Inference FPS: {fps:.2f}")

        return avg_time, fps

    def compare_models(self, fp32_model, quantized_model, method_name):
        """对比FP32和量化模型"""
        print("\n[5/5] Comparing FP32 vs Quantized Model...")
        print("="*80)

        # 模型大小对比
        fp32_size = self.get_model_size(fp32_model)
        quant_size = self.get_model_size(quantized_model)
        size_reduction = (1 - quant_size / fp32_size) * 100

        print(f"\nModel Size Comparison:")
        print(f"   FP32 Model:       {fp32_size:.2f} MB")
        print(f"   Quantized Model:  {quant_size:.2f} MB")
        print(f"   Size Reduction:   {size_reduction:.2f}%")

        # 推理速度对比
        print(f"\nInference Speed Comparison:")
        fp32_time, fp32_fps = self.benchmark_inference(fp32_model, num_iterations=50)
        quant_time, quant_fps = self.benchmark_inference(quantized_model, num_iterations=50)
        speedup = fp32_time / quant_time

        print(f"\n   FP32 Model:")
        print(f"      Time: {fp32_time*1000:.2f} ms")
        print(f"      FPS:  {fp32_fps:.2f}")
        print(f"   Quantized Model ({method_name}):")
        print(f"      Time: {quant_time*1000:.2f} ms")
        print(f"      FPS:  {quant_fps:.2f}")
        print(f"   Speedup: {speedup:.2f}x")

        print("="*80)

        return {
            'fp32_size_mb': fp32_size,
            'quant_size_mb': quant_size,
            'size_reduction_percent': size_reduction,
            'fp32_time_ms': fp32_time * 1000,
            'quant_time_ms': quant_time * 1000,
            'fp32_fps': fp32_fps,
            'quant_fps': quant_fps,
            'speedup': speedup
        }

    def quantize(self, method='dynamic'):
        """
        执行量化

        Args:
            method: 'dynamic' 或 'static'
        """
        print(f"\nQuantization Method: {method.upper()}")
        print("-"*80)

        # 加载FP32模型
        fp32_model = self.load_fp32_model()

        # 执行量化
        if method == 'dynamic':
            quantized_model = self.dynamic_quantization(fp32_model)
        elif method == 'static':
            quantized_model = self.static_quantization(fp32_model)
        else:
            raise ValueError(f"Unknown quantization method: {method}")

        # 保存量化模型
        output_path, file_size = self.save_quantized_model(quantized_model, method)

        # 对比性能
        metrics = self.compare_models(fp32_model, quantized_model, method)

        # 保存性能报告
        self.save_report(metrics, method, output_path)

        print("\n" + "="*80)
        print("Quantization Completed Successfully!")
        print("="*80)
        print(f"Quantized model saved to: {output_path}")
        print(f"Size reduction: {metrics['size_reduction_percent']:.2f}%")
        print(f"Speed improvement: {metrics['speedup']:.2f}x")
        print("="*80)

        return quantized_model, metrics

    def save_report(self, metrics, method, model_path):
        """保存量化报告"""
        report_path = self.output_dir / f"quantization_report_{method}_{time.strftime('%Y%m%d_%H%M%S')}.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Model Quantization Report\n")
            f.write("="*80 + "\n\n")

            f.write(f"Quantization Method: {method.upper()}\n")
            f.write(f"Original Model: {self.model_path}\n")
            f.write(f"Quantized Model: {model_path}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("-"*80 + "\n")
            f.write("Performance Metrics\n")
            f.write("-"*80 + "\n\n")

            f.write(f"Model Size:\n")
            f.write(f"  FP32:       {metrics['fp32_size_mb']:.2f} MB\n")
            f.write(f"  Quantized:  {metrics['quant_size_mb']:.2f} MB\n")
            f.write(f"  Reduction:  {metrics['size_reduction_percent']:.2f}%\n\n")

            f.write(f"Inference Speed:\n")
            f.write(f"  FP32 Time:       {metrics['fp32_time_ms']:.2f} ms\n")
            f.write(f"  Quantized Time:  {metrics['quant_time_ms']:.2f} ms\n")
            f.write(f"  FP32 FPS:        {metrics['fp32_fps']:.2f}\n")
            f.write(f"  Quantized FPS:   {metrics['quant_fps']:.2f}\n")
            f.write(f"  Speedup:         {metrics['speedup']:.2f}x\n\n")

            f.write("="*80 + "\n")

        print(f"\nReport saved to: {report_path}")


def main():
    """主函数 - 命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch Official Model Quantization Tool')
    parser.add_argument('--model', type=str,
                        default='./checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth',
                        help='Path to FP32 model checkpoint')
    parser.add_argument('--method', type=str, default='dynamic', choices=['dynamic', 'static'],
                        help='Quantization method (dynamic or static)')
    parser.add_argument('--output', type=str, default='quantized_models',
                        help='Output directory for quantized models')

    args = parser.parse_args()

    # 创建量化器
    quantizer = ModelQuantizer(args.model, args.output)

    # 执行量化
    quantized_model, metrics = quantizer.quantize(method=args.method)

    print("\nQuantization process completed successfully!")
    print(f"Quantized model is ready for deployment.")


if __name__ == '__main__':
    main()
