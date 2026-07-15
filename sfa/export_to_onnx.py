"""
将163轮模型导出为ONNX格式
支持FP32和量化模型导出
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from models.fpn_resnet import get_pose_net
from quantize_config import get_quantization_config


def export_fp32_to_onnx(model_path, output_path, opset_version=11):
    """导出FP32模型到ONNX格式"""
    print("="*80)
    print("FP32模型导出到ONNX")
    print("="*80)
    print(f"输入模型: {model_path}")
    print(f"输出路径: {output_path}")
    print(f"ONNX opset版本: {opset_version}")
    print()

    # 加载配置
    cfg = get_quantization_config()

    # 创建模型
    print("[1/4] 加载FP32模型...")
    model = get_pose_net(
        num_layers=cfg.num_layers,
        heads=cfg.heads,
        head_conv=cfg.head_conv,
        imagenet_pretrained=cfg.imagenet_pretrained
    )

    # 加载权重
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("   [OK] 模型加载成功")

    # 准备示例输入
    print("\n[2/4] 准备示例输入...")
    dummy_input = torch.randn(1, cfg.num_input_channels,
                             cfg.input_size[0], cfg.input_size[1])
    print(f"   输入形状: {dummy_input.shape}")

    # 导出到ONNX
    print("\n[3/4] 导出到ONNX格式...")
    try:
        # 定义输入和输出名称
        input_names = ['bev_input']
        output_names = ['hm_cen', 'cen_offset', 'direction', 'z_coor', 'dim']

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                'bev_input': {0: 'batch_size'},
                'hm_cen': {0: 'batch_size'},
                'cen_offset': {0: 'batch_size'},
                'direction': {0: 'batch_size'},
                'z_coor': {0: 'batch_size'},
                'dim': {0: 'batch_size'}
            }
        )
        print("   [OK] ONNX导出成功")
    except Exception as e:
        print(f"   [ERROR] ONNX导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 验证ONNX模型
    print("\n[4/4] 验证ONNX模型...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("   [OK] ONNX模型验证通过")

        # 显示模型信息
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n模型信息:")
        print(f"   文件大小: {file_size:.2f} MB")
        print(f"   输入节点: {input_names}")
        print(f"   输出节点: {output_names}")

    except ImportError:
        print("   [WARNING] onnx包未安装，跳过验证")
        print("   安装命令: pip install onnx")
    except Exception as e:
        print(f"   [ERROR] ONNX模型验证失败: {e}")
        return False

    print("\n" + "="*80)
    print("ONNX导出完成!")
    print("="*80)
    print(f"ONNX模型已保存到: {output_path}")
    print()
    print("使用ONNX Runtime推理:")
    print(f"  python sfa/run_onnx_inference.py --onnx_model {output_path}")
    print("="*80)

    return True


def test_onnx_inference(onnx_path):
    """测试ONNX模型推理"""
    print("\n测试ONNX推理...")

    try:
        import onnxruntime as ort
    except ImportError:
        print("[ERROR] onnxruntime未安装")
        print("安装命令: pip install onnxruntime")
        return False

    # 创建推理会话
    print(f"加载ONNX模型: {onnx_path}")
    session = ort.InferenceSession(onnx_path)

    # 获取输入输出信息
    input_name = session.get_inputs()[0].name
    print(f"输入名称: {input_name}")
    print(f"输入形状: {session.get_inputs()[0].shape}")

    print(f"\n输出节点:")
    for i, output in enumerate(session.get_outputs()):
        print(f"  {i+1}. {output.name}: {output.shape}")

    # 准备测试输入
    cfg = get_quantization_config()
    dummy_input = np.random.randn(1, cfg.num_input_channels,
                                  cfg.input_size[0], cfg.input_size[1]).astype(np.float32)

    # 推理
    print(f"\n执行推理测试...")
    import time
    start_time = time.time()
    outputs = session.run(None, {input_name: dummy_input})
    end_time = time.time()

    print(f"[OK] 推理成功")
    print(f"推理时间: {(end_time - start_time)*1000:.2f} ms")
    print(f"输出数量: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"  输出{i+1}形状: {output.shape}")

    return True


def main():
    parser = argparse.ArgumentParser(description='导出模型到ONNX格式')
    parser.add_argument('--model', type=str,
                       default='checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth',
                       help='输入模型路径')
    parser.add_argument('--output', type=str,
                       default='onnx_models/sfa3d_163_fp32.onnx',
                       help='输出ONNX文件路径')
    parser.add_argument('--opset', type=int, default=11,
                       help='ONNX opset版本 (默认: 11)')
    parser.add_argument('--test', action='store_true',
                       help='导出后测试ONNX推理')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 导出模型
    success = export_fp32_to_onnx(args.model, args.output, args.opset)

    if not success:
        print("\n[ERROR] ONNX导出失败")
        return 1

    # 测试推理
    if args.test:
        test_onnx_inference(args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
