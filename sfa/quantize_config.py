"""
量化专用配置文件
独立于原始kitti_config.py，不影响其他训练和推理脚本
"""

class QuantizationConfig:
    """量化配置类"""

    def __init__(self):
        # 模型架构配置
        self.num_layers = 18  # ResNet-18
        self.head_conv = 64
        self.imagenet_pretrained = False

        # 输入输出配置
        self.input_size = (608, 608)
        self.hm_size = (152, 152)
        self.down_ratio = 4
        self.num_input_features = 4  # 8D雷达数据映射到4维点云 (x,y,z,P)
        self.num_input_channels = 3  # BEV图像通道数 (intensity, height, density)

        # 类别配置
        self.num_classes = 3  # Car, Cyclist, Truck
        self.num_center_offset = 2
        self.num_z = 1
        self.num_dim = 3
        self.num_direction = 2

        # 检测头配置
        self.heads = {
            'hm_cen': self.num_classes,
            'cen_offset': self.num_center_offset,
            'direction': self.num_direction,
            'z_coor': self.num_z,
            'dim': self.num_dim
        }

        # 量化配置
        self.quantization_backend = 'fbgemm'  # or 'qnnpack' for mobile
        self.quantization_dtype = 'qint8'

        # 校准配置
        self.calibration_samples = 100  # 用于静态量化的校准样本数
        self.calibration_batch_size = 1

        # 基准测试配置
        self.benchmark_iterations = 100
        self.warmup_iterations = 10


def get_quantization_config():
    """获取量化配置"""
    return QuantizationConfig()
