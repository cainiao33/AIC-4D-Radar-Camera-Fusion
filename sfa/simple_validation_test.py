"""
用于测试163轮训练模型性能的简化验证测试脚本。
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

print("=== SFA3D-Modified Validation Test ===")
print(f"Project root: {project_root}")
print(f"Python path: {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    # Test imports
    from config import kitti_config as cnf
    from models.model_utils import create_model
    from data_process.kitti_dataloader import create_test_dataloader
    from utils.evaluation_utils import decode, post_processing
    from utils.torch_utils import _sigmoid

    print("✓ All imports successful")

    # Model configuration
    class Config:
        def __init__(self):
            self.arch = 'fpn_resnet_18'
            self.pretrained_path = '../checkpoints/sfa3d_8d_full_300epochs/Model_sfa3d_8d_full_300epochs_epoch_163.pth'
            self.dataset_dir = '../DRadDataset'
            self.val_subdir = 'split/val'
            self.input_size = (608, 608)
            self.hm_size = (152, 152)
            self.down_ratio = 4
            self.max_objects = 50
            self.head_conv = 64
            self.num_classes = 3
            self.num_center_offset = 2
            self.num_z = 1
            self.num_dim = 3
            self.num_direction = 2
            self.num_input_features = 4
            self.pin_memory = True
            self.distributed = False
            self.use_imagesets = False
            self.train_subdir = 'split/val'
            self.imagenet_pretrained = False

            self.heads = {
                'hm_cen': self.num_classes,
                'cen_offset': self.num_center_offset,
                'direction': self.num_direction,
                'z_coor': self.num_z,
                'dim': self.num_dim
            }

            # Device setup
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.gpu_idx = 0 if torch.cuda.is_available() else None

    config = Config()

    print(f"✓ Configuration created")
    print(f"✓ Device: {config.device}")

    # Check if model file exists
    model_path = Path(config.pretrained_path)
    if not model_path.is_absolute():
        model_path = project_root / config.pretrained_path
        config.pretrained_path = str(model_path)

    print(f"Model path: {config.pretrained_path}")
    print(f"Model exists: {model_path.exists()}")

    if not model_path.exists():
        print("❌ Model file not found!")
        sys.exit(1)

    # Create model
    print("Creating model...")
    model = create_model(config)
    print("✓ Model created")

    # Load weights
    print("Loading weights...")
    try:
        model.load_state_dict(torch.load(config.pretrained_path, map_location='cpu'))
        print("✓ Weights loaded successfully")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        sys.exit(1)

    # Move to device
    model = model.to(config.device)
    model.eval()
    print(f"✓ Model moved to {config.device}")

    # Create dataloader
    print("Creating validation dataloader...")
    try:
        dataloader = create_test_dataloader(config)
        print(f"✓ Dataloader created, dataset size: {len(dataloader)}")
    except Exception as e:
        print(f"❌ Error creating dataloader: {e}")
        sys.exit(1)

    # Test on a few samples
    print("\n=== Running Inference Test ===")
    num_samples = min(5, len(dataloader))
    total_detections = 0

    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            if i >= num_samples:
                break

            try:
                metadatas, bev_maps, _ = batch_data
                bev_maps = bev_maps.to(config.device, non_blocking=True).float()

                print(f"\nSample {i+1}/{num_samples}: Processing {bev_maps.size(0)} samples")

                # Forward pass
                outputs = model(bev_maps)
                outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
                outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])

                # Decode predictions
                detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'],
                                   outputs['z_coor'], outputs['dim'], K=50)
                detections = detections.cpu().numpy().astype(np.float32)

                # Post-processing
                detections = post_processing(detections, config.num_classes, config.down_ratio,
                                            peak_thresh=0.25, inter_class_nms=True, nms_thresh=0.2)

                # Count detections
                sample_detections = 0
                for j in range(bev_maps.size(0)):
                    for cls_id, dets in detections[j].items():
                        sample_detections += len(dets)

                total_detections += sample_detections
                print(f"  Detections: {sample_detections}")

            except Exception as e:
                print(f"❌ Error processing batch {i}: {e}")
                continue

    print(f"\n=== Test Results ===")
    print(f"Processed samples: {num_samples}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per sample: {total_detections / max(num_samples, 1):.2f}")
    print("✓ Validation test completed successfully!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)