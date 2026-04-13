#!/usr/bin/env python3
"""
Stage-2 配对数据集
统一处理 Fine-grained 和 Coarse-grained 数据
"""

from torch.utils.data import Dataset
from PIL import Image, ImageFile
import torch
import numpy as np
import os
import signal
from pathlib import Path

from .build_paired_index import build_paired_index
from .temporal_alignment import batch_align_frames

ImageFile.LOAD_TRUNCATED_IMAGES = True


class _ImageLoadTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _ImageLoadTimeout("image load timeout")


class Stage2PairedDataset(Dataset):
    """
    Stage-2 TacQuad 配对数据集
    
    特点：
    1. 支持 Fine 和 Coarse 数据（统一接口）
    2. 时间对齐（线性插值）
    3. 返回 3 个传感器的 4 帧序列
    
    目录结构：
        data_root/
        ├── banana/
        │   ├── gelsight/
        │   │   ├── 31.png
        │   │   ├── 32.png
        │   │   └── ...
        │   ├── digit/
        │   └── duragel/
    """
    
    def __init__(
        self,
        csv_file: str,
        data_root: str,
        data_type: str = 'fine',
        group_id: int = 0,
        transform=None,
        min_sequence_length: int = 4,
        base_sensor: str = 'gelsight',
        base_frame_jitter: int = 0,
        base_frame_jitter_prob: float = 0.0
    ):
        """
        Args:
            csv_file: CSV 文件路径
            data_root: 数据根目录
            data_type: 'fine' or 'coarse'（仅用于标记）
            group_id: Group-DRO分组ID (0=Fine, 1=Indoor, 2=Outdoor)
            transform: torchvision.transforms
            min_sequence_length: 最小序列长度（默认 4）
            base_sensor: 基准传感器（默认 'gelsight'）
        """
        self.csv_file = csv_file
        self.data_root = Path(data_root)
        self.data_type = data_type
        self.group_id = group_id
        self.transform = transform
        self.min_sequence_length = min_sequence_length
        self.base_sensor = base_sensor
        self.base_frame_jitter = max(0, int(base_frame_jitter))
        self.base_frame_jitter_prob = float(base_frame_jitter_prob)
        # Optional image load timeout to avoid hanging on bad/slow IO.
        self.image_load_timeout_sec = int(os.environ.get("PT_IMAGE_LOAD_TIMEOUT_SEC", "0"))
        
        # 检查数据目录是否存在
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        
        # 解析 CSV
        print(f"\n[{data_type.upper()}] Loading dataset from {csv_file}")
        self.paired_index = build_paired_index(csv_file, min_sequence_length)
        
        # 构建样本列表
        self.samples = self._build_sample_list()
        
        print(f"[{data_type.upper()}] Dataset ready:")
        print(f"  Objects: {len(self.paired_index)}")
        print(f"  Samples: {len(self.samples)}")
    
    def _build_sample_list(self):
        """
        构建样本列表
        以基准传感器为准，从第 4 帧开始（需要前 3 帧历史）
        
        Returns:
            samples: [(obj_name, base_frame), ...]
        """
        samples = []
        
        for obj_name, info in self.paired_index.items():
            base_start, base_end = info['frame_ranges'][self.base_sensor]
            
            # 从第 4 帧开始（索引 start+3），需要前 3 帧历史
            # 例如：base_start=30, 则从 33 开始（需要 30,31,32 作为历史）
            for frame_idx in range(base_start + 3, base_end + 1):
                samples.append((obj_name, frame_idx))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        加载一个样本
        
        Returns:
            {
                'gelsight': [4, 3, 224, 224],
                'digit':    [4, 3, 224, 224],
                'duragel':  [4, 3, 224, 224],
                'object':   str,
                'frame':    int,
                'pair_confidence': float,  # 时间对齐置信度（用于pair loss加权）
                'data_type': str,
                'group_label': int  # 0=Fine, 1=Indoor, 2=Outdoor
            }
        """
        obj_name, base_frame = self.samples[idx]
        frame_ranges = self.paired_index[obj_name]['frame_ranges']

        # 训练时对基准帧做轻微抖动，提升覆盖范围与鲁棒性
        if self.base_frame_jitter > 0 and self.base_frame_jitter_prob > 0:
            if np.random.rand() < self.base_frame_jitter_prob:
                base_start, base_end = frame_ranges[self.base_sensor]
                min_frame = base_start + 3  # 需要前3帧历史
                max_frame = base_end
                jitter = np.random.randint(-self.base_frame_jitter, self.base_frame_jitter + 1)
                base_frame = int(np.clip(base_frame + jitter, min_frame, max_frame))
        
        # 批量时间对齐
        aligned_frames, alignment_conf = batch_align_frames(
            self.base_sensor, base_frame, frame_ranges, return_confidence=True
        )

        # 置信度采用“最弱环节”原则：任一目标传感器对齐不稳都会降低配对可靠性
        target_conf = [
            float(alignment_conf[sensor])
            for sensor in ['gelsight', 'digit', 'duragel']
            if sensor != self.base_sensor
        ]
        pair_confidence = min(target_conf) if len(target_conf) > 0 else 1.0
        
        # 加载 3 个传感器的 4 帧序列
        images = {}
        for sensor in ['gelsight', 'digit', 'duragel']:
            sensor_frames = self._load_sensor_sequence(
                obj_name, sensor, aligned_frames[sensor]
            )
            images[sensor] = sensor_frames
        
        return {
            'gelsight': images['gelsight'],
            'digit':    images['digit'],
            'duragel':  images['duragel'],
            'object':   obj_name,
            'frame':    base_frame,
            'pair_confidence': float(pair_confidence),
            'data_type': self.data_type,
            'group_label': self.group_id  # 添加Group-DRO标签
        }
    
    def _load_sensor_sequence(self, obj_name, sensor, center_frame):
        """
        加载传感器的 4 帧序列 [t-3, t-2, t-1, t]
        
        Args:
            obj_name: 物体名称
            sensor: 传感器名称
            center_frame: 中心帧索引 (t)
        
        Returns:
            frames: [4, 3, H, W] Tensor
        """
        frames = []
        
        for offset in range(-3, 1):  # [-3, -2, -1, 0]
            frame_idx = center_frame + offset
            
            # 构建文件路径：data_root/obj_name/sensor/frame_idx.png
            frame_path = self.data_root / obj_name / sensor / f'{frame_idx}.png'
            
            try:
                # 加载图像（可选超时）
                use_timeout = self.image_load_timeout_sec > 0
                if use_timeout:
                    worker_info = torch.utils.data.get_worker_info()
                    # Disable signal alarms in DataLoader workers to avoid crashes.
                    if worker_info is not None:
                        use_timeout = False

                if use_timeout:
                    try:
                        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
                        signal.alarm(self.image_load_timeout_sec)
                    except (ValueError, RuntimeError):
                        # Signal alarms may be unsupported in this context.
                        old_handler = None
                        use_timeout = False
                else:
                    old_handler = None

                with Image.open(frame_path) as img:
                    img = img.convert('RGB')

                # 应用 transform
                if self.transform:
                    img = self.transform(img)
                else:
                    # 如果没有 transform，至少转为 tensor
                    img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                
                frames.append(img)

                if use_timeout:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

            except _ImageLoadTimeout:
                if use_timeout and old_handler is not None:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                print(f"⏱️ Image load timeout: {frame_path}")
                img = torch.zeros(3, 224, 224)
                frames.append(img)
            except FileNotFoundError:
                print(f"⚠️ File not found: {frame_path}")
                # 用零图像替代
                if self.transform:
                    # 创建与 transform 输出相同尺寸的零张量
                    # 假设 transform 输出 [3, 224, 224]
                    img = torch.zeros(3, 224, 224)
                else:
                    # 默认尺寸
                    img = torch.zeros(3, 224, 224)
                frames.append(img)
            
            except Exception as e:
                if use_timeout and old_handler is not None:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                print(f"❌ Error loading {frame_path}: {e}")
                # 用零图像替代
                img = torch.zeros(3, 224, 224)
                frames.append(img)
        
        # Stack: [4, 3, H, W]
        return torch.stack(frames, dim=0)


# ============================================
# 测试代码
# ============================================
if __name__ == '__main__':
    from torchvision import transforms
    import sys
    from pathlib import Path
    
    # 添加父目录到路径以支持导入
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from dataloader.build_paired_index import build_paired_index
    from dataloader.temporal_alignment import batch_align_frames
    
    # 定义 transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 测试 Fine-grained
    print("\n" + "="*60)
    print("测试 Fine-grained Dataset")
    print("="*60)
    
    fine_dataset = Stage2PairedDataset(
        csv_file='./Tacquad/contact_fine.csv',
        data_root='./Tacquad/data_fine',
        data_type='fine',
        transform=transform
    )
    
    print(f"\nDataset 长度: {len(fine_dataset)}")
    
    # 加载一个样本
    print("\n加载第一个样本...")
    sample = fine_dataset[0]
    
    print(f"\n样本格式:")
    print(f"  GelSight shape: {sample['gelsight'].shape}")
    print(f"  DIGIT shape: {sample['digit'].shape}")
    print(f"  DuraGel shape: {sample['duragel'].shape}")
    print(f"  Object: {sample['object']}")
    print(f"  Frame: {sample['frame']}")
    print(f"  Data type: {sample['data_type']}")
    
    # 检查数据类型和值域
    print(f"\n数据质量检查:")
    print(f"  GelSight dtype: {sample['gelsight'].dtype}")
    print(f"  GelSight range: [{sample['gelsight'].min():.3f}, {sample['gelsight'].max():.3f}]")
    print(f"  DIGIT range: [{sample['digit'].min():.3f}, {sample['digit'].max():.3f}]")
    print(f"  DuraGel range: [{sample['duragel'].min():.3f}, {sample['duragel'].max():.3f}]")
    
    # 验证 shape
    assert sample['gelsight'].shape == (4, 3, 224, 224), "GelSight shape 错误"
    assert sample['digit'].shape == (4, 3, 224, 224), "DIGIT shape 错误"
    assert sample['duragel'].shape == (4, 3, 224, 224), "DuraGel shape 错误"
    
    print("\n✅ Fine-grained Dataset 测试通过！")
    
    # 测试加载多个样本
    print("\n" + "="*60)
    print("测试加载多个样本")
    print("="*60)
    
    for i in range(min(3, len(fine_dataset))):
        sample = fine_dataset[i]
        print(f"  Sample {i}: {sample['object']:20s} Frame {sample['frame']}")
    
    print("\n✅ 所有测试通过！")
