#!/usr/bin/env python3
"""
可视化 DataLoader 输出
验证数据增强和时间对齐效果
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib.pyplot as plt
import numpy as np
from dataloader.build_dataloaders import build_stage2_dataloader


def denormalize(img_tensor):
    """
    反归一化 ImageNet 标准
    
    Args:
        img_tensor: [3, H, W] tensor (归一化后)
    
    Returns:
        img: [H, W, 3] numpy array (0-1范围)
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


def visualize_batch(save_path='./dataloader_batch_visualization.png'):
    """
    可视化一个 batch 的多传感器对齐效果
    
    布局：
    行：4个样本
    列：3个传感器 (GelSight, DIGIT, DuraGel)
    
    目的：验证3个传感器的图像是否显示相同的接触点
    """
    print("\n" + "="*60)
    print("可视化 1: 多传感器时间对齐验证")
    print("="*60)
    
    # 加载数据（小batch，CPU模式）
    loader = build_stage2_dataloader(
        stage='2a',
        batch_size=4,
        num_workers=0,      # CPU模式
        pin_memory=False,   # 不使用GPU
        shuffle=True
    )
    
    batch = next(iter(loader))
    
    print(f"\n加载的batch信息:")
    print(f"  Objects: {batch['object']}")
    print(f"  Frames: {batch['frame'].tolist()}")
    
    # 创建图表：4 个样本 × 3 个传感器
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    fig.suptitle('DataLoader Output: Time-Aligned Multi-Sensor Batch', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    for sample_idx in range(4):
        for sensor_idx, sensor in enumerate(['gelsight', 'digit', 'duragel']):
            # 获取第1帧（t-3）的图像
            img = batch[sensor][sample_idx, 0]  # [3, 224, 224]
            img = denormalize(img)
            
            ax = axes[sample_idx, sensor_idx]
            ax.imshow(img)
            
            # 设置标题
            title = (f'{sensor.upper()}\n'
                    f'Object: {batch["object"][sample_idx]}\n'
                    f'Frame: {batch["frame"][sample_idx].item()}')
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')
            
            # 添加边框（方便区分传感器）
            for spine in ax.spines.values():
                spine.set_edgecolor('gray')
                spine.set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化保存到: {save_path}")
    print(f"   请检查: 同一行的3个传感器图像是否显示相同的接触点")
    print(f"   验证点: 压痕位置、纹理特征应该相似")


def visualize_temporal_sequence(save_path='./temporal_sequence.png'):
    """
    可视化时间序列（4帧）
    
    布局：
    行：3个传感器
    列：4帧序列 [t-3, t-2, t-1, t]
    
    目的：验证4帧序列是否连续变化
    """
    print("\n" + "="*60)
    print("可视化 2: 时间序列连续性验证")
    print("="*60)
    
    loader = build_stage2_dataloader(
        stage='2a',
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle=True
    )
    
    batch = next(iter(loader))
    
    print(f"\n选择的样本:")
    print(f"  Object: {batch['object'][0]}")
    print(f"  Frame: {batch['frame'][0].item()}")
    
    # 创建图表：3 个传感器 × 4 帧
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'Temporal Sequence: {batch["object"][0]} (Frame {batch["frame"][0].item()})',
                 fontsize=18, fontweight='bold')
    
    for sensor_idx, sensor in enumerate(['gelsight', 'digit', 'duragel']):
        for frame_idx in range(4):
            img = batch[sensor][0, frame_idx]  # [3, 224, 224]
            img = denormalize(img)
            
            ax = axes[sensor_idx, frame_idx]
            ax.imshow(img)
            
            # 设置标题
            if frame_idx == 0:
                time_label = 't-3'
            elif frame_idx == 1:
                time_label = 't-2'
            elif frame_idx == 2:
                time_label = 't-1'
            else:
                time_label = 't (current)'
            
            title = f'{sensor.upper()}\n{time_label}'
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 时间序列可视化保存到: {save_path}")
    print(f"   请检查: 每行的4帧图像是否显示连续的接触过程")
    print(f"   验证点: 压痕深度、形状应该逐渐变化")


def visualize_data_augmentation(save_path='./data_augmentation.png'):
    """
    可视化数据增强效果
    
    布局：
    行：3次加载同一传感器的图像
    列：原始风格的不同增强结果
    
    目的：验证数据增强是否生效
    """
    print("\n" + "="*60)
    print("可视化 3: 数据增强效果验证")
    print("="*60)
    
    # 加载多个batch（由于shuffle和随机增强，会产生不同结果）
    loader = build_stage2_dataloader(
        stage='2a',
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle=False  # 不打乱，但增强是随机的
    )
    
    # 加载同一位置的多个样本（由于随机增强，效果会不同）
    batches = []
    for i, batch in enumerate(loader):
        if i >= 3:
            break
        batches.append(batch)
    
    print(f"\n加载了 {len(batches)} 个batch")
    print(f"  Batch 0: {batches[0]['object'][0]}")
    print(f"  Batch 1: {batches[1]['object'][0]}")
    print(f"  Batch 2: {batches[2]['object'][0]}")
    
    # 创建图表：3个batch × 3个传感器
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Data Augmentation Effect (Different Objects with Augmentation)',
                 fontsize=18, fontweight='bold')
    
    for batch_idx in range(3):
        for sensor_idx, sensor in enumerate(['gelsight', 'digit', 'duragel']):
            img = batches[batch_idx][sensor][0, 0]  # 第一帧
            img = denormalize(img)
            
            ax = axes[batch_idx, sensor_idx]
            ax.imshow(img)
            
            title = (f'{sensor.upper()}\n'
                    f'{batches[batch_idx]["object"][0]}\n'
                    f'Frame {batches[batch_idx]["frame"][0].item()}')
            ax.set_title(title, fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 数据增强可视化保存到: {save_path}")
    print(f"   说明: 每个图像都经过了ColorJitter增强（亮度、对比度、饱和度、色调）")
    print(f"   注意: 不使用RandomFlip，因为触觉图像的方向性有物理意义")


def visualize_comparison(save_path='./sensor_comparison.png'):
    """
    可视化传感器对比
    
    详细对比同一接触点在3个传感器上的表现
    """
    print("\n" + "="*60)
    print("可视化 4: 传感器详细对比")
    print("="*60)
    
    loader = build_stage2_dataloader(
        stage='2a',
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle=True
    )
    
    batch = next(iter(loader))
    
    print(f"\n对比样本:")
    print(f"  Object: {batch['object'][0]}")
    print(f"  Frame: {batch['frame'][0].item()}")
    
    # 创建图表：1行3列，更大尺寸
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Sensor Comparison: {batch["object"][0]} at Frame {batch["frame"][0].item()}',
                 fontsize=18, fontweight='bold')
    
    for sensor_idx, sensor in enumerate(['gelsight', 'digit', 'duragel']):
        img = batch[sensor][0, 0]  # 当前帧
        img = denormalize(img)
        
        ax = axes[sensor_idx]
        im = ax.imshow(img)
        
        # 详细标题
        title = (f'{sensor.upper()}\n'
                f'Object: {batch["object"][0]}\n'
                f'Frame: {batch["frame"][0].item()}\n'
                f'Shape: {batch[sensor].shape}')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('on')  # 显示坐标轴，方便对比
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 传感器对比保存到: {save_path}")
    print(f"   说明: 放大查看，验证不同传感器捕捉到的细节差异")


def main():
    """主函数"""
    print("\n" + "="*70)
    print(" "*15 + "DataLoader 可视化测试")
    print("="*70)
    print("\n这个脚本将生成4组可视化，用于验证:")
    print("  1. 多传感器时间对齐是否正确")
    print("  2. 4帧时间序列是否连续")
    print("  3. 数据增强是否生效")
    print("  4. 传感器间的差异对比")
    
    try:
        # 可视化1: 批次多传感器对齐
        visualize_batch()
        
        # 可视化2: 时间序列
        visualize_temporal_sequence()
        
        # 可视化3: 数据增强
        visualize_data_augmentation()
        
        # 可视化4: 传感器对比
        visualize_comparison()
        
        print("\n" + "="*70)
        print("✅ 所有可视化完成！")
        print("="*70)
        
        print("\n生成的文件:")
        print("  1. ./dataloader_batch_visualization.png  - 多传感器对齐验证")
        print("  2. ./temporal_sequence.png               - 时间序列连续性验证")
        print("  3. ./data_augmentation.png               - 数据增强效果")
        print("  4. ./sensor_comparison.png               - 传感器详细对比")
        
        print("\n验收要点:")
        print("  ✅ 同一行的3个传感器图像应显示相同的接触点（位置、形状相似）")
        print("  ✅ 4帧序列应该连续变化（压痕逐渐加深或减轻）")
        print("  ✅ 不同传感器有各自的视觉特征（光照、颜色、分辨率）")
        print("  ✅ 但物理信息应该一致（压痕位置、大致形状）")
        
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

