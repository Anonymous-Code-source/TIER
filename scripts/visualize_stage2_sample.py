#!/usr/bin/env python3
"""
可视化 Stage-2 Dataset 加载的样本
验证时间对齐和数据质量
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from dataloader.stage2_dataset import Stage2PairedDataset


def denormalize(img_tensor):
    """反归一化"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


def visualize_sample(dataset, idx=0):
    """
    可视化一个样本的 4 帧序列
    """
    sample = dataset[idx]
    
    # 创建图表：3 个传感器 × 4 帧
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    sensors = ['gelsight', 'digit', 'duragel']
    sensor_names = ['GelSight', 'DIGIT', 'DuraGel']
    
    for sensor_idx, (sensor, sensor_name) in enumerate(zip(sensors, sensor_names)):
        # 获取 4 帧序列 [4, 3, 224, 224]
        frames = sample[sensor]
        
        for frame_idx in range(4):
            # 获取单帧 [3, 224, 224]
            img = frames[frame_idx]
            img = denormalize(img)
            
            ax = axes[sensor_idx, frame_idx]
            ax.imshow(img)
            ax.axis('off')
            
            # 设置标题
            if sensor_idx == 0:
                ax.set_title(f'Frame t-{3-frame_idx}', fontsize=12, fontweight='bold')
            if frame_idx == 0:
                ax.set_ylabel(sensor_name, fontsize=14, fontweight='bold', rotation=0, 
                             ha='right', va='center')
    
    # 总标题
    fig.suptitle(
        f'Stage-2 Sample Visualization\n'
        f'Object: {sample["object"]} | Frame: {sample["frame"]} | Type: {sample["data_type"]}',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.tight_layout()
    
    # 保存图像
    output_path = Path(__file__).parent.parent / f'stage2_sample_{idx}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 样本可视化保存到: {output_path}")
    
    plt.close()


def visualize_temporal_alignment(dataset):
    """
    可视化时间对齐效果
    展示同一物体的连续 3 个时刻
    """
    # 取同一物体的 3 个连续样本
    samples = [dataset[i] for i in range(3)]
    
    # 创建图表：3 个时刻 × 3 个传感器
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    sensors = ['gelsight', 'digit', 'duragel']
    sensor_names = ['GelSight', 'DIGIT', 'DuraGel']
    
    for time_idx, sample in enumerate(samples):
        for sensor_idx, (sensor, sensor_name) in enumerate(zip(sensors, sensor_names)):
            # 获取当前帧（第 4 帧，索引 3） [3, 224, 224]
            img = sample[sensor][3]  # 取 t 时刻的帧
            img = denormalize(img)
            
            ax = axes[time_idx, sensor_idx]
            ax.imshow(img)
            ax.axis('off')
            
            # 设置标题
            if time_idx == 0:
                ax.set_title(sensor_name, fontsize=14, fontweight='bold')
            if sensor_idx == 0:
                ax.set_ylabel(
                    f'Frame {sample["frame"]}',
                    fontsize=12, fontweight='bold',
                    rotation=0, ha='right', va='center'
                )
    
    # 总标题
    fig.suptitle(
        f'Temporal Alignment Visualization\n'
        f'Object: {samples[0]["object"]} | Consecutive Frames',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.tight_layout()
    
    # 保存图像
    output_path = Path(__file__).parent.parent / 'stage2_temporal_alignment.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 时间对齐可视化保存到: {output_path}")
    
    plt.close()


def visualize_multi_object(dataset):
    """
    可视化多个物体
    """
    # 选择不同物体的样本（每隔 200 个样本取一个）
    indices = [i * 200 for i in range(min(3, len(dataset)//200))]
    samples = [dataset[i] for i in indices if i < len(dataset)]
    
    if len(samples) < 3:
        print("⚠️ 样本数不足，跳过多物体可视化")
        return
    
    # 创建图表：3 个物体 × 3 个传感器
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    sensors = ['gelsight', 'digit', 'duragel']
    sensor_names = ['GelSight', 'DIGIT', 'DuraGel']
    
    for obj_idx, sample in enumerate(samples):
        for sensor_idx, (sensor, sensor_name) in enumerate(zip(sensors, sensor_names)):
            # 获取当前帧（第 4 帧，索引 3）
            img = sample[sensor][3]
            img = denormalize(img)
            
            ax = axes[obj_idx, sensor_idx]
            ax.imshow(img)
            ax.axis('off')
            
            # 设置标题
            if obj_idx == 0:
                ax.set_title(sensor_name, fontsize=14, fontweight='bold')
            if sensor_idx == 0:
                ax.set_ylabel(
                    sample['object'][:15],
                    fontsize=10, fontweight='bold',
                    rotation=0, ha='right', va='center'
                )
    
    # 总标题
    fig.suptitle(
        'Multi-Object Visualization',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.tight_layout()
    
    # 保存图像
    output_path = Path(__file__).parent.parent / 'stage2_multi_object.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 多物体可视化保存到: {output_path}")
    
    plt.close()


def main():
    """主函数"""
    print("\n" + "="*60)
    print("Stage-2 Dataset 可视化验证")
    print("="*60)
    
    # 创建 dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    dataset = Stage2PairedDataset(
        csv_file='./Tacquad/contact_fine.csv',
        data_root='./Tacquad/data_fine',
        data_type='fine',
        transform=transform
    )
    
    # 1. 可视化单个样本的 4 帧序列
    print("\n生成可视化 1: 单个样本的 4 帧序列")
    visualize_sample(dataset, idx=0)
    
    # 2. 可视化时间对齐效果
    print("\n生成可视化 2: 时间对齐效果")
    visualize_temporal_alignment(dataset)
    
    # 3. 可视化多个物体
    print("\n生成可视化 3: 多物体对比")
    visualize_multi_object(dataset)
    
    print("\n" + "="*60)
    print("✅ 所有可视化完成！")
    print("="*60)
    print("\n生成的图像文件:")
    print("  1. stage2_sample_0.png")
    print("  2. stage2_temporal_alignment.png")
    print("  3. stage2_multi_object.png")


if __name__ == '__main__':
    main()

