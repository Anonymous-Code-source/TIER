#!/usr/bin/env python3
"""
可视化时间对齐
验证对齐算法的正确性
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloader.temporal_alignment import batch_align_frames
from dataloader.build_paired_index import build_paired_index


def visualize_alignment_theory():
    """
    理论可视化：展示时间对齐的原理
    使用简单的帧范围（从 0 开始）
    """
    print("\n" + "="*60)
    print("理论可视化：时间对齐原理")
    print("="*60)
    
    frame_ranges = {
        'gelsight': (0, 199),   # 200 帧
        'digit': (0, 238),      # 239 帧
        'duragel': (0, 239)     # 240 帧
    }
    
    # 生成对齐映射（每 10 帧采样）
    gelsight_frames = np.arange(0, 200, 10)
    
    digit_aligned = []
    duragel_aligned = []
    
    for gs_frame in gelsight_frames:
        aligned = batch_align_frames('gelsight', gs_frame, frame_ranges)
        digit_aligned.append(aligned['digit'])
        duragel_aligned.append(aligned['duragel'])
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 绘制对齐曲线
    ax.plot(gelsight_frames, gelsight_frames, 'o-', 
            label='GelSight (base)', linewidth=2.5, markersize=6, color='#2E86AB')
    ax.plot(gelsight_frames, digit_aligned, 's-', 
            label='DIGIT (aligned)', linewidth=2.5, markersize=6, color='#A23B72')
    ax.plot(gelsight_frames, duragel_aligned, '^-', 
            label='DuraGel (aligned)', linewidth=2.5, markersize=6, color='#F18F01')
    
    ax.set_xlabel('GelSight Frame Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Aligned Frame Index', fontsize=14, fontweight='bold')
    ax.set_title('Temporal Alignment: Linear Interpolation\n(Simple Frame Range: 0-based)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 标注关键点
    key_points = [
        (0, 0, 'Start'),
        (100, 120, 'Middle'),
        (199, 238, 'End')
    ]
    
    for gs, digit, label in key_points:
        ax.plot(gs, digit, 'ro', markersize=10, zorder=5)
        ax.annotate(
            f'{label}\n({gs}, {digit})',
            xy=(gs, digit),
            xytext=(gs+15, digit+15),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='red', lw=2)
        )
    
    plt.tight_layout()
    
    # 保存图像
    output_path = Path(__file__).resolve().parent.parent / 'alignment_theory.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 理论可视化保存到: {output_path}")
    
    plt.close()


def visualize_real_data_alignment():
    """
    真实数据可视化：展示实际 TacQuad 数据的对齐
    """
    print("\n" + "="*60)
    print("真实数据可视化：TacQuad Fine-grained")
    print("="*60)
    
    # 加载真实数据
    csv_path = Path(__file__).resolve().parent.parent / 'Tacquad' / 'contact_fine.csv'
    paired_index = build_paired_index(str(csv_path))
    
    # 取第一个物体
    obj_name = list(paired_index.keys())[0]
    frame_ranges = paired_index[obj_name]['frame_ranges']
    
    print(f"\n物体: {obj_name}")
    print(f"  GelSight: {frame_ranges['gelsight']} "
          f"({frame_ranges['gelsight'][1] - frame_ranges['gelsight'][0] + 1} 帧)")
    print(f"  DIGIT:    {frame_ranges['digit']} "
          f"({frame_ranges['digit'][1] - frame_ranges['digit'][0] + 1} 帧)")
    print(f"  DuraGel:  {frame_ranges['duragel']} "
          f"({frame_ranges['duragel'][1] - frame_ranges['duragel'][0] + 1} 帧)")
    
    # 生成对齐映射
    gs_start, gs_end = frame_ranges['gelsight']
    gelsight_frames = np.linspace(gs_start, gs_end, 20)  # 20 个采样点
    
    digit_aligned = []
    duragel_aligned = []
    
    for gs_frame in gelsight_frames:
        aligned = batch_align_frames('gelsight', int(gs_frame), frame_ranges)
        digit_aligned.append(aligned['digit'])
        duragel_aligned.append(aligned['duragel'])
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 绘制对齐曲线
    ax.plot(gelsight_frames, gelsight_frames, 'o-', 
            label=f'GelSight (base, {gs_end - gs_start + 1} frames)', 
            linewidth=2.5, markersize=6, color='#2E86AB')
    ax.plot(gelsight_frames, digit_aligned, 's-', 
            label=f'DIGIT (aligned, {frame_ranges["digit"][1] - frame_ranges["digit"][0] + 1} frames)', 
            linewidth=2.5, markersize=6, color='#A23B72')
    ax.plot(gelsight_frames, duragel_aligned, '^-', 
            label=f'DuraGel (aligned, {frame_ranges["duragel"][1] - frame_ranges["duragel"][0] + 1} frames)', 
            linewidth=2.5, markersize=6, color='#F18F01')
    
    ax.set_xlabel('GelSight Frame Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Aligned Frame Index', fontsize=14, fontweight='bold')
    ax.set_title(f'Real Data Temporal Alignment\nObject: {obj_name}', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 标注关键点（起始、中间、结束）
    key_indices = [0, len(gelsight_frames)//2, len(gelsight_frames)-1]
    labels = ['Start', 'Middle', 'End']
    
    for idx, label in zip(key_indices, labels):
        gs = int(gelsight_frames[idx])
        digit = digit_aligned[idx]
        
        ax.plot(gs, digit, 'ro', markersize=10, zorder=5)
        ax.annotate(
            f'{label}\nGS:{gs}\nDT:{digit}',
            xy=(gs, digit),
            xytext=(gs+10, digit+15),
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5)
        )
    
    plt.tight_layout()
    
    # 保存图像
    output_path = Path(__file__).resolve().parent.parent / 'alignment_real_data.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 真实数据可视化保存到: {output_path}")
    
    plt.close()


def visualize_multiple_objects():
    """
    多物体对比可视化：展示不同物体的对齐情况
    """
    print("\n" + "="*60)
    print("多物体对比可视化")
    print("="*60)
    
    # 加载真实数据
    csv_path = Path(__file__).resolve().parent.parent / 'Tacquad' / 'contact_fine.csv'
    paired_index = build_paired_index(str(csv_path))
    
    # 取前 3 个物体
    objects = list(paired_index.keys())[:3]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, obj_name in enumerate(objects):
        frame_ranges = paired_index[obj_name]['frame_ranges']
        
        # 生成对齐映射
        gs_start, gs_end = frame_ranges['gelsight']
        gelsight_frames = np.linspace(gs_start, gs_end, 15)
        
        digit_aligned = []
        duragel_aligned = []
        
        for gs_frame in gelsight_frames:
            aligned = batch_align_frames('gelsight', int(gs_frame), frame_ranges)
            digit_aligned.append(aligned['digit'])
            duragel_aligned.append(aligned['duragel'])
        
        # 绘图
        ax = axes[idx]
        ax.plot(gelsight_frames, gelsight_frames, 'o-', 
                label='GelSight', linewidth=2, markersize=4, color='#2E86AB')
        ax.plot(gelsight_frames, digit_aligned, 's-', 
                label='DIGIT', linewidth=2, markersize=4, color='#A23B72')
        ax.plot(gelsight_frames, duragel_aligned, '^-', 
                label='DuraGel', linewidth=2, markersize=4, color='#F18F01')
        
        ax.set_xlabel('GelSight Frame', fontsize=11)
        ax.set_ylabel('Aligned Frame', fontsize=11)
        ax.set_title(f'{obj_name}\n({frame_ranges["gelsight"][1] - frame_ranges["gelsight"][0] + 1} frames)', 
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Temporal Alignment Comparison: Multiple Objects', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存图像
    output_path = Path(__file__).resolve().parent.parent / 'alignment_multiple_objects.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 多物体对比可视化保存到: {output_path}")
    
    plt.close()


def main():
    """主函数"""
    print("\n" + "="*60)
    print("时间对齐可视化验证")
    print("="*60)
    
    # 1. 理论可视化
    visualize_alignment_theory()
    
    # 2. 真实数据可视化
    visualize_real_data_alignment()
    
    # 3. 多物体对比
    visualize_multiple_objects()
    
    print("\n" + "="*60)
    print("✅ 所有可视化完成！")
    print("="*60)
    print("\n生成的图像文件:")
    print("  1. alignment_theory.png")
    print("  2. alignment_real_data.png")
    print("  3. alignment_multiple_objects.png")
    print("\n请查看这些图像以验证对齐算法的正确性！")


if __name__ == '__main__':
    main()

