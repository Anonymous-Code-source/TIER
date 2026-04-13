#!/usr/bin/env python3
"""
TacQuad CSV 解析模块
构建配对索引数据结构
"""

import csv
from pathlib import Path
from typing import Dict, Tuple


def build_paired_index(
    csv_file: str,
    min_sequence_length: int = 4
) -> Dict:
    """
    解析 TacQuad CSV 文件，构建配对索引
    
    Args:
        csv_file: CSV 文件路径
            - Fine: './Tacquad/contact_fine.csv'
            - Coarse: './Tacquad/contact_indoor.csv' or 'contact_outdoor.csv'
        min_sequence_length: 最小序列长度（默认 4，需要前 3 帧历史）
    
    Returns:
        paired_index: {
            'object_folder_name': {
                'frame_ranges': {
                    'gelsight': (start_idx, end_idx),
                    'digit': (start_idx, end_idx),
                    'duragel': (start_idx, end_idx)
                },
                'text': "text description",
                'min_frames': int,  # 最小帧数
                'max_frames': int   # 最大帧数
            },
            ...
        }
    
    CSV 格式（7-8 列）：
        folder_name, 
        gelsight_start, gelsight_end, 
        digit_start, digit_end, 
        duragel_start, duragel_end, 
        [text_description]  (可选)
    """
    
    paired_index = {}
    skipped_count = 0
    
    csv_path = Path(csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        for row_idx, row in enumerate(reader, start=1):
            # 跳过空行或格式错误的行
            if len(row) < 7:
                print(f"⚠️ Row {row_idx}: 格式错误（列数 < 7），跳过")
                skipped_count += 1
                continue
            
            try:
                folder_name = row[0].strip()
                
                # 解析帧范围
                frame_ranges = {
                    'gelsight': (int(row[1]), int(row[2])),
                    'digit':    (int(row[3]), int(row[4])),
                    'duragel':  (int(row[5]), int(row[6]))
                }
                
                # 计算每个传感器的帧数
                frame_counts = {
                    sensor: end - start + 1
                    for sensor, (start, end) in frame_ranges.items()
                }
                
                min_frames = min(frame_counts.values())
                max_frames = max(frame_counts.values())
                
                # 跳过帧数太少的数据
                if min_frames < min_sequence_length:
                    print(f"⚠️ {folder_name}: 帧数太少 ({min_frames} < {min_sequence_length})，跳过")
                    skipped_count += 1
                    continue
                
                # 提取文本描述（可选）
                text_description = row[7].strip() if len(row) > 7 else ""
                
                # 保存到索引
                paired_index[folder_name] = {
                    'frame_ranges': frame_ranges,
                    'frame_counts': frame_counts,
                    'min_frames': min_frames,
                    'max_frames': max_frames,
                    'text': text_description
                }
                
            except (ValueError, IndexError) as e:
                print(f"⚠️ Row {row_idx}: 解析错误 ({e})，跳过")
                skipped_count += 1
                continue
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"CSV 解析完成: {csv_file}")
    print(f"{'='*60}")
    print(f"✅ 成功解析: {len(paired_index)} objects")
    print(f"⚠️ 跳过: {skipped_count} rows")
    
    # 统计帧数分布
    if len(paired_index) > 0:
        all_min_frames = [info['min_frames'] for info in paired_index.values()]
        all_max_frames = [info['max_frames'] for info in paired_index.values()]
        
        print(f"\n帧数统计:")
        print(f"  最小帧数: {min(all_min_frames)} - {max(all_min_frames)}")
        print(f"  最大帧数: {min(all_max_frames)} - {max(all_max_frames)}")
    
    return paired_index


def print_paired_index_summary(paired_index: Dict, num_examples: int = 3):
    """
    打印配对索引摘要
    
    Args:
        paired_index: 配对索引字典
        num_examples: 打印前 N 个示例
    """
    print(f"\n{'='*60}")
    print(f"配对索引摘要")
    print(f"{'='*60}")
    
    # 打印前 N 个示例
    for idx, (obj_name, info) in enumerate(list(paired_index.items())[:num_examples]):
        print(f"\n[{idx+1}] {obj_name}")
        print(f"    GelSight: {info['frame_ranges']['gelsight']} ({info['frame_counts']['gelsight']} 帧)")
        print(f"    DIGIT:    {info['frame_ranges']['digit']} ({info['frame_counts']['digit']} 帧)")
        print(f"    DuraGel:  {info['frame_ranges']['duragel']} ({info['frame_counts']['duragel']} 帧)")
        print(f"    Min/Max:  {info['min_frames']} / {info['max_frames']}")
        if info['text']:
            print(f"    Text:     {info['text'][:50]}...")


# ============================================
# 单元测试
# ============================================
def test_build_paired_index():
    """测试 CSV 解析功能"""
    
    print("\n" + "="*60)
    print("测试：Fine-grained CSV 解析")
    print("="*60)
    
    # 测试 Fine-grained
    fine_index = build_paired_index('./Tacquad/contact_fine.csv')
    print_paired_index_summary(fine_index, num_examples=3)
    
    assert len(fine_index) > 0, "Fine-grained 索引为空！"
    assert len(fine_index) <= 32, f"Fine-grained objects 数量异常: {len(fine_index)}"
    
    print("\n✅ Fine-grained 解析测试通过！")
    
    # 测试 Coarse-grained (indoor)
    print("\n" + "="*60)
    print("测试：Coarse-grained (indoor) CSV 解析")
    print("="*60)
    
    coarse_indoor_index = build_paired_index('./Tacquad/contact_indoor.csv')
    print_paired_index_summary(coarse_indoor_index, num_examples=2)
    
    assert len(coarse_indoor_index) > 0, "Coarse-grained (indoor) 索引为空！"
    
    print("\n✅ Coarse-grained (indoor) 解析测试通过！")
    
    # 测试 Coarse-grained (outdoor)
    print("\n" + "="*60)
    print("测试：Coarse-grained (outdoor) CSV 解析")
    print("="*60)
    
    coarse_outdoor_index = build_paired_index('./Tacquad/contact_outdoor.csv')
    print_paired_index_summary(coarse_outdoor_index, num_examples=2)
    
    assert len(coarse_outdoor_index) > 0, "Coarse-grained (outdoor) 索引为空！"
    
    print("\n✅ Coarse-grained (outdoor) 解析测试通过！")
    
    print("\n" + "="*60)
    print("✅ 所有 CSV 解析测试通过！")
    print("="*60)


if __name__ == '__main__':
    test_build_paired_index()

