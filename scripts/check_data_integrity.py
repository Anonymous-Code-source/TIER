#!/usr/bin/env python3
"""
数据完整性检查脚本
验证 TacQuad 数据是否完整下载
"""

import os
from pathlib import Path


def check_data_integrity():
    """检查数据完整性"""
    
    print("=" * 60)
    print("TacQuad 数据完整性检查")
    print("=" * 60)
    
    # 1. 检查目录结构
    # 获取脚本所在目录的父目录（PhysioTouch 根目录）
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    base_dir = project_root / 'Tacquad'
    required_dirs = [
        'data_fine',
        'data_indoor',
        'data_outdoor'
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}: 存在")
        else:
            print(f"❌ {dir_name}: 不存在！")
            all_exist = False
    
    if not all_exist:
        print("\n❌ 数据目录不完整！")
        return False
    
    # 2. 检查 CSV 文件
    required_csvs = [
        'contact_fine.csv',
        'contact_indoor.csv',
        'contact_outdoor.csv'
    ]
    
    for csv_name in required_csvs:
        csv_path = base_dir / csv_name
        if csv_path.exists():
            num_lines = sum(1 for _ in open(csv_path))
            print(f"✅ {csv_name}: 存在 ({num_lines} 行)")
        else:
            print(f"❌ {csv_name}: 不存在！")
            all_exist = False
    
    if not all_exist:
        print("\n❌ CSV 文件不完整！")
        return False
    
    # 3. 检查数据目录数量
    print("\n数据目录统计:")
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        num_objects = len(list(dir_path.glob('*')))
        print(f"   {dir_name}: {num_objects} objects")
    
    # 4. 检查示例物体的传感器完整性
    print("\n传感器完整性检查:")
    
    # Fine-grained 示例
    fine_obj = list((base_dir / 'data_fine').glob('*'))[0]
    sensors = ['gelsight', 'digit', 'duragel']
    all_sensors_exist = True
    
    print(f"   示例物体: {fine_obj.name}")
    for sensor in sensors:
        sensor_path = fine_obj / sensor
        if sensor_path.exists():
            num_frames = len(list(sensor_path.glob('*.png')))
            print(f"      {sensor}: {num_frames} 帧")
        else:
            print(f"      {sensor}: ❌ 不存在")
            all_sensors_exist = False
    
    if not all_sensors_exist:
        print("\n❌ 传感器数据不完整！")
        return False
    
    print("\n✅ 数据完整性检查通过！")
    return True


if __name__ == '__main__':
    check_data_integrity()

