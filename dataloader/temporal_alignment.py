#!/usr/bin/env python3
"""
时间对齐模块
实现线性插值对齐算法
"""

from typing import Tuple, Dict


def get_temporally_aligned_frame(
    source_sensor: str,
    source_frame: int,
    target_sensor: str,
    frame_ranges: Dict[str, Tuple[int, int]],
    return_confidence: bool = False
) -> int:
    """
    时间对齐：将源传感器的帧映射到目标传感器
    
    使用线性插值：
    假设所有传感器同时开始和结束接触，中间时间线性对应
    
    Args:
        source_sensor: 源传感器名称（'gelsight', 'digit', 'duragel'）
        source_frame: 源传感器的帧索引
        target_sensor: 目标传感器名称
        frame_ranges: 帧范围字典 {sensor: (start, end)}
    
    Returns:
        target_frame: 目标传感器的对齐帧索引
        confidence: 可选返回，对齐置信度（0-1，越高表示越可靠）
    
    Example:
        frame_ranges = {
            'gelsight': (0, 199),  # 200 帧
            'digit': (0, 238),     # 239 帧
            'duragel': (0, 239)    # 240 帧
        }
        
        # GelSight frame 100 (50%) → DIGIT frame 119 (50%)
        result = get_temporally_aligned_frame('gelsight', 100, 'digit', frame_ranges)
        # result = 119
    """
    
    # 获取源传感器和目标传感器的帧范围
    src_start, src_end = frame_ranges[source_sensor]
    tgt_start, tgt_end = frame_ranges[target_sensor]
    
    # 计算总帧数
    src_total = src_end - src_start + 1
    tgt_total = tgt_end - tgt_start + 1
    
    # 边界检查
    if source_frame < src_start or source_frame > src_end:
        raise ValueError(
            f"Source frame {source_frame} out of range "
            f"[{src_start}, {src_end}] for {source_sensor}"
        )
    
    # 计算相对位置（归一化到 [0, 1]）
    src_relative = (source_frame - src_start) / (src_total - 1)
    
    # 映射到目标传感器（连续值）
    target_frame_float = tgt_start + src_relative * (tgt_total - 1)
    
    # 四舍五入到最近的整数
    target_frame = int(round(target_frame_float))
    
    # 边界保护
    target_frame = max(tgt_start, min(target_frame, tgt_end))

    if not return_confidence:
        return target_frame

    # 对齐置信度：量化误差 + 边界距离 + 序列长度匹配度（启发式）
    quant_error = abs(float(target_frame_float) - float(target_frame))
    # rounding误差最大约0.5，误差越小置信度越高
    conf_quant = 1.0 - min(1.0, quant_error / 0.5)

    # 离边界越远越稳定，中心位置更可靠
    src_span = max(src_total - 1, 1)
    tgt_span = max(tgt_total - 1, 1)
    src_margin = min(source_frame - src_start, src_end - source_frame) / src_span
    tgt_margin = min(target_frame - tgt_start, tgt_end - target_frame) / tgt_span
    conf_margin = min(1.0, max(0.0, 2.0 * min(src_margin, tgt_margin)))

    # 序列长度越接近，线性时间映射误差通常越小
    conf_length = min(src_total, tgt_total) / max(src_total, tgt_total)

    confidence = 0.55 * conf_quant + 0.30 * conf_margin + 0.15 * conf_length
    confidence = max(0.05, min(1.0, float(confidence)))
    return target_frame, confidence


def batch_align_frames(
    base_sensor: str,
    base_frame: int,
    frame_ranges: Dict[str, Tuple[int, int]],
    sensors: list = None,
    return_confidence: bool = False
) -> Dict[str, int]:
    """
    批量对齐：将基准传感器的一帧对齐到所有其他传感器
    
    Args:
        base_sensor: 基准传感器
        base_frame: 基准帧
        frame_ranges: 帧范围字典
        sensors: 传感器列表（默认 ['gelsight', 'digit', 'duragel']）
    
    Returns:
        aligned_frames: {sensor: aligned_frame}
        alignment_confidence: 可选返回 {sensor: confidence}
    
    Example:
        frame_ranges = {
            'gelsight': (31, 199),
            'digit': (55, 238),
            'duragel': (50, 239)
        }
        
        aligned = batch_align_frames('gelsight', 100, frame_ranges)
        # 返回：{'gelsight': 100, 'digit': 130, 'duragel': 128}
    """
    if sensors is None:
        sensors = ['gelsight', 'digit', 'duragel']
    
    aligned_frames = {}
    alignment_confidence = {} if return_confidence else None
    
    for sensor in sensors:
        if sensor == base_sensor:
            aligned_frames[sensor] = base_frame
            if return_confidence:
                alignment_confidence[sensor] = 1.0
        else:
            if return_confidence:
                aligned_frame, confidence = get_temporally_aligned_frame(
                    base_sensor, base_frame, sensor, frame_ranges, return_confidence=True
                )
                aligned_frames[sensor] = aligned_frame
                alignment_confidence[sensor] = confidence
            else:
                aligned_frames[sensor] = get_temporally_aligned_frame(
                    base_sensor, base_frame, sensor, frame_ranges
                )

    if return_confidence:
        return aligned_frames, alignment_confidence
    return aligned_frames


# ============================================
# 单元测试
# ============================================
def test_temporal_alignment():
    """测试时间对齐算法"""
    
    print("\n" + "="*60)
    print("时间对齐算法单元测试")
    print("="*60)
    
    # 测试用例 1：简单的 0 开始帧范围
    frame_ranges_1 = {
        'gelsight': (0, 199),  # 200 帧
        'digit': (0, 238),     # 239 帧
        'duragel': (0, 239)    # 240 帧
    }
    
    test_cases_1 = [
        # (source_sensor, source_frame, target_sensor, expected_result)
        ('gelsight', 0, 'digit', 0),       # 起始帧
        ('gelsight', 199, 'digit', 238),   # 结束帧
        ('gelsight', 100, 'digit', 120),   # 中间帧（约50%）
        ('gelsight', 0, 'duragel', 0),
        ('gelsight', 199, 'duragel', 239),
        ('gelsight', 100, 'duragel', 120),
        ('digit', 120, 'gelsight', 100),   # 反向对齐
        ('duragel', 120, 'gelsight', 100),
    ]
    
    print("\n[测试集 1] 简单帧范围 (从 0 开始)")
    all_passed = True
    
    for idx, (src_sensor, src_frame, tgt_sensor, expected) in enumerate(test_cases_1, 1):
        result = get_temporally_aligned_frame(
            src_sensor, src_frame, tgt_sensor, frame_ranges_1
        )
        
        passed = (result == expected)
        status = "✅" if passed else "❌"
        
        print(f"{status} Test {idx}: {src_sensor}[{src_frame}] → {tgt_sensor}[{result}] "
              f"(expected {expected})")
        
        if not passed:
            all_passed = False
    
    # 测试用例 2：非 0 开始的帧范围（真实数据）
    frame_ranges_2 = {
        'gelsight': (31, 199),  # 169 帧
        'digit': (55, 238),     # 184 帧
        'duragel': (50, 239)    # 190 帧
    }
    
    print("\n[测试集 2] 真实帧范围 (非 0 开始)")
    
    test_cases_2 = [
        # 边界测试
        ('gelsight', 31, 'digit', 55),      # 起始帧
        ('gelsight', 199, 'digit', 238),    # 结束帧
        ('gelsight', 31, 'duragel', 50),
        ('gelsight', 199, 'duragel', 239),
        # 中间点测试
        ('gelsight', 115, 'digit', 146),    # 约 50%
        ('gelsight', 115, 'duragel', 145),
    ]
    
    for idx, (src_sensor, src_frame, tgt_sensor, expected) in enumerate(test_cases_2, 1):
        result = get_temporally_aligned_frame(
            src_sensor, src_frame, tgt_sensor, frame_ranges_2
        )
        
        # 允许 ±1 的误差（四舍五入）
        passed = abs(result - expected) <= 1
        status = "✅" if passed else "❌"
        
        print(f"{status} Test {idx}: {src_sensor}[{src_frame}] → {tgt_sensor}[{result}] "
              f"(expected ~{expected})")
        
        if not passed:
            all_passed = False
    
    # 测试批量对齐
    print("\n[测试集 3] 批量对齐测试")
    aligned = batch_align_frames('gelsight', 100, frame_ranges_1)
    print(f"  GelSight: {aligned['gelsight']}")
    print(f"  DIGIT:    {aligned['digit']}")
    print(f"  DuraGel:  {aligned['duragel']}")
    
    # 验证
    assert aligned['gelsight'] == 100, f"GelSight 应为 100，实际为 {aligned['gelsight']}"
    assert aligned['digit'] == 120, f"DIGIT 应为 120，实际为 {aligned['digit']}"
    assert aligned['duragel'] == 120, f"DuraGel 应为 120，实际为 {aligned['duragel']}"
    print("  ✅ 批量对齐测试通过")
    
    # 边界错误测试
    print("\n[测试集 4] 边界错误处理")
    try:
        # 超出范围的帧
        get_temporally_aligned_frame('gelsight', 300, 'digit', frame_ranges_1)
        print("  ❌ 应该抛出 ValueError，但没有")
        all_passed = False
    except ValueError as e:
        print(f"  ✅ 正确捕获边界错误: {e}")
    
    if all_passed:
        print("\n" + "="*60)
        print("✅ 所有测试通过！")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ 部分测试失败！")
        print("="*60)
        raise AssertionError("Time alignment tests failed")


def test_real_data_alignment():
    """使用真实数据进行对齐测试"""
    
    print("\n" + "="*60)
    print("真实数据对齐验证")
    print("="*60)
    
    # 从 build_paired_index 导入并测试
    try:
        from .build_paired_index import build_paired_index
        
        print("\n加载 Fine-grained 数据...")
        paired_index = build_paired_index('./Tacquad/contact_fine.csv', min_sequence_length=4)
        
        # 取第一个物体测试
        obj_name = list(paired_index.keys())[0]
        frame_ranges = paired_index[obj_name]['frame_ranges']
        
        print(f"\n测试物体: {obj_name}")
        print(f"  GelSight: {frame_ranges['gelsight']}")
        print(f"  DIGIT:    {frame_ranges['digit']}")
        print(f"  DuraGel:  {frame_ranges['duragel']}")
        
        # 测试起始帧对齐
        gs_start = frame_ranges['gelsight'][0]
        aligned = batch_align_frames('gelsight', gs_start, frame_ranges)
        
        print(f"\n对齐起始帧 (GelSight={gs_start}):")
        print(f"  GelSight: {aligned['gelsight']} (应为 {frame_ranges['gelsight'][0]})")
        print(f"  DIGIT:    {aligned['digit']} (应为 {frame_ranges['digit'][0]})")
        print(f"  DuraGel:  {aligned['duragel']} (应为 {frame_ranges['duragel'][0]})")
        
        # 验证起始帧对齐正确
        assert aligned['gelsight'] == frame_ranges['gelsight'][0]
        assert aligned['digit'] == frame_ranges['digit'][0]
        assert aligned['duragel'] == frame_ranges['duragel'][0]
        
        # 测试结束帧对齐
        gs_end = frame_ranges['gelsight'][1]
        aligned_end = batch_align_frames('gelsight', gs_end, frame_ranges)
        
        print(f"\n对齐结束帧 (GelSight={gs_end}):")
        print(f"  GelSight: {aligned_end['gelsight']} (应为 {frame_ranges['gelsight'][1]})")
        print(f"  DIGIT:    {aligned_end['digit']} (应为 {frame_ranges['digit'][1]})")
        print(f"  DuraGel:  {aligned_end['duragel']} (应为 {frame_ranges['duragel'][1]})")
        
        # 验证结束帧对齐正确
        assert aligned_end['gelsight'] == frame_ranges['gelsight'][1]
        assert aligned_end['digit'] == frame_ranges['digit'][1]
        assert aligned_end['duragel'] == frame_ranges['duragel'][1]
        
        print("\n✅ 真实数据对齐验证通过！")
        
    except ImportError:
        print("\n⚠️ 无法导入 build_paired_index，跳过真实数据测试")


if __name__ == '__main__':
    # 运行基础单元测试
    test_temporal_alignment()
    
    # 运行真实数据测试
    test_real_data_alignment()
