"""
精简版本的 process_clip.py
删除了所有未使用的函数、类和导入，只保留必要的全局变量访问函数。

版本: v1.0 - 清理冗余代码
日期: 2024
"""

# 全局配置变量，用于存储跨模块共享的配置
_global_config = {
    'NUM_FRAMES': 1, 
    'PATCH_DROPOUT': 0.0
}


def set_global_value(key, value):
    """
    设置全局配置值
    
    参数:
        key (str): 配置键名
        value: 配置值
    """
    global _global_config
    _global_config[key] = value


def get_global_value():
    """
    获取全局配置字典
    
    返回:
        dict: 全局配置字典
    """
    global _global_config
    return _global_config

