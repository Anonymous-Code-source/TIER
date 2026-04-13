"""
model/projection_head.py
Projection Head: 768-d → 128-d (L2-normalized)

将 Stage-1 encoder 的 768 维输出投影到 128 维对比学习空间
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    将 Stage-1 的 768-d 表征投影到 128-d 对比空间
    
    架构:
        input: [B, 768]
        → Linear(768, 256)
        → LayerNorm(256)
        → ReLU
        → Linear(256, 128)
        → L2 Normalize
        → output: [B, 128]
    """
    
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=128):
        """
        Args:
            input_dim: 输入维度（Stage-1 encoder 输出）
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（对比学习空间）
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, h):
        """
        Args:
            h: [B, input_dim] encoder 输出
        
        Returns:
            z: [B, output_dim] L2-normalized 表征
        """
        z = self.net(h)  # [B, output_dim]
        z = F.normalize(z, dim=1, p=2)  # L2 归一化
        return z

