"""
model/discriminator.py
Domain Discriminator: 尝试从表征预测传感器类型

用于 InfoMax-Share Loss，目标是让判别器无法从表征中识别传感器类型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainDiscriminator(nn.Module):
    """
    域判别器：从表征预测传感器类型
    
    架构:
        input: [B, 128] (表征)
        → Linear(128, 64)
        → ReLU
        → Dropout(0.1)
        → Linear(64, num_sensors)
        → LogSoftmax
        → output: [B, num_sensors] (log probabilities)
    
    目标：
        - InfoMax-Share Loss 会最小化判别器的准确率
        - 使表征分布接近均匀分布（无法预测传感器类型）
    """
    
    def __init__(self, input_dim=128, hidden_dim=64, num_sensors=3, dropout=0.1):
        """
        Args:
            input_dim: 输入维度（projection head 输出）
            hidden_dim: 隐藏层维度
            num_sensors: 传感器数量（GelSight, DIGIT, DuraGel）
            dropout: Dropout 概率（防止过拟合）
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_sensors)
        )
        
        self.num_sensors = num_sensors
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        """
        Args:
            z: [B, input_dim] L2-normalized 表征
        
        Returns:
            log_probs: [B, num_sensors] log probabilities
        """
        logits = self.net(z)  # [B, num_sensors]
        log_probs = F.log_softmax(logits, dim=1)
        return log_probs
    
    def predict(self, z):
        """
        预测传感器类型（用于评估）
        
        Args:
            z: [B, input_dim] 表征
        
        Returns:
            pred: [B] 预测的传感器索引
            probs: [B, num_sensors] 概率分布
        """
        log_probs = self.forward(z)
        probs = torch.exp(log_probs)
        pred = torch.argmax(probs, dim=1)
        return pred, probs


