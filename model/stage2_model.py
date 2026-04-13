"""
model/stage2_model.py
Stage-2 跨传感器无感表征学习模型

从 Stage-1 预训练的 TactileVideoMAE 加载 encoder，添加解耦双头:
- Contact head: 学习接触语义表征 z_contact
- Domain head: 学习域对抗分支表征 z_domain
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .projection_head import ProjectionHead
from .discriminator import DomainDiscriminator


def load_config_safe(model_name_or_path, offline_first=True):
    """
    安全加载 HuggingFace 配置，优先使用本地资源
    
    Args:
        model_name_or_path: 模型名称或路径
        offline_first: 是否优先使用离线模式
    
    Returns:
        config: HuggingFace 配置对象
    """
    from transformers import AutoConfig
    
    # 方法 1: 尝试使用项目本地的 CLIP 配置
    local_clip_paths = [
        "./CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
        "/home/shipeng/PhysioTouch/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
    ]
    
    for local_path in local_clip_paths:
        if os.path.exists(os.path.join(local_path, "config.json")):
            try:
                print(f"   📁 Using local CLIP config from: {local_path}")
                config = AutoConfig.from_pretrained(local_path, local_files_only=True)
                
                # 如果需要 ViT-B/16 的配置（Stage-1 使用的），我们需要调整
                # ViT-L/14 → ViT-B/16 的主要区别：
                # - hidden_size: 1024 → 768
                # - num_hidden_layers: 24 → 12
                # - num_attention_heads: 16 → 12
                # - patch_size: 14 → 16
                
                if hasattr(config, 'vision_config'):
                    vision_config = config.vision_config
                    # 保持 ViT-L/14 配置（与 Stage-1 一致）
                    # Stage-1 实际使用: hidden_size=1024, num_layers=24, heads=16, patch_size=14
                    # 不需要修改这些核心参数，只添加缺失的 ViT 特定属性
                    
                    # 添加 ViT 需要的额外属性（CLIP config 可能缺少）
                    if not hasattr(vision_config, 'qkv_bias'):
                        vision_config.qkv_bias = True
                    if not hasattr(vision_config, 'encoder_stride'):
                        vision_config.encoder_stride = vision_config.patch_size
                    if not hasattr(vision_config, 'attention_probs_dropout_prob'):
                        vision_config.attention_probs_dropout_prob = 0.0
                    if not hasattr(vision_config, 'hidden_dropout_prob'):
                        vision_config.hidden_dropout_prob = 0.0
                    
                    print(f"   ✅ Using ViT-L/14 config (hidden_size={vision_config.hidden_size}, patch_size={vision_config.patch_size})")
                
                return config
            except Exception as e:
                print(f"   ⚠️  Failed to load from {local_path}: {e}")
                continue
    
    # 方法 2: 尝试离线模式（使用缓存）
    if offline_first:
        try:
            print(f"   🔌 Trying offline mode (using cached files)...")
            config = AutoConfig.from_pretrained(
                model_name_or_path, 
                local_files_only=True
            )
            print(f"   ✅ Loaded from cache")
            return config
        except Exception as e:
            print(f"   ⚠️  Offline mode failed: {e}")
    
    # 方法 3: 尝试在线下载（如果前两种方法都失败）
    try:
        print(f"   🌐 Trying online download from HuggingFace Hub...")
        config = AutoConfig.from_pretrained(model_name_or_path)
        print(f"   ✅ Downloaded from HuggingFace Hub")
        return config
    except Exception as e:
        print(f"   ❌ Online download failed: {e}")
        
        # 方法 4: 如果所有方法都失败，使用手动创建的默认配置
        print(f"   🛠️  Creating default ViT-B/16 config manually...")
        from transformers import CLIPConfig, CLIPVisionConfig
        
        vision_config = CLIPVisionConfig(
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            image_size=224,
            patch_size=16,
            num_channels=3,
            hidden_act="gelu",
            layer_norm_eps=1e-5,
            attention_dropout=0.0,
            initializer_range=0.02,
        )
        
        config = CLIPConfig(
            vision_config=vision_config.to_dict(),
            projection_dim=768,
        )
        
        print(f"   ✅ Using manually created ViT-B/16 config")
        return config


def load_stage1_encoder(checkpoint_path, args=None, device='cuda'):
    """
    从 Stage-1 checkpoint 加载预训练 TactileVideoMAE encoder
    
    Args:
        checkpoint_path: Stage-1 模型权重路径
        args: 模型配置参数（如果为 None，使用默认配置）
        device: 'cuda' 或 'cpu'
    
    Returns:
        model: 完整的 TactileVideoMAE 模型（包含 encoder）
    """
    print(f"📂 Loading Stage-1 checkpoint from: {checkpoint_path}")
    
    # 加载 checkpoint
    # PyTorch 2.6+ 需要 weights_only=False 来加载包含 argparse.Namespace 的 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 提取 state_dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print(f"   Checkpoint keys: {list(checkpoint.keys())}")
    
    # 导入 Stage-1 的模型定义和配置
    from model.mae_model import TactileVideoMAE
    
    # 如果没有提供 args，从 checkpoint 中提取或创建默认配置
    if args is None:
        # 优先从 checkpoint 中提取原始 args
        if 'args' in checkpoint:
            args = checkpoint['args']
            print(f"   ✅ Using args from checkpoint")
            print(f"      use_sensor_token={args.use_sensor_token}")
            print(f"      use_video={getattr(args, 'use_video', True)}")
        else:
            # 如果 checkpoint 中没有 args，使用默认配置
            class MinimalArgs:
                use_sensor_token = False  # 根据实际 checkpoint 设置
                use_same_patchemb = False
                new_decoder_sensor_token = False
                norm_pix_loss = True
                mask_ratio = 0.0  # Stage-2 不需要 mask
            args = MinimalArgs()
            print(f"   ⚠️  Using default args (checkpoint has no args)")
    
    # 加载配置（使用安全加载函数，优先本地资源）
    print("   Loading model configuration...")
    # 从本地加载 ViT-L/14 配置（与 Stage-1 训练时一致）
    config = load_config_safe("openai/clip-vit-large-patch14", offline_first=True)
    config.vision_config.image_size = 224
    # Stage-1 使用: encoder hidden_size=1024, projection_dim=768
    config.projection_dim = 768  # Stage-1 实际使用的 projection 维度
    
    # Decoder 配置（使用安全加载函数）
    decoder_config = load_config_safe("openai/clip-vit-large-patch14", offline_first=True).vision_config
    decoder_config.hidden_size = 512
    decoder_config.num_hidden_layers = 8
    decoder_config.num_attention_heads = 8  # 512 / 8 = 64 (head_dim)
    decoder_config.intermediate_size = 2048  # 通常是 hidden_size * 4
    decoder_config.num_channels = 3
    # 确保 decoder 配置也有 ViT 需要的属性
    if not hasattr(decoder_config, 'qkv_bias'):
        decoder_config.qkv_bias = True
    if not hasattr(decoder_config, 'attention_probs_dropout_prob'):
        decoder_config.attention_probs_dropout_prob = 0.0
    if not hasattr(decoder_config, 'hidden_dropout_prob'):
        decoder_config.hidden_dropout_prob = 0.0
    
    # 创建模型
    print("   Creating TactileVideoMAE model...")
    model = TactileVideoMAE(
        args=args,
        config=config,
        decoder_config=decoder_config,
        num_frames=4,
        add_time_attn=True,
        tube_size=1
    )
    
    # 加载权重
    print("   Loading weights...")
    # 处理 DataParallel 保存的模型（可能有 'module.' 前缀）
    new_state_dict = {}
    for key, value in state_dict.items():
        # 去除 'module.' 前缀（如果有）
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    
    # 加载到模型
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"   ✅ Weights loaded successfully")
    if len(msg.missing_keys) > 0:
        print(f"      Missing keys: {len(msg.missing_keys)}")
        # print(f"      {msg.missing_keys[:5]}")  # 只打印前5个
    if len(msg.unexpected_keys) > 0:
        print(f"      Unexpected keys: {len(msg.unexpected_keys)}")
    
    model = model.to(device)
    model.eval()
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    return model


class Stage2Model(nn.Module):
    """
    Stage-2 主模型
    
    架构:
        输入: [B, 4, 3, 224, 224]
        → Stage-1 Encoder (TactileVideoMAE, 冻结前 N 层)
        → [B, 768]
        → Contact Projection Head -> z_contact [B, D_c] (L2-normalized)
        → Domain Projection Head  -> z_domain  [B, D_d] (L2-normalized)
    """
    
    def __init__(
        self, 
        stage1_model, 
        freeze_blocks=20,
        projection_dim=128,
        domain_projection_dim=None,
        discriminator_input_dim=None,
        use_discriminator=True,
        discriminator_hidden_dim=64,
        discriminator_dropout=0.1,
        num_sensors=3,
        use_matching_head=False,
        use_domain_classifier=False
    ):
        """
        Args:
            stage1_model: Stage-1 预训练的 TactileVideoMAE 模型
            freeze_blocks: 冻结 touch_model 前 N 个 Transformer 块（总共 12 块）
            projection_dim: 接触语义投影维度（默认 128）
            domain_projection_dim: 域分支投影维度（默认与 projection_dim 相同）
            use_discriminator: 是否使用域判别器
            discriminator_hidden_dim: 域判别器隐藏层维度
            discriminator_dropout: 域判别器 Dropout 概率
            num_sensors: 传感器数量
        """
        super().__init__()
        
        self.stage1_model = stage1_model
        self.freeze_blocks = freeze_blocks
        self.projection_dim = projection_dim
        self.domain_projection_dim = (
            int(domain_projection_dim) if domain_projection_dim is not None else int(projection_dim)
        )
        self.discriminator_input_dim = (
            int(discriminator_input_dim) if discriminator_input_dim is not None else self.domain_projection_dim
        )
        
        # 冻结 encoder 的部分层
        self.freeze_encoder_except_last_n_blocks(freeze_blocks)
        
        # Contact Head (768 -> D_c, L2-Norm)
        self.projection_head = ProjectionHead(
            input_dim=768,
            hidden_dim=256,
            output_dim=projection_dim
        )

        # Domain Head (768 -> D_d, L2-Norm)
        self.domain_projection_head = ProjectionHead(
            input_dim=768,
            hidden_dim=256,
            output_dim=self.domain_projection_dim
        )
        
        # Domain Discriminator (可选)
        if use_discriminator:
            self.discriminator = DomainDiscriminator(
                input_dim=self.discriminator_input_dim,
                hidden_dim=discriminator_hidden_dim,
                num_sensors=num_sensors,
                dropout=discriminator_dropout
            )
        else:
            self.discriminator = None

        # Domain Supervisor on z_domain (显式域监督分支)
        if use_domain_classifier:
            self.domain_classifier = DomainDiscriminator(
                input_dim=self.domain_projection_dim,
                hidden_dim=discriminator_hidden_dim,
                num_sensors=num_sensors,
                dropout=discriminator_dropout
            )
        else:
            self.domain_classifier = None

        if use_matching_head:
            self.match_head = nn.Sequential(
                nn.Linear(projection_dim, projection_dim),
                nn.ReLU(inplace=True),
                nn.Linear(projection_dim, 1)
            )
        else:
            self.match_head = None
        
        print(f"\n🏗️  Stage2Model initialized:")
        print(f"   - Frozen blocks: {freeze_blocks}")
        print(f"   - Contact projection dim: {projection_dim}")
        print(f"   - Domain projection dim: {self.domain_projection_dim}")
        print(f"   - Discriminator input dim: {self.discriminator_input_dim}")
        print(f"   - Use discriminator: {use_discriminator}")
        print(f"   - Use domain classifier: {use_domain_classifier}")
        print(f"   - Use matching head: {use_matching_head}")
    
    def freeze_encoder_except_last_n_blocks(self, freeze_blocks):
        """
        冻结 encoder 的前 N 层
        
        Stage-1 TactileVideoMAE 的 encoder 是 touch_model (CLIPVisionTransformer)
        包含 12 个 Transformer blocks
        
        Args:
            freeze_blocks: 从 0 开始计数，冻结前 N 个 blocks
        """
        # 如果 freeze_blocks = 0，不冻结任何层
        if freeze_blocks == 0:
            print(f"\n❄️  Freezing encoder layers...")
            print(f"   freeze_blocks=0, all layers trainable")
            return
        
        touch_model = self.stage1_model.touch_model
        
        # 检查 touch_model 是否有 encoder 属性（兼容性检查）
        if not hasattr(touch_model, 'encoder'):
            print(f"\n⚠️  Warning: touch_model has no 'encoder' attribute, skipping freeze")
            print(f"   This is expected for mock/test models")
            return
        
        encoder = touch_model.encoder  # CLIPEncoder
        
        # 获取总的 blocks 数量
        total_blocks = len(encoder.layers)
        trainable_blocks = max(0, total_blocks - freeze_blocks)
        
        print(f"\n❄️  Freezing encoder layers...")
        print(f"   Total blocks: {total_blocks}")
        print(f"   Frozen blocks: {min(freeze_blocks, total_blocks)}")
        print(f"   Trainable blocks: {trainable_blocks}")
        
        # 1. 冻结 embeddings（patch_embed, position_embed 等）
        for param in touch_model.embeddings.parameters():
            param.requires_grad = False
        print(f"   ✅ Frozen: embeddings")
        
        # 2. 冻结前 N 个 Transformer 块
        for i, layer in enumerate(encoder.layers):
            if i < freeze_blocks:
                for param in layer.parameters():
                    param.requires_grad = False
        
        if freeze_blocks > 0:
            print(f"   ✅ Frozen: layers 0-{min(freeze_blocks-1, total_blocks-1)}")
        if trainable_blocks > 0:
            print(f"   🔓 Trainable: layers {freeze_blocks}-{total_blocks-1}")
        
        # 3. 冻结 pre_layrnorm（如果存在）
        if hasattr(touch_model, 'pre_layrnorm'):
            for param in touch_model.pre_layrnorm.parameters():
                param.requires_grad = False
        
        # 4. 冻结 post_layernorm（通常在最后，保持可训练）
        # 如果需要也可以冻结
        
        # 5. touch_projection 保持可训练
        
        # 6. 统计可训练参数
        encoder_trainable = sum(
            p.numel() for p in self.stage1_model.touch_model.parameters() if p.requires_grad
        )
        encoder_total = sum(p.numel() for p in self.stage1_model.touch_model.parameters())
        projection_trainable = sum(p.numel() for p in self.stage1_model.touch_projection.parameters() if p.requires_grad)
        
        print(f"\n📊 Encoder parameters:")
        print(f"   Total: {encoder_total:,}")
        print(f"   Trainable: {encoder_trainable:,} ({encoder_trainable/encoder_total*100:.1f}%)")
        print(f"   Touch projection: {projection_trainable:,}")
    
    def forward(
        self,
        x,
        sensor_type=None,
        return_encoder_output=False,
        return_domain=False,
        return_all=False
    ):
        """
        前向传播
        
        Args:
            x: [B, 4, 3, 224, 224] 输入图像序列（4 帧）
            sensor_type: 传感器类型索引（可选）
            return_encoder_output: 是否返回 encoder 的原始输出
            return_domain: 是否同时返回域分支表征
            return_all: 是否以 dict 形式返回全部输出
        
        Returns:
            默认:
                z_contact: [B, D_c]
            可选:
                z_domain: [B, D_d]
                h: [B, 768]
        """
        # 使用 Stage-1 encoder（不使用 mask）
        # forward_encoder 返回 (out, mask, ids_restore)
        # 当 use_mask=False 时，使用 pooler_output
        with torch.set_grad_enabled(self.training):
            h, _, _ = self.stage1_model.forward_encoder(
                x, 
                sensor_type=sensor_type, 
                data_type=1,  # 1 = video mode (4 frames)
                use_mask=False  # 不使用 mask，直接提取特征
            )
        
        # h: [B, 768] (从 touch_projection 输出)
        
        # 解耦双头投影
        z_contact = self.projection_head(h)         # [B, D_c], L2-normalized
        z_domain = self.domain_projection_head(h)   # [B, D_d], L2-normalized

        if return_all:
            outputs = {
                'z_contact': z_contact,
                'z_domain': z_domain
            }
            if return_encoder_output:
                outputs['h'] = h
            return outputs

        if return_encoder_output:
            if return_domain:
                return z_contact, z_domain, h
            return z_contact, h

        if return_domain:
            return z_contact, z_domain

        return z_contact
    
    def get_trainable_params(self):
        """
        获取可训练参数列表（用于优化器）
        
        Returns:
            param_groups: dict
                {
                    'encoder': encoder 可训练参数,
                    'projection': contact head 参数,
                    'domain_projection': domain head 参数,
                    'discriminator': discriminator 参数 (如果有)
                }
        """
        # Stage-1 Encoder 可训练参数
        encoder_params = [
            p for p in self.stage1_model.parameters() if p.requires_grad
        ]
        
        # Projection Head 参数
        projection_params = list(self.projection_head.parameters())
        domain_projection_params = list(self.domain_projection_head.parameters())
        
        # Discriminator 参数
        discriminator_params = []
        if self.discriminator is not None:
            discriminator_params = list(self.discriminator.parameters())
        
        domain_classifier_params = []
        if self.domain_classifier is not None:
            domain_classifier_params = list(self.domain_classifier.parameters())

        match_params = []
        if self.match_head is not None:
            match_params = list(self.match_head.parameters())
        
        return {
            'encoder': encoder_params,
            'projection': projection_params,
            'domain_projection': domain_projection_params,
            'discriminator': discriminator_params,
            'domain_classifier': domain_classifier_params,
            'matching': match_params
        }
    
    def get_param_count(self):
        """获取参数统计信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        param_groups = self.get_trainable_params()
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'trainable_pct': trainable_params / total_params * 100,
            'encoder': sum(p.numel() for p in param_groups['encoder']),
            'projection': sum(p.numel() for p in param_groups['projection']),
            'domain_projection': sum(p.numel() for p in param_groups['domain_projection']),
            'discriminator': sum(p.numel() for p in param_groups['discriminator']),
            'domain_classifier': sum(p.numel() for p in param_groups['domain_classifier']),
            'matching': sum(p.numel() for p in param_groups['matching'])
        }
