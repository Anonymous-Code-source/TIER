import sys

import copy

import torch
import numpy as np
from einops import rearrange
from typing import Optional, Tuple, Union

from torch import nn
from transformers import CLIPModel as HFCLIPModel, CLIPVisionConfig
from transformers import AutoConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIP_VISION_INPUTS_DOCSTRING,CLIPVisionTransformer
from transformers.models.vit.modeling_vit import ViTLayer
from transformers.utils import replace_return_docstrings, add_start_docstrings_to_model_forward
from model.process_clip import get_global_value, set_global_value
from model.util.pos_embed import get_2d_sincos_pos_embed

class TactileMAE(nn.Module):
    def __init__(self, args, config, decoder_config, num_frames, add_time_attn, tube_size):
        super(TactileMAE, self).__init__()

        config.vision_config.num_frames = num_frames
        config.vision_config.tube_size = tube_size

        self.use_sensor_token = args.use_sensor_token
        self.new_decoder_sensor_token = args.new_decoder_sensor_token

        self.norm_pix_loss = False
        if args.norm_pix_loss:
            self.norm_pix_loss = args.norm_pix_loss

        self.touch_model = CLIPVisionTransformer(config.vision_config)
        self.touch_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        self.decoder_embed = nn.Linear(config.projection_dim, decoder_config.hidden_size, bias=True)
        self.num_patches = self.touch_model.embeddings.num_patches
        self.patch_size = config.vision_config.patch_size
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_config.hidden_size), requires_grad=False)
        self.touch_decoder_blocks = nn.ModuleList([ViTLayer(decoder_config) for _ in range(decoder_config.num_hidden_layers)])

        self.decoder_norm = nn.LayerNorm(decoder_config.hidden_size, eps=decoder_config.layer_norm_eps)
        self.decoder_pred = nn.Linear(decoder_config.hidden_size, decoder_config.patch_size**2 * decoder_config.num_channels, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_config.hidden_size))

        if self.use_sensor_token:
            self.sensor_token = nn.Parameter(torch.zeros(10, 5, config.vision_config.hidden_size))
            if self.new_decoder_sensor_token:
                self.sensor_token_proj = nn.Linear(config.vision_config.hidden_size, decoder_config.hidden_size, bias=False)

        self.mask_ratio = args.mask_ratio

        self.touch_model.forward = self.touch_forward
        self.touch_model.embeddings.forward = self.emb_forward


    def initialize_decoder(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)
        if self.use_sensor_token:
            torch.nn.init.normal_(self.sensor_token, std=.02)

        # 🔥 关键修复：初始化预测层权重，防止NaN
        torch.nn.init.xavier_uniform_(self.decoder_pred.weight)
        if self.decoder_pred.bias is not None:
            torch.nn.init.constant_(self.decoder_pred.bias, 0)

    def random_masking(self, sequence, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def touch_forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sensor_type = None
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        # a = self.sensor_token[sensor_type]
        # print(a.shape)
        output_attentions = output_attentions if output_attentions is not None else self.touch_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.touch_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.touch_model.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states, mask, ids_restore  = self.touch_model.embeddings(pixel_values, sensor_type = sensor_type)
        hidden_states = self.touch_model.pre_layrnorm(hidden_states)

        encoder_outputs = self.touch_model.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.touch_model.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        ), mask, ids_restore

    def emb_forward(self, pixel_values: torch.FloatTensor, noise=None, sensor_type=None) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.touch_model.embeddings.patch_embedding.weight.dtype
        patch_embeds = self.touch_model.embeddings.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        pos_emb = self.touch_model.embeddings.position_embedding(self.touch_model.embeddings.position_ids)

        embeddings = patch_embeds + pos_emb[:, 1:, :]
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)

        class_embeds = self.touch_model.embeddings.class_embedding + pos_emb[:, 0, :]
        class_embeds = class_embeds.expand(batch_size, 1, -1)

        if self.use_sensor_token:
            sensor_emb = self.sensor_token[sensor_type]
            embeddings = torch.cat([class_embeds, sensor_emb, embeddings], dim=1)
        else:
            embeddings = torch.cat([class_embeds, embeddings], dim=1)
        #embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings, mask, ids_restore

    def forward(self, x, sensor_type=None, target_sensor_type=None):
        latent, mask, ids_restore = self.forward_encoder(x, sensor_type=sensor_type)
        pred = self.forward_decoder(latent, ids_restore, sensor_type=target_sensor_type)

        loss = self.forward_loss(x, pred, mask)
        
        return loss, pred, mask

    def forward_encoder(self, x, sensor_type=None):

        x, mask, ids_restore = self.touch_model(x, sensor_type=sensor_type)
        out = self.touch_projection(x.last_hidden_state)

        return out, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore, sensor_type=None):

        x = self.decoder_embed(x)

        if self.use_sensor_token:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 6 - x.shape[1], 1)

            x_ = torch.cat([x[:, 6:, :], mask_tokens], dim=1)  # no cls token and sensor token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            if self.new_decoder_sensor_token:
                decoder_sensor = self.sensor_token_proj(self.sensor_token[sensor_type])
                x = torch.cat([x[:, :1, :], decoder_sensor, x_], dim=1)  # append cls token and sensor token
            else:
                x = torch.cat([x[:, :6, :], x_], dim=1)  # append cls token and sensor token

            x[:,0,:] += self.decoder_pos_embed[:,0,:]
            x[:,6:,:] += self.decoder_pos_embed[:,1:,:]
        else:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

            x = x + self.decoder_pos_embed

        for blk in self.touch_decoder_blocks:
            layer_outputs = blk(x)
        
            x = layer_outputs[0]

        x = self.decoder_norm(x)

        x = self.decoder_pred(x)

        if self.use_sensor_token:
            x = x[:, 6:, :]
        else:
            x = x[:, 1:, :]

        return x
    
    def forward_loss(self, x, pred, mask):
        target = self.patchify(x)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

class TactileVideoMAE(nn.Module):
    def __init__(self, args, config, decoder_config, num_frames, add_time_attn, tube_size):
        super(TactileVideoMAE, self).__init__()

        config.vision_config.num_frames = num_frames
        config.vision_config.tube_size = tube_size

        self.use_sensor_token = args.use_sensor_token
        self.use_same_patchemb = args.use_same_patchemb
        self.new_decoder_sensor_token = args.new_decoder_sensor_token

        self.norm_pix_loss = False
        if args.norm_pix_loss:
            self.norm_pix_loss = args.norm_pix_loss

        self.touch_model = CLIPVisionTransformer(config.vision_config)
        self.touch_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        self.video_patch_embedding = nn.Conv3d(
            in_channels=config.vision_config.num_channels,
            out_channels=self.touch_model.embeddings.embed_dim,
            kernel_size=(3, self.touch_model.embeddings.patch_size, self.touch_model.embeddings.patch_size),
            stride=(3, self.touch_model.embeddings.patch_size, self.touch_model.embeddings.patch_size),
            bias=False,
        )
        self.video_position_embedding = nn.Embedding(self.touch_model.embeddings.num_positions, self.touch_model.embeddings.embed_dim)

        self.decoder_embed = nn.Linear(config.projection_dim, decoder_config.hidden_size, bias=True)
        self.num_patches = self.touch_model.embeddings.num_patches
        self.patch_size = config.vision_config.patch_size
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_config.hidden_size), requires_grad=False)
        self.touch_decoder_blocks = nn.ModuleList([ViTLayer(decoder_config) for _ in range(decoder_config.num_hidden_layers)])

        self.decoder_norm = nn.LayerNorm(decoder_config.hidden_size, eps=decoder_config.layer_norm_eps)
        self.decoder_pred = nn.Linear(decoder_config.hidden_size, decoder_config.patch_size**2 * decoder_config.num_channels, bias=True)

        self.decoder_pred_video = nn.Linear(decoder_config.hidden_size, decoder_config.patch_size**2 * decoder_config.num_channels * 4, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_config.hidden_size))

        if self.use_sensor_token:
            self.sensor_token = nn.Parameter(torch.zeros(10, 5, config.vision_config.hidden_size))
            self.beta = 1.0
            if self.new_decoder_sensor_token:
                self.sensor_token_proj = nn.Linear(config.vision_config.hidden_size, decoder_config.hidden_size, bias=False)

        self.mask_ratio = args.mask_ratio

        self.touch_model.forward = self.touch_forward
        self.touch_model.embeddings.forward = self.emb_forward


    def initialize_decoder(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)
        if self.use_sensor_token:
            torch.nn.init.normal_(self.sensor_token, std=.02)

        # 🔥 关键修复：初始化预测层权重，防止NaN
        torch.nn.init.xavier_uniform_(self.decoder_pred.weight)
        if self.decoder_pred.bias is not None:
            torch.nn.init.constant_(self.decoder_pred.bias, 0)

        torch.nn.init.xavier_uniform_(self.decoder_pred_video.weight)
        if self.decoder_pred_video.bias is not None:
            torch.nn.init.constant_(self.decoder_pred_video.bias, 0)

    def random_masking(self, sequence, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def touch_forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sensor_type = None,
        data_type = None,
        use_mask = True
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        # a = self.sensor_token[sensor_type]
        # print(a.shape)
        output_attentions = output_attentions if output_attentions is not None else self.touch_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.touch_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.touch_model.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states, mask, ids_restore  = self.touch_model.embeddings(pixel_values, sensor_type = sensor_type, data_type = data_type, use_mask = use_mask)
        hidden_states = self.touch_model.pre_layrnorm(hidden_states)

        encoder_outputs = self.touch_model.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.touch_model.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        ), mask, ids_restore

    def emb_forward(self, pixel_values: torch.FloatTensor, noise=None, sensor_type=None, data_type = None, use_mask = True) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.touch_model.embeddings.patch_embedding.weight.dtype
        if data_type == 0 and (not self.use_same_patchemb):
            patch_embeds = self.touch_model.embeddings.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        else:
            patch_embeds = self.video_patch_embedding(pixel_values.to(dtype=target_dtype))

        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        pos_emb = self.touch_model.embeddings.position_embedding(self.touch_model.embeddings.position_ids)

        embeddings = patch_embeds + pos_emb[:, 1:, :]
        if use_mask:
            embeddings, mask, ids_restore = self.random_masking(embeddings, noise)
        else:
            mask = torch.ones(1)
            ids_restore = torch.ones(1)

        class_embeds = self.touch_model.embeddings.class_embedding + pos_emb[:, 0, :]
        class_embeds = class_embeds.expand(batch_size, 1, -1)

        if self.use_sensor_token:
            sensor_emb = self.sensor_token[sensor_type]
            embeddings = torch.cat([class_embeds, sensor_emb, embeddings], dim=1)
        else:
            embeddings = torch.cat([class_embeds, embeddings], dim=1)
        #embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings, mask, ids_restore

    def forward(self, x, sensor_type=None, data_type=None, target_sensor_type = None):
        # 检查是否是联合训练模式
        if data_type == 'joint':
            # 联合训练模式：使用动态模式的 encoder/decoder
            latent, mask, ids_restore = self.forward_encoder(x[:, :3], sensor_type=sensor_type, data_type=1)
            pred = self.forward_decoder(latent, ids_restore, data_type=1, sensor_type=target_sensor_type)
            # 计算损失（返回4个值：static, dynamic, recon, pred）
            loss_static, loss_dynamic, loss_recon, loss_pred = self.forward_loss_joint(x, pred, mask)
            return loss_static, loss_dynamic, loss_recon, loss_pred, pred, mask
        else:
            # 原有模式
            if data_type == 0:
                latent, mask, ids_restore = self.forward_encoder(x, sensor_type=sensor_type, data_type=data_type)
            else:
                latent, mask, ids_restore = self.forward_encoder(x[:, :3], sensor_type=sensor_type, data_type=data_type)
            pred = self.forward_decoder(latent, ids_restore, data_type=data_type, sensor_type=target_sensor_type)

            loss = self.forward_loss(x, pred, mask, data_type=data_type)
            
            return loss, pred, mask

    def forward_encoder(self, x, sensor_type=None, data_type = None, use_mask = True):
        if data_type == 0 and self.use_same_patchemb:
            x = x.unsqueeze(1).repeat(1, 3, 1, 1, 1)
            # print(x.shape)
        x, mask, ids_restore = self.touch_model(x, sensor_type=sensor_type, data_type=data_type, use_mask = use_mask)
        if use_mask:
            out = self.touch_projection(x.last_hidden_state)
        else:
            out = self.touch_projection(x.pooler_output)

        return out, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore, sensor_type=None, data_type = None):

        x = self.decoder_embed(x)

        if self.use_sensor_token:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 6 - x.shape[1], 1)

            x_ = torch.cat([x[:, 6:, :], mask_tokens], dim=1)  # no cls token and sensor token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

            if self.new_decoder_sensor_token:
                decoder_sensor = self.sensor_token_proj(self.sensor_token[sensor_type])
                x = torch.cat([x[:, :1, :], decoder_sensor, x_], dim=1)  # append cls token and sensor token
            else:
                x = torch.cat([x[:, :6, :], x_], dim=1)  # append cls token and sensor token
            #x = torch.cat([x[:, :6, :], x_], dim=1)  # append cls token and sensor token

            x[:,0,:] += self.decoder_pos_embed[:,0,:]
            x[:,6:,:] += self.decoder_pos_embed[:,1:,:]
        else:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

            x = x + self.decoder_pos_embed

        for blk in self.touch_decoder_blocks:
            layer_outputs = blk(x)
        
            x = layer_outputs[0]

        x = self.decoder_norm(x)

        if data_type == 0:
            x = self.decoder_pred(x)
        else:
            x = self.decoder_pred_video(x)

        if self.use_sensor_token:
            x = x[:, 6:, :]
        else:
            x = x[:, 1:, :]

        if data_type == 1:
            x = x.view(x.shape[0], x.shape[1], 4, -1)
        return x
    
    def forward_loss(self, x, pred, mask, data_type = None):
        target = self.patchify(x, data_type = data_type)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            # 🔥 增加epsilon提高数值稳定性
            target = (target - mean) / (var + 1.e-4)**.5
        
        if data_type == 0:
            # 静态模式
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

            # 🔥 防止除零错误
            mask_sum = mask.sum()
            if mask_sum == 0:
                print(f"⚠️ Warning: mask.sum() is 0 in static mode! Returning zero loss.")
                return torch.tensor(0.0, device=x.device, requires_grad=True)
            loss = (loss * mask).sum() / mask_sum.clamp(min=1.0)  # mean loss on removed patches
        else:
            # 动态模式
            loss = (pred - target) ** 2
            loss_pred = loss[:, :, 3, :].mean(dim=-1)
            loss_recon = loss[:, :, :3, :].mean(dim=-2).mean(dim=-1)
            
            # 🔥 防止除零错误
            mask_sum = mask.sum()
            L = loss.shape[1] * loss.shape[0]
            if mask_sum == 0:
                print(f"⚠️ Warning: mask.sum() is 0 in dynamic mode! Returning prediction loss only.")
                # 至少保留预测损失
                loss = loss_pred.sum() / max(L, 1)
            else:
                loss = (loss_recon * mask).sum() / mask_sum.clamp(min=1.0) + loss_pred.sum() / max(L, 1)
        
        return loss
    
    def forward_loss_joint(self, x, pred, mask):
        """
        联合计算静态和动态 loss
        
        Args:
            x: [B, 4, 3, H, W] 输入的 4 帧图像
            pred: [B, L, 4, patch_dim] 预测的 patches
            mask: [B, L] 掩码 (1=被掩码, 0=可见)
        
        Returns:
            loss_static: 第 1 帧的重建 loss (标量)
            loss_dynamic: 前 3 帧重建 + 第 4 帧预测 loss (标量)
            loss_recon: 前 3 帧重建 loss (标量) - 用于监控
            loss_pred: 第 4 帧预测 loss (标量) - 用于监控
        """
        # Patchify: [B, 4, 3, H, W] -> [B, L, 4, patch_dim]
        target = self.patchify(x, data_type=1)  # data_type=1 表示视频模式
        
        # 归一化 (可选)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            # 🔥 增加epsilon提高数值稳定性，防止除零
            target = (target - mean) / (var + 1.e-4)**.5
        
        # 计算 MSE: [B, L, 4, patch_dim]
        loss = (pred - target) ** 2
        
        # 🔥 关键修复：检查mask.sum()是否为0，防止除零错误
        mask_sum = mask.sum()
        if mask_sum == 0:
            # 如果没有被掩码的patches，返回零损失（理论上不应该发生）
            print(f"⚠️ Warning: mask.sum() is 0! This should not happen. Returning zero losses.")
            device = x.device
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return zero_loss, zero_loss, zero_loss, zero_loss
        
        # ========== 静态 loss：只计算第 1 帧的掩码重建 ==========
        # 提取第 1 帧: [B, L, patch_dim]
        loss_static = loss[:, :, 0, :].mean(dim=-1)  # [B, L]
        # 只计算被掩码的 patches
        # 🔥 使用clamp确保mask_sum至少为一个很小的正数
        loss_static = (loss_static * mask).sum() / mask_sum.clamp(min=1.0)  # scalar
        
        # ========== 动态 loss：前 3 帧重建 + 第 4 帧预测 ==========
        # 前 3 帧重建 (掩码区域): [B, L, 3, patch_dim] -> [B, L]
        loss_recon = loss[:, :, :3, :].mean(dim=-1).mean(dim=-1)  # [B, L]
        # 🔥 使用clamp确保mask_sum至少为1
        loss_recon = (loss_recon * mask).sum() / mask_sum.clamp(min=1.0)  # scalar
        
        # 第 4 帧预测 (全部区域): [B, L, patch_dim] -> scalar
        loss_pred = loss[:, :, 3, :].mean(dim=-1)  # [B, L]
        L = loss_pred.shape[0] * loss_pred.shape[1]  # B * L
        # 🔥 确保L至少为1
        L = max(L, 1)
        loss_pred = loss_pred.sum() / L  # scalar
        
        # 组合动态 loss
        loss_dynamic = loss_recon + loss_pred
        
        return loss_static, loss_dynamic, loss_recon, loss_pred

    def patchify(self, imgs, data_type = None):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        if data_type == 0:
            # image
            p = self.patch_size
            assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

            h = w = imgs.shape[2] // p
            x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        else:
            # video
            p = self.patch_size
            assert imgs.shape[3] == imgs.shape[4] and imgs.shape[3] % p == 0

            h = w = imgs.shape[3] // p
            x = imgs.reshape(shape=(imgs.shape[0], 4, 3, h, p, w, p))
            x = torch.einsum('ntchpwq->nhwtpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, 4, p**2 * 3))
        return x

    def unpatchify(self, x, data_type = None):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        if data_type == 0:
            # image
            p = self.patch_size
            h = w = int(x.shape[1]**.5)
            assert h * w == x.shape[1]
            
            x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
            x = torch.einsum('nhwpqc->nchpwq', x)
            imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))

        else:
            p = self.patch_size
            h = w = int(x.shape[1]**.5)
            assert h * w == x.shape[1]

            x = x.reshape(shape=(x.shape[0], h, w, 4, p, p, 3))
            x = torch.einsum('nhwtpqc->ntchpwq', x)
            imgs = x.reshape(shape=(x.shape[0], 4, 3, h * p, h * p))
        return imgs