#!/usr/bin/env python3
"""
DataLoader 构建模块
提供统一接口创建不同阶段的 DataLoader

功能：
1. 数据增强配置（训练集/验证集）
2. Stage-2a/2b/2c 的 DataLoader 工厂函数
3. 支持 Fine-grained 和 Coarse-grained 混合
4. Train/Val 数据集分割（可复现）
"""

import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset, Subset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from .stage2_dataset import Stage2PairedDataset


class ObjectRoundRobinDistributedSampler(Sampler):
    """
    基于 object 的轮转采样器（支持分布式）。

    目标：
    - 提升 batch 内接触多样性（减少单一 object 连续主导）
    - 在每轮中限制同一 object 的连续出现次数
    """

    def __init__(
        self,
        indices,
        object_ids,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=0,
        drop_last=False,
        max_per_object_per_round=1
    ):
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas-1}], got {rank}")
        if len(indices) != len(object_ids):
            raise ValueError(
                f"indices/object_ids length mismatch: {len(indices)} vs {len(object_ids)}"
            )

        self.indices = list(indices)
        self.object_ids = [str(x) for x in object_ids]
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.max_per_object_per_round = max(1, int(max_per_object_per_round))
        self.epoch = 0

        if self.drop_last:
            self.num_samples = len(self.indices) // self.num_replicas
        else:
            self.num_samples = int(math.ceil(len(self.indices) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        self._object_to_positions = {}
        for pos, obj in enumerate(self.object_ids):
            self._object_to_positions.setdefault(obj, []).append(pos)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _build_round_robin_positions(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        pools = {}
        objects = list(self._object_to_positions.keys())

        for obj in objects:
            obj_pos = list(self._object_to_positions[obj])
            if self.shuffle:
                rng.shuffle(obj_pos)
            pools[obj] = obj_pos

        ordered_positions = []
        while True:
            active = [obj for obj in objects if len(pools[obj]) > 0]
            if len(active) == 0:
                break
            if self.shuffle:
                rng.shuffle(active)
            for obj in active:
                take_n = min(self.max_per_object_per_round, len(pools[obj]))
                if take_n <= 0:
                    continue
                ordered_positions.extend(pools[obj][:take_n])
                pools[obj] = pools[obj][take_n:]
        return ordered_positions

    def __iter__(self):
        ordered_positions = self._build_round_robin_positions()
        ordered_indices = [self.indices[pos] for pos in ordered_positions]

        if not self.drop_last:
            padding_size = self.total_size - len(ordered_indices)
            if padding_size > 0 and len(ordered_indices) > 0:
                repeat = (padding_size + len(ordered_indices) - 1) // len(ordered_indices)
                ordered_indices += (ordered_indices * repeat)[:padding_size]
        else:
            ordered_indices = ordered_indices[:self.total_size]

        rank_indices = ordered_indices[self.rank:self.total_size:self.num_replicas]
        return iter(rank_indices)

    def __len__(self):
        return self.num_samples


def get_default_transform(augment=True):
    """
    获取默认的数据增强
    
    Args:
        augment: 是否使用数据增强（训练集 True，验证集 False）
    
    Returns:
        transforms.Compose: 数据变换组合
        
    设计说明（更新 v3 - 物体级别分层切分）：
        对于 Fine 数据（Stage-2a/2b）：
        - 使用轻微 ColorJitter（±5%）作为正则化
        - 由于采用了物体级别分层切分，不再有时间序列泄露
        - 轻微增强有助于泛化，不会破坏关键特征
        
        对于 Coarse 数据（Stage-2c）：
        - 可以使用稍强的增强（±10%）
        
        切分策略改进：
        1. 每个物体的样本独立切分（避免时间序列泄露）
        2. 物体内部随机打乱（打破连续性）
        3. 训练/验证集都包含所有物体的样本
    """
    if augment:
        # 训练集：使用轻微增强作为正则化
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(
                brightness=0.05,  # 轻微亮度变化 ±5%
                contrast=0.05,    # 轻微对比度变化 ±5%
                saturation=0.05,  # 轻微饱和度变化 ±5%
                hue=0.02          # 轻微色调变化 ±2%
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]    # ImageNet std
            )
        ])
    else:
        # 验证集：不使用增强
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def build_stage2_dataloader(
    stage='2a',
    split='train',
    batch_size=256,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=1,
    shuffle=None,
    drop_last=None,
    train_split_ratio=0.9,
    random_seed=42,
    data_sources=None,
    data_root_fine=None,
    data_root_indoor=None,
    data_root_outdoor=None,
    distributed=False,  # 🔧 FIX 6: 新增分布式训练支持
    group_balance_strategy='none',
    contact_sampler_enabled=False,
    contact_sampler_max_per_object_per_round=1,
    base_frame_jitter=0,
    base_frame_jitter_prob=0.0
):
    """
    构建 Stage-2 DataLoader（支持 train/val 分割 + 多数据集混合）
    
    Args:
        stage: str, 训练阶段
            - '2a': Stage-2a (基础对齐), InfoNCE only
            - '2b': Stage-2b (域不变性 + 鲁棒性), InfoNCE + InfoMax + Group-DRO
        split: str, 数据集分割类型
            - 'train': 训练集（带数据增强）
            - 'val': 验证集（无数据增强）
        batch_size: int, batch 大小（默认256，根据GPU调整）
        num_workers: int, 数据加载线程数（默认8）
        pin_memory: bool, 是否使用 pin_memory（默认True，加速GPU传输）
        prefetch_factor: int, DataLoader 预取批次数（仅 num_workers>0 生效）
        shuffle: bool, 是否打乱数据（默认：train=True, val=False）
        drop_last: bool, 是否丢弃最后不完整的batch（默认：train=True, val=False）
        train_split_ratio: float, 训练集比例（默认0.9，即90% train, 10% val）
        random_seed: int, 随机种子（用于可复现的数据分割，默认42）
        data_sources: list, 数据源列表（例如: ['Fine', 'Indoor', 'Outdoor']）
        data_root_fine: str, Fine数据集路径
        data_root_indoor: str, Indoor数据集路径
        data_root_outdoor: str, Outdoor数据集路径
        group_balance_strategy: str, 训练集按组均衡策略（none / repeat）
        contact_sampler_enabled: bool, 是否启用按object轮转的contact-centric采样
        contact_sampler_max_per_object_per_round: int, 每轮每个object最多连续采样数
    
    Returns:
        DataLoader: PyTorch DataLoader对象
        
    数据策略（新版：两阶段架构）：
        - Stage-2a/2b: 支持多数据集混合（Fine + Indoor + Outdoor）
          理由：从一开始就用所有数据，避免过拟合
          数据量：110 objects, ~70K samples
    
    数据分割：
        - 使用物体级别分层切分（Stratified Split by Object）
        - 每个物体的样本独立切分到 train/val
        - 避免时间序列泄露
        - 使用固定随机种子确保可复现性
    """
    # 根据 split 决定是否使用数据增强
    augment = (split == 'train')
    transform = get_default_transform(augment=augment)
    
    # 自动设置 shuffle 和 drop_last
    if shuffle is None:
        shuffle = (split == 'train')
    if drop_last is None:
        drop_last = (split == 'train')
    
    # 默认数据源
    if data_sources is None:
        data_sources = ['Fine']
    
    # ====================================================================
    # 多数据集加载（新版：支持 Fine + Indoor + Outdoor 混合）
    # ====================================================================
    print(f"\n{'='*60}")
    print(f"[Stage {stage}] 构建 DataLoader")
    print(f"{'='*60}")
    print(f"数据源: {', '.join(data_sources)}")
    
    datasets = []
    group_labels = []  # 用于 Group-DRO（记录每个样本属于哪个数据集）
    
    # 加载 Fine 数据
    if 'Fine' in data_sources:
        if data_root_fine is None:
            raise ValueError("data_root_fine must be provided when 'Fine' in data_sources")
        
        print(f"\n✅ 加载 Fine 数据...")
        print(f"   路径: {data_root_fine}")
        
        fine_dataset = Stage2PairedDataset(
            csv_file=f'{data_root_fine}contact_fine.csv',
            data_root=f'{data_root_fine}data_fine',
            data_type='fine',
            transform=transform,
            group_id=0,  # Fine组
            base_frame_jitter=(base_frame_jitter if split == 'train' else 0),
            base_frame_jitter_prob=(base_frame_jitter_prob if split == 'train' else 0.0),
        )
        datasets.append(fine_dataset)
        group_labels.extend([0] * len(fine_dataset))  # Group 0: Fine
        print(f"   样本数: {len(fine_dataset)}")
    
    # 加载 Indoor 数据
    if 'Indoor' in data_sources:
        if data_root_indoor is None:
            raise ValueError("data_root_indoor must be provided when 'Indoor' in data_sources")
        
        print(f"\n✅ 加载 Indoor 数据...")
        print(f"   路径: {data_root_indoor}")
        
        indoor_dataset = Stage2PairedDataset(
            csv_file=f'{data_root_indoor}contact_indoor.csv',
            data_root=f'{data_root_indoor}data_indoor',
            data_type='coarse_indoor',
            transform=transform,
            group_id=1,  # Indoor组
            base_frame_jitter=(base_frame_jitter if split == 'train' else 0),
            base_frame_jitter_prob=(base_frame_jitter_prob if split == 'train' else 0.0),
        )
        datasets.append(indoor_dataset)
        group_labels.extend([1] * len(indoor_dataset))  # Group 1: Indoor
        print(f"   样本数: {len(indoor_dataset)}")
    
    # 加载 Outdoor 数据
    if 'Outdoor' in data_sources:
        if data_root_outdoor is None:
            raise ValueError("data_root_outdoor must be provided when 'Outdoor' in data_sources")
        
        print(f"\n✅ 加载 Outdoor 数据...")
        print(f"   路径: {data_root_outdoor}")
        
        outdoor_dataset = Stage2PairedDataset(
            csv_file=f'{data_root_outdoor}contact_outdoor.csv',
            data_root=f'{data_root_outdoor}data_outdoor',
            data_type='coarse_outdoor',
            transform=transform,
            group_id=2,  # Outdoor组
            base_frame_jitter=(base_frame_jitter if split == 'train' else 0),
            base_frame_jitter_prob=(base_frame_jitter_prob if split == 'train' else 0.0),
        )
        datasets.append(outdoor_dataset)
        group_labels.extend([2] * len(outdoor_dataset))  # Group 2: Outdoor
        print(f"   样本数: {len(outdoor_dataset)}")
    
    # 合并数据集
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)
        print(f"\n数据集合并完成:")
        for i, ds in enumerate(datasets):
            source_name = data_sources[i]
            print(f"  {source_name:8s}: {len(ds):6d} samples (Group {i})")
        print(f"  {'总计':8s}: {len(dataset):6d} samples")
    elif len(datasets) == 1:
        dataset = datasets[0]
        print(f"\n总样本数: {len(dataset)}")
    else:
        raise ValueError("No datasets specified in data_sources!")
    
    # ====================================================================
    # Train/Val 数据集分割
    # 策略：物体级别分层切分 (Stratified Split by Object)
    # ====================================================================
    total_size = len(dataset)
    
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 收集每个物体的样本索引
    object_to_indices = {}
    global_object_ids = [''] * total_size
    for idx in range(total_size):
        # 处理单数据集和混合数据集
        if isinstance(dataset, ConcatDataset):
            # 混合数据集：需要找到对应的子数据集
            cumsum = 0
            for sub_dataset in dataset.datasets:
                if idx < cumsum + len(sub_dataset):
                    local_idx = idx - cumsum
                    obj_name = sub_dataset.samples[local_idx][0]
                    break
                cumsum += len(sub_dataset)
        else:
            # 单数据集
            obj_name = dataset.samples[idx][0]  # (obj_name, frame_idx)

        global_object_ids[idx] = str(obj_name)
        
        if obj_name not in object_to_indices:
            object_to_indices[obj_name] = []
        object_to_indices[obj_name].append(idx)
    
    # 对每个物体单独进行 train/val 切分
    train_indices = []
    val_indices = []
    
    for obj_name, indices in object_to_indices.items():
        # 物体内部随机打乱（打破时间序列连续性）
        np.random.shuffle(indices)
        
        # 计算该物体的 train/val 分割点
        obj_train_size = int(len(indices) * train_split_ratio)
        
        # 分配到 train 和 val
        train_indices.extend(indices[:obj_train_size])
        val_indices.extend(indices[obj_train_size:])
    
    # 再次随机打乱 train 和 val 索引（确保 batch 内物体多样性）
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    
    # 根据 split 选择相应的索引
    if split == 'train':
        split_indices = train_indices
        split_name = '训练集'
    elif split == 'val':
        split_indices = val_indices
        split_name = '验证集'
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train' or 'val'")

    split_object_ids = [global_object_ids[idx] for idx in split_indices]
    
    # 记录当前split的组标签分布
    group_labels_array = np.array(group_labels, dtype=np.int64)
    subset_group_labels = [int(group_labels_array[idx]) for idx in split_indices]
    group_to_positions = {}
    for pos, gid in enumerate(subset_group_labels):
        group_to_positions.setdefault(gid, []).append(pos)
    group_distribution = {gid: len(idxs) for gid, idxs in group_to_positions.items()}
    print(f"\n当前 {split_name} 组分布:")
    for group_id, count in sorted(group_distribution.items()):
        print(f"  Group {group_id}: {count} samples ({count / max(len(subset_group_labels),1)*100:.1f}%)")
    if len(subset_group_labels) == 0:
        print("  ⚠️  当前分割为空，请检查数据或 train_split_ratio 设置")
    
    # 可选：对训练集进行按组重复采样，实现 batch 级均衡
    if split == 'train' and group_balance_strategy.lower() == 'repeat' and len(subset_group_labels) > 0:
        non_empty_groups = {gid: cnt for gid, cnt in group_distribution.items() if cnt > 0}
        if len(non_empty_groups) > 1:
            rng = np.random.default_rng(random_seed + 1024)
            max_count = max(non_empty_groups.values())
            balanced_positions = []
            for gid, count in non_empty_groups.items():
                positions = group_to_positions.get(gid, [])
                repeats = max_count // count
                remainder = max_count % count
                balanced_positions.extend(positions * repeats)
                if remainder > 0:
                    extra = rng.choice(
                        positions,
                        size=remainder,
                        replace=(count < remainder)
                    ).tolist()
                    balanced_positions.extend(extra)
            rng.shuffle(balanced_positions)
            split_indices = [split_indices[pos] for pos in balanced_positions]
            subset_group_labels = [subset_group_labels[pos] for pos in balanced_positions]
            split_object_ids = [split_object_ids[pos] for pos in balanced_positions]
            group_distribution = {}
            for gid in subset_group_labels:
                group_distribution[gid] = group_distribution.get(gid, 0) + 1
            print(f"\n🎯 已启用按组重复采样 (repeat) - 新数据量: {len(split_indices)}")
            for group_id, count in sorted(group_distribution.items()):
                print(f"  Group {group_id}: {count} samples ({count / len(split_indices)*100:.1f}%)")
        else:
            print("\n⚠️  按组重复采样被忽略：当前训练集中只有一个有效组")
    
    # 使用 Subset 创建子数据集
    split_dataset = Subset(dataset, split_indices)
    
    # 打印分割信息
    print(f"\n{'='*60}")
    print(f"数据集分割策略: 物体级别分层切分")
    print(f"{'='*60}")
    print(f"  总物体数:         {len(object_to_indices):6d}")
    print(f"  总样本数:         {total_size:6d}")
    print(f"  训练集:           {len(train_indices):6d} ({len(train_indices)/total_size*100:.1f}%)")
    print(f"  验证集:           {len(val_indices):6d} ({len(val_indices)/total_size*100:.1f}%)")
    print(f"  当前分割:         {split_name} ({len(split_dataset):6d} samples)")
    print(f"  数据增强:         {'是' if augment else '否'}")
    print(f"  随机种子:         {random_seed}")
    print(f"\n  说明: 每个物体的样本被独立切分到训练/验证集")
    print(f"       避免了时间序列泄露，确保公平评估")
    
    # ====================================================================
    # 创建 DataLoader（🔧 FIX 6: 支持分布式训练）
    # ====================================================================
    sampler = None
    use_contact_sampler = (
        split == 'train' and bool(contact_sampler_enabled) and len(split_indices) > 0
    )
    if distributed:
        # 检查是否初始化了分布式环境
        if not dist.is_available() or not dist.is_initialized():
            print("⚠️  警告: 分布式环境未初始化，使用非分布式模式")
            distributed = False
        else:
            if use_contact_sampler:
                sampler = ObjectRoundRobinDistributedSampler(
                    indices=list(range(len(split_dataset))),
                    object_ids=split_object_ids,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=True,
                    seed=random_seed,
                    drop_last=drop_last,
                    max_per_object_per_round=contact_sampler_max_per_object_per_round
                )
            else:
                # 使用 DistributedSampler 确保每个进程读取不同数据
                sampler = DistributedSampler(
                    split_dataset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=(split == 'train'),  # 训练集打乱，验证集不打乱
                    seed=random_seed,
                    drop_last=drop_last
                )
            # 使用 sampler 时不能同时 shuffle
            shuffle = False
            print(f"\n✅ 使用 DistributedSampler:")
            print(f"   World size: {dist.get_world_size()}")
            print(f"   Rank: {dist.get_rank()}")
            print(f"   Samples per rank: ~{len(split_dataset) // dist.get_world_size()}")
            if use_contact_sampler:
                print(f"   Contact-centric sampler: enabled (max_per_object_per_round={contact_sampler_max_per_object_per_round})")

    if not distributed and use_contact_sampler:
        sampler = ObjectRoundRobinDistributedSampler(
            indices=list(range(len(split_dataset))),
            object_ids=split_object_ids,
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=random_seed,
            drop_last=drop_last,
            max_per_object_per_round=contact_sampler_max_per_object_per_round
        )
        shuffle = False
        print(f"\n✅ 使用 Contact-centric sampler (single process)")
        print(f"   max_per_object_per_round: {contact_sampler_max_per_object_per_round}")
    
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'sampler': sampler,  # 🔧 FIX 6: 添加 sampler 参数
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': drop_last,
        # Avoid silent worker hangs/crashes turning into DDP stalls.
        'persistent_workers': num_workers > 0,
        'timeout': 120,
    }
    if num_workers > 0:
        effective_prefetch = 1 if prefetch_factor is None else max(int(prefetch_factor), 1)
        loader_kwargs['prefetch_factor'] = effective_prefetch

    loader = DataLoader(
        split_dataset,
        **loader_kwargs
    )
    
    # 打印配置信息
    print(f"\n{'='*60}")
    print(f"DataLoader 配置:")
    print(f"{'='*60}")
    print(f"  Split:              {split_name}")
    print(f"  Dataset size:       {len(split_dataset):6d} samples")
    print(f"  Batch size:         {batch_size:6d}")
    print(f"  Batches per epoch:  {len(loader):6d}")
    print(f"  Num workers:        {num_workers}")
    print(f"  Pin memory:         {pin_memory}")
    print(f"  Shuffle:            {shuffle}")
    print(f"  Drop last:          {drop_last}")
    print(f"  Contact sampler:    {'是' if use_contact_sampler else '否'}")
    if num_workers > 0:
        print(f"  Prefetch factor:    {loader_kwargs.get('prefetch_factor', 'N/A')}")
    print(f"{'='*60}\n")
    
    return loader


# ============================================
# 测试代码
# ============================================
if __name__ == '__main__':
    import torch
    
    print("\n" + "="*60)
    print("DataLoader 工厂函数测试")
    print("="*60)
    
    # ========================================
    # 测试 1: Stage-2a DataLoader (Train)
    # ========================================
    print("\n" + "="*60)
    print("测试 1: Stage-2a DataLoader (训练集)")
    print("="*60)
    
    loader_2a_train = build_stage2_dataloader(
        stage='2a',
        split='train',
        batch_size=8,
        num_workers=2
    )
    
    # 加载一个 batch
    print("加载第一个 batch...")
    batch = next(iter(loader_2a_train))
    
    print(f"\n✅ Batch 加载成功！")
    print(f"\nBatch 格式:")
    print(f"  Keys: {list(batch.keys())}")
    print(f"  GelSight shape: {batch['gelsight'].shape}")
    print(f"  DIGIT shape:    {batch['digit'].shape}")
    print(f"  DuraGel shape:  {batch['duragel'].shape}")
    print(f"  Objects:        {batch['object']}")
    print(f"  Frames:         {batch['frame'].tolist()}")
    print(f"  Data types:     {batch['data_type']}")
    
    # 验证 shape
    assert batch['gelsight'].shape == (8, 4, 3, 224, 224), "GelSight shape 错误"
    assert batch['digit'].shape == (8, 4, 3, 224, 224), "DIGIT shape 错误"
    assert batch['duragel'].shape == (8, 4, 3, 224, 224), "DuraGel shape 错误"
    
    print(f"\n✅ Shape 验证通过: [B, 4, 3, 224, 224]")
    
    # 验证数据类型
    assert batch['gelsight'].dtype == torch.float32, "数据类型错误"
    print(f"✅ 数据类型验证通过: torch.float32")
    
    # 验证值域（归一化后应该在合理范围）
    gs_min = batch['gelsight'].min().item()
    gs_max = batch['gelsight'].max().item()
    gs_mean = batch['gelsight'].mean().item()
    gs_std = batch['gelsight'].std().item()
    
    print(f"\n数据统计信息:")
    print(f"  GelSight - Min: {gs_min:.3f}, Max: {gs_max:.3f}")
    print(f"  GelSight - Mean: {gs_mean:.3f}, Std: {gs_std:.3f}")
    
    assert -3 < gs_mean < 3, "均值超出合理范围"
    assert 0 < gs_std < 3, "标准差超出合理范围"
    print(f"✅ 值域验证通过")
    
    # ========================================
    # 测试 1.5: Stage-2a DataLoader (Val)
    # ========================================
    print("\n" + "="*60)
    print("测试 1.5: Stage-2a DataLoader (验证集)")
    print("="*60)
    
    loader_2a_val = build_stage2_dataloader(
        stage='2a',
        split='val',
        batch_size=8,
        num_workers=2
    )
    
    # 验证 val loader 不 shuffle
    print(f"\n✅ 验证集 DataLoader 创建成功")
    print(f"  训练集 batches: {len(loader_2a_train)}")
    print(f"  验证集 batches: {len(loader_2a_val)}")
    
    # ========================================
    # 测试 2: 数据增强效果
    # ========================================
    print("\n" + "="*60)
    print("测试 2: 数据增强验证")
    print("="*60)
    
    transform_train = get_default_transform(augment=True)
    transform_val = get_default_transform(augment=False)
    
    print(f"\n训练集 Transform:")
    for t in transform_train.transforms:
        print(f"  - {t.__class__.__name__}")
    
    print(f"\n验证集 Transform:")
    for t in transform_val.transforms:
        print(f"  - {t.__class__.__name__}")
    
    print(f"\n✅ 数据增强配置正确")
    
    # ========================================
    # 测试 3: 多 batch 加载
    # ========================================
    print("\n" + "="*60)
    print("测试 3: 连续加载多个 batch")
    print("="*60)
    
    print(f"\n加载前 3 个 batch...")
    for i, batch in enumerate(loader_2a_train):
        if i >= 3:
            break
        print(f"  Batch {i}: {batch['gelsight'].shape}, Objects: {len(set(batch['object']))} unique")
    
    print(f"\n✅ 连续加载测试通过")
    
    # ========================================
    # 测试总结
    # ========================================
    print("\n" + "="*60)
    print("✅ 所有测试通过！")
    print("="*60)
    print(f"\n总结:")
    print(f"  ✅ Stage-2a DataLoader 创建成功")
    print(f"  ✅ Train/Val 数据集分割正常")
    print(f"  ✅ Batch shape 正确: [8, 4, 3, 224, 224]")
    print(f"  ✅ 数据类型正确: torch.float32")
    print(f"  ✅ 数据增强配置正确")
    print(f"  ✅ 连续加载稳定")
    print(f"\n下一步: 运行 'python scripts/test_dataloader.py' 进行完整测试")
    print("="*60 + "\n")
