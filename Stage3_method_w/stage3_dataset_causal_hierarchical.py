#!/usr/bin/env python3
"""
Hierarchical Stage3 dataset built on top of Stage3CausalDatasetV1.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from Stage3_method_w.stage3_dataset_causal_v1 import Stage3CausalDatasetV1


def _quantile(values: List[float], q: float) -> float:
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    q = min(max(float(q), 0.0), 1.0)
    pos = q * float(len(values) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(values[low])
    frac = pos - float(low)
    return float(values[low] * (1.0 - frac) + values[high] * frac)


class Stage3CausalHierarchicalDataset(Dataset):
    """
    Wrap Stage3CausalDatasetV1 and augment samples with a continuous force target.
    """

    def __init__(
        self,
        root_dir: str = "./Causal_data",
        index_file: Optional[str] = None,
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        random_seed: int = 42,
        clip_len: int = 4,
        frame_stride: int = 1,
        train_center_frame_step: int = 1,
        eval_center_frame_step: int = 1,
        train_sequence_clips: int = 1,
        transform=None,
        augment: bool = False,
        strict_frame_loading: bool = True,
        force_num_bins: int = 3,
        force_quantiles: Optional[Tuple[float, ...]] = None,
        force_bin_edges: Optional[Tuple[float, ...]] = None,
    ) -> None:
        super().__init__()
        self.base_dataset = Stage3CausalDatasetV1(
            root_dir=root_dir,
            index_file=index_file,
            split=split,
            split_ratios=split_ratios,
            random_seed=random_seed,
            clip_len=clip_len,
            frame_stride=frame_stride,
            train_center_frame_step=train_center_frame_step,
            eval_center_frame_step=eval_center_frame_step,
            train_sequence_clips=train_sequence_clips,
            transform=transform,
            augment=augment,
            strict_frame_loading=strict_frame_loading,
        )

        self.split = split
        self.force_num_bins = max(int(force_num_bins), 1)
        self.force_quantiles = force_quantiles
        self.sequence_group_mode = bool(getattr(self.base_dataset, "sequence_group_mode", False))
        if self.sequence_group_mode:
            raise NotImplementedError(
                "Hierarchical Stage3 does not support train_sequence_clips > 1 yet. "
                "Keep train_sequence_clips=1 for this pipeline."
            )

        self.sensor_sequences = self.base_dataset.sensor_sequences
        self.selected_sequences = self.base_dataset.selected_sequences
        self.sequence_samples = self.base_dataset.sequence_samples
        self.sequence_unit_ids = self.base_dataset.sequence_unit_ids
        self.num_clip_samples = self.base_dataset.num_clip_samples
        self.num_loader_items = self.base_dataset.num_loader_items

        self._label_maps = deepcopy(self.base_dataset.get_label_maps())
        self._label_maps["force_target"] = {
            "type": "regression",
            "source": "sqrt(force_x^2 + force_y^2 + force_z^2)",
            "unit": "raw",
        }

        self.samples = self._augment_samples_with_force(self.base_dataset.samples)
        self.sequence_samples = self._group_samples_by_sensor_sequence(self.samples)
        self.sequence_unit_ids = self.base_dataset._build_sequence_unit_ids(self.sequence_samples)

        print(
            "  hierarchical label classes:"
            f" deformation={len(self._label_maps['deformation_to_idx'])},"
            f" slippage={len(self._label_maps['slippage_to_idx'])},"
            f" rotation={len(self._label_maps['rotation_to_idx'])},"
            " force=regression"
        )

    @staticmethod
    def _force_value_from_xyz(force_x: float, force_y: float, force_z: float) -> float:
        return math.sqrt(float(force_x) ** 2 + float(force_y) ** 2 + float(force_z) ** 2)

    def _augment_samples_with_force(self, samples: List[Dict[str, object]]) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for rec in samples:
            new_rec = dict(rec)
            force_value = self._force_value_from_xyz(
                force_x=float(rec["force_x"]),
                force_y=float(rec["force_y"]),
                force_z=float(rec["force_z"]),
            )
            new_rec["force"] = float(force_value)
            new_rec["force_text"] = f"{float(force_value):.6f}"
            new_rec["force_magnitude"] = float(force_value)
            out.append(new_rec)
        return out

    @staticmethod
    def _group_samples_by_sensor_sequence(samples: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
        grouped: Dict[str, List[Dict[str, object]]] = {}
        for rec in samples:
            sensor_sequence_id = str(rec["sensor_sequence_id"])
            grouped.setdefault(sensor_sequence_id, []).append(rec)
        return grouped

    def get_label_maps(self) -> Dict[str, Dict]:
        return self._label_maps

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _lookup_augmented_sample(self, sensor_sequence_id: str, center_frame: int) -> Dict[str, object]:
        for rec in self.sequence_samples[sensor_sequence_id]:
            if int(rec["center_frame"]) == int(center_frame):
                return rec
        raise KeyError(
            f"Augmented sample not found for sensor_sequence={sensor_sequence_id}, center_frame={center_frame}"
        )

    def __getitem__(self, index: int) -> Dict[str, object]:
        item = self.base_dataset[index]

        if self.sequence_group_mode:
            sensor_sequence_id = str(item["sensor_sequence_id"])
            center_frame = int(item["center_frames"][0].item())
            rec = self._lookup_augmented_sample(sensor_sequence_id, center_frame)
            targets = dict(item["targets"])
            targets["force"] = float(rec["force"])
            target_mask = dict(item["target_mask"])
            target_mask["force"] = 1
            item = dict(item)
            item["force"] = float(rec["force"])
            item["force_text"] = str(rec["force_text"])
            item["force_magnitude"] = float(rec["force_magnitude"])
            item["targets"] = targets
            item["target_mask"] = target_mask
            return item

        item = dict(item)
        force_value = self._force_value_from_xyz(
            force_x=float(item["force_x"]),
            force_y=float(item["force_y"]),
            force_z=float(item["force_z"]),
        )
        force_text = f"{float(force_value):.6f}"
        force_magnitude = float(force_value)
        targets = dict(item["targets"])
        targets["force"] = float(force_value)
        target_mask = dict(item["target_mask"])
        target_mask["force"] = 1
        item["force"] = float(force_value)
        item["force_text"] = force_text
        item["force_magnitude"] = float(force_magnitude)
        item["targets"] = targets
        item["target_mask"] = target_mask
        return item


def build_hierarchical_causal_dataloader(
    root_dir: str = "./Causal_data",
    index_file: Optional[str] = None,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: Optional[bool] = None,
    drop_last: Optional[bool] = None,
    random_seed: int = 42,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    clip_len: int = 4,
    frame_stride: int = 1,
    train_center_frame_step: int = 1,
    eval_center_frame_step: int = 1,
    train_sequence_clips: int = 1,
    transform=None,
    augment: bool = False,
    strict_frame_loading: bool = True,
    force_num_bins: int = 3,
    force_quantiles: Optional[Tuple[float, ...]] = None,
    force_bin_edges: Optional[Tuple[float, ...]] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, Stage3CausalHierarchicalDataset]:
    dataset = Stage3CausalHierarchicalDataset(
        root_dir=root_dir,
        index_file=index_file,
        split=split,
        split_ratios=split_ratios,
        random_seed=random_seed,
        clip_len=clip_len,
        frame_stride=frame_stride,
        train_center_frame_step=train_center_frame_step,
        eval_center_frame_step=eval_center_frame_step,
        train_sequence_clips=train_sequence_clips,
        transform=transform,
        augment=augment,
        strict_frame_loading=strict_frame_loading,
        force_num_bins=force_num_bins,
        force_quantiles=force_quantiles,
        force_bin_edges=force_bin_edges,
    )

    if shuffle is None:
        shuffle = split == "train"
    if drop_last is None:
        drop_last = split == "train"

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=bool(shuffle),
            drop_last=bool(drop_last),
        )
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, dataset


__all__ = [
    "Stage3CausalHierarchicalDataset",
    "build_hierarchical_causal_dataloader",
]
