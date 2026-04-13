#!/usr/bin/env python3
"""
Stage-3 V1 dataset for octopi_new.

Expected folder layout:
    octopi_new/
      index.csv
      0/
        0000000012.jpg
        ...
      1/
        ...

index.csv columns (no header required):
    0: sub-folder name
    1: start frame id (e.g. 12)
    2: end frame id (e.g. 46)
    3: hardness label (string)
    4: roughness label (string)
    5: material label (string)
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


@dataclass(frozen=True)
class SequenceRecord:
    sequence_id: str
    start_frame: int
    end_frame: int
    hardness: str
    roughness: str
    material: str


def _default_transform(augment: bool = False) -> transforms.Compose:
    if augment:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ColorJitter(
                    brightness=0.05,
                    contrast=0.05,
                    saturation=0.05,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class Stage3DatasetV1(Dataset):
    """
    Build 4-frame clips for Stage2 encoder + Stage3 heads.

    One sample corresponds to a center frame t and uses clip:
        [t-3*stride, t-2*stride, t-1*stride, t]
    """

    def __init__(
        self,
        root_dir: str = "./octopi_new",
        index_file: Optional[str] = None,
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        random_seed: int = 42,
        clip_len: int = 4,
        frame_stride: int = 1,
        train_center_frame_step: int = 1,
        eval_center_frame_step: int = 1,
        train_sequence_clips: int = 1,
        transform: Optional[transforms.Compose] = None,
        augment: bool = False,
        strict_frame_loading: bool = True,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"Unknown split={split}, expected train/val/test/all")
        if clip_len < 1:
            raise ValueError(f"clip_len must be >= 1, got {clip_len}")
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
        if train_center_frame_step < 1:
            raise ValueError(
                f"train_center_frame_step must be >= 1, got {train_center_frame_step}"
            )
        if eval_center_frame_step < 1:
            raise ValueError(
                f"eval_center_frame_step must be >= 1, got {eval_center_frame_step}"
            )
        if train_sequence_clips < 1:
            raise ValueError(
                f"train_sequence_clips must be >= 1, got {train_sequence_clips}"
            )
        if len(split_ratios) != 3:
            raise ValueError("split_ratios must be a tuple of (train, val, test)")
        if abs(sum(split_ratios) - 1.0) > 1e-6:
            raise ValueError(f"split_ratios must sum to 1.0, got {split_ratios}")

        self.root_dir = Path(root_dir)
        self.index_file = Path(index_file) if index_file is not None else self.root_dir / "index.csv"
        self.split = split
        self.split_ratios = split_ratios
        self.random_seed = int(random_seed)
        self.clip_len = int(clip_len)
        self.frame_stride = int(frame_stride)
        self.train_center_frame_step = int(train_center_frame_step)
        self.eval_center_frame_step = int(eval_center_frame_step)
        self.train_sequence_clips = int(train_sequence_clips)
        self.strict_frame_loading = bool(strict_frame_loading)
        self.sequence_group_mode = bool(split == "train" and self.train_sequence_clips > 1)

        if not self.root_dir.exists():
            raise FileNotFoundError(f"root_dir not found: {self.root_dir}")
        if not self.index_file.exists():
            raise FileNotFoundError(f"index_file not found: {self.index_file}")

        train_augment = bool(augment and split == "train")
        self.transform = transform if transform is not None else _default_transform(augment=train_augment)

        self.sequences = self._load_sequences()
        if len(self.sequences) == 0:
            raise RuntimeError(f"No valid sequence records found in {self.index_file}")

        self._label_maps = self._build_label_maps(self.sequences)
        split_ids = self._build_split_sequence_ids(self.sequences)
        selected_ids = split_ids[split]

        self.selected_sequences = [seq for seq in self.sequences if seq.sequence_id in selected_ids]
        self.samples = self._build_samples(self.selected_sequences)
        self.sequence_samples = self._group_samples_by_sequence(self.samples)
        self.sequence_unit_ids = self._build_sequence_unit_ids(self.sequence_samples)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples after building split={split}. "
                f"Try split='all' or verify index ranges and clip_len/frame_stride."
            )
        if self.sequence_group_mode and len(self.sequence_unit_ids) == 0:
            raise RuntimeError(f"No sequence groups available for split={split}")

        self.num_clip_samples = len(self.samples)
        self.num_loader_items = len(self.sequence_unit_ids) if self.sequence_group_mode else len(self.samples)

        print(f"\n[Stage3DatasetV1] root={self.root_dir}")
        print(f"  split: {split}")
        print(f"  sequences: {len(self.selected_sequences)}")
        print(f"  samples: {len(self.samples)}")
        if self.sequence_group_mode:
            print(
                f"  loader items: {self.num_loader_items}"
                f" (sequence groups, {self.train_sequence_clips} clips/item)"
            )
        print(
            "  center-frame step:"
            f" {self.train_center_frame_step if split == 'train' else self.eval_center_frame_step}"
        )
        print("  split strategy: sequence-level stratified by (material, hardness, roughness)")
        print(
            "  label classes:"
            f" material={len(self._label_maps['material_to_idx'])},"
            f" hardness={len(self._label_maps['hardness_to_idx'])},"
            f" roughness={len(self._label_maps['roughness_to_idx'])}"
        )

    @staticmethod
    def _safe_int(value: str) -> int:
        return int(str(value).strip())

    def _load_sequences(self) -> List[SequenceRecord]:
        records: List[SequenceRecord] = []
        with open(self.index_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader, start=1):
                if len(row) < 6:
                    continue

                sequence_id = str(row[0]).strip()
                try:
                    start_frame = self._safe_int(row[1])
                    end_frame = self._safe_int(row[2])
                except ValueError:
                    # likely header row, skip
                    continue

                if start_frame > end_frame:
                    continue

                hardness = str(row[3]).strip()
                roughness = str(row[4]).strip()
                material = str(row[5]).strip()

                sequence_dir = self.root_dir / sequence_id
                if not sequence_dir.exists():
                    continue

                records.append(
                    SequenceRecord(
                        sequence_id=sequence_id,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        hardness=hardness,
                        roughness=roughness,
                        material=material,
                    )
                )
        return records

    @staticmethod
    def _build_label_maps(records: List[SequenceRecord]) -> Dict[str, Dict[str, int]]:
        materials = sorted({r.material for r in records})
        hardnesses = sorted({r.hardness for r in records})
        roughnesses = sorted({r.roughness for r in records})

        material_to_idx = {name: idx for idx, name in enumerate(materials)}
        hardness_to_idx = {name: idx for idx, name in enumerate(hardnesses)}
        roughness_to_idx = {name: idx for idx, name in enumerate(roughnesses)}

        return {
            "material_to_idx": material_to_idx,
            "hardness_to_idx": hardness_to_idx,
            "roughness_to_idx": roughness_to_idx,
            "idx_to_material": {idx: name for name, idx in material_to_idx.items()},
            "idx_to_hardness": {idx: name for name, idx in hardness_to_idx.items()},
            "idx_to_roughness": {idx: name for name, idx in roughness_to_idx.items()},
        }

    @staticmethod
    def _sequence_sort_key(sequence_id: str) -> Tuple[int, str]:
        return (0, f"{int(sequence_id):012d}") if sequence_id.isdigit() else (1, sequence_id)

    @classmethod
    def _sort_sequence_ids(cls, sequence_ids: List[str]) -> List[str]:
        return sorted(sequence_ids, key=cls._sequence_sort_key)

    @staticmethod
    def _sequence_label_signature(record: SequenceRecord) -> Tuple[str, str, str]:
        return (record.material, record.hardness, record.roughness)

    def _build_sequence_label_map(self, records: List[SequenceRecord]) -> Dict[str, Tuple[str, str, str]]:
        sequence_labels: Dict[str, Tuple[str, str, str]] = {}
        for rec in records:
            signature = self._sequence_label_signature(rec)
            prev = sequence_labels.get(rec.sequence_id)
            if prev is None:
                sequence_labels[rec.sequence_id] = signature
                continue
            if prev != signature:
                raise ValueError(
                    "Inconsistent labels found for the same sequence_id. "
                    f"sequence_id={rec.sequence_id}, first={prev}, current={signature}"
                )
        return sequence_labels

    def _allocate_group_counts(
        self,
        group_size: int,
        remaining_targets: List[int],
    ) -> List[int]:
        ideal = [float(group_size) * float(ratio) for ratio in self.split_ratios]
        counts = [min(int(x), remaining_targets[idx]) for idx, x in enumerate(ideal)]
        assigned = sum(counts)

        while assigned < group_size:
            candidates = [
                idx for idx, remaining in enumerate(remaining_targets) if counts[idx] < remaining
            ]
            if len(candidates) == 0:
                raise RuntimeError(
                    "Unable to allocate stratified split counts. "
                    f"group_size={group_size}, remaining_targets={remaining_targets}, counts={counts}"
                )

            best_idx = max(
                candidates,
                key=lambda idx: (
                    ideal[idx] - counts[idx],
                    remaining_targets[idx] - counts[idx],
                    -idx,
                ),
            )
            counts[best_idx] += 1
            assigned += 1

        return counts

    def _build_split_sequence_ids(self, records: List[SequenceRecord]) -> Dict[str, set]:
        sequence_labels = self._build_sequence_label_map(records)
        unique_ids = self._sort_sequence_ids(list(sequence_labels.keys()))
        n_total = len(unique_ids)
        n_train = int(n_total * self.split_ratios[0])
        n_val = int(n_total * self.split_ratios[1])
        n_test = n_total - n_train - n_val

        split_names = ("train", "val", "test")
        split_targets = [n_train, n_val, n_test]
        split_ids = {name: [] for name in split_names}
        remaining_targets = split_targets[:]

        strata: Dict[Tuple[str, str, str], List[str]] = {}
        for sequence_id, signature in sequence_labels.items():
            strata.setdefault(signature, []).append(sequence_id)

        rng = random.Random(self.random_seed)
        ordered_strata = sorted(
            strata.items(),
            key=lambda item: (
                len(item[1]),
                item[0][0],
                item[0][1],
                item[0][2],
            ),
        )

        for _, group_ids in ordered_strata:
            shuffled_ids = self._sort_sequence_ids(list(group_ids))
            rng.shuffle(shuffled_ids)
            group_counts = self._allocate_group_counts(
                group_size=len(shuffled_ids),
                remaining_targets=remaining_targets,
            )

            offset = 0
            for split_idx, split_name in enumerate(split_names):
                count = group_counts[split_idx]
                if count <= 0:
                    continue
                split_ids[split_name].extend(shuffled_ids[offset : offset + count])
                remaining_targets[split_idx] -= count
                offset += count

        if any(remaining_targets):
            raise RuntimeError(
                "Stratified split assignment did not exhaust split targets. "
                f"remaining_targets={remaining_targets}"
            )

        train_ids = set(split_ids["train"])
        val_ids = set(split_ids["val"])
        test_ids = set(split_ids["test"])
        all_ids = set(unique_ids)

        return {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
            "all": all_ids,
        }

    def _build_samples(self, records: List[SequenceRecord]) -> List[Dict[str, object]]:
        samples: List[Dict[str, object]] = []
        min_center_offset = (self.clip_len - 1) * self.frame_stride
        center_frame_step = self.train_center_frame_step if self.split == "train" else self.eval_center_frame_step

        for rec in records:
            min_center = rec.start_frame + min_center_offset
            if min_center > rec.end_frame:
                continue

            material_idx = self._label_maps["material_to_idx"][rec.material]
            hardness_idx = self._label_maps["hardness_to_idx"][rec.hardness]
            roughness_idx = self._label_maps["roughness_to_idx"][rec.roughness]

            for center_frame in range(min_center, rec.end_frame + 1, center_frame_step):
                samples.append(
                    {
                        "sequence_id": rec.sequence_id,
                        "center_frame": int(center_frame),
                        "start_frame": int(rec.start_frame),
                        "end_frame": int(rec.end_frame),
                        "material": material_idx,
                        "hardness": hardness_idx,
                        "roughness": roughness_idx,
                        "material_text": rec.material,
                        "hardness_text": rec.hardness,
                        "roughness_text": rec.roughness,
                    }
                )
        return samples

    @staticmethod
    def _group_samples_by_sequence(samples: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
        grouped: Dict[str, List[Dict[str, object]]] = {}
        for rec in samples:
            sequence_id = str(rec["sequence_id"])
            grouped.setdefault(sequence_id, []).append(rec)
        return grouped

    def _build_sequence_unit_ids(self, grouped: Dict[str, List[Dict[str, object]]]) -> List[str]:
        sequence_ids = self._sort_sequence_ids(list(grouped.keys()))
        if not self.sequence_group_mode:
            return sequence_ids

        unit_ids: List[str] = []
        for sequence_id in sequence_ids:
            num_clips = max(len(grouped[sequence_id]), 1)
            repeats = max(1, int(math.ceil(float(num_clips) / float(self.train_sequence_clips))))
            unit_ids.extend([sequence_id] * repeats)
        return unit_ids

    def get_label_maps(self) -> Dict[str, Dict]:
        return self._label_maps

    def __len__(self) -> int:
        return self.num_loader_items

    def _resolve_frame_path(self, sequence_id: str, frame_idx: int) -> Path:
        seq_dir = self.root_dir / sequence_id
        candidates = [
            seq_dir / f"{frame_idx:010d}.jpg",
            seq_dir / f"{frame_idx}.jpg",
            seq_dir / f"{frame_idx:010d}.png",
            seq_dir / f"{frame_idx}.png",
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Frame file not found for sequence={sequence_id}, frame={frame_idx}, "
            f"checked {len(candidates)} candidates."
        )

    def _load_one_frame(self, sequence_id: str, frame_idx: int) -> torch.Tensor:
        frame_path = self._resolve_frame_path(sequence_id, frame_idx)
        with Image.open(frame_path) as img:
            img = img.convert("RGB")
        return self.transform(img)

    def _build_clip(self, sequence_id: str, center_frame: int) -> torch.Tensor:
        frames: List[torch.Tensor] = []
        for idx in range(self.clip_len):
            frame_idx = center_frame - (self.clip_len - 1 - idx) * self.frame_stride
            try:
                frame = self._load_one_frame(sequence_id, frame_idx)
            except Exception:
                if self.strict_frame_loading:
                    raise
                frame = torch.zeros(3, 224, 224, dtype=torch.float32)
            frames.append(frame)
        return torch.stack(frames, dim=0)

    def __getitem__(self, index: int) -> Dict[str, object]:
        if self.sequence_group_mode:
            sequence_id = self.sequence_unit_ids[index]
            grouped_samples = self.sequence_samples[sequence_id]
            if len(grouped_samples) >= self.train_sequence_clips:
                chosen = random.sample(grouped_samples, self.train_sequence_clips)
            else:
                chosen = [random.choice(grouped_samples) for _ in range(self.train_sequence_clips)]
            chosen = sorted(chosen, key=lambda rec: int(rec["center_frame"]))
            clips = [
                self._build_clip(sequence_id, int(rec["center_frame"]))
                for rec in chosen
            ]
            first = chosen[0]
            return {
                "touch": torch.stack(clips, dim=0),  # [num_clips, clip_len, 3, 224, 224]
                "sequence_id": sequence_id,
                "center_frames": torch.tensor(
                    [int(rec["center_frame"]) for rec in chosen],
                    dtype=torch.long,
                ),
                "material": int(first["material"]),
                "hardness": int(first["hardness"]),
                "roughness": int(first["roughness"]),
                "material_text": str(first["material_text"]),
                "hardness_text": str(first["hardness_text"]),
                "roughness_text": str(first["roughness_text"]),
            }

        rec = self.samples[index]
        sequence_id = str(rec["sequence_id"])
        center_frame = int(rec["center_frame"])
        clip = self._build_clip(sequence_id, center_frame)

        return {
            "touch": clip,  # [clip_len, 3, 224, 224]
            "sequence_id": sequence_id,
            "center_frame": center_frame,
            "start_frame": int(rec["start_frame"]),
            "end_frame": int(rec["end_frame"]),
            "sample_id": f"{sequence_id}:{center_frame}",
            "material": int(rec["material"]),
            "hardness": int(rec["hardness"]),
            "roughness": int(rec["roughness"]),
            "material_text": str(rec["material_text"]),
            "hardness_text": str(rec["hardness_text"]),
            "roughness_text": str(rec["roughness_text"]),
        }


def build_octopi_new_dataloader(
    root_dir: str = "./octopi_new",
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
    transform: Optional[transforms.Compose] = None,
    augment: bool = False,
    strict_frame_loading: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, Stage3DatasetV1]:
    dataset = Stage3DatasetV1(
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


if __name__ == "__main__":
    dl, ds = build_octopi_new_dataloader(
        root_dir="./octopi_new",
        split="train",
        batch_size=2,
        num_workers=0,
        strict_frame_loading=True,
    )
    print(f"dataset len: {len(ds)}")
    batch = next(iter(dl))
    print("touch shape:", tuple(batch["touch"].shape))
    print("material:", batch["material"])
    print("hardness:", batch["hardness"])
    print("roughness:", batch["roughness"])
