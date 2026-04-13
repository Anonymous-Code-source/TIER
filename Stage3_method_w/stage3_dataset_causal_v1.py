#!/usr/bin/env python3
"""
Stage-3 dataset for Causal_data.

Expected layout:
    Causal_data/
      index.csv
      <object>/<sequence>/<sensor>/
        info.csv
        1.png
        2.png
        ...

index.csv columns:
    info_csv,start_frame,end_frame

Per-sensor info.csv columns:
    index,image_name,x,y,z,Material,Hardness,Roughness,Deformation,Slippage,Rotation
"""

from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


@dataclass(frozen=True)
class FrameRecord:
    frame_index: int
    image_name: str
    x: float
    y: float
    z: float
    material: str
    hardness: str
    roughness: str
    deformation: str
    slippage: str
    rotation: str


@dataclass(frozen=True)
class SensorSequenceRecord:
    sequence_group_id: str
    sensor_sequence_id: str
    object_name: str
    sequence_name: str
    sensor_name: str
    start_frame: int
    end_frame: int
    info_csv_path: str
    image_dir: str
    frames: Tuple[FrameRecord, ...]


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


class Stage3CausalDatasetV1(Dataset):
    """
    Build 4-frame clips from Causal_data sensor sequences.

    Split is performed at the sequence-group level (object/sequence), so the two
    sensors of the same physical trial are always assigned to the same split.
    """

    LABEL_KEYS = (
        "material",
        "hardness",
        "roughness",
        "deformation",
        "slippage",
        "rotation",
    )

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

        self.sensor_sequences = self._load_sensor_sequences()
        if len(self.sensor_sequences) == 0:
            raise RuntimeError(f"No valid Causal_data records found in {self.index_file}")

        self._label_maps = self._build_label_maps(self.sensor_sequences)
        split_group_ids = self._build_split_group_ids(self.sensor_sequences)
        selected_group_ids = split_group_ids[split]

        self.selected_sequences = [
            seq for seq in self.sensor_sequences if seq.sequence_group_id in selected_group_ids
        ]
        self.samples = self._build_samples(self.selected_sequences)
        self.sequence_samples = self._group_samples_by_sensor_sequence(self.samples)
        self.sequence_unit_ids = self._build_sequence_unit_ids(self.sequence_samples)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples after building split={split}. "
                "Verify clip_len/frame_stride and Causal_data continuity."
            )
        if self.sequence_group_mode and len(self.sequence_unit_ids) == 0:
            raise RuntimeError(f"No sequence groups available for split={split}")

        self.num_clip_samples = len(self.samples)
        self.num_loader_items = len(self.sequence_unit_ids) if self.sequence_group_mode else len(self.samples)

        sensor_names = sorted({seq.sensor_name for seq in self.selected_sequences})
        print(f"\n[Stage3CausalDatasetV1] root={self.root_dir}")
        print(f"  split: {split}")
        print(f"  sequence groups: {len({seq.sequence_group_id for seq in self.selected_sequences})}")
        print(f"  sensor sequences: {len(self.selected_sequences)}")
        print(f"  samples: {len(self.samples)}")
        print(f"  sensors: {sensor_names}")
        if self.sequence_group_mode:
            print(
                f"  loader items: {self.num_loader_items}"
                f" (sensor-sequence groups, {self.train_sequence_clips} clips/item)"
            )
        print(
            "  center-frame step:"
            f" {self.train_center_frame_step if split == 'train' else self.eval_center_frame_step}"
        )
        print("  split strategy: sequence-group level stratified by label signature")
        print(
            "  physical label classes:"
            f" material={len(self._label_maps['material_to_idx'])},"
            f" hardness={len(self._label_maps['hardness_to_idx'])},"
            f" roughness={len(self._label_maps['roughness_to_idx'])}"
        )

    @staticmethod
    def _safe_int(value: str) -> int:
        return int(str(value).strip())

    @staticmethod
    def _safe_float(value: str) -> float:
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _clean_label(value: str) -> str:
        out = str(value).strip()
        return out if out != "" else "unknown"

    @classmethod
    def _frame_record_from_row(cls, row: Dict[str, str]) -> FrameRecord:
        image_name = str(row.get("image_name", "")).strip()
        frame_index_raw = row.get("index", "")
        if str(frame_index_raw).strip() == "" and image_name != "":
            frame_index_raw = Path(image_name).stem
        return FrameRecord(
            frame_index=cls._safe_int(frame_index_raw),
            image_name=image_name,
            x=cls._safe_float(row.get("x", "0")),
            y=cls._safe_float(row.get("y", "0")),
            z=cls._safe_float(row.get("z", "0")),
            material=cls._clean_label(row.get("Material", "")),
            hardness=cls._clean_label(row.get("Hardness", "")),
            roughness=cls._clean_label(row.get("Roughness", "")),
            deformation=cls._clean_label(row.get("Deformation", "")),
            slippage=cls._clean_label(row.get("Slippage", "")),
            rotation=cls._clean_label(row.get("Rotation", "")),
        )

    def _load_sensor_sequences(self) -> List[SensorSequenceRecord]:
        sequences: List[SensorSequenceRecord] = []
        with open(self.index_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                info_csv_rel = str(row.get("info_csv", "")).strip()
                if info_csv_rel == "":
                    continue

                info_csv_path = self.root_dir / info_csv_rel
                if not info_csv_path.exists():
                    continue

                parts = Path(info_csv_rel).parts
                if len(parts) < 4:
                    continue
                object_name, sequence_name, sensor_name = parts[0], parts[1], parts[2]
                sequence_group_id = f"{object_name}/{sequence_name}"
                sensor_sequence_id = f"{sequence_group_id}/{sensor_name}"

                try:
                    start_frame = self._safe_int(row.get("start_frame", ""))
                    end_frame = self._safe_int(row.get("end_frame", ""))
                except ValueError:
                    continue
                if start_frame > end_frame:
                    continue

                with open(info_csv_path, "r", encoding="utf-8") as info_f:
                    frame_rows = [self._frame_record_from_row(frame_row) for frame_row in csv.DictReader(info_f)]

                frame_rows = [
                    fr for fr in frame_rows if start_frame <= int(fr.frame_index) <= end_frame
                ]
                frame_rows.sort(key=lambda fr: int(fr.frame_index))
                if len(frame_rows) == 0:
                    continue

                sequences.append(
                    SensorSequenceRecord(
                        sequence_group_id=sequence_group_id,
                        sensor_sequence_id=sensor_sequence_id,
                        object_name=object_name,
                        sequence_name=sequence_name,
                        sensor_name=sensor_name,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        info_csv_path=str(info_csv_path),
                        image_dir=str(info_csv_path.parent),
                        frames=tuple(frame_rows),
                    )
                )
        return sequences

    @classmethod
    def _build_label_maps(cls, sequences: List[SensorSequenceRecord]) -> Dict[str, Dict]:
        label_values = {key: set() for key in cls.LABEL_KEYS}
        for seq in sequences:
            for frame in seq.frames:
                label_values["material"].add(frame.material)
                label_values["hardness"].add(frame.hardness)
                label_values["roughness"].add(frame.roughness)
                label_values["deformation"].add(frame.deformation)
                label_values["slippage"].add(frame.slippage)
                label_values["rotation"].add(frame.rotation)

        out: Dict[str, Dict] = {}
        for key, values in label_values.items():
            ordered = sorted(values)
            to_idx = {name: idx for idx, name in enumerate(ordered)}
            out[f"{key}_to_idx"] = to_idx
            out[f"idx_to_{key}"] = {idx: name for name, idx in to_idx.items()}
        return out

    @staticmethod
    def _sequence_sort_key(sequence_id: str) -> Tuple[str, str]:
        parts = sequence_id.split("/")
        if len(parts) >= 2 and parts[1].isdigit():
            return parts[0], f"{int(parts[1]):012d}"
        return parts[0], sequence_id

    @classmethod
    def _sort_sequence_ids(cls, sequence_ids: List[str]) -> List[str]:
        return sorted(sequence_ids, key=cls._sequence_sort_key)

    @classmethod
    def _group_label_signature(cls, sequences: List[SensorSequenceRecord]) -> Tuple[str, ...]:
        counters = {key: Counter() for key in cls.LABEL_KEYS}
        for seq in sequences:
            for frame in seq.frames:
                counters["material"][frame.material] += 1
                counters["hardness"][frame.hardness] += 1
                counters["roughness"][frame.roughness] += 1
                counters["deformation"][frame.deformation] += 1
                counters["slippage"][frame.slippage] += 1
                counters["rotation"][frame.rotation] += 1

        signature = []
        for key in cls.LABEL_KEYS:
            if len(counters[key]) == 0:
                signature.append("unknown")
            else:
                signature.append(counters[key].most_common(1)[0][0])
        return tuple(signature)

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

    def _build_split_group_ids(self, sequences: List[SensorSequenceRecord]) -> Dict[str, set]:
        grouped_sequences: Dict[str, List[SensorSequenceRecord]] = defaultdict(list)
        for seq in sequences:
            grouped_sequences[seq.sequence_group_id].append(seq)

        unique_group_ids = self._sort_sequence_ids(list(grouped_sequences.keys()))
        n_total = len(unique_group_ids)
        n_train = int(n_total * self.split_ratios[0])
        n_val = int(n_total * self.split_ratios[1])
        n_test = n_total - n_train - n_val

        split_names = ("train", "val", "test")
        split_targets = [n_train, n_val, n_test]
        split_ids = {name: [] for name in split_names}
        remaining_targets = split_targets[:]

        strata: Dict[Tuple[str, ...], List[str]] = {}
        for group_id, group_seqs in grouped_sequences.items():
            signature = self._group_label_signature(group_seqs)
            strata.setdefault(signature, []).append(group_id)

        rng = random.Random(self.random_seed)
        ordered_strata = sorted(
            strata.items(),
            key=lambda item: (len(item[1]),) + tuple(item[0]),
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
        all_ids = set(unique_group_ids)
        return {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
            "all": all_ids,
        }

    def _build_samples(self, sequences: List[SensorSequenceRecord]) -> List[Dict[str, object]]:
        samples: List[Dict[str, object]] = []
        center_frame_step = self.train_center_frame_step if self.split == "train" else self.eval_center_frame_step

        for seq in sequences:
            frame_map = {int(frame.frame_index): frame for frame in seq.frames}
            min_center = int(seq.start_frame) + (self.clip_len - 1) * self.frame_stride
            if min_center > int(seq.end_frame):
                continue

            for center_frame in range(min_center, int(seq.end_frame) + 1, center_frame_step):
                frame_indices = [
                    center_frame - (self.clip_len - 1 - idx) * self.frame_stride
                    for idx in range(self.clip_len)
                ]
                if any(frame_idx not in frame_map for frame_idx in frame_indices):
                    continue

                center_meta = frame_map[int(center_frame)]
                samples.append(
                    {
                        "sequence_group_id": seq.sequence_group_id,
                        "sensor_sequence_id": seq.sensor_sequence_id,
                        "object_name": seq.object_name,
                        "sequence_name": seq.sequence_name,
                        "sensor_name": seq.sensor_name,
                        "center_frame": int(center_frame),
                        "frame_indices": frame_indices,
                        "start_frame": int(seq.start_frame),
                        "end_frame": int(seq.end_frame),
                        "material": self._label_maps["material_to_idx"][center_meta.material],
                        "hardness": self._label_maps["hardness_to_idx"][center_meta.hardness],
                        "roughness": self._label_maps["roughness_to_idx"][center_meta.roughness],
                        "deformation": self._label_maps["deformation_to_idx"][center_meta.deformation],
                        "slippage": self._label_maps["slippage_to_idx"][center_meta.slippage],
                        "rotation": self._label_maps["rotation_to_idx"][center_meta.rotation],
                        "material_text": center_meta.material,
                        "hardness_text": center_meta.hardness,
                        "roughness_text": center_meta.roughness,
                        "deformation_text": center_meta.deformation,
                        "slippage_text": center_meta.slippage,
                        "rotation_text": center_meta.rotation,
                        "force_x": float(center_meta.x),
                        "force_y": float(center_meta.y),
                        "force_z": float(center_meta.z),
                    }
                )
        return samples

    @staticmethod
    def _group_samples_by_sensor_sequence(samples: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
        grouped: Dict[str, List[Dict[str, object]]] = {}
        for rec in samples:
            sensor_sequence_id = str(rec["sensor_sequence_id"])
            grouped.setdefault(sensor_sequence_id, []).append(rec)
        return grouped

    def _build_sequence_unit_ids(self, grouped: Dict[str, List[Dict[str, object]]]) -> List[str]:
        sensor_sequence_ids = sorted(grouped.keys())
        if not self.sequence_group_mode:
            return sensor_sequence_ids

        unit_ids: List[str] = []
        for sensor_sequence_id in sensor_sequence_ids:
            num_clips = max(len(grouped[sensor_sequence_id]), 1)
            repeats = max(1, int(math.ceil(float(num_clips) / float(self.train_sequence_clips))))
            unit_ids.extend([sensor_sequence_id] * repeats)
        return unit_ids

    def get_label_maps(self) -> Dict[str, Dict]:
        return self._label_maps

    def __len__(self) -> int:
        return self.num_loader_items

    def _resolve_frame_path(self, sensor_sequence_id: str, frame_idx: int) -> Path:
        seq = self._sensor_sequence_by_id[sensor_sequence_id]
        frame_meta = self._frame_map_by_sensor_id[sensor_sequence_id][int(frame_idx)]
        image_dir = Path(seq.image_dir)

        candidates = []
        if frame_meta.image_name != "":
            candidates.append(image_dir / frame_meta.image_name)
        candidates.extend(
            [
                image_dir / f"{frame_idx}.png",
                image_dir / f"{frame_idx}.jpg",
                image_dir / f"{frame_idx:010d}.png",
                image_dir / f"{frame_idx:010d}.jpg",
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Frame file not found for sensor_sequence={sensor_sequence_id}, frame={frame_idx}."
        )

    def _load_one_frame(self, sensor_sequence_id: str, frame_idx: int) -> torch.Tensor:
        frame_path = self._resolve_frame_path(sensor_sequence_id, frame_idx)
        with Image.open(frame_path) as img:
            img = img.convert("RGB")
        return self.transform(img)

    def _build_clip(self, sensor_sequence_id: str, frame_indices: List[int]) -> torch.Tensor:
        frames: List[torch.Tensor] = []
        for frame_idx in frame_indices:
            try:
                frame = self._load_one_frame(sensor_sequence_id, int(frame_idx))
            except Exception:
                if self.strict_frame_loading:
                    raise
                frame = torch.zeros(3, 224, 224, dtype=torch.float32)
            frames.append(frame)
        return torch.stack(frames, dim=0)

    def __getitem__(self, index: int) -> Dict[str, object]:
        if self.sequence_group_mode:
            sensor_sequence_id = self.sequence_unit_ids[index]
            grouped_samples = self.sequence_samples[sensor_sequence_id]
            if len(grouped_samples) >= self.train_sequence_clips:
                chosen = random.sample(grouped_samples, self.train_sequence_clips)
            else:
                chosen = [random.choice(grouped_samples) for _ in range(self.train_sequence_clips)]
            chosen = sorted(chosen, key=lambda rec: int(rec["center_frame"]))
            clips = [
                self._build_clip(sensor_sequence_id, list(rec["frame_indices"]))
                for rec in chosen
            ]
            first = chosen[0]
            targets = {
                "material": int(first["material"]),
                "hardness": int(first["hardness"]),
                "roughness": int(first["roughness"]),
                "deformation": int(first["deformation"]),
                "slippage": int(first["slippage"]),
                "rotation": int(first["rotation"]),
            }
            target_mask = {
                "material": 1,
                "hardness": 1,
                "roughness": 1,
                "deformation": 1,
                "slippage": 1,
                "rotation": 1,
                "force": 0,
            }
            return {
                "touch": torch.stack(clips, dim=0),
                "sequence_group_id": str(first["sequence_group_id"]),
                "sensor_sequence_id": str(sensor_sequence_id),
                "sensor_name": str(first["sensor_name"]),
                "center_frames": torch.tensor(
                    [int(rec["center_frame"]) for rec in chosen],
                    dtype=torch.long,
                ),
                "material": int(first["material"]),
                "hardness": int(first["hardness"]),
                "roughness": int(first["roughness"]),
                "deformation": int(first["deformation"]),
                "slippage": int(first["slippage"]),
                "rotation": int(first["rotation"]),
                "targets": targets,
                "target_mask": target_mask,
                "force_xyz": torch.tensor(
                    [float(first["force_x"]), float(first["force_y"]), float(first["force_z"])],
                    dtype=torch.float32,
                ),
                "sample_id": f"{sensor_sequence_id}:{int(first['center_frame'])}",
            }

        rec = self.samples[index]
        sensor_sequence_id = str(rec["sensor_sequence_id"])
        center_frame = int(rec["center_frame"])
        clip = self._build_clip(sensor_sequence_id, list(rec["frame_indices"]))
        targets = {
            "material": int(rec["material"]),
            "hardness": int(rec["hardness"]),
            "roughness": int(rec["roughness"]),
            "deformation": int(rec["deformation"]),
            "slippage": int(rec["slippage"]),
            "rotation": int(rec["rotation"]),
        }
        target_mask = {
            "material": 1,
            "hardness": 1,
            "roughness": 1,
            "deformation": 1,
            "slippage": 1,
            "rotation": 1,
            "force": 0,
        }
        return {
            "touch": clip,
            "sequence_group_id": str(rec["sequence_group_id"]),
            "sensor_sequence_id": sensor_sequence_id,
            "sequence_id": sensor_sequence_id,
            "object_name": str(rec["object_name"]),
            "sequence_name": str(rec["sequence_name"]),
            "sensor_name": str(rec["sensor_name"]),
            "center_frame": center_frame,
            "start_frame": int(rec["start_frame"]),
            "end_frame": int(rec["end_frame"]),
            "sample_id": f"{sensor_sequence_id}:{center_frame}",
            "material": int(rec["material"]),
            "hardness": int(rec["hardness"]),
            "roughness": int(rec["roughness"]),
            "deformation": int(rec["deformation"]),
            "slippage": int(rec["slippage"]),
            "rotation": int(rec["rotation"]),
            "targets": targets,
            "target_mask": target_mask,
            "material_text": str(rec["material_text"]),
            "hardness_text": str(rec["hardness_text"]),
            "roughness_text": str(rec["roughness_text"]),
            "deformation_text": str(rec["deformation_text"]),
            "slippage_text": str(rec["slippage_text"]),
            "rotation_text": str(rec["rotation_text"]),
            "force_x": float(rec["force_x"]),
            "force_y": float(rec["force_y"]),
            "force_z": float(rec["force_z"]),
            "force_xyz": torch.tensor(
                [float(rec["force_x"]), float(rec["force_y"]), float(rec["force_z"])],
                dtype=torch.float32,
            ),
        }

    @property
    def _sensor_sequence_by_id(self) -> Dict[str, SensorSequenceRecord]:
        if not hasattr(self, "__sensor_sequence_by_id_cache"):
            self.__sensor_sequence_by_id_cache = {
                seq.sensor_sequence_id: seq for seq in self.sensor_sequences
            }
        return self.__sensor_sequence_by_id_cache

    @property
    def _frame_map_by_sensor_id(self) -> Dict[str, Dict[int, FrameRecord]]:
        if not hasattr(self, "__frame_map_by_sensor_id_cache"):
            self.__frame_map_by_sensor_id_cache = {
                seq.sensor_sequence_id: {int(frame.frame_index): frame for frame in seq.frames}
                for seq in self.sensor_sequences
            }
        return self.__frame_map_by_sensor_id_cache


def build_causal_dataloader(
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
    transform: Optional[transforms.Compose] = None,
    augment: bool = False,
    strict_frame_loading: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, Stage3CausalDatasetV1]:
    dataset = Stage3CausalDatasetV1(
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
    dl, ds = build_causal_dataloader(
        root_dir="./Causal_data",
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
