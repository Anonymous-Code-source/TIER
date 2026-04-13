#!/usr/bin/env python3
"""
Task specification for hierarchical Stage3 training.
"""

from __future__ import annotations

from typing import Dict, Iterable, List


LAYER_TASKS: Dict[str, List[str]] = {
    "physical": ["material", "hardness", "roughness"],
    "interaction": ["deformation"],
    "contact": ["slippage", "rotation", "force"],
}

TASK_TO_LAYER: Dict[str, str] = {
    task: layer
    for layer, tasks in LAYER_TASKS.items()
    for task in tasks
}

TASK_LABEL_MAP_KEYS: Dict[str, str] = {
    "material": "material_to_idx",
    "hardness": "hardness_to_idx",
    "roughness": "roughness_to_idx",
    "deformation": "deformation_to_idx",
    "slippage": "slippage_to_idx",
    "rotation": "rotation_to_idx",
    "force": "force_to_idx",
}

TASK_IDX_LABEL_MAP_KEYS: Dict[str, str] = {
    "material": "idx_to_material",
    "hardness": "idx_to_hardness",
    "roughness": "idx_to_roughness",
    "deformation": "idx_to_deformation",
    "slippage": "idx_to_slippage",
    "rotation": "idx_to_rotation",
    "force": "idx_to_force",
}

TASK_ORDER: List[str] = [
    "material",
    "hardness",
    "roughness",
    "deformation",
    "slippage",
    "rotation",
    "force",
]

REGRESSION_TASKS = {"force"}
CLASSIFICATION_TASKS = [task for task in TASK_ORDER if task not in REGRESSION_TASKS]

SYMBOL_LEVELS: List[str] = ["obs", "physical", "interaction", "contact"]


def get_task_names() -> List[str]:
    return list(TASK_ORDER)


def get_layer_names() -> List[str]:
    return list(LAYER_TASKS.keys())


def tasks_for_layer(layer_name: str) -> List[str]:
    return list(LAYER_TASKS[layer_name])


def label_count_for_task(label_maps: Dict[str, Dict], task_name: str) -> int:
    if task_name in REGRESSION_TASKS:
        return 1
    key = TASK_LABEL_MAP_KEYS[task_name]
    return int(len(label_maps.get(key, {})))


def idx_to_name_map_for_task(label_maps: Dict[str, Dict], task_name: str) -> Dict[int, str]:
    return dict(label_maps.get(TASK_IDX_LABEL_MAP_KEYS[task_name], {}))


def active_tasks_from_mask(mask_dict: Dict[str, int | float]) -> List[str]:
    active = []
    for task in TASK_ORDER:
        if float(mask_dict.get(task, 0)) > 0.0:
            active.append(task)
    return active


def iter_layer_tasks(layer_names: Iterable[str] | None = None):
    if layer_names is None:
        layer_names = get_layer_names()
    for layer_name in layer_names:
        yield layer_name, tasks_for_layer(layer_name)


__all__ = [
    "LAYER_TASKS",
    "TASK_TO_LAYER",
    "TASK_LABEL_MAP_KEYS",
    "TASK_IDX_LABEL_MAP_KEYS",
    "TASK_ORDER",
    "REGRESSION_TASKS",
    "CLASSIFICATION_TASKS",
    "SYMBOL_LEVELS",
    "get_task_names",
    "get_layer_names",
    "tasks_for_layer",
    "label_count_for_task",
    "idx_to_name_map_for_task",
    "active_tasks_from_mask",
    "iter_layer_tasks",
]
