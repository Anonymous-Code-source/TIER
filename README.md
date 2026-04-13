# PhysioTouch

PhysioTouch is a tactile representation learning project for multi-stage tactile perception. The current codebase keeps the main training path for three stages:

1. Stage 1: tactile video MAE pretraining
2. Stage 2: cross-sensor tactile representation alignment
3. Stage 3: hierarchical tactile attribute prediction

The Stage 3 pipeline predicts physical, interaction, and contact attributes in a structured hierarchy:

```text
touch sequence -> Stage2 backbone -> observation symbols
              -> physical layer -> interaction layer -> contact layer
```

The current hierarchical Stage 3 tasks are:

- Physical layer: material, hardness, roughness
- Interaction layer: deformation
- Contact layer: slippage, rotation, force



## Main Files

Stage 1:

```text
Main_stage1.py
Stage1_engine.py
train_joint_balanced_weights.sh
```

Stage 2:

```text
Stage2_main.py
Stage2_engine.py
scripts/train_stage2a_mixed.sh
scripts/train_stage2b_phase1.sh
scripts/train_stage2b_phase2.sh
```

Stage 3 hierarchical training:

```text
Stage3_method_w/train_stage3_hierarchical.sh
Stage3_method_w/main_stage3_hierarchical.py
Stage3_method_w/model_stage3_hierarchical.py
Stage3_method_w/engine_stage3_hierarchical.py
Stage3_method_w/stage3_dataset_causal_hierarchical.py
Stage3_method_w/stage3_dataset_causal_v1.py
Stage3_method_w/stage3_task_spec.py
```

## Data And Checkpoints

Stage 1 expects the pretraining datasets under:

```text
/data/tactile_datasets/S1_dataset/
```

Stage 2 uses TacQuad data by default:

```text
./Tacquad/
```

Stage 3 uses:

```text
./Causal_data/
```

Stage 3 also needs a Stage 2 checkpoint. By default:

```text
./log/stage2b_phase2/best.pth
```

## Training

Run Stage 1:

```bash
bash train_joint_balanced_weights.sh
```

Run Stage 2:

```bash
bash scripts/train_stage2a_mixed.sh
bash scripts/train_stage2b_phase1.sh
bash scripts/train_stage2b_phase2.sh
```

Run Stage 3:

```bash
bash Stage3_method_w/train_stage3_hierarchical.sh
```

## Notes

- Large datasets, logs, caches, and checkpoints should normally stay out of Git.
- `force` metrics are regression metrics such as `mae_force` and `rmse_force`.
- `Stage3_method_w` is the active Stage 3 implementation.
