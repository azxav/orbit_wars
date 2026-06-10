
# Orbit Wars BC / PPO Training Pipeline

This repository contains the full training and evaluation pipeline for an Orbit Wars Kaggle agent.

The project is built around one main idea:

> The ML policy chooses **what to do**.  
> The geometry engine decides **how to execute it safely**.

The model does not directly predict launch angles. It predicts high-level source decisions:

- choose target planet or no-op
- choose amount bin
- rely on geometry solver for launch angle, trajectory, sun/bounds checks, and move decoding

This keeps behavior cloning, PPO, and final game execution aligned under the same action contract.

---

## Project structure

```text
orbit_wars/
├── orbit_geometry_skeleton/
│   └── geometry_skeleton.py
│
├── orbit_training_prep/
│   ├── replay_io.py
│   ├── target_inference.py
│   ├── features.py
│   ├── dataset_builder.py
│   ├── validate_dataset.py
│   ├── split_episodes.py
│   ├── materialize_splits.py
│   └── audit_exact_targets.py
│
├── orbit_bc_training/
│   ├── model.py
│   ├── dataset.py
│   ├── losses.py
│   ├── train_bc_policy.py
│   ├── eval_bc_policy.py
│   └── checkpoints.py
│
├── orbit_bc_eval/
│   ├── bc_agent_runtime.py
│   ├── run_local_matches.py
│   ├── eval_report.py
│   ├── rollout_metrics.py
│   └── compare_metric_relevance.py
│
├── orbit_ppo_training/
│   ├── train_ppo.py
│   ├── eval_ppo.py
│   ├── smoke_test.py
│   ├── policy.py
│   ├── rollout_worker.py
│   └── ppo_loss.py
│
├── tests/
└── README.md
````

---

## Main components

### 1. Geometry skeleton

Location:

```text
orbit_geometry_skeleton/
```

Responsible for:

* launch angle generation
* trajectory / intercept handling
* ETA estimation
* collision / sun / bounds feasibility
* converting model decisions into valid environment moves

The model should not learn raw angle prediction unless the whole action contract is redesigned.

---

### 2. Dataset preparation

Location:

```text
orbit_training_prep/
```

Responsible for:

* loading Kaggle replay JSON files
* normalizing observations
* aligning replay action at index `t` with observation at index `t-1`
* mapping raw replay actions into training labels
* building JSONL rows and dense `.npz` tensors
* validating label quality

Important replay alignment:

```text
Kaggle replay action at row t belongs to the observation from row t-1.
```

Without this alignment, ship counts and source ownership can be wrong after launches.

---

### 3. Behavior cloning

Location:

```text
orbit_bc_training/
```

The BC model is an entity-based policy.

It receives:

* planet features
* global game features
* target-state features
* source-target pair features
* source slot

It predicts:

* target planet slot or no-op
* amount bin

The amount bins are:

```text
none
one_ship
capture_plus_one
quarter
half
three_quarter
all
```

The policy granularity is:

```text
one decision per owned source planet per turn
```

---

### 4. BC local evaluation

Location:

```text
orbit_bc_eval/
```

Responsible for running trained BC checkpoints in real local Kaggle matches.

It reports gameplay metrics such as:

* win rate
* average rank
* average final reward
* owned planets
* total ships
* launch count
* illegal actions
* timeout count
* decode / no-op / launch behavior
* opening behavior

Use this for real model comparison. Training loss alone is not reliable enough.

---

### 5. PPO fine-tuning

Location:

```text
orbit_ppo_training/
```

PPO starts from a BC checkpoint.

The PPO policy keeps the same BC action contract:

```text
source planet -> target/no-op + amount bin -> geometry decoder -> env action
```

PPO should improve decision quality without changing the action interface.

---

## Feature contract

The project should use one compact feature contract.

Avoid keeping multiple feature versions such as `v1`, `v2`, `v3`. The model, dataset builder, runtime agent, BC training, and PPO should all use the same feature path.

Recommended compact feature groups:

```text
planet features
global features
target-state features
source-target pair features
```

The feature set should prioritize:

* ownership
* source ships
* target ships
* production
* capture cost
* target value
* threat
* projected garrison
* safe sendable ships
* enemy timing
* no-op decision

Avoid fake or redundant features such as:

* constant geometry flags
* duplicated ETA variants
* misleading comet fields
* repeated 10-step / 20-step copies when one horizon is enough
* reconstructed ship values that do not invert log normalization correctly

---

## Dataset outputs

A built dataset directory contains:

```text
launch_rows.jsonl
source_turn_rows.jsonl
state_rows.jsonl
dense_bc_arrays.npz
metadata.json
validation_report.json
validation_report.md
```

### `launch_rows.jsonl`

One row per actual replay launch.

Useful for:

* debugging target inference
* auditing geometry labels
* checking first-hit accuracy

### `source_turn_rows.jsonl`

Primary BC training table.

One row per owned source planet per player step.

Contains:

* source slot
* target label
* amount label
* winner / loser information
* geometry inference metadata
* sample weight

### `state_rows.jsonl`

One row per player observation state.

Useful for:

* checking state-level replay coverage
* debugging episode splits

### `dense_bc_arrays.npz`

Dense tensor file used by neural training.

Contains fixed-size arrays for:

* planet features
* global features
* target-state features
* target labels
* amount labels
* source mask
* feature names

### `metadata.json`

Schema and dataset build metadata.

Contains:

* action space
* feature names
* build settings
* replay paths
* stats
* target inference method counts

---

## Recommended workflow

### Step 1: Install project

```powershell
python -m pip install -e .
```

### Step 2: Put replays into a folder

Example:

```text
replays/
├── episode_001-replay.json
├── episode_002-replay.json
└── ...
```

### Step 3: Build combined dataset

```powershell
python -m orbit_training_prep.dataset_builder `
  --replay .\replays `
  --out-dir .\orbit_dataset_work\combined `
  --horizon 160 `
  --device cpu `
  --workers 8
```

Use CUDA only if exact target inference needs it:

```powershell
python -m orbit_training_prep.dataset_builder `
  --replay .\replays `
  --out-dir .\orbit_dataset_work\combined `
  --horizon 160 `
  --device cuda `
  --workers 1
```

### Step 4: Validate dataset

```powershell
python -m orbit_training_prep.validate_dataset `
  --out-dir .\orbit_dataset_work\combined
```

This writes:

```text
validation_report.json
validation_report.md
```

### Step 5: Split episodes

```powershell
python -m orbit_training_prep.split_episodes `
  --dataset_root .\orbit_dataset_work\combined `
  --valid_frac 0.15 `
  --seed 42 `
  --out .\orbit_dataset_work\splits.json
```

### Step 6: Materialize train/valid files

```powershell
python -m orbit_training_prep.materialize_splits `
  --dataset_root .\orbit_dataset_work\combined `
  --splits .\orbit_dataset_work\splits.json `
  --out .\orbit_dataset_work\split_dataset
```

This creates:

```text
orbit_dataset_work/split_dataset/train/source_turn_rows.jsonl
orbit_dataset_work/split_dataset/valid/source_turn_rows.jsonl
```

### Step 7: Train BC model

```powershell
python -m orbit_bc_training.train_bc_policy `
  --train_dir .\orbit_dataset_work\split_dataset\train `
  --valid_dir .\orbit_dataset_work\split_dataset\valid `
  --out_dir .\bc_checkpoints\compact_bc_v1 `
  --batch_size 512 `
  --epochs 20 `
  --lr 3e-4 `
  --weight_decay 1e-4 `
  --grad_clip 1.0 `
  --hidden_size 128 `
  --num_layers 2 `
  --num_heads 4 `
  --mlp_size 256 `
  --dropout 0.0 `
  --seed 42 `
  --device auto `
  --num_workers 4
```

Important outputs:

```text
bc_checkpoints/compact_bc_v1/latest/checkpoint.pt
bc_checkpoints/compact_bc_v1/best/checkpoint.pt
bc_checkpoints/compact_bc_v1/metrics.jsonl
```

### Step 8: Evaluate BC on validation data

```powershell
python -m orbit_bc_training.eval_bc_policy `
  --checkpoint .\bc_checkpoints\compact_bc_v1\best\checkpoint.pt `
  --valid_dir .\orbit_dataset_work\split_dataset\valid `
  --out .\bc_checkpoints\compact_bc_v1\offline_eval.json `
  --device auto
```

This evaluates imitation accuracy, not real gameplay strength.

### Step 9: Evaluate BC in real local matches

```powershell
python -m orbit_bc_eval.run_local_matches `
  --bc_checkpoint .\bc_checkpoints\compact_bc_v1\best\checkpoint.pt `
  --opponent heuristic_path `
  --players both `
  --num_games 20 `
  --seed_start 1000 `
  --out_dir .\bc_eval_runs\compact_bc_v1_vs_heuristic `
  --device cpu `
  --debug_game
```

Use this as the main BC quality check.

### Step 10: Compare metric relevance between runs

Example:

```powershell
python -m orbit_bc_eval.compare_metric_relevance `
  .\bc_eval_runs\old_bc\2p\games.jsonl `
  .\bc_eval_runs\compact_bc_v1_vs_heuristic\2p\games.jsonl
```

Use this to find which metrics actually move with reward/rank.

### Step 11: PPO smoke test

Before full PPO training:

```powershell
python -m orbit_ppo_training.smoke_test `
  --bc_checkpoint .\bc_checkpoints\compact_bc_v1\best\checkpoint.pt `
  --out_dir .\ppo_runs\smoke_compact_bc_v1 `
  --device cpu
```

### Step 12: Train PPO from BC

```powershell
python -m orbit_ppo_training.train_ppo `
  --bc_checkpoint .\bc_checkpoints\compact_bc_v1\best\checkpoint.pt `
  --out_dir .\ppo_runs\compact_bc_v1_ppo `
  --players 4 `
  --opponent heuristic_path `
  --num_envs 8 `
  --rollout_games_per_update 32 `
  --updates 50 `
  --lr 2e-5 `
  --clip_range 0.10 `
  --entropy_coef 0.01 `
  --kl_to_bc_coef 0.02 `
  --target_kl 0.03 `
  --ppo_epochs 2 `
  --minibatch_size 512 `
  --eval_interval_updates 5 `
  --save_interval_updates 5 `
  --eval_games 10 `
  --seed 42 `
  --device cpu `
  --heuristic_path .\orbit_wars_base.py `
  --max_episode_steps 500
```

Important outputs:

```text
ppo_runs/compact_bc_v1_ppo/latest/checkpoint.pt
ppo_runs/compact_bc_v1_ppo/best/checkpoint.pt
ppo_runs/compact_bc_v1_ppo/metrics.jsonl
ppo_runs/compact_bc_v1_ppo/eval_summary.json
```

### Step 13: Evaluate PPO checkpoint

```powershell
python -m orbit_ppo_training.eval_ppo `
  --checkpoint .\ppo_runs\compact_bc_v1_ppo\best `
  --opponent heuristic_path `
  --players 4 `
  --num_games 50 `
  --out_dir .\ppo_eval_runs\compact_bc_v1_ppo_vs_heuristic `
  --seed 2000 `
  --device cpu `
  --save_replays 3 `
  --save_html_replays
```

---

## What to trust

### Useful training metrics

Use these only for debugging:

```text
target_loss
amount_loss
target_accuracy
amount_accuracy
target_non_noop_accuracy
launch_vs_noop_accuracy
noop_rate_predicted
noop_rate_true
```

They do not always translate to real gameplay.

### Useful gameplay metrics

Use these for real model comparison:

```text
average_final_reward
average_rank
winrate
average_owned_planets
average_total_ships
average_final_ship_count
average_planets_captured
average_launches_per_game
early_launch_rate
illegal_action_count
timeout_count
predicted_launch_rate
actual_returned_move_count
```

### Best checkpoint selection

Prefer gameplay-based selection when possible.

Offline validation accuracy is not enough because the model can imitate replay labels while still failing to produce strong real-game behavior.

---

## Development rules

### Keep one action contract

Do not create separate action spaces for BC and PPO.

Correct:

```text
BC target/no-op + amount bin
PPO target/no-op + amount bin
Geometry decoder converts both into env moves
```

Wrong:

```text
BC predicts target
PPO predicts angle
Runtime uses heuristic amount
```

That creates train/eval mismatch.

### Keep one feature contract

Avoid feature version branches.

Correct:

```text
features.py -> one FeatureState -> one dense dataset schema -> one runtime batch builder
```

Wrong:

```text
feature_version=v1/v2/v3
old dense fallback
runtime branch for old checkpoints
```

### Rebuild datasets after feature changes

If feature names or dimensions change, rebuild:

```powershell
python -m orbit_training_prep.dataset_builder `
  --replay .\replays `
  --out-dir .\orbit_dataset_work\combined `
  --horizon 160 `
  --device cpu `
  --workers 8
```

Do not train new models from old dense arrays.

---

## Debugging checklist

### Dataset has too many no-ops

Check:

```powershell
python -m orbit_training_prep.validate_dataset `
  --out-dir .\orbit_dataset_work\combined
```

Look at:

```text
noop_source_rate
positive_source_turns
amount_bin_distribution_source_turns
```

### Target labels look wrong

Run exact target audit:

```powershell
python -m orbit_training_prep.audit_exact_targets `
  --dataset_dir .\orbit_dataset_work\combined `
  --replay_dir .\replays `
  --sample_size 2000 `
  --out .\orbit_dataset_work\exact_target_audit.json `
  --seed 0 `
  --horizon 200
```

### BC has good validation but bad gameplay

Run real local matches:

```powershell
python -m orbit_bc_eval.run_local_matches `
  --bc_checkpoint .\bc_checkpoints\compact_bc_v1\best\checkpoint.pt `
  --opponent heuristic_path `
  --players both `
  --num_games 20 `
  --seed_start 1000 `
  --out_dir .\bc_eval_runs\debug_bc `
  --device cpu `
  --debug_game
```

Then inspect:

```text
games.jsonl
summary.json
debug/*.json
```

### PPO collapses

Check:

```text
illegal_action_count
timeout_count
clip_frac
approx_kl
collapse_clip_streak
eval_average_final_reward
```

If collapse happens:

* lower learning rate
* increase KL-to-BC coefficient
* reduce PPO epochs
* reduce clip range
* start from stronger BC checkpoint

---

# Useful commands

## Install

```powershell
python -m pip install -e .
```

## Build dataset

```powershell
python -m orbit_training_prep.dataset_builder `
  --replay .\replays `
  --out-dir .\orbit_dataset_work\combined `
  --horizon 160 `
  --device cpu `
  --batch-size 256 `
  --workers 8
```

## Build dataset from one replay

```powershell
python -m orbit_training_prep.dataset_builder `
  --replay .\replays\example-replay.json `
  --out-dir .\orbit_dataset_work\single `
  --horizon 160 `
  --device cpu `
  --batch-size 256 `
  --workers 1
```

## Validate dataset

```powershell
python -m orbit_training_prep.validate_dataset `
  --out-dir .\orbit_dataset_work\combined
```

## Create episode split

```powershell
python -m orbit_training_prep.split_episodes `
  --dataset_root .\orbit_dataset_work\combined `
  --valid_frac 0.15 `
  --seed 42 `
  --out .\orbit_dataset_work\splits.json
```

## Materialize train/valid split

```powershell
python -m orbit_training_prep.materialize_splits `
  --dataset_root .\orbit_dataset_work\combined `
  --splits .\orbit_dataset_work\splits.json `
  --out .\orbit_dataset_work\split_dataset
```

## Audit exact target inference

```powershell
python -m orbit_training_prep.audit_exact_targets `
  --dataset_dir .\orbit_dataset_work\combined `
  --replay_dir .\replays `
  --sample_size 2000 `
  --out .\orbit_dataset_work\exact_target_audit.json `
  --seed 0 `
  --horizon 200
```

## Train BC

```powershell
python -m orbit_bc_training.train_bc_policy `
  --train_dir .\orbit_dataset_work\split_dataset\train `
  --valid_dir .\orbit_dataset_work\split_dataset\valid `
  --out_dir .\bc_checkpoints\compact_bc_v1 `
  --batch_size 512 `
  --epochs 20 `
  --lr 3e-4 `
  --weight_decay 1e-4 `
  --grad_clip 1.0 `
  --hidden_size 128 `
  --num_layers 2 `
  --num_heads 4 `
  --mlp_size 256 `
  --dropout 0.0 `
  --seed 42 `
  --device auto `
  --num_workers 4
```

## Evaluate BC offline

```powershell
python -m orbit_bc_training.eval_bc_policy `
  --checkpoint .\bc_checkpoints\compact_bc_v1\best\checkpoint.pt `
  --valid_dir .\orbit_dataset_work\split_dataset\valid `
  --out .\bc_checkpoints\compact_bc_v1\offline_eval.json `
  --device auto
```

## Run BC local matches vs heuristic

```powershell
python -m orbit_bc_eval.run_local_matches `
  --bc_checkpoint .\bc_checkpoints\compact_bc_v1\best\checkpoint.pt `
  --opponent heuristic_path `
  --players both `
  --num_games 20 `
  --seed_start 1000 `
  --out_dir .\bc_eval_runs\compact_bc_v1_vs_heuristic `
  --device cpu `
  --debug_game
```

## Run BC local matches vs simple expand

```powershell
python -m orbit_bc_eval.run_local_matches `
  --bc_checkpoint .\bc_checkpoints\compact_bc_v1\best\checkpoint.pt `
  --opponent simple_expand `
  --players both `
  --num_games 20 `
  --seed_start 1000 `
  --out_dir .\bc_eval_runs\compact_bc_v1_vs_simple_expand `
  --device cpu
```

## Run BC local matches vs another BC checkpoint

```powershell
python -m orbit_bc_eval.run_local_matches `
  --bc_checkpoint .\bc_checkpoints\compact_bc_v1\best\checkpoint.pt `
  --opponent bc_checkpoint `
  --opponent_bc_checkpoint .\bc_checkpoints\baseline_bc\best\checkpoint.pt `
  --players both `
  --num_games 20 `
  --seed_start 1000 `
  --out_dir .\bc_eval_runs\compact_vs_baseline `
  --device cpu
```

## Compare metric relevance

```powershell
python -m orbit_bc_eval.compare_metric_relevance `
  .\bc_eval_runs\baseline\2p\games.jsonl `
  .\bc_eval_runs\compact_bc_v1_vs_heuristic\2p\games.jsonl
```

## PPO smoke test

```powershell
python -m orbit_ppo_training.smoke_test `
  --bc_checkpoint .\bc_checkpoints\compact_bc_v1\best\checkpoint.pt `
  --out_dir .\ppo_runs\smoke_compact_bc_v1 `
  --device cpu
```

## Train PPO

```powershell
python -m orbit_ppo_training.train_ppo `
  --bc_checkpoint .\bc_checkpoints\compact_bc_v1\best\checkpoint.pt `
  --out_dir .\ppo_runs\compact_bc_v1_ppo `
  --players 4 `
  --opponent heuristic_path `
  --num_envs 8 `
  --rollout_games_per_update 32 `
  --updates 50 `
  --lr 2e-5 `
  --clip_range 0.10 `
  --entropy_coef 0.01 `
  --kl_to_bc_coef 0.02 `
  --target_kl 0.03 `
  --ppo_epochs 2 `
  --minibatch_size 512 `
  --eval_interval_updates 5 `
  --save_interval_updates 5 `
  --eval_games 10 `
  --seed 42 `
  --device cpu `
  --heuristic_path .\orbit_wars_base.py `
  --max_episode_steps 500
```

## Evaluate PPO

```powershell
python -m orbit_ppo_training.eval_ppo `
  --checkpoint .\ppo_runs\compact_bc_v1_ppo\best `
  --opponent heuristic_path `
  --players 4 `
  --num_games 50 `
  --out_dir .\ppo_eval_runs\compact_bc_v1_ppo_vs_heuristic `
  --seed 2000 `
  --device cpu
```

## Evaluate PPO and save replays

```powershell
python -m orbit_ppo_training.eval_ppo `
  --checkpoint .\ppo_runs\compact_bc_v1_ppo\best `
  --opponent heuristic_path `
  --players 4 `
  --num_games 50 `
  --out_dir .\ppo_eval_runs\compact_bc_v1_ppo_vs_heuristic `
  --seed 2000 `
  --device cpu `
  --save_replays 3 `
  --save_html_replays
```

## Run tests

```powershell
python -m pytest
```

## Run only BC tests

```powershell
python -m pytest tests/test_bc_training.py tests/test_bc_eval_pipeline.py
```

```

Use this as `README.md`. Note: if the compact single-feature refactor has not been implemented yet, this README describes the intended cleaned architecture, not the current versioned feature code.
```
