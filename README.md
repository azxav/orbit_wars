# Orbit Wars BC / PPO Training Pipeline

This repository contains the behavior-cloning, local-evaluation, and PPO training pipeline for an Orbit Wars Kaggle agent.

The ML policy uses one action contract everywhere:

```text
owned source planet -> target/no-op + amount bin -> geometry decoder -> environment move
```

The model does **not** directly predict launch angles. The geometry layer handles angle search, ETA, sun/bounds feasibility, and decoding.

---

## Current dataset contract

The dataset format is now:

```text
source_turn_memmap_v1
```

The old training artifact is removed:

```text
dense_bc_arrays.npz
```

Do not train from old dense datasets. Rebuild datasets after this change.

---

## Dataset builder backends

The dataset builder has two backends:

```text
--backend lite   # default, fast movement-cache heuristic backend
--backend exact  # old exact simulator-based backend
```

### Lite backend, recommended

The Lite backend is the default path for large replay datasets. It removes the expensive exact source-target-amount simulation from dataset construction and uses an `orbit_lite` movement-cache heuristic backend instead.

It considers dynamic movement geometry through:

```text
future planet positions
cross-time source-target distances
movement-cache ETA approximation
projected owner / garrison near estimated arrival
```

It does **not** perform full exact collision / first-hit simulation:

```text
no ExactTargetSimulator
no exact first-hit search
no exact source-target-amount aim grid
no exact sun/bounds/comet/body collision resolution
```

Use Lite for fast training dataset generation. Use Exact only for small audit datasets or when you explicitly need the old simulator-derived labels.

### Player-perspective canonicalization, default

The dataset builder canonicalizes every player observation into a shared P0 frame by default. This is required to reduce map/seat bias for both BC and PPO.

For a sample from player `p`, the builder now:

```text
rotates planet/fleet/comet coordinates by -2*pi*p/num_players around board center
remaps owner ids so the acting player becomes owner 0
reorders planet slots deterministically in canonical coordinates
rotates replay launch angles into the same canonical frame
trains labels/features from player_id=0 after the transform
```

This means P0/P1 in 2-player games and P0/P1/P2/P3 in 4-player games are presented to the model in the same coordinate convention. Live BC evaluation and PPO rollouts apply the same transform before policy inference, then rotate decoded moves back to the environment frame.

Canonicalization is enabled by default. Only disable it for legacy comparisons:

```bash
--no-canonicalize-perspective
```

### Proportion correction, default

The dataset builder also applies proportion correction by default to reduce replay and source-turn label bias:

```text
2-player / 4-player replay mix: target 50 / 50
noop / op source-turn rows: target 40 / 60
fleet amount bins: automatic square-root-smoothed balancing over non-noop rows
```

Replay balancing happens before worker sharding. Source-turn balancing happens after the final `samples/*.npy` dataset is materialized, so serial and parallel builds use the same global quotas. If the requested ratios are impossible because one class is missing or rare, the builder keeps the closest feasible deterministic subset and records the shortfall in `metadata.json`.

The relevant metadata is under:

```text
proportion_correction
proportion_correction.replay_selection
proportion_correction.sample_balance
proportion_correction.unbalanced_stats
```

The final top-level `sample_count`, `amount_bin_counts`, and `stats.source_turn_rows` describe the train-ready filtered dataset. Pre-filter counts are preserved in `proportion_correction.unbalanced_stats`.

Use the default correction for normal training datasets. Disable it only for raw-distribution audits or legacy comparisons:

```bash
--no-balance-proportions
```

Deterministic balancing can be controlled with:

```bash
--balance-seed 42
--noop-ratio 0.40
--op-ratio 0.60
```

Parallel replay-level building is now supported with `--workers N`. Each worker builds a compact source-turn shard, then the main process merges `states/*.npy` and `samples/*.npy` while correcting `samples/state_index.npy` offsets. CUDA exact builds still fall back to serial to avoid multiple GPU contexts.

---

## Project structure

```text
orbit_wars/
├── orbit_training_prep/
│   ├── dataset_builder.py
│   ├── source_turn_store.py
│   ├── training_io.py
│   ├── lite_backend.py
│   ├── canonical.py
│   ├── features.py
│   ├── split_episodes.py
│   ├── materialize_splits.py
│   └── validate_dataset.py
│
├── orbit_bc_training/
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── train_bc_policy.py
│   └── eval_bc_policy.py
│
├── orbit_bc_eval/
│   ├── bc_agent_runtime.py
│   ├── run_local_matches.py
│   └── eval_report.py
│
├── orbit_ppo_training/
│   ├── policy.py
│   ├── train_ppo.py
│   ├── eval_ppo.py
│   └── smoke_test.py
│
├── orbit_geometry_skeleton/
├── orbit_jax_env/
├── orbit_lite/                    # movement-cache heuristic backend
├── tests/
└── README.md
```

---

## Dataset output layout

A built dataset directory contains:

```text
orbit_dataset_work/combined_lite/
├── metadata.json
├── states/
│   ├── planet_features.npy
│   ├── global_features.npy
│   ├── target_state_features.npy
│   └── episode_id.npy
├── samples/
│   ├── state_index.npy
│   ├── source_slot.npy
│   ├── target_label.npy
│   ├── amount_label.npy
│   ├── sample_weight.npy
│   ├── step.npy
│   ├── pair_features.npy
│   ├── target_mask.npy
│   └── amount_mask.npy
└── debug/                         # only when --write-debug-jsonl is used
    ├── launch_rows.jsonl
    ├── source_turn_rows.jsonl
    └── state_rows.jsonl
```

Main shapes:

```text
states/planet_features.npy          float32 [N_states, 64, 16]
states/global_features.npy          float32 [N_states, 10]
states/target_state_features.npy    float32 [N_states, 64, 9]
states/episode_id.npy               str     [N_states]

samples/state_index.npy             uint32  [N_samples]
samples/source_slot.npy             uint8   [N_samples]
samples/target_label.npy            uint8   [N_samples]
samples/amount_label.npy            uint8   [N_samples]
samples/sample_weight.npy           float32 [N_samples]
samples/step.npy                    uint16  [N_samples]
samples/pair_features.npy           float16 [N_samples, 65, 15]
samples/target_mask.npy             bool    [N_samples, 65]
samples/amount_mask.npy             bool    [N_samples, 7]
```

Each sample is one **source-turn**:

```text
one owned source planet at one player observation state
```

---

## Compact pair feature contract

`pair_features.npy` uses 15 source-target features:

```text
capture_ratio
surplus_after_capture
roi_prod_per_ship
distance
angle_sin
angle_cos
geom_viable_amount_frac
safe_sendable_ships
post_send_frac_capture
our_eta_norm
enemy_ships_before_our_arrival
friendly_ships_before_our_arrival
projected_garrison_at_our_arrival
projected_owner_at_our_arrival
target_capture_margin_at_arrival
```

This replaces the old 30-feature dense pair tensor.

Old:

```text
[state, 64 sources, 65 targets, 30 features]
```

New:

```text
[source-turn sample, 65 targets, 15 features]
```

---

## Environment setup

Run commands from the repository root.

### PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pytest torch
$env:PYTHONPATH = (Get-Location).Path
```

### Ubuntu / Bash

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pytest torch
export PYTHONPATH="$PWD"
```

Install Kaggle/local environment dependencies separately if running full environment matches or comet parity tests.

---

## Recommended workflow

### 1. Build combined dataset with Lite backend, recommended

PowerShell:

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:TORCH_NUM_THREADS="1"

python -m orbit_training_prep.dataset_builder `
  --replay .\replays `
  --out-dir .\orbit_dataset_work\combined_lite `
  --horizon 160 `
  --device cpu `
  --batch-size 256 `
  --backend lite `
  --workers 16
```

Ubuntu / Bash:

```bash
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
TORCH_NUM_THREADS=1 \
python -m orbit_training_prep.dataset_builder \
  --replay ./replays \
  --out-dir ./orbit_dataset_work/combined_lite \
  --horizon 160 \
  --device cpu \
  --batch-size 256 \
  --backend lite \
  --workers 16
```

This is the default high-throughput path. It uses movement-cache heuristics from `orbit_lite`, avoids exact first-hit simulation during dataset construction, and parallelizes across replay files when `--workers > 1`.

It also applies the default anti-bias proportion correction. The output dataset is already filtered to the selected 2P/4P replay mix, noop/op ratio, and softened amount-bin distribution.

For a 128 GB RAM CPU build machine, start with `--workers 16`; increase toward 20-24 only if CPU utilization and disk throughput remain healthy. Keep `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `TORCH_NUM_THREADS` at `1` so workers do not oversubscribe CPU threads.

Legacy comparison build without canonicalization:

PowerShell:

```powershell
python -m orbit_training_prep.dataset_builder `
  --replay .\replays `
  --out-dir .\orbit_dataset_work\combined_lite_legacy_perspective `
  --horizon 160 `
  --device cpu `
  --batch-size 256 `
  --backend lite `
  --workers 16 `
  --no-canonicalize-perspective
```

Ubuntu / Bash:

```bash
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
TORCH_NUM_THREADS=1 \
python -m orbit_training_prep.dataset_builder \
  --replay ./replays \
  --out-dir ./orbit_dataset_work/combined_lite_legacy_perspective \
  --horizon 160 \
  --device cpu \
  --batch-size 256 \
  --backend lite \
  --workers 16 \
  --no-canonicalize-perspective
```

Use this only to compare old seat/map-biased behavior against the default canonicalized dataset.

Raw-distribution build without proportion correction:

PowerShell:

```powershell
python -m orbit_training_prep.dataset_builder `
  --replay .\replays `
  --out-dir .\orbit_dataset_work\combined_lite_unbalanced `
  --horizon 160 `
  --device cpu `
  --batch-size 256 `
  --backend lite `
  --workers 16 `
  --no-balance-proportions
```

Ubuntu / Bash:

```bash
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
TORCH_NUM_THREADS=1 \
python -m orbit_training_prep.dataset_builder \
  --replay ./replays \
  --out-dir ./orbit_dataset_work/combined_lite_unbalanced \
  --horizon 160 \
  --device cpu \
  --batch-size 256 \
  --backend lite \
  --workers 16 \
  --no-balance-proportions
```

Use this only when you need to inspect the original replay/source-turn distribution.

Optional: build a small exact audit dataset:

PowerShell:

```powershell
python -m orbit_training_prep.dataset_builder `
  --replay .\replays `
  --out-dir .\orbit_dataset_work\combined_lite_exact_small `
  --horizon 160 `
  --device cpu `
  --batch-size 256 `
  --backend exact `
  --max-files 5
```

Ubuntu / Bash:

```bash
python -m orbit_training_prep.dataset_builder \
  --replay ./replays \
  --out-dir ./orbit_dataset_work/combined_lite_exact_small \
  --horizon 160 \
  --device cpu \
  --batch-size 256 \
  --backend exact \
  --max-files 5
```

With debug JSONL rows:

PowerShell:

```powershell
python -m orbit_training_prep.dataset_builder `
  --replay .\replays `
  --out-dir .\orbit_dataset_work\combined_lite_debug `
  --horizon 160 `
  --device cpu `
  --batch-size 256 `
  --backend lite `
  --write-debug-jsonl
```

Ubuntu / Bash:

```bash
python -m orbit_training_prep.dataset_builder \
  --replay ./replays \
  --out-dir ./orbit_dataset_work/combined_lite_debug \
  --horizon 160 \
  --device cpu \
  --batch-size 256 \
  --backend lite \
  --write-debug-jsonl
```

Exact backend with CUDA target inference, one worker:

PowerShell:

```powershell
python -m orbit_training_prep.dataset_builder `
  --replay .\replays `
  --out-dir .\orbit_dataset_work\combined_lite_cuda `
  --horizon 160 `
  --device cuda `
  --batch-size 256 `
  --backend exact `
  --workers 1
```

Ubuntu / Bash:

```bash
python -m orbit_training_prep.dataset_builder \
  --replay ./replays \
  --out-dir ./orbit_dataset_work/combined_lite_cuda \
  --horizon 160 \
  --device cuda \
  --batch-size 256 \
  --backend exact \
  --workers 1
```

---

### 2. Validate dataset

PowerShell:

```powershell
python -m orbit_training_prep.validate_dataset `
  --out-dir .\orbit_dataset_work\combined_lite
```

Ubuntu / Bash:

```bash
python -m orbit_training_prep.validate_dataset \
  --out-dir ./orbit_dataset_work/combined_lite
```

This writes:

```text
validation_report.json
validation_report.md
```

---

### 3. Split episodes

PowerShell:

```powershell
python -m orbit_training_prep.split_episodes `
  --dataset_root .\orbit_dataset_work\combined_lite `
  --valid_frac 0.15 `
  --seed 42 `
  --out .\orbit_dataset_work\splits.json
```

Ubuntu / Bash:

```bash
python -m orbit_training_prep.split_episodes \
  --dataset_root ./orbit_dataset_work/combined_lite \
  --valid_frac 0.15 \
  --seed 42 \
  --out ./orbit_dataset_work/splits.json
```

---

### 4. Materialize train/valid split

PowerShell:

```powershell
python -m orbit_training_prep.materialize_splits `
  --dataset_root .\orbit_dataset_work\combined_lite `
  --splits .\orbit_dataset_work\splits.json `
  --out .\orbit_dataset_work\split_dataset
```

Ubuntu / Bash:

```bash
python -m orbit_training_prep.materialize_splits \
  --dataset_root ./orbit_dataset_work/combined_lite \
  --splits ./orbit_dataset_work/splits.json \
  --out ./orbit_dataset_work/split_dataset
```

This creates:

```text
orbit_dataset_work/split_dataset/train/
orbit_dataset_work/split_dataset/valid/
```

Each split keeps the same `source_turn_memmap_v1` layout.

---

### 5. Train BC model

PowerShell:

```powershell
python -m orbit_bc_training.train_bc_policy `
  --train_dir .\orbit_dataset_work\split_dataset\train `
  --valid_dir .\orbit_dataset_work\split_dataset\valid `
  --out_dir .\bc_checkpoints\lite_bc_v1 `
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

Ubuntu / Bash:

```bash
python -m orbit_bc_training.train_bc_policy \
  --train_dir ./orbit_dataset_work/split_dataset/train \
  --valid_dir ./orbit_dataset_work/split_dataset/valid \
  --out_dir ./bc_checkpoints/lite_bc_v1 \
  --batch_size 512 \
  --epochs 20 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --hidden_size 128 \
  --num_layers 2 \
  --num_heads 4 \
  --mlp_size 256 \
  --dropout 0.0 \
  --seed 42 \
  --device auto \
  --num_workers 4
```

Outputs:

```text
bc_checkpoints/lite_bc_v1/latest/checkpoint.pt
bc_checkpoints/lite_bc_v1/best/checkpoint.pt
bc_checkpoints/lite_bc_v1/metrics.jsonl
```

---

### 6. Evaluate BC offline

PowerShell:

```powershell
python -m orbit_bc_training.eval_bc_policy `
  --checkpoint .\bc_checkpoints\lite_bc_v1\best\checkpoint.pt `
  --valid_dir .\orbit_dataset_work\split_dataset\valid `
  --out .\bc_checkpoints\lite_bc_v1\offline_eval.json `
  --device auto
```

Ubuntu / Bash:

```bash
python -m orbit_bc_training.eval_bc_policy \
  --checkpoint ./bc_checkpoints/lite_bc_v1/best/checkpoint.pt \
  --valid_dir ./orbit_dataset_work/split_dataset/valid \
  --out ./bc_checkpoints/lite_bc_v1/offline_eval.json \
  --device auto
```

Offline validation checks imitation accuracy. It is not a replacement for local gameplay evaluation.

---

### 7. Evaluate BC in local matches

PowerShell:

```powershell
python -m orbit_bc_eval.run_local_matches `
  --bc_checkpoint .\bc_checkpoints\lite_bc_v1\best\checkpoint.pt `
  --opponent heuristic_path `
  --heuristic_path .\orbit_wars_base.py `
  --players both `
  --num_games 20 `
  --seed_start 1000 `
  --out_dir .\bc_eval_runs\lite_bc_v1_vs_heuristic `
  --device cpu `
  --debug_game
```

Ubuntu / Bash:

```bash
python -m orbit_bc_eval.run_local_matches \
  --bc_checkpoint ./bc_checkpoints/lite_bc_v1/best/checkpoint.pt \
  --opponent heuristic_path \
  --heuristic_path ./orbit_wars_base.py \
  --players both \
  --num_games 20 \
  --seed_start 1000 \
  --out_dir ./bc_eval_runs/lite_bc_v1_vs_heuristic \
  --device cpu \
  --debug_game
```

Save HTML replays:

PowerShell:

```powershell
python -m orbit_bc_eval.run_local_matches `
  --bc_checkpoint .\bc_checkpoints\lite_bc_v1\best\checkpoint.pt `
  --opponent heuristic_path `
  --heuristic_path .\orbit_wars_base.py `
  --players 2 `
  --num_games 5 `
  --seed_start 42 `
  --out_dir .\bc_eval_runs\lite_bc_visual `
  --device cpu `
  --render_html `
  --render_html_games 3
```

Ubuntu / Bash:

```bash
python -m orbit_bc_eval.run_local_matches \
  --bc_checkpoint ./bc_checkpoints/lite_bc_v1/best/checkpoint.pt \
  --opponent heuristic_path \
  --heuristic_path ./orbit_wars_base.py \
  --players 2 \
  --num_games 5 \
  --seed_start 42 \
  --out_dir ./bc_eval_runs/lite_bc_visual \
  --device cpu \
  --render_html \
  --render_html_games 3
```

---

### 8. PPO smoke test

PowerShell:

```powershell
python -m orbit_ppo_training.smoke_test `
  --bc_checkpoint .\bc_checkpoints\lite_bc_v1\best\checkpoint.pt `
  --out_dir .\ppo_runs\smoke_lite_bc_v1 `
  --device cpu
```

Ubuntu / Bash:

```bash
python -m orbit_ppo_training.smoke_test \
  --bc_checkpoint ./bc_checkpoints/lite_bc_v1/best/checkpoint.pt \
  --out_dir ./ppo_runs/smoke_lite_bc_v1 \
  --device cpu
```

---

### 9. Train PPO from BC

PowerShell:

```powershell
python -m orbit_ppo_training.train_ppo `
  --bc_checkpoint .\bc_checkpoints\lite_bc_v1\best\checkpoint.pt `
  --out_dir .\ppo_runs\lite_bc_v1_ppo `
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

Ubuntu / Bash:

```bash
python -m orbit_ppo_training.train_ppo \
  --bc_checkpoint ./bc_checkpoints/lite_bc_v1/best/checkpoint.pt \
  --out_dir ./ppo_runs/lite_bc_v1_ppo \
  --players 4 \
  --opponent heuristic_path \
  --num_envs 8 \
  --rollout_games_per_update 32 \
  --updates 50 \
  --lr 2e-5 \
  --clip_range 0.10 \
  --entropy_coef 0.01 \
  --kl_to_bc_coef 0.02 \
  --target_kl 0.03 \
  --ppo_epochs 2 \
  --minibatch_size 512 \
  --eval_interval_updates 5 \
  --save_interval_updates 5 \
  --eval_games 10 \
  --seed 42 \
  --device cpu \
  --heuristic_path ./orbit_wars_base.py \
  --max_episode_steps 500
```

---

### 10. Evaluate PPO

PowerShell:

```powershell
python -m orbit_ppo_training.eval_ppo `
  --checkpoint .\ppo_runs\lite_bc_v1_ppo\best `
  --opponent heuristic_path `
  --players 4 `
  --num_games 50 `
  --out_dir .\ppo_eval_runs\lite_bc_v1_ppo_vs_heuristic `
  --seed 2000 `
  --device cpu `
  --save_replays 3 `
  --save_html_replays
```

Ubuntu / Bash:

```bash
python -m orbit_ppo_training.eval_ppo \
  --checkpoint ./ppo_runs/lite_bc_v1_ppo/best \
  --opponent heuristic_path \
  --players 4 \
  --num_games 50 \
  --out_dir ./ppo_eval_runs/lite_bc_v1_ppo_vs_heuristic \
  --seed 2000 \
  --device cpu \
  --save_replays 3 \
  --save_html_replays
```

---

### 11. CUDA JAX PPO from BC

JAX CUDA training is intended for WSL/Linux. Native Windows JAX currently runs CPU-only in this workspace, so use the setup script from WSL:

```bash
cd /mnt/d/Projects/orbit_dataset_prep
bash scripts/setup_jax_cuda_wsl.sh
source .venv-jax/bin/activate
python scripts/check_jax_cuda.py --require-cuda
```

Run one short BC-initialized PPO update on GPU against the pure-JAX proxy opponent, with checkpoint selection evaluated against `orbit_wars_base.py`:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 python -m orbit_ppo_jax.train \
  --require_cuda \
  --bc_checkpoint bc_checkpoints/lite_bc_v1_500/best/checkpoint.pt \
  --out_dir ppo_runs/jax_compact_bc_v1_smoke \
  --opponent jax_proxy \
  --eval_heuristic_path orbit_wars_base.py \
  --players 4 \
  --envs 8 \
  --rollout_steps 32 \
  --episode_steps 500 \
  --updates 1 \
  --eval_games 2
```

Longer training run:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 python -m orbit_ppo_jax.train \
  --require_cuda \
  --bc_checkpoint bc_checkpoints/lite_bc_v1_500/best/checkpoint.pt \
  --out_dir ppo_runs/jax_compact_bc_v1 \
  --opponent jax_proxy \
  --eval_heuristic_path orbit_wars_base.py \
  --players 4 \
  --envs 32 \
  --rollout_steps 128 \
  --episode_steps 500 \
  --updates 1000 \
  --eval_games 20 \
  --eval_interval_updates 25 \
  --save_interval_updates 25
```

To train from official Kaggle initial maps instead of the synthetic JAX reset template, build a state bank and pass it to the trainer:

```bash
python -m orbit_jax_env.build_official_state_bank \
  --out data/official_state_bank_4p.npz \
  --players 4 \
  --seeds 0:1024
```

```bash
python -m orbit_ppo_jax.train \
  --bc_checkpoint bc_checkpoints/lite_bc_v1_500/best/checkpoint.pt \
  --out_dir ppo_runs/jax_compact_bc_v1 \
  --opponent jax_proxy \
  --players 4 \
  --initial_state_bank data/official_state_bank_4p.npz \
  --state_bank_mode random \
  --rollout_steps 128 \
  --episode_steps 500
```

`--steps` remains accepted as a legacy alias for `--rollout_steps`, but new runs should use `--rollout_steps` and `--episode_steps` explicitly. `--enable_comets` uses the approximate JAX-native comet schedule unless states were imported with official comet path metadata; runs record `comet_mode` and `comet_warning` in config and metrics.

Evaluate a saved JAX PPO checkpoint against the real heuristic:

```bash
python -m orbit_ppo_jax.eval_vs_heuristic \
  --checkpoint ppo_runs/jax_compact_bc_v1/latest \
  --heuristic_path orbit_wars_base.py \
  --games 4 \
  --players 4 \
  --episode_steps 500 \
  --out_dir ppo_eval_runs/jax_compact_bc_v1
```

---

## Debugging commands

### Audit exact target inference

PowerShell:

```powershell
python -m orbit_training_prep.audit_exact_targets `
  --dataset_dir .\orbit_dataset_work\combined_lite `
  --replay_dir .\replays `
  --sample_size 2000 `
  --out .\orbit_dataset_work\exact_target_audit.json `
  --seed 0 `
  --horizon 200
```

Ubuntu / Bash:

```bash
python -m orbit_training_prep.audit_exact_targets \
  --dataset_dir ./orbit_dataset_work/combined_lite \
  --replay_dir ./replays \
  --sample_size 2000 \
  --out ./orbit_dataset_work/exact_target_audit.json \
  --seed 0 \
  --horizon 200
```

### Compare metric relevance

PowerShell:

```powershell
python -m orbit_bc_eval.compare_metric_relevance `
  .\bc_eval_runs\baseline\2p\games.jsonl `
  .\bc_eval_runs\lite_bc_v1_vs_heuristic\2p\games.jsonl
```

Ubuntu / Bash:

```bash
python -m orbit_bc_eval.compare_metric_relevance \
  ./bc_eval_runs/baseline/2p/games.jsonl \
  ./bc_eval_runs/lite_bc_v1_vs_heuristic/2p/games.jsonl
```

---

## Test commands

### Focused compact-dataset verification

PowerShell:

```powershell
python -m pytest -q `
  tests/test_source_turn_store.py `
  tests/test_training_io.py `
  tests/test_bc_training.py `
  tests/test_dataset_builder_streaming.py `
  tests/test_dataset_builder_multiple_replays.py `
  tests/test_dataset_builder_workers.py `
  tests/test_dataset_builder_batching.py `
  tests/test_lite_dataset_backend.py `
  tests/test_perspective_canonicalization.py `
  tests/test_split_materialize.py `
  tests/test_validate_dataset.py `
  tests/test_pair_eta_features.py `
  tests/test_bc_eval_pipeline.py `
  tests/test_ppo_debug.py
```

Ubuntu / Bash:

```bash
python -m pytest -q \
  tests/test_source_turn_store.py \
  tests/test_training_io.py \
  tests/test_bc_training.py \
  tests/test_dataset_builder_streaming.py \
  tests/test_dataset_builder_multiple_replays.py \
  tests/test_dataset_builder_workers.py \
  tests/test_dataset_builder_batching.py \
  tests/test_lite_dataset_backend.py \
  tests/test_perspective_canonicalization.py \
  tests/test_split_materialize.py \
  tests/test_validate_dataset.py \
  tests/test_pair_eta_features.py \
  tests/test_bc_eval_pipeline.py \
  tests/test_ppo_debug.py
```

Expected focused result from the canonicalized Lite-parallel implementation:

```text
86 passed in focused verification chunks
```

The focused verification includes compact source-turn storage, Lite backend, real worker shard merging, BC dataset/training/eval, PPO rollout batching, and player-perspective canonicalization.

### Canonicalization-specific verification

PowerShell:

```powershell
python -m pytest -q tests/test_perspective_canonicalization.py
```

Ubuntu / Bash:

```bash
python -m pytest -q tests/test_perspective_canonicalization.py
```

This verifies that replay BC rows, live BC runtime, and PPO rollout observations all use the same player-perspective P0-frame transform and that decoded live moves are rotated back to the environment frame.

### Full test suite

PowerShell:

```powershell
python -m pytest -q
```

Ubuntu / Bash:

```bash
python -m pytest -q
```

Full suite may require optional environment dependencies such as `kaggle_environments`.

---

## Metrics to trust

Training metrics are useful for debugging:

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

Gameplay metrics are better for checkpoint selection:

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

---

## Rules for future changes

1. Keep one action contract for BC, PPO, and runtime.
2. Keep one feature contract in `orbit_training_prep/features.py`.
3. Rebuild datasets after changing feature names or dimensions.
4. Do not add fallback support for `dense_bc_arrays.npz`.
5. Keep player-perspective canonicalization shared across dataset building, live BC runtime, and PPO rollout collection.
6. Keep default proportion correction deterministic and recorded in dataset metadata.
7. Use local gameplay evaluation before trusting checkpoint quality.

---

## Lite backend verification commands

PowerShell:

```powershell
pytest -q tests/test_lite_dataset_backend.py
pytest -q tests/test_source_turn_store.py tests/test_training_io.py tests/test_bc_training.py
```

Ubuntu / Bash:

```bash
pytest -q tests/test_lite_dataset_backend.py
pytest -q tests/test_source_turn_store.py tests/test_training_io.py tests/test_bc_training.py
```

Known full-suite note: comet parity tests require the Kaggle environment package. If `kaggle_environments` is not installed, run the focused tests above instead of the full suite.
