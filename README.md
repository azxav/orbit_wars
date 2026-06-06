# Orbit Wars Dataset + Pre-Training Preparation Pipeline

This project contains both pieces needed to prepare BC/RL-compatible Orbit Wars
datasets:

- `orbit_geometry_skeleton/`: deterministic game geometry and mechanics.
- `orbit_training_prep/`: replay loading, label inference, dataset building, and validation.

The packages live side by side so the training pipeline can import the geometry
skeleton directly. No separate checkout or hard-coded `PYTHONPATH` is required
when commands are run from this project root.

## Core split

ML decides:

- per owned source planet: target/no-op
- amount bin
- priority/timing later

Geometry skeleton decides:

- launch angle
- intercept/ETA
- sun/bounds/collision feasibility

The dataset and training contract use the same action space to prevent a mismatch
between behavior cloning and PPO fine-tuning.

## Main files

- `orbit_geometry_skeleton/geometry_skeleton.py` - public geometry/mechanics interface.
- `orbit_training_prep/replay_io.py` - Kaggle replay loader and normalized per-player step iterator.
- `orbit_training_prep/target_inference.py` - maps raw `[from_planet_id, angle, ships]` moves to `(source_slot, inferred_target_slot, amount_bin)` labels using fast projected-angle geometry.
- `orbit_training_prep/features.py` - stable planet/pair features for baseline checks.
- `orbit_training_prep/dataset_builder.py` - builds all JSONL/NPZ datasets.
- `orbit_training_prep/validate_dataset.py` - sanity report for label quality and trainability.
- `TRAINING_CONTRACT.md` - exact model/training assumptions to keep IL and RL aligned.

## Setup

From this folder:

```powershell
python -m pip install -e .
```

## Important replay alignment

Kaggle replay rows store `action` beside the post-action observation. The builder
pairs `action` from replay index `t` with observation from `t-1`. Without this,
ship counts become invalid after launches.

## Build a dataset

Put replay JSON files in `replays/` under this project folder, then run:

```powershell
python -m orbit_training_prep.dataset_builder `
  --replay .\replays `
  --out-dir .\orbit_dataset_work\combined `
  --horizon 160
```

`--replay` can also point to one replay JSON file if you want to build a dataset
from a single episode.

## Validate

```powershell
python -m orbit_training_prep.validate_dataset `
  --out-dir .\orbit_dataset_work\combined
```

## Output datasets

- `launch_rows.jsonl`: one row per actual replay launch. Used to debug action inference.
- `source_turn_rows.jsonl`: primary BC dataset, one row per owned source per player step.
- `pair_rank_rows.jsonl`: pairwise source-target candidate dataset for an MLP/ranker baseline.
- `dense_bc_arrays.npz`: fixed-size arrays for direct neural training.
- `metadata.json`: schema, action space, feature names, stats.
- `validation_report.{json,md}`: quality checks.

## First training usage

1. Train a pairwise MLP/ranker on `pair_rank_rows.jsonl` to verify labels.
2. Train the entity policy on `dense_bc_arrays.npz` / `source_turn_rows.jsonl`.
3. Decode model output through the geometry skeleton, not through learned angle prediction.
4. Initialize PPO from the BC checkpoint.

## Training IO helpers

`orbit_training_prep/training_io.py` provides framework-neutral loaders:

```python
from orbit_training_prep.training_io import load_dense_bc_arrays, make_pair_rank_numpy

bc = load_dense_bc_arrays(r".\orbit_dataset_work\combined")
rank = make_pair_rank_numpy(r".\orbit_dataset_work\combined")
```

Use these in JAX, PyTorch, LightGBM, or quick NumPy checks without changing the
dataset schema.
