# JAX PPO Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume JAX PPO training from the last completed update across completed passes and process interruptions, including 2P/4P mode switches.

**Architecture:** Keep `params.npz` as the portable model/eval artifact and add an optional `trainer_state.npz` sidecar for PPO-only state. On resume, always restore learner params and optimizer/RNG/update counters when compatible; restore vector env state only when env shape and player mode match, otherwise reset envs for the new mode while preserving the learner.

**Tech Stack:** Python dataclasses, JAX/Optax PyTrees, existing JAX PPO checkpoint format, pytest.

---

### Task 1: Regression Coverage

**Files:**
- Modify: `tests/test_ppo_jax_readiness.py`

- [x] Add a test that runs a 1-update PPO pass twice against the same `out_dir` and expects the second pass to append update `2`, preserve the env episode step, and write `latest/trainer_state.npz`.
- [x] Add a test that runs a 4P pass followed by a 2P pass in the same `out_dir` and expects the second pass to resume as update `2` while resetting incompatible env state.

### Task 2: Training-State Sidecar

**Files:**
- Modify: `orbit_ppo_jax/checkpointing.py`

- [x] Add save/load helpers for optimizer state, PRNG key, vector env state, state-bank cycle index, update index, cumulative env steps, best score, and env compatibility metadata.
- [x] Rebuild saved PyTrees against runtime templates so Optax and `EnvState` structure remain explicit and version-sensitive.

### Task 3: Resume Training Loop

**Files:**
- Modify: `orbit_ppo_jax/train.py`

- [x] Add `resume` and `resume_from` config fields with CLI controls `--no_resume` and `--resume_from`.
- [x] Auto-resume from `out_dir/latest` when `params.npz` exists.
- [x] Treat `--updates` as additional updates after the resumed global update index.
- [x] Save `trainer_state.npz` after every completed update.
- [x] Include resume metadata in metrics.

### Task 4: Player-Mode Stability

**Files:**
- Modify: `orbit_ppo_jax/train.py`

- [x] Restore env state only when player count, env count, episode steps, comet setting, state-bank path, and state-bank mode match.
- [x] Reset env state on 2P/4P switches while preserving params, optimizer state, RNG, and global update count.
- [x] Avoid reusing PFSP manifests with mismatched player counts; keep legacy `league/manifest.json` for fresh compatible runs and use `league/<players>p/manifest.json` when switching modes.

### Task 5: Verification

**Files:**
- Verify: `tests/test_ppo_jax_readiness.py`
- Verify: `tests/test_ppo_jax_pfsp.py`

- [ ] Run focused continuation tests.
- [ ] Run checkpoint/PFSP readiness tests as available in the local environment.
