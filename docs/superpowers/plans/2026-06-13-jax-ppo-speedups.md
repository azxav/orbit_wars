# JAX PPO Speedups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Increase JAX PPO training throughput by removing avoidable feature-builder and policy-forward overhead identified in `deep-research-report.md`.

**Architecture:** Keep the existing compiled PPO update shape and training semantics unchanged. Refactor feature construction to batch source-pair feature rows with `jax.vmap`, and refactor rollout action sampling so the sampled target/amount forward passes are reused to compute log-probability, value, and entropy instead of running a third identical encoder pass.

**Tech Stack:** Python, JAX, Optax, existing `orbit_ppo_jax` functional BC policy, pytest.

---

## File Structure

- Modify `orbit_ppo_jax/features.py`: add `_pair_features_for_sources()` as the single batched implementation for source-pair features and call it from `build_bc_features_for_seat()`.
- Modify `orbit_ppo_jax/train.py`: add `_policy_eval_from_forward_outputs()` and use it from both `_policy_eval()` and `_policy_sample_act()`.
- Modify `tests/test_ppo_jax_readiness.py`: add regression coverage for batched pair feature equivalence and the two-forward rollout sampling path.

### Task 1: Vectorize Pair Feature Construction

**Files:**
- Modify: `tests/test_ppo_jax_readiness.py`
- Modify: `orbit_ppo_jax/features.py`

- [x] **Step 1: Add a failing helper test**

Add `test_batched_pair_features_match_per_source_rows()` to `tests/test_ppo_jax_readiness.py`. It should import `_pair_features_for_sources`, build a small manual state, compare the helper output against an explicit stack of `pair_features_for_source()` calls for two source slots, and assert all rows match.

- [x] **Step 2: Run the test to verify red**

Run:

```bash
pytest tests/test_ppo_jax_readiness.py::test_batched_pair_features_match_per_source_rows -q
```

Expected before implementation: import failure because `_pair_features_for_sources` does not exist.

- [x] **Step 3: Implement the batched helper**

In `orbit_ppo_jax/features.py`, add:

```python
def _pair_features_for_sources(planet_features, target_state_features, source_slots, amount_mask):
    return jax.vmap(
        lambda source_slot, source_amount_mask: pair_features_for_source(
            planet_features,
            target_state_features,
            source_slot,
            source_amount_mask,
        )
    )(source_slots, amount_mask)
```

Then replace the Python list comprehension in `build_bc_features_for_seat()` with this helper.

- [x] **Step 4: Verify feature coverage**

Run:

```bash
pytest tests/test_ppo_jax_readiness.py::test_batched_pair_features_match_per_source_rows tests/test_ppo_jax_readiness.py::test_jax_feature_contract_matches_dense_python_features tests/test_ppo_jax_readiness.py::test_compact_features_select_top_ship_active_owned_sources tests/test_ppo_jax_pfsp.py::test_dynamic_seat_features_are_jittable -q
```

Expected after implementation: all pass.

### Task 2: Reuse Rollout Policy Forward Outputs

**Files:**
- Modify: `tests/test_ppo_jax_readiness.py`
- Modify: `orbit_ppo_jax/train.py`

- [x] **Step 1: Add failing call-count coverage**

Add `test_policy_sample_act_uses_two_bc_forwards()` to `tests/test_ppo_jax_readiness.py`. It should load a tiny BC checkpoint, monkeypatch `orbit_ppo_jax.train.bc_forward` with a counting wrapper, call `_policy_sample_act()` once on a manual 2-player state with `source_cap=2`, and assert `calls == 2`.

- [x] **Step 2: Run the test to verify red**

Run:

```bash
pytest tests/test_ppo_jax_readiness.py::test_policy_sample_act_uses_two_bc_forwards -q
```

Expected before implementation: assertion failure with 3 calls.

- [x] **Step 3: Add reusable evaluation helper**

In `orbit_ppo_jax/train.py`, add `_policy_eval_from_forward_outputs(params, features, target_out, amount_out, target_idx, amount_idx)`. It should apply masks, compute target and amount log-probabilities, entropy, and value using `value_apply(params["value"], target_out["global_ctx"][0])`.

- [x] **Step 4: Route both policy paths through the helper**

Update `_policy_eval()` to call `bc_forward()` once with `target_idx`, then pass that output as both `target_out` and `amount_out`. Update `_policy_sample_act()` to compute `target_out` and `amount_out` exactly once each, sample from them, and pass both outputs to `_policy_eval_from_forward_outputs()`.

- [x] **Step 5: Verify policy and training coverage**

Run:

```bash
pytest tests/test_ppo_jax_readiness.py::test_policy_sample_act_uses_two_bc_forwards tests/test_ppo_jax_readiness.py::test_tiny_train_writes_checkpoint_and_metrics tests/test_ppo_jax_readiness.py::test_tiny_train_with_small_source_cap_records_compact_metrics tests/test_ppo_jax_pfsp.py::test_tiny_train_pfsp_writes_manifest_metrics_and_latest_checkpoint -q
```

Expected after implementation: all pass.

### Task 3: Final Verification

**Files:**
- Verify: `tests/test_ppo_jax_readiness.py`
- Verify: `tests/test_ppo_jax_pfsp.py`

- [x] Run:

```bash
pytest tests/test_ppo_jax_readiness.py tests/test_ppo_jax_pfsp.py -q
```

Expected: all pass in the local environment.

## Self-Review

- Spec coverage: addresses the report's explicit `build_bc_features_for_seat()` Python loop bottleneck and removes one redundant BC policy forward from each learner rollout action sample.
- Deliberately deferred: mixed precision, rematerialization, pmap, and mask recomputation are higher-risk training-surface changes and should be benchmarked separately after these low-risk compute reductions.
- Placeholder scan: all tasks name exact files, functions, and verification commands.
- Type consistency: helper names and signatures match the existing feature and policy code.
