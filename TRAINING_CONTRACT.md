# Orbit Wars BC/RL Training Contract

This contract is the shared interface between dataset creation, behavior cloning, and RL fine-tuning.

## Action space

The model does **not** predict raw `[from_planet_id, angle, num_ships]`.

The model predicts per owned source planet:

```text
source_slot is fixed by the row/source mask
 target_head: 0..63 planet slots, 64 = no-op
 amount_head: 0 none, 1 one_ship, 2 capture_plus_one, 3 quarter, 4 half, 5 three_quarter, 6 all
```

The geometry skeleton converts:

```text
source_slot + target_slot + decoded amount_bin -> [from_planet_id, angle, num_ships]
```

## Why per-source rows

Orbit Wars allows multiple launches per turn. A first stable model should emit one decision per owned source planet per turn. This captures coordinated multi-planet turns without making the policy generate variable-length action lists.

If a replay has multiple launches from the same source in one turn, the dataset keeps the largest launch as the v1 primary label and marks the row with:

```text
ambiguous_multi_launch=true
drop_for_v1_bc=true
```

Those rows are retained for later sequence/multi-launch modeling but should be excluded from the first BC run.

## Primary BC data

Use `source_turn_rows.jsonl` as the authoritative supervised labels:

- `source_slot`
- `target_slot_label`
- `amount_bin_label`
- `train_weight`
- `drop_for_v1_bc`

For dense model training, use `dense_bc_arrays.npz`:

- `planet_features`: `[N, 64, planet_feature_dim]`
- `target_labels`: `[N, 64]`, no-op slot = 64
- `amount_labels`: `[N, 64]`
- `source_mask`: `[N, 64]`

The loss must be masked by `source_mask` and should ignore rows marked `drop_for_v1_bc=true` when using JSONL labels.

## Baseline ranker data

Use `pair_rank_rows.jsonl` first to validate the dataset:

```text
group_uid = one source planet at one state
label = 1 for the chosen target/no-op, 0 for alternatives
```

A simple pairwise MLP/ranker should beat trivial nearest/capture baselines before investing in a transformer.

## Target inference quality

`target_inference_method=first_contact` is strongest.

`target_inference_method=angular_nearest` is fallback when raw replay angle did not produce a first contact inside the geometry horizon. These rows are useful but lower confidence.

Recommended v1 BC filter:

```text
drop_for_v1_bc == false
winner_action == true OR train_weight-based sampling
```

## RL compatibility

PPO must use the same decoder:

```text
policy logits -> target_slot/amount_bin -> geometry skeleton -> raw env moves
```

Do not add an angle head until logs show:

```text
correct target + correct amount, but fleet misses target often
```

## Value pretraining

Initial value labels can use `final_reward` from replay:

```text
winner = +1
loser = -1
```

For 4-player ranking later, replace this with normalized final ship score/rank if available from replay analysis.
