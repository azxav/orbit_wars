# JAX PPO Training Pipeline Structure

The codex/jax-pfsp branch implements a single JIT‑compiled update loop that performs both environment rollouts and gradient updates on the GPU.  The training script jit‑compiles an `update` function (via `jax.jit`) that, for each update, calls a rollout routine and then computes losses and gradients.  In `rollout()`, the code uses `jax.lax.scan` over time-steps, and within each step it vectorizes (“`jax.vmap`”) over all parallel environments.  Concretely, in each step the code runs `_learner_act` to sample actions (calling the JAX BC policy network), steps the environment state (via `orbit_jax_env.step`), and accumulates observations/features into a trajectory buffer `traj`.  After the scan, it computes advantages (`_compute_gae`) and re-evaluates the policy/value for every stored state-action pair using nested `jax.vmap` (vectorizing over time *and* over the source locations).  Finally it applies `jax.value_and_grad` to this loss and updates parameters with an Optax optimizer.  All of this happens on the GPU once per update, blocking on `jax.block_until_ready` to measure time per update.  

This implementation fully exploits JAX’s parallelism (everything from feature computation to environment stepping to network inference is done inside XLA).  However, it also allocates **very large intermediate arrays**.  For example, every update accumulates a trajectory of shape `[rollout_steps, envs, …]` for many fields: planet features, global features, target-state features, masks, and pairwise features of dimension ≈ `(source_slots, P_MAX+1, 14)`.  With 8 environments and 32 steps, the sample trajectory arrays already consume significant GPU memory.  Profiling (via JAX’s profiler or manual timing) would likely show that most time is spent in the rollout (the `scan + vmap` loop) and the nested `vmap` policy/value evaluation.  

# Profiling and Bottleneck Identification

To diagnose bottlenecks, one should time the key subroutines with `jax.block_until_ready()`: (1) **rollout loop** (time to run the scan over `rollout_steps` with all envs), (2) **policy/value forward and backward pass** (time for computing log‑probs, values, and gradients), and (3) **feature generation**.  In this code, feature construction (in `build_bc_features_for_seat`) is non-trivial: it builds many per-planet and per-pair features using JAX ops.  For example, it computes pairwise distances and attack heuristics for each source slot.  Although the entire update is JIT‑compiled, the **initial JIT compile time** is amortized, so we focus on per‑update compute. The Reddit RL community notes that naive JAX code (with Python loops) can be much slower than optimized JAX code. Here, the `update` function is jitted, but it includes a Python list comprehension over source slots when building pair-features. This loop (though static length) may degrade vectorization. 

Memory is another clear constraint. The user observed that increasing `envs` beyond 8 causes OOMs.  This matches our calculation: doubling environments would double all trajectory buffers and quadruple some (pair_features scales with `envs*source_cap`). Thus the GPU is likely saturated at envs=8. Current throughput (~120 steps/s) corresponds to \(8 \times 32 / 2.04\) (env_steps / seconds) – indeed ~125 steps/sec as logged. To reach 1000–5000 sps, we must reduce work per update or parallelize further.

# Data Pipeline and Vectorization Inefficiencies

The data pipeline uses **full GPU-environment simulation and feature extraction**, which avoids CPU-GPU transfer overhead but packs everything into GPU memory.  Key inefficiencies include: 

- **Large trajectory storage**: The code stores *all* intermediate features and masks for the entire rollout. For example, `pair_features` is an array of shape `[source_slots, P_MAX+1, feature_dim]` per environment, and the code stacks these into a tensor of shape `[rollout_steps, envs, source_slots, P_MAX+1, feature_dim]`. This is very memory-intensive.  

- **Python loops in jitted code**: The `build_bc_features_for_seat` function calls `pair_features_for_source` in a Python loop (`for row in range(source_slots.shape[0])`). Even though the loop index is fixed (source_cap=16), this prevents using a single `vmap` on that axis and may cause suboptimal kernel fusion. Rewriting this as a `jax.vmap(pair_features_for_source)` would likely improve efficiency and memory access patterns, as recommended by JAX best practices. 

- **Mask storage and use**: The code builds boolean masks (`target_mask`, `amount_mask`) for allowable actions and stores them in the trajectory. These masks consume memory (size ~ `[rollout, envs, source_slots, P_MAX+1]`). In principle, some mask logic could be merged into the policy computation (e.g. using `NEG` penalties), or regenerated on the fly during loss computation, to avoid storing full mask arrays.

- **Feature precision**: All features and network activations are in full float32. Using lower precision (e.g. float16 or bfloat16) for the network and possibly parts of the feature pipeline can reduce memory. JAX supports mixed precision (e.g. setting `jax_default_matmul_precision`) which can yield ~2× throughput on GPUs with Tensor Cores.

# JAX Optimization Strategies

To drastically increase throughput, we suggest the following optimizations:

- **Mixed Precision / Reduced Precision**: Convert model parameters and most computations to float16/bfloat16. For example, use `jax.config.update("jax_default_matmul_precision", jax.lax.Precision.HIGHEST)` and cast network weights to `jnp.float16`. Mixed precision can double arithmetic throughput and halve tensor sizes on modern GPUs.

- **Increase Parallelism**:  Use more GPUs (if available) via `jax.pmap`. The code could replicate the network across devices and shard the batch (e.g. split envs across GPUs). Multi-host JAX (PMAP/sharding) is explicitly supported. Even if single‑GPU, one can try to increase `envs` + reduce `rollout_steps` or vice versa; e.g. more but shorter rollouts to feed GPU more parallel tasks. The XLA memory fraction flag (`XLA_PYTHON_CLIENT_MEM_FRACTION`) can be tuned to allow larger live batches.

- **Vectorize Python loops**: Replace any remaining Python-side loops with JAX parallel constructs. Specifically, replace the pair-feature loop by `jax.vmap(pair_features_for_source, in_axes=(None, None, 0, 0))` or similar. This ensures that the pair-features for all source slots are computed in one batched kernel, rather than sequentially. Similarly, ensure that resetting done environments inside the scan (the `reset_many` logic) is fused.

- **Checkpointing / Rematerialization**: If memory is the tightest constraint, use `jax.checkpoint` (remat) on parts of the computation. For example, one could rematerialize the trajectory inside the gradient computation to avoid storing all intermediate activations. This will trade some extra compute for lower peak memory usage. JAX’s training cookbook shows how to apply `jax.checkpoint` on heavy functions. In PPO, one might rematerialize the policy and value network outputs or recompute masks during backprop.

- **Profile to Identify Hotspots**: Use `jax.profiler.trace` or built-in GPU profiling to see which operations dominate. It may reveal that, for instance, the environment physics or collision resolution is a bottleneck. If so, one could simplify or approximate the environment (e.g. precompute static geometry, or simplify physics for training). Profiling can also check for inefficient memory transfers or padding.

- **Optimize Data Shapes**:  If some dimensions (like number of planets) are frequently smaller than `P_MAX`, consider compressing state representation. For example, pack active planets only (though this sacrifices static shape). Alternatively, reduce `source_cap` if fewer decisions are useful.

- **Asynchronous or Multi-threaded Data Collection**: Though the environment is already on the GPU, one could overlap data collection and training by using two copies of the network/state and ping-pong between them (like double buffering). While one batch is being used for gradient computation, the next batch of rollouts could be prepared on the GPU. This is more complex but can increase utilization.

# Experimental Benchmarks and Recommendations

No benchmarks can be provided here without running code, but we can estimate gains. For example, moving to float16 often yields ~2× speed on modern GPUs (via Tensor Cores). Doubling `envs` (if memory allows) should nearly double throughput. CleanRL’s documentation notes PPO can scale roughly linearly with vector environments up to GPU limits. With aggressive optimizations (mixed precision, full vectorization, more devices) reaching **1000+ sps** is feasible – indeed, a purely JAX RL library (**PureJAXRL**) reports thousands-fold speedups by fully vectorizing on GPU.

In summary, to achieve >1000 sps, we recommend: 

- Converting key computations to **lower precision** and using XLA’s highest matmul precision flag.  
- Refactoring loops into `jax.vmap` to **fully utilize GPU** (e.g. remove the Python loop in `build_bc_features_for_seat`).  
- Experimenting with larger batched environments (balanced against memory) or **multi-GPU training** via `jax.pmap`.  
- Possibly using gradient **checkpointing** to cut memory use.  
- Tuning memory settings (`XLA_PYTHON_CLIENT_MEM_FRACTION`) and ensuring JIT is warm.  

These changes, together with profiling feedback, should bring substantial speedups over the current ~120 sps. 

**Sources:** We examined the training code in `orbit_ppo_jax/train.py` and feature-builder `orbit_ppo_jax/features.py` (codex/jax-pfsp branch) to identify bottlenecks, and applied JAX performance guidelines (e.g. full vmapping and reduced precision) as in JAX documentation and community reports. These sources inform the above recommendations.