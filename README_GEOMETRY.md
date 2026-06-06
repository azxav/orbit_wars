# Orbit Wars Geometry Skeleton Extraction

This package isolates deterministic geometry/mechanics from the uploaded open heuristic solution.

## Intended split

ML/BC/RL should decide:

- source planet
- target planet/comet/no-op
- amount/bin
- timing/priority

The geometry skeleton handles:

- fleet speed from ship count
- future planet positions, including orbiting planets
- comet path projection
- source-target intercept angle
- sun / out-of-bounds / planet first-contact checks
- ETA estimation
- planned launch simulation
- optional garrison/fleet-arrival projection

## Important files

- `orbit_geometry_skeleton/open_best_heuristic_original.py`  
  Exact uploaded `submission.py`, preserved for traceability.

- `orbit_geometry_skeleton/geometry_skeleton.py`  
  Clean public interface over the geometry/mechanics pieces. This is the file the next dataset-construction step should import.

- `geometry_manifest.json`  
  List of extracted components and what they are used for.

## Minimal usage

```python
import torch
from orbit_geometry_skeleton import GeometrySkeleton

geo = GeometrySkeleton()
obs_tensors = geo.obs_to_tensors(obs, player_id=obs["player"])
movement = geo.build_or_update_movement(obs_tensors)

out = geo.aim_source_to_target(
    source_slots=torch.tensor([0]),
    target_slots=torch.tensor([5]),
    fleet_sizes=torch.tensor([20.0]),
)

# out["angle"], out["eta"], out["viable"]
```

## Next dataset-construction use

For every replay step:

1. Convert observation with `obs_to_tensors`.
2. Build movement with `build_or_update_movement`.
3. Map expert raw move `[from_planet_id, angle, num_ships]` to source slot.
4. Use first-contact / planned-launch inference to infer intended target and ETA.
5. Store supervised labels: `source_slot`, `target_slot`, `amount`, `angle`, `eta`, `viable`.

## Do not train ML to learn these pieces first

These are deterministic mechanics. The model should initially learn decision-making on top of them, not rediscover angle physics.
