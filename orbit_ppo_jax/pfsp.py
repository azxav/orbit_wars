from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from orbit_jax_env.config import MAX_PLAYERS

OPP_NONE = 0
OPP_SIMPLE_HEURISTIC = 1
OPP_JAX_PROXY = 2
OPP_FROZEN_POLICY = 3


@dataclass(frozen=True)
class JaxMatchPlan:
    learner_seat: jnp.ndarray
    opponent_kind: jnp.ndarray
    opponent_slot: jnp.ndarray


@dataclass(frozen=True)
class PFSPEntry:
    id: str
    kind: str
    slot: int | None
    anchor: bool
    active: bool
    path: str | None
    added_update: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PFSPEntry":
        return cls(
            id=str(data["id"]),
            kind=str(data["kind"]),
            slot=None if data.get("slot") is None else int(data["slot"]),
            anchor=bool(data.get("anchor", False)),
            active=bool(data.get("active", True)),
            path=None if data.get("path") is None else str(data["path"]),
            added_update=int(data.get("added_update", 0)),
        )


@dataclass(frozen=True)
class PFSPEntryStats:
    games: int = 0
    score_sum: float = 0.0
    reward_sum: float = 0.0
    rank_sum: float = 0.0
    last_played_update: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PFSPEntryStats":
        return cls(
            games=int(data.get("games", 0)),
            score_sum=float(data.get("score_sum", 0.0)),
            reward_sum=float(data.get("reward_sum", 0.0)),
            rank_sum=float(data.get("rank_sum", 0.0)),
            last_played_update=int(data.get("last_played_update", 0)),
        )


@dataclass(frozen=True)
class PFSPManifest:
    version: int
    players: int
    max_policy_slots: int
    entries: list[PFSPEntry]
    stats: dict[str, PFSPEntryStats]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PFSPManifest":
        return cls(
            version=int(data["version"]),
            players=int(data["players"]),
            max_policy_slots=int(data["max_policy_slots"]),
            entries=[PFSPEntry.from_dict(row) for row in data.get("entries", [])],
            stats={str(k): PFSPEntryStats.from_dict(v) for k, v in data.get("stats", {}).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "players": int(self.players),
            "max_policy_slots": int(self.max_policy_slots),
            "entries": [asdict(entry) for entry in self.entries],
            "stats": {entry_id: asdict(stats) for entry_id, stats in self.stats.items()},
        }


def save_manifest(path: str | Path, manifest: PFSPManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)


def load_manifest(path: str | Path) -> PFSPManifest:
    with open(path, "r", encoding="utf-8") as f:
        return PFSPManifest.from_dict(json.load(f))


def build_initial_manifest(*, players: int, max_policy_slots: int, bc_checkpoint: str, include_anchors: bool = False) -> PFSPManifest:
    entries = [
        PFSPEntry("initial_bc", "frozen_policy", 0, True, True, bc_checkpoint, 0),
    ]
    if include_anchors:
        entries = [
            PFSPEntry("anchor_simple_heuristic_jax", "simple_heuristic_jax", None, True, True, None, 0),
            PFSPEntry("anchor_jax_proxy", "jax_proxy", None, True, True, None, 0),
            *entries,
        ]
    return PFSPManifest(
        version=1,
        players=int(players),
        max_policy_slots=int(max_policy_slots),
        entries=entries,
        stats={entry.id: PFSPEntryStats() for entry in entries},
    )


def pfsp_weight(
    stats: PFSPEntryStats,
    *,
    hard_low: float,
    hard_high: float,
    hard_bonus: float,
    exploration_bonus: float,
    prior: float = 1.0,
) -> float:
    games = max(int(stats.games), 0)
    p = (float(stats.score_sum) + float(prior)) / (float(games) + 2.0 * float(prior))
    p = min(max(p, 0.0), 1.0)
    base = float(np.sqrt(p * (1.0 - p)))
    hard = float(hard_bonus) if float(hard_low) <= p <= float(hard_high) else 0.0
    explore = float(exploration_bonus) / float(np.sqrt(games + 1.0))
    return base + hard + explore


def normalized_result(rank: float, players: int) -> float:
    if int(players) == 2:
        return 1.0 if int(rank) == 0 else 0.0
    return float(players - 1 - rank) / float(max(players - 1, 1))


def update_manifest_from_slot_stats(
    manifest: PFSPManifest,
    *,
    slot_games: list[int],
    slot_score_sum: list[float],
    slot_reward_sum: list[float],
    slot_rank_sum: list[float],
    kind_games: dict[int, int] | None = None,
    kind_score_sum: dict[int, float] | None = None,
    kind_reward_sum: dict[int, float] | None = None,
    kind_rank_sum: dict[int, float] | None = None,
    update_index: int,
) -> PFSPManifest:
    stats = dict(manifest.stats)
    for entry in manifest.entries:
        if entry.kind != "frozen_policy" or not entry.active or entry.slot is None:
            continue
        slot = int(entry.slot)
        if slot < 0 or slot >= len(slot_games):
            continue
        games = int(slot_games[slot])
        if games <= 0:
            continue
        current = stats.get(entry.id, PFSPEntryStats())
        stats[entry.id] = PFSPEntryStats(
            games=int(current.games) + games,
            score_sum=float(current.score_sum) + float(slot_score_sum[slot]),
            reward_sum=float(current.reward_sum) + float(slot_reward_sum[slot]),
            rank_sum=float(current.rank_sum) + float(slot_rank_sum[slot]),
            last_played_update=int(update_index),
        )
    kind_to_entry_id = {
        OPP_SIMPLE_HEURISTIC: "anchor_simple_heuristic_jax",
        OPP_JAX_PROXY: "anchor_jax_proxy",
    }
    kind_games = kind_games or {}
    kind_score_sum = kind_score_sum or {}
    kind_reward_sum = kind_reward_sum or {}
    kind_rank_sum = kind_rank_sum or {}
    for kind, entry_id in kind_to_entry_id.items():
        games = int(kind_games.get(kind, 0))
        if games <= 0 or not any(entry.id == entry_id and entry.active for entry in manifest.entries):
            continue
        current = stats.get(entry_id, PFSPEntryStats())
        stats[entry_id] = PFSPEntryStats(
            games=int(current.games) + games,
            score_sum=float(current.score_sum) + float(kind_score_sum.get(kind, 0.0)),
            reward_sum=float(current.reward_sum) + float(kind_reward_sum.get(kind, 0.0)),
            rank_sum=float(current.rank_sum) + float(kind_rank_sum.get(kind, 0.0)),
            last_played_update=int(update_index),
        )
    return PFSPManifest(
        version=manifest.version,
        players=manifest.players,
        max_policy_slots=manifest.max_policy_slots,
        entries=manifest.entries,
        stats=stats,
    )


def prune_entries(
    entries: list[PFSPEntry],
    *,
    max_policy_slots: int,
    best_entry_id: str | None,
    latest_entry_id: str | None,
) -> list[PFSPEntry]:
    keep_ids: set[str] = set()
    for entry in entries:
        if entry.anchor:
            keep_ids.add(entry.id)
    if best_entry_id:
        keep_ids.add(best_entry_id)
    if latest_entry_id:
        keep_ids.add(latest_entry_id)

    frozen = [entry for entry in entries if entry.kind == "frozen_policy" and entry.active]
    required_frozen = [entry for entry in frozen if entry.id in keep_ids]
    remaining_capacity = max(int(max_policy_slots) - len(required_frozen), 0)
    recent = [
        entry
        for entry in sorted(frozen, key=lambda e: int(e.added_update), reverse=True)
        if entry.id not in keep_ids
    ]
    for entry in recent[:remaining_capacity]:
        keep_ids.add(entry.id)

    return [entry for entry in entries if entry.anchor or entry.id in keep_ids]


def add_snapshot_entry(
    manifest: PFSPManifest,
    *,
    entry_id: str,
    path: str,
    update_index: int,
    protected_entry_ids: set[str] | None = None,
) -> PFSPManifest:
    if any(entry.id == entry_id for entry in manifest.entries):
        return manifest
    protected_entry_ids = protected_entry_ids or set()
    used_slots = {
        int(entry.slot)
        for entry in manifest.entries
        if entry.kind == "frozen_policy" and entry.active and entry.slot is not None
    }
    slot = next(
        (candidate for candidate in range(1, int(manifest.max_policy_slots)) if candidate not in used_slots),
        None,
    )
    if slot is None:
        reusable = [
            entry
            for entry in manifest.entries
            if entry.kind == "frozen_policy"
            and entry.active
            and not entry.anchor
            and entry.id not in protected_entry_ids
            and entry.slot is not None
        ]
        if not reusable:
            raise RuntimeError("PFSP policy bank has no free slot for promoted snapshot")
        victim = min(reusable, key=lambda entry: int(entry.added_update))
        slot = int(victim.slot)
        base_entries = [
            PFSPEntry(entry.id, entry.kind, entry.slot, entry.anchor, False, entry.path, entry.added_update)
            if entry.id == victim.id
            else entry
            for entry in manifest.entries
        ]
    else:
        base_entries = list(manifest.entries)
    entries = [
        *base_entries,
        PFSPEntry(entry_id, "frozen_policy", slot, False, True, path, int(update_index)),
    ]
    stats = dict(manifest.stats)
    stats.setdefault(entry_id, PFSPEntryStats())
    return PFSPManifest(
        version=manifest.version,
        players=manifest.players,
        max_policy_slots=manifest.max_policy_slots,
        entries=entries,
        stats=stats,
    )


def _entry_kind_and_slot(entry: PFSPEntry) -> tuple[int, int]:
    if entry.kind == "simple_heuristic_jax":
        return OPP_SIMPLE_HEURISTIC, -1
    if entry.kind == "jax_proxy":
        return OPP_JAX_PROXY, -1
    if entry.kind == "frozen_policy":
        return OPP_FROZEN_POLICY, int(entry.slot if entry.slot is not None else 0)
    raise RuntimeError(f"unsupported PFSP entry kind: {entry.kind}")


def _active_entries(manifest: PFSPManifest) -> list[PFSPEntry]:
    return [entry for entry in manifest.entries if entry.active]


def _weighted_frozen_entry(
    manifest: PFSPManifest,
    frozen_entries: list[PFSPEntry],
    rng: np.random.Generator,
    *,
    min_games_per_entry: int,
    hard_low: float,
    hard_high: float,
    hard_bonus: float,
    exploration_bonus: float,
) -> PFSPEntry:
    under_sampled = [
        entry
        for entry in frozen_entries
        if int(manifest.stats.get(entry.id, PFSPEntryStats()).games) < int(min_games_per_entry)
    ]
    if under_sampled:
        return _cycle_entry(under_sampled, int(rng.integers(0, len(under_sampled))))
    if len(frozen_entries) == 1:
        return frozen_entries[0]
    weights = np.asarray(
        [
            pfsp_weight(
                manifest.stats.get(entry.id, PFSPEntryStats()),
                hard_low=hard_low,
                hard_high=hard_high,
                hard_bonus=hard_bonus,
                exploration_bonus=exploration_bonus,
            )
            for entry in frozen_entries
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(weights)) or float(np.sum(weights)) <= 0.0:
        weights = np.ones((len(frozen_entries),), dtype=np.float64)
    probs = weights / np.sum(weights)
    return frozen_entries[int(rng.choice(len(frozen_entries), p=probs))]


def _cycle_entry(entries: list[PFSPEntry], index: int) -> PFSPEntry:
    return entries[index % len(entries)]


def _learner_seats(*, rng: np.random.Generator, envs: int, players: int, mode: str) -> np.ndarray:
    if mode == "fixed0":
        return np.zeros((envs,), dtype=np.int32)
    if mode == "rotate":
        return np.arange(envs, dtype=np.int32) % int(players)
    if mode == "random":
        return rng.integers(0, int(players), size=(envs,), dtype=np.int32)
    raise RuntimeError("--pfsp_learner_seat_mode must be fixed0, rotate, or random")


def build_match_plan(
    manifest: PFSPManifest,
    *,
    rng: np.random.Generator,
    envs: int,
    players: int,
    learner_seat_mode: str,
    anchor_fraction: float,
    layout: str,
    min_games_per_entry: int = 0,
    hard_low: float = 0.20,
    hard_high: float = 0.55,
    hard_bonus: float = 0.15,
    exploration_bonus: float = 0.10,
) -> JaxMatchPlan:
    if int(players) not in {2, 4}:
        raise RuntimeError("PFSP match plans support 2 or 4 players")
    if layout != "one_pfsp_two_anchors":
        raise RuntimeError("--pfsp_4p_layout must be one_pfsp_two_anchors")

    envs = int(envs)
    players = int(players)
    learner_seat = _learner_seats(rng=rng, envs=envs, players=players, mode=learner_seat_mode)
    opponent_kind = np.zeros((envs, MAX_PLAYERS), dtype=np.int32)
    opponent_slot = np.full((envs, MAX_PLAYERS), -1, dtype=np.int32)
    entries = _active_entries(manifest)
    if not entries:
        raise RuntimeError("PFSP manifest has no active entries")
    anchor_entries = [entry for entry in entries if entry.anchor and entry.kind != "frozen_policy"]
    frozen_entries = [entry for entry in entries if entry.kind == "frozen_policy"]

    for env_i in range(envs):
        available_seats = [seat for seat in range(players) if seat != int(learner_seat[env_i])]
        selected_entries = []
        for _offset, _seat in enumerate(available_seats):
            use_anchor = bool(rng.random() < float(anchor_fraction)) or not frozen_entries
            if use_anchor and anchor_entries:
                entry = _cycle_entry(anchor_entries, env_i + len(selected_entries))
            elif frozen_entries:
                entry = _weighted_frozen_entry(
                    manifest,
                    frozen_entries,
                    rng,
                    min_games_per_entry=min_games_per_entry,
                    hard_low=hard_low,
                    hard_high=hard_high,
                    hard_bonus=hard_bonus,
                    exploration_bonus=exploration_bonus,
                )
            else:
                entry = _cycle_entry(entries, env_i + len(selected_entries))
            selected_entries.append(entry)

        for seat, entry in zip(available_seats, selected_entries, strict=True):
            kind, slot = _entry_kind_and_slot(entry)
            opponent_kind[env_i, seat] = kind
            opponent_slot[env_i, seat] = slot
        opponent_kind[env_i, int(learner_seat[env_i])] = OPP_NONE
        opponent_slot[env_i, int(learner_seat[env_i])] = -1

    return JaxMatchPlan(
        learner_seat=jnp.asarray(learner_seat, dtype=jnp.int32),
        opponent_kind=jnp.asarray(opponent_kind, dtype=jnp.int32),
        opponent_slot=jnp.asarray(opponent_slot, dtype=jnp.int32),
    )
