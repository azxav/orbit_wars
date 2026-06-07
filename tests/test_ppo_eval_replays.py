from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace


def test_save_game_replay_saves_json_when_html_render_fails(tmp_path, caplog) -> None:
    from orbit_ppo_training.eval_ppo import save_game_replay

    class FakeEnv:
        def toJSON(self):
            return {"finished": True}

        def render(self, *, mode):
            assert mode == "html"
            raise RuntimeError("html unavailable")

    caplog.set_level(logging.WARNING)

    paths = save_game_replay(FakeEnv(), tmp_path / "replays", 0, save_html=True)

    assert json.loads((tmp_path / "replays" / "game_000.json").read_text(encoding="utf-8")) == {"finished": True}
    assert not (tmp_path / "replays" / "game_000.html").exists()
    assert paths == {"json": str(tmp_path / "replays" / "game_000.json"), "html": None}
    assert "Failed to render HTML replay for game 0" in caplog.text


def test_evaluate_saves_selected_replays_in_summary_and_rows(tmp_path, monkeypatch) -> None:
    from orbit_ppo_training import eval_ppo

    class FakeEnv:
        def __init__(self, game_idx: int):
            self.game_idx = game_idx

        def toJSON(self):
            return {"game": self.game_idx}

        def render(self, *, mode):
            assert mode == "html"
            return f"<html>game {self.game_idx}</html>"

    def fake_load_ppo_checkpoint(checkpoint, *, device):
        return object(), SimpleNamespace(opponent="", players=0, device="", seed=0), {}

    def fake_collect_rollouts(policy, config, *, games, deterministic, seed_start, replay_callback=None):
        rows = []
        for game_idx in range(int(games)):
            row = {"game_id": f"game_{game_idx:05d}", "win": game_idx == 0}
            if replay_callback is not None:
                replay_callback(FakeEnv(game_idx), game_idx, row)
            rows.append(row)
        return SimpleNamespace(rows=rows, summary={"num_games": len(rows), "winrate": 0.5})

    monkeypatch.setattr(eval_ppo, "load_ppo_checkpoint", fake_load_ppo_checkpoint)
    monkeypatch.setattr(eval_ppo, "collect_rollouts", fake_collect_rollouts)

    summary = eval_ppo.evaluate(
        "checkpoint",
        opponent="orbit_wars_base",
        players=4,
        num_games=2,
        out_dir=str(tmp_path),
        save_replays=1,
        save_html_replays=True,
    )

    rows = [json.loads(line) for line in (tmp_path / "games.jsonl").read_text(encoding="utf-8").splitlines()]
    summary_json = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert summary["replay_paths"] == [rows[0]["replay_paths"]]
    assert Path(rows[0]["replay_paths"]["json"]).parts[-2:] == ("replays", "game_000.json")
    assert Path(rows[0]["replay_paths"]["html"]).parts[-2:] == ("replays", "game_000.html")
    assert "replay_paths" not in rows[1]
    assert summary_json["replay_paths"] == [rows[0]["replay_paths"]]
    assert json.loads((tmp_path / "replays" / "game_000.json").read_text(encoding="utf-8")) == {"game": 0}
    assert (tmp_path / "replays" / "game_000.html").read_text(encoding="utf-8") == "<html>game 0</html>"
