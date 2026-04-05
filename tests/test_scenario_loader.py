# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for scenario loading, normalization, and procedural generation."""

from server.scenario_loader import (
    ScenarioRepository,
    build_repository,
    load_scenarios,
    normalize_scenario,
)
from server.scenario_generator import generate_scenario, _session_is_feasible, _parse_avail


# ---------------------------------------------------------------------------
# normalize_scenario
# ---------------------------------------------------------------------------

class TestNormalizeScenario:
    def test_fills_defaults(self):
        raw = {
            "sessions": [
                {"session_id": "s1", "teacher_id": "T1", "group_id": "G1"}
            ]
        }
        result = normalize_scenario(raw)
        session = result["sessions"][0]
        assert session["duration_in_slots"] == 1
        assert session["must_schedule"] is True
        assert session["valid_room_types"] == []
        assert result["initial_assignments"] == []

    def test_preserves_existing_values(self):
        raw = {
            "sessions": [
                {"session_id": "s1", "teacher_id": "T1", "group_id": "G1",
                 "duration_in_slots": 2, "must_schedule": False,
                 "valid_room_types": ["lab"]}
            ],
            "initial_assignments": [{"session_id": "s1"}],
        }
        result = normalize_scenario(raw)
        session = result["sessions"][0]
        assert session["duration_in_slots"] == 2
        assert session["must_schedule"] is False
        assert session["valid_room_types"] == ["lab"]

    def test_deep_copy(self):
        raw = {"sessions": [{"session_id": "s1", "teacher_id": "T1", "group_id": "G1"}]}
        result = normalize_scenario(raw)
        result["sessions"][0]["teacher_id"] = "CHANGED"
        assert raw["sessions"][0]["teacher_id"] == "T1"


# ---------------------------------------------------------------------------
# load_scenarios / build_repository
# ---------------------------------------------------------------------------

class TestLoadScenarios:
    def test_load_returns_list(self):
        scenarios = load_scenarios()
        assert isinstance(scenarios, list)
        assert len(scenarios) >= 8

    def test_all_have_required_fields(self):
        for scenario in load_scenarios():
            assert "scenario_id" in scenario
            assert "task_name" in scenario
            assert "grid" in scenario
            assert "sessions" in scenario


class TestScenarioRepository:
    def test_round_robin(self):
        repo = ScenarioRepository(scenarios=[{"id": "a"}, {"id": "b"}])
        assert repo.next_scenario()["id"] == "a"
        assert repo.next_scenario()["id"] == "b"
        assert repo.next_scenario()["id"] == "a"

    def test_empty_raises(self):
        repo = ScenarioRepository(scenarios=[])
        try:
            repo.next_scenario()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_get_scenario_by_id(self):
        repo = build_repository()
        s = repo.get_scenario("easy_1")
        assert s["scenario_id"] == "easy_1"
        assert s["task_name"] == "easy"

    def test_get_scenario_by_id_generated(self):
        repo = build_repository()
        s = repo.get_scenario("gen_medium_2001")
        assert s["task_name"] == "medium"

    def test_get_scenario_unknown_raises(self):
        repo = build_repository()
        try:
            repo.get_scenario("nonexistent_99")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_first_scenario_for_task(self):
        repo = build_repository()
        for task in ("easy", "medium", "hard", "expert"):
            s = repo.first_scenario_for_task(task)
            assert s["task_name"] == task

    def test_first_scenario_for_task_unknown_raises(self):
        repo = build_repository()
        try:
            repo.first_scenario_for_task("nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_build_repository_includes_generated(self):
        repo = build_repository()
        seen_ids = set()
        for _ in range(20):
            s = repo.next_scenario()
            seen_ids.add(s["scenario_id"])
        gen_ids = [sid for sid in seen_ids if sid.startswith("gen_")]
        assert len(gen_ids) > 0


# ---------------------------------------------------------------------------
# Procedural generation
# ---------------------------------------------------------------------------

class TestGenerateScenario:
    def test_easy_scenario(self):
        s = generate_scenario("easy", seed=42)
        assert s["task_name"] == "easy"
        assert len(s["sessions"]) > 0
        assert len(s["teachers"]) > 0

    def test_medium_has_availability(self):
        s = generate_scenario("medium", seed=42)
        for t in s["teachers"]:
            assert "available_slots" in t

    def test_hard_has_baseline_and_disruption(self):
        s = generate_scenario("hard", seed=42)
        assert "baseline_assignments" in s
        assert "disruption" in s
        assert len(s["baseline_assignments"]) > 0

    def test_deterministic_with_seed(self):
        s1 = generate_scenario("medium", seed=123)
        s2 = generate_scenario("medium", seed=123)
        assert s1["sessions"] == s2["sessions"]
        assert s1["teachers"] == s2["teachers"]

    def test_expert_has_baseline_and_disruption(self):
        s = generate_scenario("expert", seed=42)
        assert "baseline_assignments" in s
        assert "disruption" in s
        assert len(s["baseline_assignments"]) > 0

    def test_all_must_schedule_sessions_are_feasible(self):
        for seed in range(50):
            for task in ("easy", "medium", "hard", "expert"):
                s = generate_scenario(task, seed=seed)
                day_count = len(s["grid"]["days"])
                spd = s["grid"]["slots_per_day"]
                teacher_map = {t["teacher_id"]: t for t in s["teachers"]}
                rooms_by_type = {}
                for r in s["rooms"]:
                    rooms_by_type.setdefault(r["room_type"], []).append(r)

                for session in s["sessions"]:
                    if not session.get("must_schedule", True):
                        continue
                    teacher = teacher_map[session["teacher_id"]]
                    valid_types = session.get("valid_room_types", [])
                    compat = []
                    for rt in valid_types:
                        compat.extend(rooms_by_type.get(rt, []))
                    assert _session_is_feasible(
                        session, teacher, compat, day_count, spd,
                    ), f"Infeasible session {session['session_id']} in {s['scenario_id']}"
