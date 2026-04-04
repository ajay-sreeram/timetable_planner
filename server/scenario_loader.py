# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scenario loading, normalization, and procedural generation utilities."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .scenario_generator import generate_scenario

_SCENARIO_PATH = Path(__file__).resolve().parent / "scenarios" / "scenarios.json"

_GENERATED_SEEDS: List[Dict[str, Any]] = [
    {"task_name": "easy", "seed": 1001},
    {"task_name": "easy", "seed": 1002},
    {"task_name": "medium", "seed": 2001},
    {"task_name": "medium", "seed": 2002},
    {"task_name": "hard", "seed": 3001},
    {"task_name": "hard", "seed": 3002},
]


@dataclass
class ScenarioRepository:
    scenarios: List[Dict[str, Any]]
    _index: int = 0
    _gen_seeds: List[Dict[str, Any]] = field(default_factory=list)

    def next_scenario(self) -> Dict[str, Any]:
        total = len(self.scenarios) + len(self._gen_seeds)
        if total == 0:
            raise ValueError("No scenarios found for timetable planner.")

        combined_index = self._index % total
        self._index += 1

        if combined_index < len(self.scenarios):
            return copy.deepcopy(self.scenarios[combined_index])

        gen_cfg = self._gen_seeds[combined_index - len(self.scenarios)]
        return normalize_scenario(
            generate_scenario(task_name=gen_cfg["task_name"], seed=gen_cfg["seed"])
        )


def load_scenarios() -> List[Dict[str, Any]]:
    with _SCENARIO_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    scenarios = payload.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise ValueError("Scenario file must contain a list under 'scenarios'.")
    return [normalize_scenario(item) for item in scenarios]


def normalize_scenario(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Fill defaults for optional fields to keep runtime logic simple."""
    normalized = copy.deepcopy(scenario)
    sessions = normalized.get("sessions", [])
    for session in sessions:
        session.setdefault("duration_in_slots", 1)
        session.setdefault("must_schedule", True)
        session.setdefault("valid_room_types", [])
    normalized.setdefault("initial_assignments", [])
    return normalized


def build_repository() -> ScenarioRepository:
    return ScenarioRepository(
        scenarios=load_scenarios(),
        _gen_seeds=list(_GENERATED_SEEDS),
    )
