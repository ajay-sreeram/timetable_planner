# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Timetable Planner Environment Implementation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

import logging
import os

try:
    from ..models import TimetablePlannerAction, TimetablePlannerObservation
except ImportError:
    from models import TimetablePlannerAction, TimetablePlannerObservation

from .scenario_loader import build_repository
from .scoring import (
    build_maps,
    compute_compactness_score,
    compute_conflicts,
    compute_coverage_score,
    compute_daily_balance_score,
    compute_preference_score,
    compute_room_fit_score,
    compute_score_breakdown,
    compute_stability_score,
    parse_available_slots,
)


_ALLOWED_ACTIONS = {
    "assign_session",
    "move_session",
    "swap_sessions",
    "unassign_session",
    "submit_timetable",
}

_INVALID_ACTION_PENALTY = -0.02
_NOOP_ACTION_PENALTY = -0.01
_SUBMIT_BONUS_THRESHOLD = 0.90
_SUBMIT_BONUS = 0.05

_LOG_LEVEL = os.getenv("TIMETABLE_PLANNER_LOG_LEVEL", "INFO").upper()
_LOG_EVERY_STEP = os.getenv("TIMETABLE_PLANNER_LOG_EVERY_STEP", "1") == "1"
_LOG_ACTIONS = os.getenv("TIMETABLE_PLANNER_LOG_ACTIONS", "1") == "1"

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=_LOG_LEVEL)
logger.setLevel(_LOG_LEVEL)


class TimetablePlannerEnvironment(Environment):
    """
    Timetable planning environment with deterministic scenarios and dense rewards.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._repository = build_repository()
        self._scenario: Dict[str, Any] = {}
        self._teachers: Dict[str, Dict[str, Any]] = {}
        self._groups: Dict[str, Dict[str, Any]] = {}
        self._rooms: Dict[str, Dict[str, Any]] = {}
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._sessions_list: List[Dict[str, Any]] = []
        self._assignments: Dict[str, Dict[str, Any]] = {}
        self._baseline_assignments: Dict[str, Dict[str, Any]] = {}
        self._remaining_steps = 0
        self._previous_score = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs: Any,
    ) -> TimetablePlannerObservation:
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._reset_count += 1

        try:
            self._load_scenario(
                scenario_id=(
                    scenario_id.strip()
                    if isinstance(scenario_id, str) and scenario_id.strip()
                    else None
                ),
                task_name=(
                    task_name.lower().strip()
                    if isinstance(task_name, str) and task_name.strip()
                    else None
                ),
            )
        except (ValueError, KeyError, TypeError):
            logger.warning(
                "Invalid reset params scenario_id=%r task_name=%r — falling back to next_scenario",
                scenario_id,
                task_name,
            )
            self._load_scenario()

        self._previous_score = 0.0

        conflicts, score_breakdown = self._evaluate()
        self._previous_score = score_breakdown.get("overall_score", 0.0)

        if _LOG_ACTIONS:
            logger.info(
                "reset scenario=%s task=%s step_budget=%s score=%.3f conflicts=%s",
                self._scenario.get("scenario_id"),
                self._scenario.get("task_name"),
                self._remaining_steps,
                score_breakdown.get("overall_score", 0.0),
                conflicts,
            )

        return self._build_observation(
            conflicts=conflicts,
            score_breakdown=score_breakdown,
            reward=0.0,
            done=False,
        )

    def step(
        self,
        action: TimetablePlannerAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TimetablePlannerObservation:
        if not self._scenario:
            logger.warning("step() called before reset() — auto-resetting")
            self.reset()

        self._state.step_count += 1

        if self._remaining_steps <= 0:
            conflicts, score_breakdown = self._evaluate()
            return self._build_observation(
                conflicts=conflicts,
                score_breakdown=score_breakdown,
                reward=0.0,
                done=True,
                action_error="step_budget_exhausted",
            )

        penalty = 0.0
        action_error = None
        normalized_action, parse_error = self._normalize_action(action)

        if parse_error:
            penalty += _INVALID_ACTION_PENALTY
            action_error = parse_error
            action_type = "invalid"
        else:
            action_type = normalized_action["action_type"]
            is_valid, is_noop, error_message = self._validate_action(normalized_action)

            if not is_valid:
                penalty += _INVALID_ACTION_PENALTY
                action_error = error_message
            else:
                if is_noop:
                    penalty += _NOOP_ACTION_PENALTY
                else:
                    self._apply_action(normalized_action)

        self._remaining_steps = max(0, self._remaining_steps - 1)

        conflicts, score_breakdown = self._evaluate()
        score = score_breakdown.get("overall_score", 0.0)
        reward = (score - self._previous_score) + penalty

        done = False
        if action_type == "submit_timetable":
            done = True
            if score >= _SUBMIT_BONUS_THRESHOLD:
                reward += _SUBMIT_BONUS
        elif self._remaining_steps == 0:
            done = True

        self._previous_score = score

        if _LOG_EVERY_STEP:
            action_log = (
                normalized_action if normalized_action else {"action_type": "invalid"}
            )
            if action_log.get("action_type") in {"assign_session", "move_session"}:
                session_id = action_log.get("session_id")
                duration = self._sessions.get(session_id, {}).get(
                    "duration_in_slots", 1
                )
                action_log = dict(action_log)
                action_log["duration_in_slots"] = duration
            logger.info(
                "step=%s scenario=%s action=%s reward=%.3f score=%.3f penalty=%.3f done=%s conflicts=%s error=%s",
                self._state.step_count,
                self._scenario.get("scenario_id"),
                action_log,
                reward,
                score,
                penalty,
                done,
                conflicts,
                action_error,
            )

        return self._build_observation(
            conflicts=conflicts,
            score_breakdown=score_breakdown,
            reward=reward,
            done=done,
            action_error=action_error,
            action_penalty=penalty if penalty != 0.0 else None,
        )

    @property
    def state(self) -> State:
        return self._state

    def _load_scenario(
        self,
        scenario_id: Optional[str] = None,
        task_name: Optional[str] = None,
    ) -> None:
        if scenario_id:
            scenario = self._repository.get_scenario(scenario_id)
        elif task_name:
            scenario = self._repository.first_scenario_for_task(task_name)
        else:
            scenario = self._repository.next_scenario()
        self._scenario = scenario
        self._remaining_steps = int(scenario.get("step_budget", 30))

        self._teachers = build_maps(scenario.get("teachers", []), "teacher_id")
        self._groups = build_maps(scenario.get("groups", []), "group_id")
        self._rooms = build_maps(scenario.get("rooms", []), "room_id")
        self._sessions = build_maps(scenario.get("sessions", []), "session_id")
        self._sessions_list = list(self._sessions.values())

        self._baseline_assignments = self._build_assignment_map(
            scenario.get("baseline_assignments", [])
        )

        self._assignments = self._build_assignment_map(
            scenario.get("initial_assignments", [])
        )

        self._ensure_availability_defaults()

        if scenario.get("task_name") in {"hard", "expert"}:
            self._apply_disruption(scenario.get("disruption"))

    def _ensure_availability_defaults(self) -> None:
        grid = self._scenario.get("grid", {})
        day_count = len(grid.get("days", []))
        slots_per_day = grid.get("slots_per_day", 0)

        def full_slots() -> Dict[str, List[int]]:
            return {str(day): list(range(slots_per_day)) for day in range(day_count)}

        for teacher in self._teachers.values():
            if teacher.get("available_slots") is None:
                teacher["available_slots"] = full_slots()
            if teacher.get("preferred_slots") is None:
                teacher["preferred_slots"] = {
                    day: list(slots)
                    for day, slots in teacher["available_slots"].items()
                }

        for room in self._rooms.values():
            if room.get("available_slots") is None:
                room["available_slots"] = full_slots()

    def _apply_disruption(self, disruption: Any) -> None:
        if not disruption:
            return
        disruption_type = disruption.get("type")
        if disruption_type == "teacher_unavailable":
            teacher_id = disruption.get("teacher_id")
            slots = disruption.get("slots", {})
            self._remove_availability(self._teachers.get(teacher_id), slots)
        elif disruption_type == "room_unavailable":
            room_id = disruption.get("room_id")
            slots = disruption.get("slots", {})
            self._remove_availability(self._rooms.get(room_id), slots)
        elif disruption_type == "new_session_added":
            session = disruption.get("session")
            if session:
                session_id = session.get("session_id")
                if session_id and session_id not in self._sessions:
                    self._sessions[session_id] = dict(session)
                    self._sessions_list.append(self._sessions[session_id])

    def _remove_availability(self, entity: Any, slots: Dict[str, List[int]]) -> None:
        if not entity:
            return
        grid = self._scenario.get("grid", {})
        day_count = len(grid.get("days", []))
        slots_per_day = grid.get("slots_per_day", 0)
        current_slots = parse_available_slots(
            entity.get("available_slots"), day_count, slots_per_day
        )
        remove_slots = parse_available_slots(slots, day_count, slots_per_day)
        updated = current_slots - remove_slots
        entity["available_slots"] = self._slots_to_dict(updated)
        pref_raw = entity.get("preferred_slots")
        if pref_raw is not None:
            pref_slots = parse_available_slots(pref_raw, day_count, slots_per_day)
            entity["preferred_slots"] = self._slots_to_dict(pref_slots & updated)

    def _slots_to_dict(self, slots: set[Tuple[int, int]]) -> Dict[str, List[int]]:
        by_day: Dict[str, List[int]] = {}
        for day, slot in sorted(slots):
            by_day.setdefault(str(day), []).append(slot)
        return by_day

    def _normalize_action(
        self, action: TimetablePlannerAction
    ) -> Tuple[Dict[str, Any] | None, str | None]:
        action_type = self._coerce_str(action.action_type)
        if not action_type:
            return None, "action_type is required"
        action_type = action_type.strip()
        if action_type not in _ALLOWED_ACTIONS:
            return None, f"Unknown action_type: {action_type}"

        normalized: Dict[str, Any] = {"action_type": action_type}

        if action_type in {"assign_session", "move_session"}:
            session_id = self._coerce_str(action.session_id)
            room_id = self._coerce_str(action.room_id)
            day = self._coerce_day(action.day)
            start_slot = self._coerce_int(action.start_slot)
            missing = [
                k
                for k, v in [
                    ("session_id", session_id),
                    ("day (0-based int or name like Mon)", day),
                    ("start_slot", start_slot),
                    ("room_id", room_id),
                ]
                if v is None
            ]
            if missing:
                return None, f"missing required fields: {', '.join(missing)}"
            normalized.update(
                {
                    "session_id": session_id,
                    "room_id": room_id,
                    "day": day,
                    "start_slot": start_slot,
                }
            )
            return normalized, None

        if action_type == "swap_sessions":
            session_id = self._coerce_str(action.session_id)
            target_id = self._coerce_str(action.target_session_id)
            if session_id is None or target_id is None:
                return None, "swap_sessions requires session_id and target_session_id"
            normalized["session_id"] = session_id
            normalized["target_session_id"] = target_id
            return normalized, None

        if action_type == "unassign_session":
            session_id = self._coerce_str(action.session_id)
            if session_id is None:
                return None, "unassign_session requires session_id"
            normalized["session_id"] = session_id
            return normalized, None

        return normalized, None

    def _coerce_str(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip() or None
        try:
            text = str(value).strip()
            return text or None
        except Exception:
            return None

    def _coerce_int(self, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        try:
            if isinstance(value, float):
                return int(value)
            return int(str(value).strip())
        except Exception:
            return None

    def _coerce_day(self, value: Any) -> int | None:
        """Accept a 0-based int or a day name (Mon, Tue, …) and return the index."""
        as_int = self._coerce_int(value)
        if as_int is not None:
            return as_int
        text = self._coerce_str(value)
        if text is None:
            return None
        days = self._scenario.get("grid", {}).get("days", [])
        lower_days = [d.lower() for d in days]
        needle = text.lower()
        if needle in lower_days:
            return lower_days.index(needle)
        return None

    def _validate_action(self, action: Dict[str, Any]) -> Tuple[bool, bool, str | None]:
        action_type = action["action_type"]

        if action_type in {"assign_session", "move_session"}:
            session_id = action["session_id"]
            room_id = action["room_id"]
            day = action["day"]
            start_slot = action["start_slot"]

            if session_id not in self._sessions:
                return False, False, f"Unknown session_id: {session_id}"
            if room_id not in self._rooms:
                return False, False, f"Unknown room_id: {room_id}"

            day_count = len(self._scenario.get("grid", {}).get("days", []))
            slots_per_day = self._scenario.get("grid", {}).get("slots_per_day", 0)
            if day < 0 or day >= day_count:
                return False, False, "day is out of range"

            duration = self._sessions[session_id].get("duration_in_slots", 1)
            if start_slot < 0 or start_slot + duration > slots_per_day:
                return False, False, "start_slot is out of range for session duration"

            already_assigned = session_id in self._assignments
            if action_type == "assign_session" and already_assigned:
                return (
                    False,
                    False,
                    "session already assigned; use move_session to relocate",
                )
            if action_type == "move_session" and not already_assigned:
                return (
                    False,
                    False,
                    "session not assigned yet; use assign_session first",
                )

            existing = self._assignments.get(session_id)
            if (
                existing
                and existing["day"] == day
                and existing["start_slot"] == start_slot
                and existing["room_id"] == room_id
            ):
                return True, True, None
            return True, False, None

        if action_type == "swap_sessions":
            sid_a = action["session_id"]
            sid_b = action["target_session_id"]
            if sid_a not in self._sessions:
                return False, False, f"Unknown session_id: {sid_a}"
            if sid_b not in self._sessions:
                return False, False, f"Unknown target session_id: {sid_b}"
            if sid_a not in self._assignments:
                return False, False, f"session {sid_a} is not assigned"
            if sid_b not in self._assignments:
                return False, False, f"session {sid_b} is not assigned"
            if sid_a == sid_b:
                return True, True, None
            grid = self._scenario.get("grid", {})
            slots_per_day = grid.get("slots_per_day", 0)
            a_assignment = self._assignments[sid_a]
            b_assignment = self._assignments[sid_b]
            a_duration = self._sessions[sid_a].get("duration_in_slots", 1)
            b_duration = self._sessions[sid_b].get("duration_in_slots", 1)
            if a_assignment["start_slot"] + b_duration > slots_per_day:
                return (
                    False,
                    False,
                    "swap target slot is out of range for target duration",
                )
            if b_assignment["start_slot"] + a_duration > slots_per_day:
                return (
                    False,
                    False,
                    "swap target slot is out of range for source duration",
                )
            return True, False, None

        if action_type == "unassign_session":
            session_id = action["session_id"]
            if session_id not in self._sessions:
                return False, False, f"Unknown session_id: {session_id}"
            if session_id not in self._assignments:
                return True, True, None
            return True, False, None

        return True, False, None

    def _apply_action(self, action: Dict[str, Any]) -> None:
        action_type = action["action_type"]
        if action_type in {"assign_session", "move_session"}:
            self._assignments[action["session_id"]] = {
                "session_id": action["session_id"],
                "day": int(action["day"]),
                "start_slot": int(action["start_slot"]),
                "room_id": action["room_id"],
            }
        elif action_type == "swap_sessions":
            sid_a = action["session_id"]
            sid_b = action["target_session_id"]
            a = self._assignments[sid_a]
            b = self._assignments[sid_b]
            self._assignments[sid_a] = {
                "session_id": sid_a,
                "day": b["day"],
                "start_slot": b["start_slot"],
                "room_id": b["room_id"],
            }
            self._assignments[sid_b] = {
                "session_id": sid_b,
                "day": a["day"],
                "start_slot": a["start_slot"],
                "room_id": a["room_id"],
            }
        elif action_type == "unassign_session":
            self._assignments.pop(action["session_id"], None)

    def _build_assignment_map(
        self, assignments: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        return {item["session_id"]: dict(item) for item in assignments}

    def _evaluate(self) -> Tuple[Dict[str, int], Dict[str, float]]:
        grid = self._scenario.get("grid", {})
        conflicts = compute_conflicts(
            self._assignments,
            self._sessions,
            self._teachers,
            self._rooms,
            grid,
        )

        coverage_score = compute_coverage_score(self._sessions, self._assignments)
        compactness_score = compute_compactness_score(
            self._assignments,
            self._sessions,
            self._teachers,
            self._groups,
            grid,
        )
        room_fit_score = compute_room_fit_score(
            self._assignments,
            self._sessions,
            self._rooms,
        )
        stability_score = compute_stability_score(
            self._assignments,
            (
                self._baseline_assignments
                if self._scenario.get("task_name") in {"hard", "expert"}
                else {}
            ),
        )
        preference_score = compute_preference_score(
            self._assignments,
            self._sessions,
            self._teachers,
        )
        daily_balance_score = compute_daily_balance_score(
            self._assignments,
            grid,
        )

        total_assigned_sessions = len(self._assignments)
        total_assigned_slots = 0
        for session_id in self._assignments:
            session = self._sessions.get(session_id)
            if session:
                total_assigned_slots += session.get("duration_in_slots", 1)

        score_breakdown = compute_score_breakdown(
            self._scenario.get("task_name", "easy"),
            conflicts,
            coverage_score,
            compactness_score,
            room_fit_score,
            stability_score,
            total_assigned_sessions,
            total_assigned_slots,
            preference_score=preference_score,
            daily_balance_score=daily_balance_score,
        )

        return conflicts, score_breakdown

    def _build_observation(
        self,
        conflicts: Dict[str, int],
        score_breakdown: Dict[str, float],
        reward: float,
        done: bool,
        action_error: str | None = None,
        action_penalty: float | None = None,
    ) -> TimetablePlannerObservation:
        unscheduled = sorted(
            [
                session_id
                for session_id, session in self._sessions.items()
                if session.get("must_schedule", True)
                and session_id not in self._assignments
            ]
        )

        timetable = sorted(
            [dict(assignment) for assignment in self._assignments.values()],
            key=lambda item: (item["day"], item["start_slot"], item["room_id"]),
        )

        grid_data = self._scenario.get("grid", {})
        day_labels = list(grid_data.get("days", []))

        def _format_slots(slots: Any) -> Any:
            if not isinstance(slots, dict) or not day_labels:
                return slots
            formatted: Dict[str, List[int]] = {}
            # Preserve day label order when possible.
            index_map = {str(i): label for i, label in enumerate(day_labels)}
            for key, values in slots.items():
                label = index_map.get(str(key), str(key))
                formatted[label] = list(values)
            ordered: Dict[str, List[int]] = {}
            for label in day_labels:
                if label in formatted:
                    ordered[label] = formatted[label]
            for key, values in formatted.items():
                if key not in ordered:
                    ordered[key] = values
            return ordered

        def _format_entity(entity: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(entity)
            if "available_slots" in out:
                out["available_slots"] = _format_slots(out.get("available_slots"))
            if "preferred_slots" in out:
                out["preferred_slots"] = _format_slots(out.get("preferred_slots"))
            return out

        baseline = (
            sorted(
                [dict(a) for a in self._baseline_assignments.values()],
                key=lambda item: (item["day"], item["start_slot"], item["room_id"]),
            )
            if self._baseline_assignments
            else []
        )

        return TimetablePlannerObservation(
            task_name=self._scenario.get("task_name", ""),
            scenario_id=self._scenario.get("scenario_id", ""),
            grid=grid_data,
            teachers=[_format_entity(t) for t in self._teachers.values()],
            groups=list(self._groups.values()),
            rooms=[_format_entity(r) for r in self._rooms.values()],
            sessions=self._sessions_list,
            current_timetable=timetable,
            baseline_timetable=baseline,
            unscheduled_sessions=unscheduled,
            conflict_summary=conflicts,
            score_breakdown=score_breakdown,
            remaining_step_budget=self._remaining_steps,
            done=done,
            reward=reward,
            action_error=action_error,
            action_penalty=action_penalty,
        )
