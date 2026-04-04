# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scoring and constraint evaluation utilities for the timetable planner."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Tuple

Slot = Tuple[int, int]


def parse_available_slots(
    raw: Any, day_count: int, slots_per_day: int
) -> set[Slot]:
    if raw is None:
        return {(day, slot) for day in range(day_count) for slot in range(slots_per_day)}

    if isinstance(raw, dict):
        slots: set[Slot] = set()
        for day_key, slot_list in raw.items():
            day = int(day_key)
            for slot in slot_list:
                slots.add((day, int(slot)))
        return slots

    slots = set()
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            slots.add((int(item[0]), int(item[1])))
    return slots


def build_maps(items: Iterable[Mapping[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    return {str(item[key]): dict(item) for item in items}


def expand_assignment_slots(start_slot: int, duration: int) -> List[int]:
    return [start_slot + offset for offset in range(duration)]


def build_occupancy(
    assignments: Mapping[str, Dict[str, Any]],
    sessions: Mapping[str, Dict[str, Any]],
) -> Tuple[Dict[Tuple[str, int, int], List[str]], Dict[Tuple[str, int, int], List[str]], Dict[Tuple[str, int, int], List[str]]]:
    teacher_occ: Dict[Tuple[str, int, int], List[str]] = {}
    group_occ: Dict[Tuple[str, int, int], List[str]] = {}
    room_occ: Dict[Tuple[str, int, int], List[str]] = {}

    for session_id, assignment in assignments.items():
        session = sessions.get(session_id)
        if not session:
            continue
        day = assignment["day"]
        start_slot = assignment["start_slot"]
        duration = session.get("duration_in_slots", 1)
        for slot in expand_assignment_slots(start_slot, duration):
            teacher_key = (session["teacher_id"], day, slot)
            group_key = (session["group_id"], day, slot)
            room_key = (assignment["room_id"], day, slot)
            teacher_occ.setdefault(teacher_key, []).append(session_id)
            group_occ.setdefault(group_key, []).append(session_id)
            room_occ.setdefault(room_key, []).append(session_id)

    return teacher_occ, group_occ, room_occ


def count_overlaps(occupancy: Mapping[Tuple[str, int, int], List[str]]) -> int:
    return sum(max(0, len(session_ids) - 1) for session_ids in occupancy.values())


def compute_conflicts(
    assignments: Mapping[str, Dict[str, Any]],
    sessions: Mapping[str, Dict[str, Any]],
    teachers: Mapping[str, Dict[str, Any]],
    rooms: Mapping[str, Dict[str, Any]],
    grid: Mapping[str, Any],
) -> Dict[str, int]:
    day_count = len(grid["days"])
    slots_per_day = grid["slots_per_day"]

    teacher_occ, group_occ, room_occ = build_occupancy(assignments, sessions)
    conflicts = {
        "teacher_overlap": count_overlaps(teacher_occ),
        "group_overlap": count_overlaps(group_occ),
        "room_overlap": count_overlaps(room_occ),
        "room_capacity_violation": 0,
        "room_type_violation": 0,
        "teacher_availability_violation": 0,
        "room_availability_violation": 0,
    }

    teacher_availability = {
        teacher_id: parse_available_slots(teacher.get("available_slots"), day_count, slots_per_day)
        for teacher_id, teacher in teachers.items()
    }
    room_availability = {
        room_id: parse_available_slots(room.get("available_slots"), day_count, slots_per_day)
        for room_id, room in rooms.items()
    }

    for session_id, assignment in assignments.items():
        session = sessions.get(session_id)
        room = rooms.get(assignment["room_id"])
        if not session or not room:
            continue
        required_capacity = session.get("required_capacity", 0)
        if room.get("capacity", 0) < required_capacity:
            conflicts["room_capacity_violation"] += 1
        if room.get("room_type") not in session.get("valid_room_types", []):
            conflicts["room_type_violation"] += 1

        day = assignment["day"]
        start_slot = assignment["start_slot"]
        duration = session.get("duration_in_slots", 1)
        for slot in expand_assignment_slots(start_slot, duration):
            if (day, slot) not in teacher_availability.get(session["teacher_id"], set()):
                conflicts["teacher_availability_violation"] += 1
            if (day, slot) not in room_availability.get(assignment["room_id"], set()):
                conflicts["room_availability_violation"] += 1

    return conflicts


def compute_hard_constraint_score(
    conflicts: Mapping[str, int],
    task_name: str,
    total_assigned_sessions: int,
    total_assigned_slots: int,
) -> float:
    active_types = ["teacher_overlap", "group_overlap", "room_overlap"]
    if task_name in {"medium", "hard"}:
        active_types += [
            "room_capacity_violation",
            "room_type_violation",
            "teacher_availability_violation",
            "room_availability_violation",
        ]

    if not active_types:
        return 1.0

    rates = []
    for violation_type in active_types:
        count = conflicts.get(violation_type, 0)
        if violation_type in {"room_capacity_violation", "room_type_violation"}:
            normalizer = max(1, total_assigned_sessions)
        else:
            normalizer = max(1, total_assigned_slots)
        rates.append(min(1.0, count / normalizer))

    avg_rate = sum(rates) / len(rates)
    return max(0.0, 1.0 - avg_rate)


def compute_coverage_score(sessions: Mapping[str, Dict[str, Any]], assignments: Mapping[str, Dict[str, Any]]) -> float:
    required_sessions = [s for s in sessions.values() if s.get("must_schedule", True)]
    if not required_sessions:
        return 1.0
    scheduled_required = sum(1 for s in required_sessions if s["session_id"] in assignments)
    return scheduled_required / len(required_sessions)


def compute_compactness_score(
    assignments: Mapping[str, Dict[str, Any]],
    sessions: Mapping[str, Dict[str, Any]],
    teachers: Mapping[str, Dict[str, Any]],
    groups: Mapping[str, Dict[str, Any]],
    grid: Mapping[str, Any],
) -> float:
    day_count = len(grid["days"])
    slots_per_day = grid["slots_per_day"]

    teacher_slots: Dict[str, Dict[int, set[int]]] = {teacher_id: {} for teacher_id in teachers}
    group_slots: Dict[str, Dict[int, set[int]]] = {group_id: {} for group_id in groups}

    for session_id, assignment in assignments.items():
        session = sessions.get(session_id)
        if not session:
            continue
        day = assignment["day"]
        start_slot = assignment["start_slot"]
        duration = session.get("duration_in_slots", 1)
        slots = expand_assignment_slots(start_slot, duration)
        teacher_slots.setdefault(session["teacher_id"], {}).setdefault(day, set()).update(slots)
        group_slots.setdefault(session["group_id"], {}).setdefault(day, set()).update(slots)

    total_score = 0.0
    scored_days = 0
    for slots_by_day in list(teacher_slots.values()) + list(group_slots.values()):
        for day in range(day_count):
            slots = sorted(slots_by_day.get(day, set()))
            if len(slots) < 2:
                continue
            gaps = 0
            for current, nxt in zip(slots, slots[1:]):
                gaps += max(0, nxt - current - 1)
            max_gap = max(0, slots_per_day - len(slots))
            if max_gap == 0:
                day_score = 1.0
            else:
                day_score = max(0.0, 1.0 - min(1.0, gaps / max_gap))
            total_score += day_score
            scored_days += 1

    if scored_days == 0:
        return 1.0
    return total_score / scored_days


def compute_room_fit_score(
    assignments: Mapping[str, Dict[str, Any]],
    sessions: Mapping[str, Dict[str, Any]],
    rooms: Mapping[str, Dict[str, Any]],
) -> float:
    if not assignments:
        return 0.0

    scores = []
    for session_id, assignment in assignments.items():
        session = sessions.get(session_id)
        room = rooms.get(assignment["room_id"])
        if not session or not room:
            continue
        capacity = room.get("capacity", 0)
        required = session.get("required_capacity", 0)
        type_ok = room.get("room_type") in session.get("valid_room_types", [])
        capacity_ok = capacity >= required
        if not type_ok or not capacity_ok:
            continue
        waste = max(0, capacity - required)
        waste_score = 1.0 - min(1.0, waste / max(1, capacity))
        scores.append(0.7 + 0.3 * waste_score)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def compute_stability_score(
    assignments: Mapping[str, Dict[str, Any]],
    baseline_assignments: Mapping[str, Dict[str, Any]],
) -> float:
    if not baseline_assignments:
        return 1.0

    total_score = 0.0
    for session_id, baseline in baseline_assignments.items():
        current = assignments.get(session_id)
        if not current:
            continue_score = 0.0
        elif (
            current["day"] == baseline["day"]
            and current["start_slot"] == baseline["start_slot"]
            and current["room_id"] == baseline["room_id"]
        ):
            continue_score = 1.0
        else:
            continue_score = 0.5
        total_score += continue_score

    return total_score / len(baseline_assignments)


def compute_preference_score(
    assignments: Mapping[str, Dict[str, Any]],
    sessions: Mapping[str, Dict[str, Any]],
    teachers: Mapping[str, Dict[str, Any]],
) -> float:
    """Score based on teacher preferred_slots (soft constraint).

    Teachers may declare ``preferred_slots`` (a subset of ``available_slots``).
    Sessions placed in preferred slots score 1.0; sessions in available but
    non-preferred slots score 0.5; sessions with no preference data score 1.0.
    """
    if not assignments:
        return 0.0

    def _slots_from(raw: Any) -> set[Slot]:
        slots: set[Slot] = set()
        if isinstance(raw, dict):
            for day_key, slot_list in raw.items():
                for s in slot_list:
                    slots.add((int(day_key), int(s)))
        return slots

    scores: List[float] = []
    for session_id, assignment in assignments.items():
        session = sessions.get(session_id)
        if not session:
            continue
        teacher = teachers.get(session["teacher_id"])
        if not teacher:
            scores.append(1.0)
            continue
        pref_raw = teacher["preferred_slots"]
        if pref_raw is None or (isinstance(pref_raw, dict) and len(pref_raw) == 0):
            scores.append(1.0)
            continue

        pref = _slots_from(pref_raw)
        avail_raw = teacher.get("available_slots")
        if avail_raw is not None:
            avail = _slots_from(avail_raw)
            if pref == avail:
                scores.append(1.0)
                continue

        duration = session.get("duration_in_slots", 1)
        day = assignment["day"]
        start = assignment["start_slot"]
        slots = [(day, start + off) for off in range(duration)]
        if all(s in pref for s in slots):
            scores.append(1.0)
        else:
            scores.append(0.5)

    return sum(scores) / len(scores) if scores else 0.0


def compute_daily_balance_score(
    assignments: Mapping[str, Dict[str, Any]],
    grid: Mapping[str, Any],
) -> float:
    """Reward even distribution of sessions across days.

    Perfect balance (equal sessions per day) gives 1.0.  Heavy clustering on
    a few days reduces the score.
    """
    if not assignments:
        return 0.0

    day_count = len(grid.get("days", []))
    if day_count == 0:
        return 1.0

    counts = [0] * day_count
    for assignment in assignments.values():
        day = assignment.get("day", 0)
        if 0 <= day < day_count:
            counts[day] += 1

    total = sum(counts)
    if total == 0:
        return 1.0

    ideal = total / day_count
    deviation = sum(abs(c - ideal) for c in counts)
    max_deviation = 2.0 * total * (day_count - 1) / day_count
    if max_deviation == 0:
        return 1.0
    return max(0.0, 1.0 - deviation / max_deviation)


def compute_score_breakdown(
    task_name: str,
    conflicts: Mapping[str, int],
    coverage_score: float,
    compactness_score: float,
    room_fit_score: float,
    stability_score: float,
    total_assigned_sessions: int,
    total_assigned_slots: int,
    preference_score: float = 1.0,
    daily_balance_score: float = 1.0,
) -> Dict[str, float]:
    hard_constraint_score = compute_hard_constraint_score(
        conflicts, task_name, total_assigned_sessions, total_assigned_slots
    )

    if task_name == "easy":
        overall = (
            0.55 * hard_constraint_score
            + 0.25 * coverage_score
            + 0.10 * compactness_score
            + 0.10 * daily_balance_score
        )
    elif task_name == "medium":
        overall = (
            0.40 * hard_constraint_score
            + 0.20 * coverage_score
            + 0.10 * compactness_score
            + 0.10 * room_fit_score
            + 0.10 * preference_score
            + 0.10 * daily_balance_score
        )
    else:
        overall = (
            0.35 * hard_constraint_score
            + 0.15 * coverage_score
            + 0.10 * compactness_score
            + 0.10 * room_fit_score
            + 0.15 * stability_score
            + 0.05 * preference_score
            + 0.10 * daily_balance_score
        )

    return {
        "overall_score": max(0.0, min(1.0, overall)),
        "hard_constraint_score": hard_constraint_score,
        "coverage_score": coverage_score,
        "compactness_score": compactness_score,
        "room_fit_score": room_fit_score,
        "stability_score": stability_score,
        "preference_score": preference_score,
        "daily_balance_score": daily_balance_score,
    }
