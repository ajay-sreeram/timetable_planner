# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Procedural scenario generation with feasibility guarantees."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

_ROOM_TYPES = ["lecture", "lab", "seminar"]
_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def generate_scenario(
    task_name: str = "easy",
    seed: Optional[int] = None,
    num_teachers: int = 0,
    num_groups: int = 0,
    num_rooms: int = 0,
    num_sessions: int = 0,
) -> Dict[str, Any]:
    """Generate a random but feasible scenario.

    Uses defaults per difficulty if counts are 0.  Always verifies that every
    ``must_schedule`` session has at least one valid placement so the scenario
    is never impossible.
    """
    rng = random.Random(seed)

    defaults = _difficulty_defaults(task_name)
    num_teachers = num_teachers or defaults["teachers"]
    num_groups = num_groups or defaults["groups"]
    num_rooms = num_rooms or defaults["rooms"]
    num_sessions = num_sessions or defaults["sessions"]

    slots_per_day = defaults["slots_per_day"]
    day_count = len(_DAY_NAMES)

    teachers = _generate_teachers(rng, num_teachers, task_name, day_count, slots_per_day)
    groups = _generate_groups(rng, num_groups)
    rooms = _generate_rooms(rng, num_rooms, task_name, day_count, slots_per_day)
    sessions = _generate_sessions(
        rng, num_sessions, teachers, groups, rooms, task_name, day_count, slots_per_day,
    )

    scenario: Dict[str, Any] = {
        "scenario_id": f"gen_{task_name}_{seed or rng.randint(0, 99999)}",
        "task_name": task_name,
        "grid": {"days": _DAY_NAMES[:day_count], "slots_per_day": slots_per_day},
        "step_budget": defaults["step_budget"],
        "teachers": teachers,
        "groups": groups,
        "rooms": rooms,
        "sessions": sessions,
        "initial_assignments": [],
    }

    if task_name in {"hard", "expert"}:
        baseline = _build_greedy_baseline(
            sessions, teachers, rooms, day_count, slots_per_day, rng,
        )
        scenario["baseline_assignments"] = baseline
        scenario["disruption"] = _generate_disruption(
            rng, teachers, groups, rooms, day_count, slots_per_day,
        )

    return scenario


def _difficulty_defaults(task_name: str) -> Dict[str, int]:
    if task_name == "easy":
        return {"teachers": 4, "groups": 4, "rooms": 5, "sessions": 14,
                "slots_per_day": 4, "step_budget": 25}
    if task_name == "medium":
        return {"teachers": 5, "groups": 5, "rooms": 6, "sessions": 18,
                "slots_per_day": 5, "step_budget": 35}
    if task_name == "expert":
        return {"teachers": 7, "groups": 7, "rooms": 8, "sessions": 24,
                "slots_per_day": 6, "step_budget": 55}
    return {"teachers": 6, "groups": 6, "rooms": 7, "sessions": 20,
            "slots_per_day": 5, "step_budget": 45}


def _generate_teachers(
    rng: random.Random,
    count: int,
    task_name: str,
    day_count: int,
    slots_per_day: int,
) -> List[Dict[str, Any]]:
    teachers = []
    for i in range(1, count + 1):
        teacher: Dict[str, Any] = {"teacher_id": f"T{i}"}
        if task_name != "easy":
            avail = _random_availability(rng, day_count, slots_per_day, min_days=3, min_slots_per_day=2)
            teacher["available_slots"] = avail
            pref: Dict[str, List[int]] = {}
            for day_key, slot_list in avail.items():
                if len(slot_list) > 1:
                    n = rng.randint(1, max(1, len(slot_list) - 1))
                    pref[day_key] = sorted(rng.sample(slot_list, n))
                else:
                    pref[day_key] = list(slot_list)
            teacher["preferred_slots"] = pref
        teachers.append(teacher)
    return teachers


def _generate_groups(rng: random.Random, count: int) -> List[Dict[str, Any]]:
    return [{"group_id": f"G{i}", "size": rng.randint(12, 32)} for i in range(1, count + 1)]


def _generate_rooms(
    rng: random.Random,
    count: int,
    task_name: str,
    day_count: int,
    slots_per_day: int,
) -> List[Dict[str, Any]]:
    rooms = []
    type_cycle = _ROOM_TYPES * ((count // len(_ROOM_TYPES)) + 1)
    rng.shuffle(type_cycle)
    for i in range(1, count + 1):
        room: Dict[str, Any] = {
            "room_id": f"R{i}",
            "capacity": rng.randint(18, 40),
            "room_type": type_cycle[i - 1],
        }
        if task_name != "easy":
            room["available_slots"] = _random_availability(
                rng, day_count, slots_per_day, min_days=3, min_slots_per_day=2,
            )
        rooms.append(room)
    return rooms


def _random_availability(
    rng: random.Random,
    day_count: int,
    slots_per_day: int,
    min_days: int = 3,
    min_slots_per_day: int = 2,
) -> Dict[str, List[int]]:
    days = list(range(day_count))
    num_days = rng.randint(min_days, day_count)
    chosen_days = sorted(rng.sample(days, num_days))
    avail: Dict[str, List[int]] = {}
    all_slots = list(range(slots_per_day))
    for d in chosen_days:
        n = rng.randint(min_slots_per_day, slots_per_day)
        avail[str(d)] = sorted(rng.sample(all_slots, n))
    return avail


def _generate_sessions(
    rng: random.Random,
    count: int,
    teachers: List[Dict[str, Any]],
    groups: List[Dict[str, Any]],
    rooms: List[Dict[str, Any]],
    task_name: str,
    day_count: int,
    slots_per_day: int,
) -> List[Dict[str, Any]]:
    room_types_available = list({r["room_type"] for r in rooms})
    rooms_by_type: Dict[str, List[Dict[str, Any]]] = {}
    for r in rooms:
        rooms_by_type.setdefault(r["room_type"], []).append(r)

    sessions: List[Dict[str, Any]] = []
    idx = 0
    max_attempts_total = count * 20
    attempts = 0

    while len(sessions) < count and attempts < max_attempts_total:
        attempts += 1
        idx += 1
        teacher = rng.choice(teachers)
        group = rng.choice(groups)
        room_type = rng.choice(room_types_available)
        compatible_rooms = rooms_by_type.get(room_type, [])
        if not compatible_rooms:
            continue

        max_cap = max(r["capacity"] for r in compatible_rooms)
        required_cap = min(group["size"], max_cap)
        duration = 1 if task_name == "easy" else rng.choice([1, 1, 1, 2])
        must_schedule = rng.random() < 0.85

        session = {
            "session_id": f"S{idx}",
            "teacher_id": teacher["teacher_id"],
            "group_id": group["group_id"],
            "valid_room_types": [room_type],
            "required_capacity": required_cap,
            "duration_in_slots": duration,
            "must_schedule": must_schedule,
        }

        if must_schedule and not _session_is_feasible(
            session, teacher, compatible_rooms, day_count, slots_per_day,
        ):
            continue

        sessions.append(session)

    return sessions


def _session_is_feasible(
    session: Dict[str, Any],
    teacher: Dict[str, Any],
    compatible_rooms: List[Dict[str, Any]],
    day_count: int,
    slots_per_day: int,
) -> bool:
    """Check that at least one (day, slot, room) placement exists."""
    duration = session.get("duration_in_slots", 1)
    required_cap = session.get("required_capacity", 0)

    teacher_avail = _parse_avail(teacher.get("available_slots"), day_count, slots_per_day)

    for room in compatible_rooms:
        if room["capacity"] < required_cap:
            continue
        room_avail = _parse_avail(room.get("available_slots"), day_count, slots_per_day)
        for day in range(day_count):
            for start in range(slots_per_day - duration + 1):
                slots = [(day, start + off) for off in range(duration)]
                if all(s in teacher_avail and s in room_avail for s in slots):
                    return True
    return False


def _parse_avail(
    raw: Any, day_count: int, slots_per_day: int,
) -> set[Tuple[int, int]]:
    if raw is None:
        return {(d, s) for d in range(day_count) for s in range(slots_per_day)}
    result: set[Tuple[int, int]] = set()
    if isinstance(raw, dict):
        for day_key, slot_list in raw.items():
            for s in slot_list:
                result.add((int(day_key), int(s)))
    return result


def _build_greedy_baseline(
    sessions: List[Dict[str, Any]],
    teachers: List[Dict[str, Any]],
    rooms: List[Dict[str, Any]],
    day_count: int,
    slots_per_day: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Build a valid baseline assignment greedily (for hard scenarios)."""
    teacher_map = {t["teacher_id"]: t for t in teachers}
    rooms_by_type: Dict[str, List[Dict[str, Any]]] = {}
    for r in rooms:
        rooms_by_type.setdefault(r["room_type"], []).append(r)

    teacher_occ: set[Tuple[str, int, int]] = set()
    group_occ: set[Tuple[str, int, int]] = set()
    room_occ: set[Tuple[str, int, int]] = set()
    assignments: List[Dict[str, Any]] = []

    must_schedule = [s for s in sessions if s.get("must_schedule", True)]
    rng.shuffle(must_schedule)

    for session in must_schedule:
        duration = session.get("duration_in_slots", 1)
        required_cap = session.get("required_capacity", 0)
        teacher = teacher_map.get(session["teacher_id"])
        if not teacher:
            continue
        teacher_avail = _parse_avail(teacher.get("available_slots"), day_count, slots_per_day)
        placed = False

        valid_types = session.get("valid_room_types", [])
        candidate_rooms = []
        for rt in valid_types:
            candidate_rooms.extend(rooms_by_type.get(rt, []))

        for room in candidate_rooms:
            if room["capacity"] < required_cap:
                continue
            room_avail = _parse_avail(room.get("available_slots"), day_count, slots_per_day)
            for day in range(day_count):
                for start in range(slots_per_day - duration + 1):
                    slots = [(day, start + off) for off in range(duration)]
                    if not all(s in teacher_avail and s in room_avail for s in slots):
                        continue
                    if any((session["teacher_id"], d, s) in teacher_occ for d, s in slots):
                        continue
                    if any((session["group_id"], d, s) in group_occ for d, s in slots):
                        continue
                    if any((room["room_id"], d, s) in room_occ for d, s in slots):
                        continue
                    for d, s in slots:
                        teacher_occ.add((session["teacher_id"], d, s))
                        group_occ.add((session["group_id"], d, s))
                        room_occ.add((room["room_id"], d, s))
                    assignments.append({
                        "session_id": session["session_id"],
                        "day": day,
                        "start_slot": start,
                        "room_id": room["room_id"],
                    })
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break

    return assignments


def _generate_disruption(
    rng: random.Random,
    teachers: List[Dict[str, Any]],
    groups: List[Dict[str, Any]],
    rooms: List[Dict[str, Any]],
    day_count: int,
    slots_per_day: int,
) -> Dict[str, Any]:
    disruption_type = rng.choice(["teacher_unavailable", "room_unavailable", "new_session_added"])

    if disruption_type == "teacher_unavailable":
        teacher = rng.choice(teachers)
        avail = _parse_avail(teacher.get("available_slots"), day_count, slots_per_day)
        if avail:
            remove_count = max(1, len(avail) // 6)
            to_remove = rng.sample(sorted(avail), min(remove_count, len(avail)))
            remove_dict: Dict[str, List[int]] = {}
            for d, s in to_remove:
                remove_dict.setdefault(str(d), []).append(s)
            return {"type": "teacher_unavailable", "teacher_id": teacher["teacher_id"], "slots": remove_dict}

    if disruption_type == "room_unavailable":
        room = rng.choice(rooms)
        avail = _parse_avail(room.get("available_slots"), day_count, slots_per_day)
        if avail:
            remove_count = max(1, len(avail) // 6)
            to_remove = rng.sample(sorted(avail), min(remove_count, len(avail)))
            remove_dict = {}
            for d, s in to_remove:
                remove_dict.setdefault(str(d), []).append(s)
            return {"type": "room_unavailable", "room_id": room["room_id"], "slots": remove_dict}

    room_types_available = list({r["room_type"] for r in rooms})
    rooms_by_type: Dict[str, List[Dict[str, Any]]] = {}
    for r in rooms:
        rooms_by_type.setdefault(r["room_type"], []).append(r)
    rt = rng.choice(room_types_available)
    max_cap = max(r["capacity"] for r in rooms_by_type[rt])
    return {
        "type": "new_session_added",
        "session": {
            "session_id": f"S_new_{rng.randint(100, 999)}",
            "teacher_id": rng.choice(teachers)["teacher_id"],
            "group_id": rng.choice(groups)["group_id"],
            "valid_room_types": [rt],
            "required_capacity": min(rng.randint(14, 28), max_cap),
            "duration_in_slots": 1,
            "must_schedule": True,
        },
    }
