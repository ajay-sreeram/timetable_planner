# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the scoring module."""

import pytest

from server.scoring import (
    build_maps,
    build_occupancy,
    compute_compactness_score,
    compute_conflicts,
    compute_coverage_score,
    compute_daily_balance_score,
    compute_hard_constraint_score,
    compute_preference_score,
    compute_room_fit_score,
    compute_score_breakdown,
    compute_stability_score,
    count_overlaps,
    expand_assignment_slots,
    parse_available_slots,
)


# ---------------------------------------------------------------------------
# parse_available_slots
# ---------------------------------------------------------------------------

class TestParseAvailableSlots:
    def test_none_returns_all_slots(self):
        result = parse_available_slots(None, 3, 4)
        assert len(result) == 12
        assert (0, 0) in result
        assert (2, 3) in result

    def test_dict_format(self):
        raw = {"0": [0, 2], "2": [1]}
        result = parse_available_slots(raw, 3, 4)
        assert result == {(0, 0), (0, 2), (2, 1)}

    def test_list_of_pairs(self):
        raw = [[0, 1], [1, 2]]
        result = parse_available_slots(raw, 3, 4)
        assert result == {(0, 1), (1, 2)}


# ---------------------------------------------------------------------------
# build_maps / expand_assignment_slots
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_build_maps(self):
        items = [{"id": "a", "val": 1}, {"id": "b", "val": 2}]
        result = build_maps(items, "id")
        assert set(result.keys()) == {"a", "b"}
        assert result["a"]["val"] == 1

    def test_expand_assignment_slots(self):
        assert expand_assignment_slots(1, 3) == [1, 2, 3]
        assert expand_assignment_slots(0, 1) == [0]


# ---------------------------------------------------------------------------
# build_occupancy / count_overlaps
# ---------------------------------------------------------------------------

class TestOccupancy:
    def test_no_overlap(self):
        sessions = {
            "s1": {"session_id": "s1", "teacher_id": "T1", "group_id": "G1", "duration_in_slots": 1},
            "s2": {"session_id": "s2", "teacher_id": "T2", "group_id": "G2", "duration_in_slots": 1},
        }
        assignments = {
            "s1": {"session_id": "s1", "day": 0, "start_slot": 0, "room_id": "R1"},
            "s2": {"session_id": "s2", "day": 0, "start_slot": 1, "room_id": "R1"},
        }
        t, g, r = build_occupancy(assignments, sessions)
        assert count_overlaps(t) == 0
        assert count_overlaps(g) == 0
        assert count_overlaps(r) == 0

    def test_teacher_overlap(self):
        sessions = {
            "s1": {"session_id": "s1", "teacher_id": "T1", "group_id": "G1", "duration_in_slots": 1},
            "s2": {"session_id": "s2", "teacher_id": "T1", "group_id": "G2", "duration_in_slots": 1},
        }
        assignments = {
            "s1": {"session_id": "s1", "day": 0, "start_slot": 0, "room_id": "R1"},
            "s2": {"session_id": "s2", "day": 0, "start_slot": 0, "room_id": "R2"},
        }
        t, g, r = build_occupancy(assignments, sessions)
        assert count_overlaps(t) == 1


# ---------------------------------------------------------------------------
# compute_conflicts
# ---------------------------------------------------------------------------

class TestComputeConflicts:
    def _grid(self):
        return {"days": ["Mon", "Tue"], "slots_per_day": 3}

    def test_clean_schedule(self):
        sessions = {"s1": {"session_id": "s1", "teacher_id": "T1", "group_id": "G1",
                           "duration_in_slots": 1, "required_capacity": 10, "valid_room_types": ["lecture"]}}
        teachers = {"T1": {"teacher_id": "T1"}}
        rooms = {"R1": {"room_id": "R1", "capacity": 30, "room_type": "lecture"}}
        assignments = {"s1": {"session_id": "s1", "day": 0, "start_slot": 0, "room_id": "R1"}}
        conflicts = compute_conflicts(assignments, sessions, teachers, rooms, self._grid())
        assert all(v == 0 for v in conflicts.values())

    def test_capacity_violation(self):
        sessions = {"s1": {"session_id": "s1", "teacher_id": "T1", "group_id": "G1",
                           "duration_in_slots": 1, "required_capacity": 50, "valid_room_types": ["lecture"]}}
        teachers = {"T1": {"teacher_id": "T1"}}
        rooms = {"R1": {"room_id": "R1", "capacity": 20, "room_type": "lecture"}}
        assignments = {"s1": {"session_id": "s1", "day": 0, "start_slot": 0, "room_id": "R1"}}
        conflicts = compute_conflicts(assignments, sessions, teachers, rooms, self._grid())
        assert conflicts["room_capacity_violation"] == 1

    def test_type_violation(self):
        sessions = {"s1": {"session_id": "s1", "teacher_id": "T1", "group_id": "G1",
                           "duration_in_slots": 1, "required_capacity": 10, "valid_room_types": ["lab"]}}
        teachers = {"T1": {"teacher_id": "T1"}}
        rooms = {"R1": {"room_id": "R1", "capacity": 30, "room_type": "lecture"}}
        assignments = {"s1": {"session_id": "s1", "day": 0, "start_slot": 0, "room_id": "R1"}}
        conflicts = compute_conflicts(assignments, sessions, teachers, rooms, self._grid())
        assert conflicts["room_type_violation"] == 1


# ---------------------------------------------------------------------------
# Sub-score functions
# ---------------------------------------------------------------------------

class TestSubScores:
    def test_hard_constraint_perfect(self):
        conflicts = {"teacher_overlap": 0, "group_overlap": 0, "room_overlap": 0}
        assert compute_hard_constraint_score(conflicts, "easy", 5, 5) == 1.0

    def test_hard_constraint_with_violations(self):
        conflicts = {"teacher_overlap": 2, "group_overlap": 0, "room_overlap": 0}
        score = compute_hard_constraint_score(conflicts, "easy", 5, 5)
        assert 0.0 < score < 1.0

    def test_coverage_all_scheduled(self):
        sessions = {"s1": {"session_id": "s1", "must_schedule": True}}
        assignments = {"s1": {}}
        assert compute_coverage_score(sessions, assignments) == 1.0

    def test_coverage_none_scheduled(self):
        sessions = {"s1": {"session_id": "s1", "must_schedule": True}}
        assert compute_coverage_score(sessions, {}) == 0.0

    def test_coverage_optional_only(self):
        sessions = {"s1": {"session_id": "s1", "must_schedule": False}}
        assert compute_coverage_score(sessions, {}) == 1.0

    def test_room_fit_perfect(self):
        sessions = {"s1": {"session_id": "s1", "required_capacity": 20, "valid_room_types": ["lecture"]}}
        rooms = {"R1": {"room_id": "R1", "capacity": 20, "room_type": "lecture"}}
        assignments = {"s1": {"session_id": "s1", "room_id": "R1"}}
        score = compute_room_fit_score(assignments, sessions, rooms)
        assert score == 1.0

    def test_room_fit_empty(self):
        assert compute_room_fit_score({}, {}, {}) == 0.0

    def test_stability_unchanged(self):
        baseline = {"s1": {"day": 0, "start_slot": 0, "room_id": "R1"}}
        current = {"s1": {"day": 0, "start_slot": 0, "room_id": "R1"}}
        assert compute_stability_score(current, baseline) == 1.0

    def test_stability_changed(self):
        baseline = {"s1": {"day": 0, "start_slot": 0, "room_id": "R1"}}
        current = {"s1": {"day": 1, "start_slot": 0, "room_id": "R1"}}
        assert compute_stability_score(current, baseline) == 0.5

    def test_stability_no_baseline(self):
        assert compute_stability_score({}, {}) == 1.0

    def test_stability_unassigned(self):
        baseline = {"s1": {"day": 0, "start_slot": 0, "room_id": "R1"}}
        assert compute_stability_score({}, baseline) == 0.0


# ---------------------------------------------------------------------------
# New sub-scores
# ---------------------------------------------------------------------------

class TestNewSubScores:
    def test_preference_no_pref_data(self):
        sessions = {"s1": {"session_id": "s1", "teacher_id": "T1", "duration_in_slots": 1}}
        teachers = {"T1": {"teacher_id": "T1"}}
        assignments = {"s1": {"session_id": "s1", "day": 0, "start_slot": 0}}
        assert compute_preference_score(assignments, sessions, teachers) == 1.0

    def test_preference_in_preferred(self):
        sessions = {"s1": {"session_id": "s1", "teacher_id": "T1", "duration_in_slots": 1}}
        teachers = {"T1": {"teacher_id": "T1", "preferred_slots": {"0": [0, 1]}}}
        assignments = {"s1": {"session_id": "s1", "day": 0, "start_slot": 0}}
        assert compute_preference_score(assignments, sessions, teachers) == 1.0

    def test_preference_not_preferred(self):
        sessions = {"s1": {"session_id": "s1", "teacher_id": "T1", "duration_in_slots": 1}}
        teachers = {"T1": {"teacher_id": "T1", "preferred_slots": {"0": [2, 3]}}}
        assignments = {"s1": {"session_id": "s1", "day": 0, "start_slot": 0}}
        assert compute_preference_score(assignments, sessions, teachers) == 0.5

    def test_daily_balance_even(self):
        assignments = {
            "s1": {"day": 0}, "s2": {"day": 1}, "s3": {"day": 2},
        }
        grid = {"days": ["Mon", "Tue", "Wed"]}
        score = compute_daily_balance_score(assignments, grid)
        assert score == 1.0

    def test_daily_balance_clustered(self):
        assignments = {"s1": {"day": 0}, "s2": {"day": 0}, "s3": {"day": 0}}
        grid = {"days": ["Mon", "Tue", "Wed"]}
        score = compute_daily_balance_score(assignments, grid)
        assert score < 1.0

    def test_daily_balance_empty(self):
        assert compute_daily_balance_score({}, {"days": ["Mon"]}) == 0.0


# ---------------------------------------------------------------------------
# compute_score_breakdown
# ---------------------------------------------------------------------------

class TestScoreBreakdown:
    def test_easy_score_range(self):
        conflicts = {"teacher_overlap": 0, "group_overlap": 0, "room_overlap": 0}
        result = compute_score_breakdown("easy", conflicts, 1.0, 1.0, 1.0, 1.0, 5, 5)
        assert 0.0 <= result["overall_score"] <= 1.0

    def test_medium_includes_all_components(self):
        conflicts = {"teacher_overlap": 0, "group_overlap": 0, "room_overlap": 0,
                     "room_capacity_violation": 0, "room_type_violation": 0,
                     "teacher_availability_violation": 0, "room_availability_violation": 0}
        result = compute_score_breakdown("medium", conflicts, 1.0, 1.0, 1.0, 1.0, 5, 5)
        assert result["overall_score"] == pytest.approx(1.0, abs=0.01)

    def test_hard_includes_stability(self):
        conflicts = {"teacher_overlap": 0, "group_overlap": 0, "room_overlap": 0,
                     "room_capacity_violation": 0, "room_type_violation": 0,
                     "teacher_availability_violation": 0, "room_availability_violation": 0}
        result = compute_score_breakdown("hard", conflicts, 1.0, 1.0, 1.0, 0.5, 5, 5)
        assert result["stability_score"] == 0.5
        assert result["overall_score"] < 1.0

    def test_breakdown_has_all_keys(self):
        conflicts = {"teacher_overlap": 0, "group_overlap": 0, "room_overlap": 0}
        result = compute_score_breakdown("easy", conflicts, 1.0, 1.0, 1.0, 1.0, 5, 5)
        expected_keys = {
            "overall_score", "hard_constraint_score", "coverage_score",
            "compactness_score", "room_fit_score", "stability_score",
            "preference_score", "daily_balance_score",
        }
        assert set(result.keys()) == expected_keys
