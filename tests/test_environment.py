# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the TimetablePlannerEnvironment."""

import pytest

from models import TimetablePlannerAction
from server.timetable_planner_environment import TimetablePlannerEnvironment


@pytest.fixture
def env():
    return TimetablePlannerEnvironment()


class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset()
        assert obs.task_name in {"easy", "medium", "hard"}
        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.remaining_step_budget > 0

    def test_reset_produces_sessions(self, env):
        obs = env.reset()
        assert len(obs.sessions) > 0
        assert len(obs.rooms) > 0
        assert len(obs.teachers) > 0

    def test_reset_score_breakdown(self, env):
        obs = env.reset()
        assert "overall_score" in obs.score_breakdown
        assert 0.0 <= obs.score_breakdown["overall_score"] <= 1.0

    def test_multiple_resets_cycle(self, env):
        ids = set()
        for _ in range(20):
            obs = env.reset()
            ids.add(obs.scenario_id)
        assert len(ids) > 1

    def test_reset_has_grid(self, env):
        obs = env.reset()
        assert obs.grid is not None
        assert len(obs.grid.days) > 0
        assert obs.grid.slots_per_day > 0

    def test_reset_has_scenario_id(self, env):
        obs = env.reset()
        assert obs.scenario_id != ""


class TestStep:
    def test_invalid_action_type(self, env):
        env.reset()
        action = TimetablePlannerAction(action_type="invalid_type")
        obs = env.step(action)
        assert obs.reward < 0
        assert obs.action_error is not None

    def test_assign_session(self, env):
        obs = env.reset()
        if not obs.unscheduled_sessions:
            return
        session_id = obs.unscheduled_sessions[0]
        room_id = obs.rooms[0].room_id
        action = TimetablePlannerAction(
            action_type="assign_session",
            session_id=session_id,
            day=0,
            start_slot=0,
            room_id=room_id,
        )
        obs2 = env.step(action)
        assert obs2.remaining_step_budget < obs.remaining_step_budget

    def test_assign_already_assigned_fails(self, env):
        obs = env.reset()
        if not obs.unscheduled_sessions:
            return
        session_id = obs.unscheduled_sessions[0]
        room_id = obs.rooms[0].room_id
        action = TimetablePlannerAction(
            action_type="assign_session",
            session_id=session_id, day=0, start_slot=0, room_id=room_id,
        )
        env.step(action)
        obs2 = env.step(action)
        assert obs2.action_error is not None
        assert "already assigned" in obs2.action_error

    def test_move_unassigned_fails(self, env):
        obs = env.reset()
        if not obs.unscheduled_sessions:
            return
        session_id = obs.unscheduled_sessions[0]
        room_id = obs.rooms[0].room_id
        action = TimetablePlannerAction(
            action_type="move_session",
            session_id=session_id, day=0, start_slot=0, room_id=room_id,
        )
        obs2 = env.step(action)
        assert obs2.action_error is not None
        assert "not assigned" in obs2.action_error

    def test_move_assigned_succeeds(self, env):
        obs = env.reset()
        if not obs.unscheduled_sessions:
            return
        session_id = obs.unscheduled_sessions[0]
        room_id = obs.rooms[0].room_id
        env.step(TimetablePlannerAction(
            action_type="assign_session",
            session_id=session_id, day=0, start_slot=0, room_id=room_id,
        ))
        obs2 = env.step(TimetablePlannerAction(
            action_type="move_session",
            session_id=session_id, day=1, start_slot=0, room_id=room_id,
        ))
        assert obs2.action_error is None

    def test_unassign_session(self, env):
        obs = env.reset()
        if not obs.unscheduled_sessions:
            return
        session_id = obs.unscheduled_sessions[0]
        room_id = obs.rooms[0].room_id
        env.step(TimetablePlannerAction(
            action_type="assign_session",
            session_id=session_id, day=0, start_slot=0, room_id=room_id,
        ))
        obs2 = env.step(TimetablePlannerAction(
            action_type="unassign_session", session_id=session_id,
        ))
        assert session_id in obs2.unscheduled_sessions

    def test_unassign_noop(self, env):
        obs = env.reset()
        if not obs.unscheduled_sessions:
            return
        session_id = obs.unscheduled_sessions[0]
        obs2 = env.step(TimetablePlannerAction(
            action_type="unassign_session", session_id=session_id,
        ))
        assert obs2.reward < 0

    def test_submit_ends_episode(self, env):
        env.reset()
        obs = env.step(TimetablePlannerAction(action_type="submit_timetable"))
        assert obs.done is True

    def test_step_budget_exhaustion(self, env):
        obs = env.reset()
        budget = obs.remaining_step_budget
        for _ in range(budget + 2):
            obs = env.step(TimetablePlannerAction(action_type="submit_timetable"))
            if obs.done:
                break
        assert obs.done is True

    def test_swap_sessions(self, env):
        obs = env.reset()
        if len(obs.unscheduled_sessions) < 2:
            return
        s1 = obs.unscheduled_sessions[0]
        s2 = obs.unscheduled_sessions[1]
        r1 = obs.rooms[0].room_id
        r2 = obs.rooms[1].room_id if len(obs.rooms) > 1 else r1

        env.step(TimetablePlannerAction(
            action_type="assign_session", session_id=s1, day=0, start_slot=0, room_id=r1,
        ))
        env.step(TimetablePlannerAction(
            action_type="assign_session", session_id=s2, day=1, start_slot=0, room_id=r2,
        ))
        obs3 = env.step(TimetablePlannerAction(
            action_type="swap_sessions", session_id=s1, target_session_id=s2,
        ))
        assert obs3.action_error is None

    def test_swap_unassigned_fails(self, env):
        obs = env.reset()
        if len(obs.unscheduled_sessions) < 2:
            return
        s1 = obs.unscheduled_sessions[0]
        s2 = obs.unscheduled_sessions[1]
        obs2 = env.step(TimetablePlannerAction(
            action_type="swap_sessions", session_id=s1, target_session_id=s2,
        ))
        assert obs2.action_error is not None

    def test_unknown_session_id(self, env):
        env.reset()
        obs = env.step(TimetablePlannerAction(
            action_type="assign_session",
            session_id="NONEXISTENT", day=0, start_slot=0, room_id="R1",
        ))
        assert obs.action_error is not None

    def test_day_out_of_range(self, env):
        obs = env.reset()
        if not obs.unscheduled_sessions:
            return
        session_id = obs.unscheduled_sessions[0]
        room_id = obs.rooms[0].room_id
        obs2 = env.step(TimetablePlannerAction(
            action_type="assign_session",
            session_id=session_id, day=99, start_slot=0, room_id=room_id,
        ))
        assert obs2.action_error is not None


class TestScoreRange:
    def test_scores_between_0_and_1(self, env):
        obs = env.reset()
        for key, val in obs.score_breakdown.items():
            assert 0.0 <= val <= 1.0, f"{key} = {val}"

    def test_reward_is_float(self, env):
        obs = env.reset()
        assert isinstance(obs.reward, float)
