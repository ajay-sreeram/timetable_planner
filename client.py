# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Timetable Planner Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import TimetablePlannerAction, TimetablePlannerObservation
except ImportError:
    from models import TimetablePlannerAction, TimetablePlannerObservation


class TimetablePlannerEnv(
    EnvClient[TimetablePlannerAction, TimetablePlannerObservation, State]
):
    """
    Client for the Timetable Planner Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.
    """

    def _step_payload(self, action: TimetablePlannerAction) -> Dict:
        """
        Convert TimetablePlannerAction to JSON payload for step message.

        Args:
            action: TimetablePlannerAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload = {
            "action_type": action.action_type,
        }
        if action.session_id is not None:
            payload["session_id"] = action.session_id
        if action.day is not None:
            payload["day"] = action.day
        if action.start_slot is not None:
            payload["start_slot"] = action.start_slot
        if action.room_id is not None:
            payload["room_id"] = action.room_id
        if action.target_session_id is not None:
            payload["target_session_id"] = action.target_session_id
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[TimetablePlannerObservation]:
        """
        Parse server response into StepResult[TimetablePlannerObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with TimetablePlannerObservation
        """
        obs_data = payload.get("observation", {})
        observation = TimetablePlannerObservation(
            task_name=obs_data.get("task_name", ""),
            scenario_id=obs_data.get("scenario_id", ""),
            grid=obs_data.get("grid", {}),
            teachers=obs_data.get("teachers", []),
            groups=obs_data.get("groups", []),
            rooms=obs_data.get("rooms", []),
            sessions=obs_data.get("sessions", []),
            current_timetable=obs_data.get("current_timetable", []),
            baseline_timetable=obs_data.get("baseline_timetable", []),
            unscheduled_sessions=obs_data.get("unscheduled_sessions", []),
            conflict_summary=obs_data.get("conflict_summary", {}),
            score_breakdown=obs_data.get("score_breakdown", {}),
            remaining_step_budget=obs_data.get("remaining_step_budget", 0),
            action_error=obs_data.get("action_error"),
            action_penalty=obs_data.get("action_penalty"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
