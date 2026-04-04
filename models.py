# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Timetable Planner Environment.

The timetable_planner environment is a deterministic benchmark for scheduling
classes into rooms and time slots.
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class Teacher(BaseModel):
    teacher_id: str
    available_slots: Optional[Dict[str, List[int]]] = None
    preferred_slots: Optional[Dict[str, List[int]]] = None


class StudentGroup(BaseModel):
    group_id: str
    size: int


class Room(BaseModel):
    room_id: str
    capacity: int
    room_type: str
    available_slots: Optional[Dict[str, List[int]]] = None


class ClassSession(BaseModel):
    session_id: str
    teacher_id: str
    group_id: str
    valid_room_types: List[str]
    required_capacity: int
    duration_in_slots: int = 1
    must_schedule: bool = True


class Assignment(BaseModel):
    session_id: str
    day: int
    start_slot: int
    room_id: str


class TimetablePlannerAction(Action):
    """Action for the Timetable Planner environment."""

    action_type: Optional[Any] = Field(
        default=None,
        description=(
            "assign_session, move_session, swap_sessions, "
            "unassign_session, or submit_timetable"
        ),
    )
    session_id: Optional[Any] = Field(default=None, description="Session identifier")
    day: Optional[Any] = Field(default=None, description="Day index (0-based)")
    start_slot: Optional[Any] = Field(default=None, description="Start slot index (0-based)")
    room_id: Optional[Any] = Field(
        default=None,
        description="Room identifier (for assign/move)",
    )
    target_session_id: Optional[Any] = Field(
        default=None,
        description="Target session id (for swap_sessions)",
    )


class Grid(BaseModel):
    days: List[str] = Field(default_factory=list)
    slots_per_day: int = 0


class TimetablePlannerObservation(Observation):
    """Observation from the Timetable Planner environment."""

    task_name: str = Field(default="", description="easy, medium, or hard")
    scenario_id: str = Field(default="", description="Identifier for the current scenario")
    grid: Grid = Field(default_factory=Grid, description="Day names and slots per day")
    teachers: List[Teacher] = Field(default_factory=list)
    groups: List[StudentGroup] = Field(default_factory=list)
    rooms: List[Room] = Field(default_factory=list)
    sessions: List[ClassSession] = Field(default_factory=list)
    current_timetable: List[Assignment] = Field(default_factory=list)
    baseline_timetable: List[Assignment] = Field(
        default_factory=list,
        description="Reference timetable for hard scenarios — place sessions close to this for stability bonus",
    )
    unscheduled_sessions: List[str] = Field(default_factory=list)
    conflict_summary: Dict[str, int] = Field(default_factory=dict)
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    remaining_step_budget: int = Field(default=0)
    action_error: Optional[str] = Field(default=None, description="Error from the last action, if any")
    action_penalty: Optional[float] = Field(default=None, description="Penalty applied, if any")
