---
title: Timetable Planner Environment Server
emoji: 📅
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Timetable Planner Environment

A deterministic benchmark for building weekly timetables. The environment assigns class sessions to time slots and rooms under teacher, student-group, and room constraints. It exposes a dense reward signal with a deterministic score in the range 0.0 to 1.0.

## Quick Start

```python
from timetable_planner import TimetablePlannerAction, TimetablePlannerEnv

with TimetablePlannerEnv(base_url="http://localhost:8000") as env:
    reset_result = env.reset()
    obs = reset_result.observation
    print(obs.task_name)
    print(obs.unscheduled_sessions)

    action = TimetablePlannerAction(
        action_type="assign_session",
        session_id=obs.unscheduled_sessions[0],
        day=0,
        start_slot=0,
        room_id=obs.rooms[0].room_id,
    )

    step_result = env.step(action)
    print(step_result.observation.score_breakdown)
```

## Actions

Use one of the following `action_type` values:

1. `assign_session` — place an **unassigned** session at (day, start_slot, room_id)
2. `move_session` — relocate an **already assigned** session to a new slot/room
3. `swap_sessions` — exchange time-slots of two assigned sessions (pass `session_id` and `target_session_id`)
4. `unassign_session` — remove a session from the timetable
5. `submit_timetable` — end the episode (bonus reward if score ≥ 0.90)

Payload fields:

- `session_id`: required for assign/move/unassign/swap
- `target_session_id`: required for swap
- `day`: required for assign/move
- `start_slot`: required for assign/move
- `room_id`: required for assign/move

## Observation

Each step returns:

- `task_name`
- `teachers`, `groups`, `rooms`, `sessions`
- `current_timetable`
- `unscheduled_sessions`
- `conflict_summary`
- `score_breakdown` (8 sub-scores including `preference_score` and `daily_balance_score`)
- `remaining_step_budget`
- `scenario_id`, `grid` (top-level, not in metadata)
- `action_error`, `action_penalty` (top-level, populated when an action fails)
- `baseline_timetable` (hard scenarios only — reference schedule for stability scoring)

The environment provides raw state and reward only — no strategy hints. The agent must learn from the `conflict_summary` counts and `score_breakdown` sub-scores which actions improve the timetable.

Availability semantics:
- `available_slots` is always present for teachers/rooms; only listed slots are allowed.
- `preferred_slots` is a soft preference; if it matches `available_slots`, there is no extra preference.
- `available_slots` and `preferred_slots` keys use day labels from `grid.days` (e.g., Mon, Tue).

## Tasks

The environment includes 12 hand-crafted scenarios (including `medium_lab_block_v1` and `hard_room_outage_v1`) plus 6 procedurally generated ones (deterministic seeds). Scenarios cycle round-robin on each `reset()`.

- **Easy** (step budget 20–25): overlap constraints only, all slots available
- **Medium** (step budget 30–35): adds room type/capacity, teacher/room availability, and teacher preferred slots (soft constraint)
- **Hard** (step budget 40–45): starts from a valid baseline, then applies a disruption (teacher/room unavailable, or new session added); scored on stability vs. baseline

## Scoring

Scores are deterministic in [0.0, 1.0] and combine 8 sub-scores:

| Sub-score | Easy | Medium | Hard |
|---|---|---|---|
| hard_constraint_score | 55% | 40% | 35% |
| coverage_score | 25% | 20% | 15% |
| compactness_score | 10% | 10% | 10% |
| room_fit_score | — | 10% | 10% |
| preference_score | — | 10% | 5% |
| daily_balance_score | 10% | 10% | 10% |
| stability_score | — | — | 15% |

Reward is `current_score - previous_score` with small penalties for invalid (-0.02) or no-op (-0.01) actions.

Notes:
- `compactness_score` is computed only over teacher/group day pairs with ≥2 scheduled sessions.
- `room_fit_score` is computed only over assignments that already satisfy room type and capacity.
- `stability_score` gives 1.0 for unchanged baseline placements, 0.5 for moved placements, and 0.0 for unassigned baseline sessions.

## Running Inference

### Prerequisites

Set the following environment variables (or create a `.env` file):

```bash
export HF_TOKEN="hf_..."            # Hugging Face / API key
export API_BASE_URL="https://router.huggingface.co/v1"  # LLM endpoint
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"           # Model to use
```

### Option A: Localhost (server running locally)

Start the server in one terminal:

```bash
cd timetable_planner
uv run --project . server
```

Run inference in another terminal:

```bash
python inference.py            # from the project root (outside timetable_planner/)
DEBUG=1 python inference.py    # with debug output on stderr
```

### Option B: Docker image

```bash
export IMAGE_NAME="timetable_planner-env:latest"
cd timetable_planner
python inference.py            # uses from_docker_image() internally
```

### Stdout output format

The script emits structured lines for automated parsing:

```
[START] task=timetable-planning env=timetable_planner model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"action_type":"assign_session",...} reward=0.03 done=false error=null
[STEP] step=2 ...
[END] success=true steps=11 score=0.973 rewards=0.03,0.03,...
```

## Running Tests

```bash
cd timetable_planner
PYTHONPATH=. .venv/bin/python -m pytest tests/ -v
```

### Smoke Test

```bash
cd timetable_planner
bash scripts/smoke_test.sh
```

## Building the Docker Image

```bash
docker build -t timetable_planner-env:latest .
```

## Deploying to Hugging Face Spaces

```bash
openenv push
```
