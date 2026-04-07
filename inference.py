"""
Inference Script for Timetable Planner (Docker version)
=======================================================
Loads the environment from a Docker image via from_docker_image().

MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME     The Docker image name for the environment.

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
"""

import asyncio
import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from client import TimetablePlannerEnv
from models import TimetablePlannerAction

HF_SPACE_URL = "https://huggingface.co/spaces/sreeramajay/timetable_planner-env"

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASK_NAMES = os.getenv("TIMETABLE_PLANNER_TASKS", "easy,medium,hard,expert").split(",")
BENCHMARK = os.getenv("TIMETABLE_PLANNER_BENCHMARK", "timetable_planner")
MAX_STEPS = 30
TEMPERATURE = 0.3
MAX_TOKENS = 320
SUCCESS_SCORE_THRESHOLD = 0.85
DEBUG = os.getenv("DEBUG", "0") == "1"


def debug(msg: str) -> None:
    """Print debug info to stderr so it never pollutes structured stdout."""
    if DEBUG:
        print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)


SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are scheduling class sessions into a weekly timetable.
    Maximize overall_score (0.0–1.0). Reward = score delta per step.

    Actions (return ONE JSON object, nothing else) like below:

    {"action_type":"assign_session","session_id":"S1","day":0,"start_slot":0,"room_id":"R1"}
    {"action_type":"move_session","session_id":"S1","day":2,"start_slot":1,"room_id":"R2"}
    {"action_type":"swap_sessions","session_id":"S1","target_session_id":"S2"}
    {"action_type":"unassign_session","session_id":"S1"}
    {"action_type":"submit_timetable"}

    Rules:
    - day: integer 0-4 (Mon=0, Tue=1, Wed=2, Thu=3, Fri=4)
    - start_slot: integer 0 to slots_per_day-1
    - assign_session: session must be unassigned
    - move_session: session must be already assigned
    - swap_sessions: both sessions must be assigned (use target_session_id)
    - submit_timetable: ends episode, bonus if score >= 0.90

    Slot semantics: availability is hard; preferences are soft (if preferred == available, there is no extra preference).
    """
).strip()


# ---------------------------------------------------------------------------
# Logging helpers (stdout protocol)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def to_dict(item: Any) -> Dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "dict"):
        return item.dict()
    return dict(item)


def action_to_string(action: Dict[str, Any]) -> str:
    try:
        return json.dumps(action, separators=(",", ":"), sort_keys=True)
    except Exception:
        return str(action)


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


_VALID_ACTIONS = {
    "assign_session",
    "move_session",
    "swap_sessions",
    "unassign_session",
    "submit_timetable",
}


def normalize_action(action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not action or "action_type" not in action:
        return None
    action_type = action.get("action_type")
    if action_type not in _VALID_ACTIONS:
        return None
    normalized: Dict[str, Any] = {"action_type": action_type}
    if action_type in {"assign_session", "move_session"}:
        for key in ("session_id", "day", "start_slot", "room_id"):
            if key not in action:
                return None
            normalized[key] = action[key]
        return normalized
    if action_type == "swap_sessions":
        for key in ("session_id", "target_session_id"):
            if key not in action:
                return None
            normalized[key] = action[key]
        return normalized
    if action_type == "unassign_session":
        if "session_id" not in action:
            return None
        normalized["session_id"] = action["session_id"]
        return normalized
    return normalized


# ---------------------------------------------------------------------------
# Prompt building & model call
# ---------------------------------------------------------------------------


def build_user_prompt(
    obs: Any,
    last_action: Optional[Dict[str, Any]],
    last_reward: float,
    history: List[str],
) -> str:
    grid = to_dict(obs.grid) if hasattr(obs.grid, "model_dump") else (obs.grid or {})
    sessions = {s["session_id"]: s for s in (to_dict(i) for i in obs.sessions)}
    rooms = [to_dict(r) for r in obs.rooms]

    unscheduled = list(obs.unscheduled_sessions)
    unscheduled_preview = unscheduled[:8]
    timetable_preview = [to_dict(i) for i in obs.current_timetable[:12]]
    session_preview = []
    for sid in unscheduled_preview:
        s = sessions.get(sid)
        if s:
            session_preview.append(
                {
                    "session_id": s["session_id"],
                    "teacher_id": s["teacher_id"],
                    "group_id": s["group_id"],
                    "valid_room_types": s.get("valid_room_types", []),
                    "required_capacity": s.get("required_capacity", 0),
                    "duration_in_slots": s.get("duration_in_slots", 1),
                }
            )

    history_block = "\n".join(history[-6:]) if history else "None"
    last_action_str = action_to_string(last_action) if last_action else "null"
    rooms_summary = [
        {
            "room_id": r["room_id"],
            "room_type": r["room_type"],
            "capacity": r["capacity"],
        }
        for r in rooms
    ]

    return textwrap.dedent(
        f"""
        Task: {obs.task_name}
        Grid: {grid}
        Remaining steps: {obs.remaining_step_budget}
        Conflicts: {obs.conflict_summary}
        Scores: {obs.score_breakdown}
        Last action: {last_action_str}
        Last reward: {last_reward:.2f}
        Recent history:
        {history_block}
        Unscheduled: {unscheduled_preview}
        Session details: {session_preview}
        Timetable: {timetable_preview}
        Rooms: {rooms_summary}
        Timetable size: {len(obs.current_timetable)}
        """
    ).strip()


def get_model_action(
    client: OpenAI,
    obs: Any,
    last_action: Optional[Dict[str, Any]],
    last_reward: float,
    history: List[str],
) -> Dict[str, Any]:
    user_prompt = build_user_prompt(obs, last_action, last_reward, history)
    debug(f"PROMPT:\n{user_prompt}")
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        debug(f"LLM raw response: {text}")
        parsed = parse_action(text)
        debug(f"Parsed action: {parsed}")
        action = normalize_action(parsed or {})
        if action:
            debug(f"Normalized action: {action}")
            return action
        debug("Failed to normalize — falling back to submit_timetable")
    except Exception as exc:
        debug(f"Model request failed: {exc}")

    return {"action_type": "submit_timetable"}


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------


async def run_episode(
    env: Any,
    client: OpenAI,
    task_name: str,
) -> None:
    """Run one full episode for *task_name*, emitting [START] / [STEP]* / [END]."""
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_reward = 0.0
    last_action: Optional[Dict[str, Any]] = None
    obs = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_name=task_name)
        obs = result.observation
        debug(
            f"RESET: scenario={obs.scenario_id} task={obs.task_name} "
            f"grid={to_dict(obs.grid) if hasattr(obs.grid, 'model_dump') else obs.grid} "
            f"sessions={len(obs.sessions)} unscheduled={len(obs.unscheduled_sessions)} "
            f"budget={obs.remaining_step_budget} score={obs.score_breakdown.get('overall_score', 0):.3f}"
        )

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs = result.observation
            action_dict = get_model_action(
                client,
                obs,
                last_action,
                last_reward,
                history,
            )

            action = TimetablePlannerAction(**action_dict)
            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = obs.action_error

            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            last_action = action_dict

            log_step(
                step=step,
                action=action_to_string(action_dict),
                reward=reward,
                done=done,
                error=error,
            )

            debug(
                f"STEP {step}: score={obs.score_breakdown.get('overall_score', 0):.3f} "
                f"reward={reward:+.3f} conflicts={obs.conflict_summary} "
                f"unscheduled={len(obs.unscheduled_sessions)} error={error}"
            )

            history.append(
                f"Step {step}: action={action_to_string(action_dict)} "
                f"reward={reward:+.02f} score={obs.score_breakdown.get('overall_score', 0.0):.3f}"
            )

            if done:
                break

        if obs is not None and steps_taken > 0:
            score = float(obs.score_breakdown.get("overall_score", 0.0))
            score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if LOCAL_IMAGE_NAME:
        if LOCAL_IMAGE_NAME == "http://127.0.0.1:8000":
            env = TimetablePlannerEnv(base_url=LOCAL_IMAGE_NAME)
        else:
            env = await TimetablePlannerEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env = await TimetablePlannerEnv.from_env(
            HF_SPACE_URL,
            use_docker=False,
        )

    try:
        for task_name in TASK_NAMES:
            await run_episode(env, client, task_name)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (cleanup): {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
