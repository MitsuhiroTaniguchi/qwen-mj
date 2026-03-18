from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from random import Random
from typing import Any, Protocol, Sequence, TextIO

import numpy as np

from .encoding import ObservationEncoder
from .environment import MahjongSelfPlayEnv
from .match import MahjongMatchEnv
from .types import Action, Transition


class Policy(Protocol):
    def select_action(self, observation: dict[str, Any], legal_actions: Sequence[Action]) -> Action: ...


@dataclass(slots=True)
class FirstLegalPolicy:
    def select_action(self, observation: dict[str, Any], legal_actions: Sequence[Action]) -> Action:
        if not legal_actions:
            raise ValueError("no legal actions available")
        return legal_actions[0]


@dataclass(slots=True)
class RandomPolicy:
    seed: int | None = None
    _rng: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = Random(self.seed)

    def select_action(self, observation: dict[str, Any], legal_actions: Sequence[Action]) -> Action:
        if not legal_actions:
            raise ValueError("no legal actions available")
        return self._rng.choice(list(legal_actions))


class JsonlRolloutLogger:
    def __init__(self, path: str | Path | TextIO):
        self._owns_handle = False
        if hasattr(path, "write"):
            self._handle = path  # type: ignore[assignment]
        else:
            self._handle = Path(path).open("a", encoding="utf-8")
            self._owns_handle = True

    def write(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._owns_handle:
            self._handle.close()

    def __enter__(self) -> "JsonlRolloutLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _policy_for_seat(policies: Policy | Sequence[Policy], seat: int) -> Policy:
    if isinstance(policies, Sequence):
        return policies[seat]
    return policies


def _serialize_action(action: Action) -> dict[str, Any]:
    return {
        "kind": action.kind.value,
        "tile": action.tile,
        "source_seat": action.source_seat,
        "bias": action.bias,
        "meta": dict(action.meta),
    }


def _serialize_transition(transition: Transition) -> dict[str, Any]:
    return {
        "seat": transition.seat,
        "action": _serialize_action(transition.action),
        "reward": transition.reward,
        "terminated": transition.terminated,
        "truncated": transition.truncated,
        "info": dict(transition.info),
        "observation": transition.observation,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def play_hand(
    env: MahjongSelfPlayEnv,
    policies: Policy | Sequence[Policy],
    reset_kwargs: dict[str, Any] | None = None,
    logger: JsonlRolloutLogger | None = None,
    encoder: ObservationEncoder | None = None,
    episode_id: int = 0,
    max_steps: int = 1000,
) -> dict[str, Any]:
    observation = env.reset(**(reset_kwargs or {}))
    records: list[dict[str, Any]] = []
    encoder = encoder or ObservationEncoder()

    for step_index in range(max_steps):
        seat = env.state.current_seat
        legal_actions = env.legal_actions(seat)
        policy = _policy_for_seat(policies, seat)
        action = policy.select_action(observation, legal_actions)
        result = env.step(action)
        record = {
            "episode_id": episode_id,
            "step_index": step_index,
            "seat": seat,
            "action": _serialize_action(action),
            "result": {
                "reward": result.reward,
                "terminated": result.terminated,
                "truncated": result.truncated,
                "info": dict(result.info),
            },
            "observation": result.observation,
            "encoded": encoder.encode(result.observation),
            "text": encoder.render_text(result.observation, env.legal_actions(result.observation["current_seat"])),
        }
        records.append(record)
        if logger is not None:
            logger.write(record)
        if result.terminated or result.truncated:
            break
        observation = result.observation

    return {
        "final_observation": env.observe(),
        "records": records,
        "transitions": [_serialize_transition(item) for item in env.history],
    }


def play_match(
    env: MahjongMatchEnv,
    policies: Policy | Sequence[Policy],
    reset_kwargs: dict[str, Any] | None = None,
    logger: JsonlRolloutLogger | None = None,
    encoder: ObservationEncoder | None = None,
    episode_id: int = 0,
    max_steps: int = 10000,
) -> dict[str, Any]:
    observation = env.reset(**(reset_kwargs or {}))
    records: list[dict[str, Any]] = []
    encoder = encoder or ObservationEncoder()

    for step_index in range(max_steps):
        if env.state.terminated or env.state.truncated:
            break
        if env.hand_env.state.terminated or env.hand_env.state.truncated:
            observation = env.advance_hand()
            continue

        seat = env.hand_env.state.current_seat
        legal_actions = env.legal_actions(seat)
        policy = _policy_for_seat(policies, seat)
        action = policy.select_action(env.hand_env.observe(seat), legal_actions)
        result = env.step(action)
        hand_observation = result.observation
        match_observation = env.observe()
        record = {
            "episode_id": episode_id,
            "step_index": step_index,
            "seat": seat,
            "action": _serialize_action(action),
            "result": {
                "reward": result.reward,
                "terminated": result.terminated,
                "truncated": result.truncated,
                "info": dict(result.info),
            },
            "observation": {
                "match": match_observation["match"],
                "hand": hand_observation,
            },
            "encoded": encoder.encode(hand_observation),
            "text": encoder.render_text(hand_observation, env.legal_actions(hand_observation["current_seat"])),
        }
        records.append(record)
        if logger is not None:
            logger.write(record)
        observation = match_observation

    return {
        "final_observation": observation,
        "records": records,
        "transitions": [_serialize_transition(item) for item in env.history],
    }


__all__ = [
    "FirstLegalPolicy",
    "JsonlRolloutLogger",
    "Policy",
    "RandomPolicy",
    "play_hand",
    "play_match",
]
