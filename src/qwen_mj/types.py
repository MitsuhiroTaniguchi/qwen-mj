from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

Seat = int
Tile = int


class Phase(str, Enum):
    DEALING = "dealing"
    DRAW = "draw"
    SELF_DECISION = "self_decision"
    REACTION = "reaction"
    HAND_END = "hand_end"
    ROUND_END = "round_end"
    GAME_END = "game_end"


class ActionKind(str, Enum):
    DISCARD = "discard"
    RIICHI = "riichi"
    TSUMO = "tsumo"
    ANKAN = "ankan"
    KAKAN = "kakan"
    CHI = "chi"
    PON = "pon"
    MINKAN = "minkan"
    RON = "ron"
    PASS = "pass"


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    tile: Tile | None = None
    source_seat: Seat | None = None
    bias: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StepResult:
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass(slots=True)
class Transition:
    seat: Seat
    action: Action
    reward: float
    observation: dict[str, Any]
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)

