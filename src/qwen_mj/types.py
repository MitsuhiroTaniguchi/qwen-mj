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
    KYUSHUKYUHAI = "kyushukyuhai"
    PENUKI = "penuki"


@dataclass(frozen=True, slots=True)
class TileInstance:
    tile: Tile
    red: bool = False


@dataclass(frozen=True, slots=True)
class Meld:
    kind: ActionKind
    tile: Tile
    source_seat: Seat | None = None
    bias: int | None = None
    red: bool = False


@dataclass(slots=True)
class PlayerState:
    tiles: list[TileInstance] = field(default_factory=list)
    melds: list[Meld] = field(default_factory=list)
    discards: list[TileInstance] = field(default_factory=list)
    riichi: bool = False
    riichi_declared_turn: int | None = None
    closed_kans: int = 0
    open_meld_count: int = 0
    open_pon_tiles: list[int] = field(default_factory=list)
    own_discard_tiles: list[int] = field(default_factory=list)
    temp_furiten_tiles: list[int] = field(default_factory=list)
    first_turn_open_calls_seen: bool = False
    drawn_tile: TileInstance | None = None
    last_discard: TileInstance | None = None
    has_drawn_this_turn: bool = False
    is_menzen_locked: bool = False
    last_call_kind: ActionKind | None = None
    last_call_source: Seat | None = None
    last_call_tile: Tile | None = None


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


@dataclass(slots=True)
class ReactionOpportunity:
    seat: Seat
    kind: ActionKind
    mask: int
    source_seat: Seat
    tile: Tile
    bias: int | None = None


@dataclass(slots=True)
class WinEvent:
    seat: Seat
    action: Action
    tsumo: bool
    tile: Tile
    source_seat: Seat | None = None
    has_hupai: bool = False
    yaku: list[tuple[str, int]] = field(default_factory=list)
    fanshu: int = 0
    fu: int = 0
    damanguan: int = 0
