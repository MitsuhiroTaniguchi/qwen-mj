from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .rules import PyMahjongRulesAdapter
from .types import Action, ActionKind, Phase, Seat, StepResult, Transition


@dataclass(slots=True)
class TableState:
    dealer: Seat = 0
    phase: Phase = Phase.DEALING
    current_seat: Seat = 0
    honba: int = 0
    riichi_sticks: int = 0
    wall_remaining: int = 70
    scores: list[int] = field(default_factory=lambda: [25000, 25000, 25000, 25000])
    hands: list[list[int]] = field(default_factory=lambda: [[] for _ in range(4)])
    discards: list[list[int]] = field(default_factory=lambda: [[] for _ in range(4)])
    melds: list[list[dict[str, Any]]] = field(default_factory=lambda: [[] for _ in range(4)])
    last_discard: tuple[Seat, int] | None = None
    pending_reactions: list[Seat] = field(default_factory=list)
    turn_index: int = 0


class MahjongSelfPlayEnv:
    """A Tenhou-rule environment scaffold.

    This class intentionally separates table flow from rule evaluation.
    The first implementation target is a reproducible self-play harness,
    not a complete model-facing API.
    """

    def __init__(self, rules: PyMahjongRulesAdapter | None = None):
        self.rules = rules or PyMahjongRulesAdapter()
        self.state = TableState()
        self.history: list[Transition] = []

    def reset(self, dealer: Seat = 0) -> dict[str, Any]:
        self.state = TableState(dealer=dealer, current_seat=dealer)
        self.history.clear()
        self.state.phase = Phase.DRAW
        return self.observe()

    def observe(self, seat: Seat | None = None) -> dict[str, Any]:
        seat = self.state.current_seat if seat is None else seat
        return {
            "seat": seat,
            "phase": self.state.phase.value,
            "dealer": self.state.dealer,
            "current_seat": self.state.current_seat,
            "honba": self.state.honba,
            "riichi_sticks": self.state.riichi_sticks,
            "wall_remaining": self.state.wall_remaining,
            "scores": list(self.state.scores),
            "hands": [list(hand) for hand in self.state.hands],
            "discards": [list(discard) for discard in self.state.discards],
            "melds": [[dict(meld) for meld in melds] for melds in self.state.melds],
            "phase_name": self.state.phase.value,
        }

    def legal_actions(self, seat: Seat | None = None) -> list[Action]:
        seat = self.state.current_seat if seat is None else seat
        if self.state.phase == Phase.DRAW:
            return [Action(ActionKind.PASS), Action(ActionKind.TSUMO)]
        if self.state.phase == Phase.SELF_DECISION:
            return [Action(ActionKind.DISCARD), Action(ActionKind.RIICHI)]
        if self.state.phase == Phase.REACTION:
            return [Action(ActionKind.PASS), Action(ActionKind.RON)]
        return [Action(ActionKind.PASS)]

    def step(self, action: Action) -> StepResult:
        actor_seat = self.state.current_seat
        info: dict[str, Any] = {"phase_before": self.state.phase.value}
        info["actor_seat"] = actor_seat
        reward = 0.0
        terminated = False
        truncated = False

        if action.kind == ActionKind.PASS:
            self._advance_phase()
        elif action.kind == ActionKind.DISCARD:
            self._record_discard(action)
            self._advance_after_discard()
        elif action.kind in {ActionKind.TSUMO, ActionKind.RON}:
            self.state.phase = Phase.HAND_END
            reward = 1.0
            terminated = True
        else:
            info["note"] = "action accepted by scaffold, but full rule resolution is not implemented yet"
            self._advance_phase()

        observation = self.observe()
        result = StepResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        self.history.append(
            Transition(
                seat=actor_seat,
                action=action,
                reward=reward,
                observation=observation,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        )
        return result

    def _record_discard(self, action: Action) -> None:
        seat = self.state.current_seat
        tile = 0 if action.tile is None else action.tile
        self.state.discards[seat].append(tile)
        self.state.last_discard = (seat, tile)

    def _advance_after_discard(self) -> None:
        self.state.pending_reactions = [((self.state.current_seat + offset) % 4) for offset in range(1, 4)]
        self.state.phase = Phase.REACTION
        self.state.current_seat = self.state.pending_reactions[0]

    def _advance_phase(self) -> None:
        if self.state.phase == Phase.REACTION and self.state.pending_reactions:
            self.state.pending_reactions.pop(0)
            if self.state.pending_reactions:
                self.state.current_seat = self.state.pending_reactions[0]
                return
            self.state.phase = Phase.DRAW
            self.state.current_seat = (self.state.current_seat + 1) % 4
            return

        if self.state.phase in {Phase.DEALING, Phase.DRAW, Phase.SELF_DECISION}:
            self.state.phase = Phase.DRAW
            self.state.current_seat = (self.state.current_seat + 1) % 4
