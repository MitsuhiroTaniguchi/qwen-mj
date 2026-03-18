from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .environment import MahjongSelfPlayEnv
from .rules import PyMahjongRulesAdapter
from .types import Action, Phase, Seat, StepResult, Transition

PLAYER_COUNT = 4
DEFAULT_MAX_ROUND_WIND = 2


@dataclass(slots=True)
class MatchState:
    dealer: Seat = 0
    round_wind: int = 0
    honba: int = 0
    riichi_sticks: int = 0
    scores: list[int] = field(default_factory=lambda: [25000 for _ in range(PLAYER_COUNT)])
    hand_index: int = 0
    max_round_wind: int = DEFAULT_MAX_ROUND_WIND
    phase: Phase = Phase.DEALING
    terminated: bool = False
    truncated: bool = False


class MahjongMatchEnv:
    """A thin match-level wrapper around the hand-level self-play env."""

    def __init__(
        self,
        rules: PyMahjongRulesAdapter | None = None,
        seed: int | None = None,
        max_round_wind: int = DEFAULT_MAX_ROUND_WIND,
    ) -> None:
        self.rules = rules or PyMahjongRulesAdapter()
        self._seed = seed
        self.hand_env = MahjongSelfPlayEnv(rules=self.rules, seed=seed)
        self.state = MatchState(max_round_wind=max_round_wind)
        self.history: list[Transition] = self.hand_env.history
        self._last_hand_result: StepResult | None = None

    def reset(
        self,
        dealer: Seat = 0,
        seed: int | None = None,
        scores: list[int] | None = None,
        round_wind: int = 0,
        honba: int = 0,
        riichi_sticks: int = 0,
    ) -> dict[str, Any]:
        if seed is not None:
            self._seed = seed
        self.state = MatchState(
            dealer=dealer,
            round_wind=round_wind,
            honba=honba,
            riichi_sticks=riichi_sticks,
            scores=list(scores) if scores is not None else [25000 for _ in range(PLAYER_COUNT)],
            hand_index=0,
            max_round_wind=self.state.max_round_wind,
            phase=Phase.DEALING,
        )
        self.hand_env.reset(
            dealer=dealer,
            seed=seed,
            scores=self.state.scores,
            round_wind=round_wind,
            honba=honba,
            riichi_sticks=riichi_sticks,
        )
        self.history = self.hand_env.history
        self._last_hand_result = None
        return self.observe()

    def observe(self, seat: Seat | None = None) -> dict[str, Any]:
        hand_observation = self.hand_env.observe(seat=seat)
        return {
            "match": {
                "dealer": self.state.dealer,
                "round_wind": self.state.round_wind,
                "honba": self.state.honba,
                "riichi_sticks": self.state.riichi_sticks,
                "scores": list(self.state.scores),
                "hand_index": self.state.hand_index,
                "max_round_wind": self.state.max_round_wind,
                "phase": self.state.phase.value,
                "terminated": self.state.terminated,
                "truncated": self.state.truncated,
            },
            "hand": hand_observation,
        }

    def legal_actions(self, seat: Seat | None = None):
        return self.hand_env.legal_actions(seat=seat)

    def step(self, action: Action) -> StepResult:
        if self.state.terminated or self.state.truncated:
            raise ValueError("cannot step a finished match")
        result = self.hand_env.step(action)
        self.state.scores = list(self.hand_env.state.scores)
        self.state.riichi_sticks = self.hand_env.state.riichi_sticks
        self.state.honba = self.hand_env.state.honba
        self.state.phase = self.hand_env.state.phase
        self.state.terminated = self.hand_env.state.terminated
        self.state.truncated = self.hand_env.state.truncated
        self._last_hand_result = result if (result.terminated or result.truncated) else None
        return result

    def advance_hand(self) -> dict[str, Any]:
        if self._last_hand_result is None:
            if not (self.hand_env.state.terminated or self.hand_env.state.truncated):
                raise ValueError("current hand is not finished")
            raise ValueError("no finished hand result is available")

        result = self._last_hand_result
        outcome = result.info.get("settlement", {})
        dealer_repeats = False

        if result.terminated:
            wins = outcome.get("wins", [])
            dealer_repeats = any(win.get("seat") == self.state.dealer for win in wins)
        elif result.truncated:
            dealer_repeats = self.state.dealer in set(outcome.get("tenpai_seats", []))

        next_dealer = self.state.dealer if dealer_repeats else (self.state.dealer + 1) % PLAYER_COUNT
        next_round_wind = self.state.round_wind
        if not dealer_repeats and next_dealer == 0:
            next_round_wind += 1

        if dealer_repeats or result.truncated:
            next_honba = self.state.honba + 1
        else:
            next_honba = 0

        if next_round_wind >= self.state.max_round_wind:
            self.state.phase = Phase.GAME_END
            self.state.terminated = True
            self.state.truncated = False
            return self.observe()

        next_scores = list(self.state.scores)
        next_riichi_sticks = self.hand_env.state.riichi_sticks
        self.state = MatchState(
            dealer=next_dealer,
            round_wind=next_round_wind,
            honba=next_honba,
            riichi_sticks=next_riichi_sticks,
            scores=next_scores,
            hand_index=self.state.hand_index + 1,
            max_round_wind=self.state.max_round_wind,
            phase=Phase.DEALING,
        )
        self.hand_env.reset(
            dealer=next_dealer,
            scores=next_scores,
            round_wind=next_round_wind,
            honba=next_honba,
            riichi_sticks=next_riichi_sticks,
        )
        self.history = self.hand_env.history
        self._last_hand_result = None
        return self.observe()


__all__ = ["MahjongMatchEnv", "MatchState"]
