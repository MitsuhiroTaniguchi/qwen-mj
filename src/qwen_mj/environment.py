from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any

from .rules import PyMahjongRulesAdapter
from .types import (
    Action,
    ActionKind,
    Meld,
    Phase,
    PlayerState,
    ReactionOpportunity,
    Seat,
    StepResult,
    TileInstance,
    Transition,
    WinEvent,
)

RED_TILE_INDICES = {4, 13, 22}
NUM_TILES = 34
PLAYER_COUNT = 4
INITIAL_HAND_SIZE = 13
STARTING_SCORE = 25000
RIICHI_STICK_COST = 1000


@dataclass(slots=True)
class TableState:
    dealer: Seat = 0
    round_wind: int = 0
    honba: int = 0
    riichi_sticks: int = 0
    current_seat: Seat = 0
    phase: Phase = Phase.DEALING
    live_wall: list[TileInstance] = field(default_factory=list)
    dead_wall: list[TileInstance] = field(default_factory=list)
    players: list[PlayerState] = field(default_factory=lambda: [PlayerState() for _ in range(PLAYER_COUNT)])
    scores: list[int] = field(default_factory=lambda: [STARTING_SCORE for _ in range(PLAYER_COUNT)])
    last_discard: tuple[Seat, TileInstance] | None = None
    reaction_queue: list[ReactionOpportunity] = field(default_factory=list)
    pending_wins: list[WinEvent] = field(default_factory=list)
    pending_call_action: Action | None = None
    pending_call_seat: Seat | None = None
    pending_draw_from_dead_wall: bool = False
    last_draw_was_gangzimo: bool = False
    first_turn_open_calls_seen: bool = False
    turn_index: int = 0
    terminated: bool = False
    truncated: bool = False


class MahjongSelfPlayEnv:
    """A Tenhou-rule self-play scaffold.

    The environment focuses on legal turn progression and rule-adapter checks.
    Point settlement is intentionally kept conservative until a full scoring
    layer is added, so the environment does not invent payments it cannot
    verify.
    """

    def __init__(self, rules: PyMahjongRulesAdapter | None = None, seed: int | None = None):
        self.rules = rules or PyMahjongRulesAdapter()
        self._seed = seed
        self._rng = Random(seed)
        self.state = TableState()
        self.history: list[Transition] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
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
            self._rng = Random(seed)

        deck = self._build_deck()
        self.state = TableState(
            dealer=dealer,
            round_wind=round_wind,
            honba=honba,
            riichi_sticks=riichi_sticks,
            scores=list(scores) if scores is not None else [STARTING_SCORE for _ in range(PLAYER_COUNT)],
        )
        self.history.clear()

        self.state.live_wall = deck[:-14]
        self.state.dead_wall = deck[-14:]
        self.state.current_seat = dealer
        self.state.phase = Phase.SELF_DECISION

        # Deal 13 tiles to each player, then the dealer's extra tile.
        for _ in range(INITIAL_HAND_SIZE):
            for seat in self._deal_order_from_dealer(dealer):
                self._draw_from_live_wall(seat)

        # Dealer starts with the 14th tile and must decide how to discard it.
        self._draw_from_live_wall(dealer, is_initial_dealer_draw=True)

        for seat in range(PLAYER_COUNT):
            self.state.players[seat].has_drawn_this_turn = seat == dealer

        return self.observe()

    def observe(self, seat: Seat | None = None) -> dict[str, Any]:
        seat = self.state.current_seat if seat is None else seat
        player = self.state.players[seat]
        return {
            "seat": seat,
            "phase": self.state.phase.value,
            "dealer": self.state.dealer,
            "current_seat": self.state.current_seat,
            "round_wind": self.state.round_wind,
            "honba": self.state.honba,
            "riichi_sticks": self.state.riichi_sticks,
            "turn_index": self.state.turn_index,
            "scores": list(self.state.scores),
            "wall_remaining": len(self.state.live_wall),
            "dead_wall_remaining": len(self.state.dead_wall),
            "last_discard": self._serialize_tile_instance(self.state.last_discard[1]) if self.state.last_discard else None,
            "reaction_queue": [
                {
                    "seat": opportunity.seat,
                    "kind": opportunity.kind.value,
                    "mask": opportunity.mask,
                    "source_seat": opportunity.source_seat,
                    "tile": opportunity.tile,
                    "bias": opportunity.bias,
                }
                for opportunity in self.state.reaction_queue
            ],
            "pending_wins": [
                {
                    "seat": win.seat,
                    "kind": win.action.kind.value,
                    "tile": win.tile,
                    "source_seat": win.source_seat,
                    "has_hupai": win.has_hupai,
                    "yaku": list(win.yaku),
                }
                for win in self.state.pending_wins
            ],
            "self": self._serialize_player(seat, private=True),
            "players": [self._serialize_player(i, private=False) for i in range(PLAYER_COUNT)],
            "terminated": self.state.terminated,
            "truncated": self.state.truncated,
        }

    def legal_actions(self, seat: Seat | None = None) -> list[Action]:
        seat = self.state.current_seat if seat is None else seat
        if seat != self.state.current_seat:
            return []

        if self.state.terminated or self.state.truncated:
            return []

        if self.state.phase == Phase.DRAW:
            return [Action(ActionKind.PASS)]

        if self.state.phase == Phase.SELF_DECISION:
            return self._legal_self_actions(seat)

        if self.state.phase == Phase.REACTION:
            return self._legal_reaction_actions()

        return []

    def step(self, action: Action) -> StepResult:
        if self.state.terminated or self.state.truncated:
            raise ValueError("cannot step a finished environment")

        actor = self.state.current_seat
        info: dict[str, Any] = {
            "actor_seat": actor,
            "phase_before": self.state.phase.value,
        }
        reward = 0.0
        terminated = False
        truncated = False

        if self.state.phase == Phase.DRAW:
            if action.kind != ActionKind.PASS:
                raise ValueError(f"expected PASS during DRAW phase, got {action.kind.value}")
            if self.state.pending_draw_from_dead_wall:
                if not self.state.dead_wall:
                    settlement = self._settle_exhaustive_draw()
                    self.state.phase = Phase.ROUND_END
                    self.state.truncated = True
                    truncated = True
                    info["exhausted"] = "dead_wall"
                    info["settlement"] = settlement
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
                            seat=actor,
                            action=action,
                            reward=reward,
                            observation=observation,
                            terminated=terminated,
                            truncated=truncated,
                            info=info,
                        )
                    )
                    return result
            elif not self.state.live_wall:
                settlement = self._settle_exhaustive_draw()
                self.state.phase = Phase.ROUND_END
                self.state.truncated = True
                truncated = True
                info["exhausted"] = "live_wall"
                info["settlement"] = settlement
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
                        seat=actor,
                        action=action,
                        reward=reward,
                        observation=observation,
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                    )
                )
                return result
            self._perform_draw(actor)
            info["drawn_tile"] = self._serialize_tile_instance(self.state.players[actor].drawn_tile)
            self.state.phase = Phase.SELF_DECISION

        elif self.state.phase == Phase.SELF_DECISION:
            result = self._step_self_decision(actor, action)
            reward = result["reward"]
            terminated = result["terminated"]
            truncated = result["truncated"]
            info.update(result["info"])

        elif self.state.phase == Phase.REACTION:
            result = self._step_reaction(actor, action)
            reward = result["reward"]
            terminated = result["terminated"]
            truncated = result["truncated"]
            info.update(result["info"])

        else:
            raise ValueError(f"unsupported phase: {self.state.phase.value}")

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
                seat=actor,
                action=action,
                reward=reward,
                observation=observation,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        )
        return result

    # ------------------------------------------------------------------
    # Reset / setup helpers
    # ------------------------------------------------------------------
    def _build_deck(self) -> list[TileInstance]:
        deck: list[TileInstance] = []
        for tile in range(NUM_TILES):
            for copy_index in range(4):
                deck.append(TileInstance(tile=tile, red=tile in RED_TILE_INDICES and copy_index == 0))
        self._rng.shuffle(deck)
        return deck

    def _deal_order_from_dealer(self, dealer: Seat) -> list[Seat]:
        return [(dealer + offset) % PLAYER_COUNT for offset in range(PLAYER_COUNT)]

    def _draw_from_live_wall(self, seat: Seat, is_initial_dealer_draw: bool = False) -> TileInstance:
        if not self.state.live_wall:
            raise ValueError("live wall is empty")
        tile = self.state.live_wall.pop(0)
        player = self.state.players[seat]
        player.tiles.append(tile)
        player.drawn_tile = tile
        player.has_drawn_this_turn = True
        if is_initial_dealer_draw:
            player.is_menzen_locked = False
        return tile

    def _draw_from_dead_wall(self, seat: Seat) -> TileInstance:
        if not self.state.dead_wall:
            raise ValueError("dead wall is empty")
        tile = self.state.dead_wall.pop(0)
        player = self.state.players[seat]
        player.tiles.append(tile)
        player.drawn_tile = tile
        player.has_drawn_this_turn = True
        self.state.last_draw_was_gangzimo = True
        return tile

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def _serialize_tile_instance(self, tile: TileInstance | None, red: bool | None = None) -> dict[str, Any] | None:
        if tile is None:
            return None
        return {"tile": tile.tile, "red": tile.red if red is None else red}

    def _serialize_player(self, seat: Seat, private: bool) -> dict[str, Any]:
        player = self.state.players[seat]
        counts = self._player_counts(player)
        result = {
            "seat": seat,
            "hand_counts": list(counts) if private else None,
            "hand_size": len(player.tiles),
            "melds": [self._serialize_meld(meld) for meld in player.melds],
            "discards": [self._serialize_tile_instance(tile) for tile in player.discards],
            "riichi": player.riichi,
            "riichi_declared_turn": player.riichi_declared_turn,
            "closed_kans": player.closed_kans,
            "open_meld_count": player.open_meld_count,
            "open_pon_tiles": list(player.open_pon_tiles),
            "own_discard_tiles": list(player.own_discard_tiles),
            "temp_furiten_tiles": list(player.temp_furiten_tiles),
            "first_turn_open_calls_seen": player.first_turn_open_calls_seen,
            "has_drawn_this_turn": player.has_drawn_this_turn,
            "drawn_tile": self._serialize_tile_instance(player.drawn_tile) if private else None,
            "last_discard": self._serialize_tile_instance(player.last_discard) if private else None,
            "is_menzen": self._player_is_menzen(player),
            "is_furiten": self._player_is_furiten(seat),
        }
        if not private:
            result["hand_size"] = len(player.tiles)
        return result

    def _serialize_meld(self, meld: Meld) -> dict[str, Any]:
        return {
            "kind": meld.kind.value,
            "tile": meld.tile,
            "source_seat": meld.source_seat,
            "bias": meld.bias,
            "red": meld.red,
        }

    # ------------------------------------------------------------------
    # State conversion helpers
    # ------------------------------------------------------------------
    def _player_counts(self, player: PlayerState) -> list[int]:
        counts = [0] * NUM_TILES
        for tile in player.tiles:
            counts[tile.tile] += 1
        return counts

    def _player_meld_tuples(self, player: PlayerState) -> list[tuple[int, int]]:
        pm = self.rules.pm
        melds: list[tuple[int, int]] = []
        for meld in player.melds:
            if meld.kind == ActionKind.CHI:
                melds.append((pm.FuluType.chi.value, meld.tile))
            elif meld.kind == ActionKind.PON:
                melds.append((pm.FuluType.peng.value, meld.tile))
            elif meld.kind in {ActionKind.MINKAN, ActionKind.KAKAN}:
                melds.append((pm.FuluType.minggang.value, meld.tile))
            elif meld.kind == ActionKind.ANKAN:
                melds.append((pm.FuluType.angang.value, meld.tile))
        return melds

    def _player_shoupai(self, player: PlayerState):
        pm = self.rules.pm
        counts = self._player_counts(player)
        fulu = [self._meld_to_mianzi(meld) for meld in player.melds]
        return pm.Shoupai(tuple(counts), fulu)

    def _player_shoupai_with_tile(self, player: PlayerState, tile: int):
        pm = self.rules.pm
        counts = self._player_counts(player)
        counts[tile] += 1
        fulu = [self._meld_to_mianzi(meld) for meld in player.melds]
        return pm.Shoupai(tuple(counts), fulu)

    def _meld_to_mianzi(self, meld: Meld):
        pm = self.rules.pm
        if meld.kind == ActionKind.CHI:
            return pm.Mianzi(pm.FuluType.chi, meld.tile)
        if meld.kind == ActionKind.PON:
            return pm.Mianzi(pm.FuluType.peng, meld.tile)
        if meld.kind in {ActionKind.MINKAN, ActionKind.KAKAN}:
            return pm.Mianzi(pm.FuluType.minggang, meld.tile)
        if meld.kind == ActionKind.ANKAN:
            return pm.Mianzi(pm.FuluType.angang, meld.tile)
        raise ValueError(f"unsupported meld kind: {meld.kind.value}")

    def _player_tuple_for_reaction(self, seat: Seat) -> tuple[Any, ...]:
        player = self.state.players[seat]
        counts = self._player_counts(player)
        melds = self._player_meld_tuples(player)
        return (
            tuple(counts),
            melds,
            player.riichi,
            self._player_is_menzen(player),
            self._player_is_furiten(seat),
            self.state.scores[seat],
            player.closed_kans,
        )

    def _player_is_menzen(self, player: PlayerState) -> bool:
        return player.open_meld_count == 0

    def _player_wait_mask(self, seat: Seat) -> int:
        player = self.state.players[seat]
        counts = self._player_counts(player)
        meld_count = len(player.melds)
        return self.rules.wait_mask(tuple(counts), meld_count)

    def _player_is_furiten(self, seat: Seat) -> bool:
        player = self.state.players[seat]
        wait_mask = self._player_wait_mask(seat)
        if wait_mask == 0:
            return False
        blocked_tiles = set(player.own_discard_tiles) | set(player.temp_furiten_tiles)
        return any(((wait_mask >> tile) & 1) for tile in blocked_tiles)

    def _current_self_option_mask(self, seat: Seat) -> int:
        player = self.state.players[seat]
        counts = self._player_counts(player)
        melds = self._player_meld_tuples(player)
        drawn_tile = player.drawn_tile.tile if player.drawn_tile is not None else 0
        pm = self.rules.pm
        return self.rules.compute_self_options(
            tuple(counts),
            melds,
            drawn_tile,
            player.riichi,
            self.state.scores[seat],
            self.state.round_wind,
            seat,
            len(self.state.live_wall),
            player.closed_kans,
            player.open_meld_count,
            list(player.open_pon_tiles),
            player.first_turn_open_calls_seen is False and self.state.turn_index == 0 and seat == self.state.dealer,
            player.first_turn_open_calls_seen,
            self.state.last_draw_was_gangzimo,
            False,
        )

    def _can_win(self, seat: Seat, tile: int, is_tsumo: bool) -> bool:
        player = self.state.players[seat]
        counts = self._player_counts(player)
        if not is_tsumo:
            counts[tile] += 1
        melds = self._player_meld_tuples(player)
        return self.rules.has_hupai(
            tuple(counts),
            melds,
            tile,
            is_tsumo,
            self._player_is_menzen(player),
            player.riichi,
            self.state.round_wind,
            seat,
            len(self.state.live_wall) == 0,
            self.state.last_draw_was_gangzimo and is_tsumo,
            False,
        )

    def _win_event(self, seat: Seat, action: Action, tile: int, source_seat: Seat | None, tsumo: bool) -> WinEvent:
        player = self.state.players[seat]
        shoupai = self._player_shoupai(player) if tsumo else self._player_shoupai_with_tile(player, tile)
        pm = self.rules.pm
        option = pm.HuleOption(self.state.round_wind, seat)
        option.is_menqian = self._player_is_menzen(player)
        option.is_lizhi = player.riichi
        option.is_lingshang = tsumo and self.state.last_draw_was_gangzimo
        option.is_haidi = len(self.state.live_wall) == 0
        option.is_qianggang = source_seat is not None and action.kind == ActionKind.RON and self.state.pending_call_action is not None
        result = pm.Hule(shoupai, pm.Action(pm.ActionType.zimohu if tsumo else pm.ActionType.ronghu, tile), option)
        return WinEvent(
            seat=seat,
            action=action,
            tsumo=tsumo,
            tile=tile,
            source_seat=source_seat,
            has_hupai=bool(result.has_hupai),
            yaku=result.hupai.tolist() if hasattr(result.hupai, "tolist") else [],
            fanshu=int(result.fanshu),
            fu=int(result.fu),
            damanguan=int(result.damanguan),
        )

    # ------------------------------------------------------------------
    # Legal action enumeration
    # ------------------------------------------------------------------
    def _legal_self_actions(self, seat: Seat) -> list[Action]:
        player = self.state.players[seat]
        mask = self._current_self_option_mask(seat)
        actions: list[Action] = []

        if player.riichi and player.drawn_tile is not None:
            tile_index = self._tile_instance_index(player, player.drawn_tile)
            if tile_index is not None:
                actions.append(
                    Action(
                        ActionKind.DISCARD,
                        tile=player.drawn_tile.tile,
                        meta={"tile_index": tile_index, "red": player.drawn_tile.red, "riichi_locked": True},
                    )
                )
            if mask & self.rules.pm.SELF_OPT_TSUMO:
                actions.append(Action(ActionKind.TSUMO, tile=player.drawn_tile.tile, meta={"from_draw": True}))
            return actions

        for index, tile in enumerate(player.tiles):
            actions.append(
                Action(
                    ActionKind.DISCARD,
                    tile=tile.tile,
                    meta={"tile_index": index, "red": tile.red},
                )
            )

        if mask & self.rules.pm.SELF_OPT_TSUMO:
            drawn = player.drawn_tile
            actions.append(Action(ActionKind.TSUMO, tile=drawn.tile if drawn else None, meta={"from_draw": True}))

        if mask & self.rules.pm.SELF_OPT_RIICHI:
            for action in self._riichi_discard_actions(seat):
                actions.append(action)

        if mask & self.rules.pm.SELF_OPT_ANKAN:
            for action in self._ankan_actions(seat):
                actions.append(action)

        if mask & self.rules.pm.SELF_OPT_KAKAN:
            for action in self._kakan_actions(seat):
                actions.append(action)

        if mask & self.rules.pm.SELF_OPT_KYUSHUKYUHAI:
            actions.append(Action(ActionKind.KYUSHUKYUHAI))

        if mask & self.rules.pm.SELF_OPT_PENUKI:
            actions.append(Action(ActionKind.PENUKI))

        return actions

    def _riichi_discard_actions(self, seat: Seat) -> list[Action]:
        player = self.state.players[seat]
        if player.riichi:
            return []
        actions: list[Action] = []
        for index, tile in enumerate(player.tiles):
            if self._discard_keeps_tenpai(player, index):
                actions.append(
                    Action(
                        ActionKind.RIICHI,
                        tile=tile.tile,
                        meta={"tile_index": index, "red": tile.red},
                    )
                )
        return actions

    def _discard_keeps_tenpai(self, player: PlayerState, tile_index: int) -> bool:
        if tile_index < 0 or tile_index >= len(player.tiles):
            return False
        reduced_tiles = [tile for idx, tile in enumerate(player.tiles) if idx != tile_index]
        counts = [0] * NUM_TILES
        for tile in reduced_tiles:
            counts[tile.tile] += 1
        wait_mask = self.rules.wait_mask(tuple(counts), len(player.melds))
        return wait_mask != 0

    def _ankan_actions(self, seat: Seat) -> list[Action]:
        player = self.state.players[seat]
        actions: list[Action] = []
        counts = self._player_counts(player)
        for tile, count in enumerate(counts):
            if count >= 4:
                actions.append(Action(ActionKind.ANKAN, tile=tile, meta={"tile": tile}))
        return actions

    def _kakan_actions(self, seat: Seat) -> list[Action]:
        player = self.state.players[seat]
        actions: list[Action] = []
        counts = self._player_counts(player)
        for tile in sorted(set(player.open_pon_tiles)):
            if counts[tile] >= 1:
                actions.append(Action(ActionKind.KAKAN, tile=tile, meta={"tile": tile}))
        return actions

    def _reaction_priority(self, kind: ActionKind) -> int:
        if kind == ActionKind.MINKAN:
            return 3
        if kind == ActionKind.PON:
            return 2
        if kind == ActionKind.CHI:
            return 1
        return 0

    def _legal_reaction_actions(self) -> list[Action]:
        if not self.state.reaction_queue:
            return [Action(ActionKind.PASS)]

        opportunity = self.state.reaction_queue[0]
        player = self.state.players[opportunity.seat]
        actions = [Action(ActionKind.PASS)]

        if opportunity.kind == ActionKind.RON:
            actions.append(Action(ActionKind.RON, tile=opportunity.tile, source_seat=opportunity.source_seat))
            return actions

        if opportunity.mask & self.rules.pm.REACT_OPT_MINKAN:
            actions.append(Action(ActionKind.MINKAN, tile=opportunity.tile, source_seat=opportunity.source_seat))
        if opportunity.mask & self.rules.pm.REACT_OPT_PON:
            actions.append(Action(ActionKind.PON, tile=opportunity.tile, source_seat=opportunity.source_seat))
        if opportunity.mask & self.rules.pm.REACT_OPT_CHI:
            actions.extend(self._chi_actions_for_opportunity(opportunity.seat, opportunity.tile, opportunity.source_seat))

        return actions

    def _chi_actions_for_opportunity(self, seat: Seat, tile: int, source_seat: Seat) -> list[Action]:
        if tile >= 27:
            return []
        player = self.state.players[seat]
        counts = self._player_counts(player)
        actions: list[Action] = []

        for bias in (0, 1, 2):
            try:
                needed = self._chi_tiles_for_bias(tile, bias)
            except ValueError:
                continue
            if all(counts[idx] >= 1 for idx in needed):
                actions.append(Action(ActionKind.CHI, tile=tile, source_seat=source_seat, bias=bias, meta={"needed": needed}))
        return actions

    # ------------------------------------------------------------------
    # Turn transitions
    # ------------------------------------------------------------------
    def _step_self_decision(self, actor: Seat, action: Action) -> dict[str, Any]:
        player = self.state.players[actor]
        info: dict[str, Any] = {}
        reward = 0.0
        terminated = False
        truncated = False

        if action.kind == ActionKind.TSUMO:
            if not self._can_win(actor, player.drawn_tile.tile if player.drawn_tile else 0, True):
                raise ValueError("illegal tsumo")
            win = self._win_event(actor, action, player.drawn_tile.tile if player.drawn_tile else 0, None, True)
            self.state.pending_wins = [win]
            settlement = self._settle_pending_wins()
            reward = 1.0
            terminated = True
            self.state.terminated = True
            self.state.phase = Phase.HAND_END
            info["win"] = self._serialize_win(win)
            info["settlement"] = settlement
            return {"reward": reward, "terminated": terminated, "truncated": truncated, "info": info}

        if action.kind == ActionKind.KYUSHUKYUHAI:
            self.state.truncated = True
            self.state.phase = Phase.ROUND_END
            truncated = True
            info["abort"] = "kyushukyuhai"
            return {"reward": reward, "terminated": terminated, "truncated": truncated, "info": info}

        if action.kind == ActionKind.ANKAN:
            self._apply_ankan(actor, action)
            self._prepare_rinshan_draw(actor)
            info["kan"] = "ankan"
            return {"reward": reward, "terminated": terminated, "truncated": truncated, "info": info}

        if action.kind == ActionKind.KAKAN:
            self._apply_kakan(actor, action)
            self._prepare_rinshan_draw(actor)
            info["kan"] = "kakan"
            return {"reward": reward, "terminated": terminated, "truncated": truncated, "info": info}

        if action.kind in {ActionKind.DISCARD, ActionKind.RIICHI}:
            discard_result = self._apply_discard(actor, action)
            info.update(discard_result)
            self._prepare_reaction_queue(actor, discard_result["tile"].tile)
            if self.state.reaction_queue:
                self.state.phase = Phase.REACTION
                self.state.current_seat = self.state.reaction_queue[0].seat
            else:
                self._advance_to_next_draw(after_discard=True)
            return {"reward": reward, "terminated": terminated, "truncated": truncated, "info": info}

        raise ValueError(f"illegal action in self decision: {action.kind.value}")

    def _step_reaction(self, actor: Seat, action: Action) -> dict[str, Any]:
        if not self.state.reaction_queue:
            raise ValueError("reaction queue is empty")
        opportunity = self.state.reaction_queue[0]
        if actor != opportunity.seat:
            raise ValueError("unexpected reacting seat")

        info: dict[str, Any] = {"opportunity": self._serialize_reaction_opportunity(opportunity)}
        reward = 0.0
        terminated = False
        truncated = False

        if action.kind == ActionKind.PASS:
            if opportunity.kind == ActionKind.RON:
                player = self.state.players[actor]
                if opportunity.tile not in player.temp_furiten_tiles:
                    player.temp_furiten_tiles.append(opportunity.tile)
            self.state.reaction_queue.pop(0)
            if not self.state.reaction_queue:
                if self.state.pending_wins:
                    settlement = self._settle_pending_wins()
                    info["settlement"] = settlement
                    self.state.phase = Phase.HAND_END
                    self.state.terminated = True
                    terminated = True
                else:
                    if self.state.pending_call_action is not None:
                        self._resolve_pending_call_without_reactions()
                        info["resolved_call"] = self.state.pending_call_action.kind.value if self.state.pending_call_action else None
                    else:
                        self._advance_to_next_draw(after_discard=True)
            else:
                self.state.current_seat = self.state.reaction_queue[0].seat
            return {"reward": reward, "terminated": terminated, "truncated": truncated, "info": info}

        if opportunity.kind == ActionKind.RON:
            if action.kind != ActionKind.RON:
                raise ValueError("expected ron or pass in ron opportunity")
            if not self._can_win(actor, opportunity.tile, False):
                raise ValueError("illegal ron")
            win = self._win_event(actor, action, opportunity.tile, opportunity.source_seat, False)
            self.state.pending_wins.append(win)
            info["win"] = self._serialize_win(win)
            self.state.reaction_queue.pop(0)
            if not self.state.reaction_queue:
                settlement = self._settle_pending_wins()
                info["settlement"] = settlement
                self.state.phase = Phase.HAND_END
                self.state.terminated = True
                terminated = True
            else:
                self.state.current_seat = self.state.reaction_queue[0].seat
            reward = 1.0
            return {"reward": reward, "terminated": terminated, "truncated": truncated, "info": info}

        if action.kind not in {ActionKind.CHI, ActionKind.PON, ActionKind.MINKAN}:
            raise ValueError(f"illegal reaction action: {action.kind.value}")

        if action.kind == ActionKind.CHI and not (opportunity.mask & self.rules.pm.REACT_OPT_CHI):
            raise ValueError("chi is not legal")
        if action.kind == ActionKind.PON and not (opportunity.mask & self.rules.pm.REACT_OPT_PON):
            raise ValueError("pon is not legal")
        if action.kind == ActionKind.MINKAN and not (opportunity.mask & self.rules.pm.REACT_OPT_MINKAN):
            raise ValueError("minkan is not legal")

        self._apply_call(opportunity.seat, action, opportunity)
        self.state.reaction_queue.clear()
        self.state.pending_call_action = action
        self.state.pending_call_seat = opportunity.seat
        if action.kind in {ActionKind.CHI, ActionKind.PON}:
            self.state.current_seat = opportunity.seat
            self.state.phase = Phase.SELF_DECISION
        else:
            self.state.current_seat = opportunity.seat
            self._prepare_rinshan_draw(opportunity.seat)
        info["call"] = action.kind.value
        return {"reward": reward, "terminated": terminated, "truncated": truncated, "info": info}

    def _serialize_reaction_opportunity(self, opportunity: ReactionOpportunity) -> dict[str, Any]:
        return {
            "seat": opportunity.seat,
            "kind": opportunity.kind.value,
            "mask": opportunity.mask,
            "source_seat": opportunity.source_seat,
            "tile": opportunity.tile,
            "bias": opportunity.bias,
        }

    def _serialize_win(self, win: WinEvent) -> dict[str, Any]:
        return {
            "seat": win.seat,
            "kind": win.action.kind.value,
            "tile": win.tile,
            "source_seat": win.source_seat,
            "tsumo": win.tsumo,
            "has_hupai": win.has_hupai,
            "fanshu": win.fanshu,
            "fu": win.fu,
            "damanguan": win.damanguan,
            "yaku": list(win.yaku),
        }

    def _round_up_100(self, value: int) -> int:
        return ((value + 99) // 100) * 100

    def _base_points(self, win: WinEvent) -> int:
        if win.damanguan > 0:
            return 8000 * win.damanguan
        if win.fanshu >= 13:
            return 8000
        if win.fanshu >= 11:
            return 6000
        if win.fanshu >= 8:
            return 4000
        if win.fanshu >= 6:
            return 3000
        base = win.fu * (2 ** (win.fanshu + 2))
        if win.fanshu >= 5 or base >= 2000:
            return 2000
        return base

    def _tsumo_payments(self, winner_seat: Seat, base_points: int) -> dict[Seat, int]:
        payments: dict[Seat, int] = {}
        if winner_seat == self.state.dealer:
            payment = self._round_up_100(base_points * 2)
            for seat in range(PLAYER_COUNT):
                if seat == winner_seat:
                    continue
                payments[seat] = payment + self.state.honba * 100
            return payments

        dealer_payment = self._round_up_100(base_points * 2) + self.state.honba * 100
        nondealer_payment = self._round_up_100(base_points) + self.state.honba * 100
        for seat in range(PLAYER_COUNT):
            if seat == winner_seat:
                continue
            payments[seat] = dealer_payment if seat == self.state.dealer else nondealer_payment
        return payments

    def _ron_payment(self, winner_seat: Seat, base_points: int) -> int:
        multiplier = 6 if winner_seat == self.state.dealer else 4
        return self._round_up_100(base_points * multiplier) + self.state.honba * 300

    def _settle_pending_wins(self) -> dict[str, Any]:
        if not self.state.pending_wins:
            return {
                "kind": "win",
                "score_deltas": [0, 0, 0, 0],
                "riichi_pot": 0,
                "riichi_pot_awarded_to": None,
                "wins": [],
            }

        deltas = [0, 0, 0, 0]
        wins: list[dict[str, Any]] = []
        riichi_pot = self.state.riichi_sticks * RIICHI_STICK_COST
        riichi_winner = self.state.pending_wins[0].seat if riichi_pot else None

        for win in self.state.pending_wins:
            base_points = self._base_points(win)
            if win.tsumo:
                payments = self._tsumo_payments(win.seat, base_points)
                for payer, payment in payments.items():
                    deltas[payer] -= payment
                    deltas[win.seat] += payment
                wins.append(
                    {
                        "seat": win.seat,
                        "kind": "tsumo",
                        "base_points": base_points,
                        "payments": payments,
                    }
                )
                continue

            if win.source_seat is None:
                raise ValueError("ron win missing source seat")
            payment = self._ron_payment(win.seat, base_points)
            deltas[win.source_seat] -= payment
            deltas[win.seat] += payment
            wins.append(
                {
                    "seat": win.seat,
                    "kind": "ron",
                    "base_points": base_points,
                    "payment": payment,
                    "source_seat": win.source_seat,
                }
            )

        if riichi_pot and riichi_winner is not None:
            deltas[riichi_winner] += riichi_pot
            self.state.riichi_sticks = 0

        for seat, delta in enumerate(deltas):
            self.state.scores[seat] += delta

        return {
            "kind": "win",
            "score_deltas": deltas,
            "riichi_pot": riichi_pot,
            "riichi_pot_awarded_to": riichi_winner,
            "wins": wins,
        }

    def _settle_exhaustive_draw(self) -> dict[str, Any]:
        tenpai_seats = [seat for seat in range(PLAYER_COUNT) if self._player_wait_mask(seat) != 0]
        deltas = [0, 0, 0, 0]

        if len(tenpai_seats) in {1, 2, 3}:
            if len(tenpai_seats) == 1:
                tenpai_delta, noten_delta = 3000, -1000
            elif len(tenpai_seats) == 2:
                tenpai_delta, noten_delta = 1500, -1500
            else:
                tenpai_delta, noten_delta = 1000, -3000
            tenpai = set(tenpai_seats)
            for seat in range(PLAYER_COUNT):
                deltas[seat] = tenpai_delta if seat in tenpai else noten_delta
                self.state.scores[seat] += deltas[seat]

        return {
            "kind": "draw",
            "tenpai_seats": tenpai_seats,
            "score_deltas": deltas,
        }

    def _apply_discard(self, seat: Seat, action: Action) -> dict[str, Any]:
        player = self.state.players[seat]
        tile = self._resolve_tile_instance(player, action)
        if tile is None:
            raise ValueError("discard tile could not be resolved")

        if action.kind == ActionKind.RIICHI:
            if player.riichi:
                raise ValueError("already in riichi")
            if self._player_is_menzen(player) is False:
                raise ValueError("riichi requires a closed hand")
            if self.state.scores[seat] < RIICHI_STICK_COST:
                raise ValueError("riichi requires at least 1000 points")
            if not self._discard_keeps_tenpai(player, self._tile_instance_index(player, tile) or 0):
                raise ValueError("riichi discard must keep tenpai")
            player.riichi = True
            player.riichi_declared_turn = self.state.turn_index
            self.state.riichi_sticks += 1
            self.state.scores[seat] -= RIICHI_STICK_COST

        self._remove_tile_instance(player, tile)
        player.discards.append(tile)
        player.own_discard_tiles.append(tile.tile)
        player.last_discard = tile
        self.state.last_discard = (seat, tile)
        player.drawn_tile = None
        player.has_drawn_this_turn = False
        if self.state.turn_index == 0:
            player.first_turn_open_calls_seen = False
        return {"tile": tile}

    def _apply_ankan(self, seat: Seat, action: Action) -> None:
        player = self.state.players[seat]
        tile = action.tile
        if tile is None:
            raise ValueError("ankan requires a tile")
        matching = [instance for instance in player.tiles if instance.tile == tile]
        if len(matching) < 4:
            raise ValueError("ankan requires four matching tiles")
        for _ in range(4):
            self._remove_first_tile_by_value(player, tile)
        player.melds.append(Meld(ActionKind.ANKAN, tile=tile, source_seat=seat))
        player.closed_kans += 1
        player.last_call_kind = ActionKind.ANKAN
        player.last_call_source = seat
        player.last_call_tile = tile

    def _apply_kakan(self, seat: Seat, action: Action) -> None:
        player = self.state.players[seat]
        tile = action.tile
        if tile is None:
            raise ValueError("kakan requires a tile")
        if tile not in player.open_pon_tiles:
            raise ValueError("kakan requires an existing pon tile")
        if sum(1 for instance in player.tiles if instance.tile == tile) < 1:
            raise ValueError("kakan requires a matching tile in hand")
        self._remove_first_tile_by_value(player, tile)
        player.melds = [
            Meld(ActionKind.KAKAN, tile=meld.tile, source_seat=meld.source_seat, bias=meld.bias, red=meld.red)
            if meld.kind == ActionKind.PON and meld.tile == tile
            else meld
            for meld in player.melds
        ]
        player.last_call_kind = ActionKind.KAKAN
        player.last_call_source = seat
        player.last_call_tile = tile

    def _apply_call(self, seat: Seat, action: Action, opportunity: ReactionOpportunity) -> None:
        player = self.state.players[seat]
        tile = opportunity.tile
        if action.kind == ActionKind.CHI:
            if action.bias is None:
                raise ValueError("chi requires a bias")
            needed = self._chi_tiles_for_bias(tile, action.bias)
            for needed_tile in needed:
                self._remove_first_tile_by_value(player, needed_tile)
            player.melds.append(
                Meld(ActionKind.CHI, tile=min(needed), source_seat=opportunity.source_seat, bias=action.bias, red=False)
            )
            player.open_meld_count += 1
        elif action.kind == ActionKind.PON:
            for _ in range(2):
                self._remove_first_tile_by_value(player, tile)
            player.melds.append(Meld(ActionKind.PON, tile=tile, source_seat=opportunity.source_seat))
            player.open_meld_count += 1
            player.open_pon_tiles.append(tile)
        elif action.kind == ActionKind.MINKAN:
            for _ in range(3):
                self._remove_first_tile_by_value(player, tile)
            player.melds.append(Meld(ActionKind.MINKAN, tile=tile, source_seat=opportunity.source_seat))
            player.open_meld_count += 1
        else:
            raise ValueError(f"unsupported call action: {action.kind.value}")

        player.first_turn_open_calls_seen = True
        self.state.first_turn_open_calls_seen = True
        player.drawn_tile = None
        player.has_drawn_this_turn = False
        player.is_menzen_locked = False

    def _resolve_pending_call_without_reactions(self) -> None:
        if self.state.pending_call_action is None or self.state.pending_call_seat is None:
            self.state.pending_call_action = None
            self.state.pending_call_seat = None
            return
        action = self.state.pending_call_action
        seat = self.state.pending_call_seat
        if action.kind in {ActionKind.CHI, ActionKind.PON}:
            self.state.phase = Phase.SELF_DECISION
            self.state.current_seat = seat
        elif action.kind == ActionKind.MINKAN:
            self._draw_from_dead_wall(seat)
            self.state.phase = Phase.SELF_DECISION
            self.state.current_seat = seat
        self.state.pending_call_action = None
        self.state.pending_call_seat = None

    def _prepare_rinshan_draw(self, seat: Seat) -> None:
        self.state.pending_draw_from_dead_wall = True
        self.state.phase = Phase.DRAW
        self.state.current_seat = seat

    def _advance_to_next_draw(self, after_discard: bool = False) -> None:
        self.state.pending_call_action = None
        self.state.pending_call_seat = None
        self.state.reaction_queue.clear()
        next_seat = (self.state.last_discard[0] + 1) % PLAYER_COUNT if self.state.last_discard else self.state.current_seat
        self.state.current_seat = next_seat
        self.state.phase = Phase.DRAW
        self.state.turn_index += 1
        self.state.last_draw_was_gangzimo = False
        if after_discard:
            for player in self.state.players:
                player.temp_furiten_tiles = []

    # ------------------------------------------------------------------
    # Primitive tile helpers
    # ------------------------------------------------------------------
    def _resolve_tile_instance(self, player: PlayerState, action: Action) -> TileInstance | None:
        index = action.meta.get("tile_index")
        if isinstance(index, int) and 0 <= index < len(player.tiles):
            candidate = player.tiles[index]
            if action.tile is not None and candidate.tile != action.tile:
                raise ValueError("tile index does not match action tile")
            if "red" in action.meta and bool(action.meta["red"]) != candidate.red:
                raise ValueError("tile index does not match red flag")
            return candidate

        tile = action.tile
        if tile is None:
            return None
        red = action.meta.get("red")
        for candidate in player.tiles:
            if candidate.tile != tile:
                continue
            if red is not None and bool(red) != candidate.red:
                continue
            return candidate
        return None

    def _tile_instance_index(self, player: PlayerState, tile: TileInstance) -> int | None:
        for index, candidate in enumerate(player.tiles):
            if candidate == tile:
                return index
        return None

    def _remove_tile_instance(self, player: PlayerState, tile: TileInstance) -> None:
        for index, candidate in enumerate(player.tiles):
            if candidate == tile:
                del player.tiles[index]
                return
        raise ValueError("tile instance not found in hand")

    def _remove_first_tile_by_value(self, player: PlayerState, tile: int) -> TileInstance:
        for index, candidate in enumerate(player.tiles):
            if candidate.tile == tile:
                removed = player.tiles.pop(index)
                return removed
        raise ValueError(f"tile {tile} not found in hand")

    def _perform_draw(self, seat: Seat) -> None:
        player = self.state.players[seat]
        player.temp_furiten_tiles = []
        if self.state.pending_draw_from_dead_wall:
            tile = self._draw_from_dead_wall(seat)
            self.state.pending_draw_from_dead_wall = False
        else:
            tile = self._draw_from_live_wall(seat)
        player.temp_furiten_tiles = []
        player.drawn_tile = tile
        player.has_drawn_this_turn = True

    def _prepare_reaction_queue(self, discarder: Seat, tile: int) -> None:
        ron_queue: list[ReactionOpportunity] = []
        call_candidates: list[ReactionOpportunity] = []

        for offset in range(1, PLAYER_COUNT):
            seat = (discarder + offset) % PLAYER_COUNT
            player = self.state.players[seat]
            if player.riichi:
                if self._can_win(seat, tile, False):
                    ron_queue.append(
                        ReactionOpportunity(seat=seat, kind=ActionKind.RON, mask=self.rules.pm.REACT_OPT_RON, source_seat=discarder, tile=tile)
                    )
                continue

            if self._can_win(seat, tile, False):
                ron_queue.append(
                    ReactionOpportunity(seat=seat, kind=ActionKind.RON, mask=self.rules.pm.REACT_OPT_RON, source_seat=discarder, tile=tile)
                )

            counts = self._player_counts(player)
            call_kind = None
            if seat == (discarder + 1) % PLAYER_COUNT and tile < 27:
                for bias in (0, 1, 2):
                    try:
                        needed = self._chi_tiles_for_bias(tile, bias)
                    except ValueError:
                        continue
                    if all(counts[idx] >= 1 for idx in needed):
                        call_kind = ActionKind.CHI
                        break
            if call_kind is None:
                if counts[tile] >= 3:
                    call_kind = ActionKind.MINKAN
                elif counts[tile] >= 2:
                    call_kind = ActionKind.PON

            if call_kind is not None:
                mask = 0
                if call_kind == ActionKind.CHI:
                    mask |= self.rules.pm.REACT_OPT_CHI
                if call_kind == ActionKind.PON:
                    mask |= self.rules.pm.REACT_OPT_PON
                if call_kind == ActionKind.MINKAN:
                    mask |= self.rules.pm.REACT_OPT_MINKAN
                call_candidates.append(
                    ReactionOpportunity(
                        seat=seat,
                        kind=call_kind,
                        mask=mask,
                        source_seat=discarder,
                        tile=tile,
                    )
                )

        ron_queue.sort(key=lambda opportunity: (opportunity.seat - discarder) % PLAYER_COUNT)
        call_candidates.sort(
            key=lambda opportunity: (-self._reaction_priority(opportunity.kind), (opportunity.seat - discarder) % PLAYER_COUNT)
        )

        self.state.reaction_queue = ron_queue if ron_queue else call_candidates[:1]

    def _best_call_kind(self, mask: int) -> ActionKind | None:
        if mask & self.rules.pm.REACT_OPT_MINKAN:
            return ActionKind.MINKAN
        if mask & self.rules.pm.REACT_OPT_PON:
            return ActionKind.PON
        if mask & self.rules.pm.REACT_OPT_CHI:
            return ActionKind.CHI
        return None

    def _chi_tiles_for_bias(self, tile: int, bias: int) -> tuple[int, int]:
        if bias == 2:
            return (tile - 2, tile - 1)
        if bias == 1:
            return (tile - 1, tile + 1)
        if bias == 0:
            return (tile + 1, tile + 2)
        raise ValueError(f"invalid chi bias: {bias}")

    def _step_reaction_queue(self, actor: Seat, action: Action) -> dict[str, Any]:
        if not self.state.reaction_queue:
            raise ValueError("reaction queue is empty")
        opportunity = self.state.reaction_queue[0]
        if actor != opportunity.seat:
            raise ValueError("unexpected reaction seat")

        info: dict[str, Any] = {"opportunity": self._serialize_reaction_opportunity(opportunity)}
        reward = 0.0
        terminated = False
        truncated = False

        if action.kind == ActionKind.PASS:
            if opportunity.kind == ActionKind.RON:
                player = self.state.players[actor]
                if opportunity.tile not in player.temp_furiten_tiles:
                    player.temp_furiten_tiles.append(opportunity.tile)
            self.state.reaction_queue.pop(0)
            if self.state.reaction_queue:
                self.state.current_seat = self.state.reaction_queue[0].seat
            else:
                if self.state.pending_wins:
                    settlement = self._settle_pending_wins()
                    info["settlement"] = settlement
                    self.state.phase = Phase.HAND_END
                    self.state.terminated = True
                    terminated = True
                else:
                    if self.state.pending_call_action is not None:
                        resolved_call = self.state.pending_call_action.kind.value
                        self._resolve_pending_call_without_reactions()
                        info["resolved_call"] = resolved_call
                    else:
                        self._advance_to_next_draw(after_discard=True)
            return {"reward": reward, "terminated": terminated, "truncated": truncated, "info": info}

        if opportunity.kind == ActionKind.RON:
            if action.kind != ActionKind.RON:
                raise ValueError("expected ron or pass for this reaction")
            if not self._can_win(actor, opportunity.tile, False):
                raise ValueError("illegal ron")
            win = self._win_event(actor, action, opportunity.tile, opportunity.source_seat, False)
            self.state.pending_wins.append(win)
            info["win"] = self._serialize_win(win)
            self.state.reaction_queue.pop(0)
            if self.state.reaction_queue:
                self.state.current_seat = self.state.reaction_queue[0].seat
            else:
                settlement = self._settle_pending_wins()
                info["settlement"] = settlement
                self.state.phase = Phase.HAND_END
                self.state.terminated = True
                terminated = True
            reward = 1.0
            return {"reward": reward, "terminated": terminated, "truncated": truncated, "info": info}

        if action.kind not in {ActionKind.CHI, ActionKind.PON, ActionKind.MINKAN}:
            raise ValueError(f"illegal reaction action: {action.kind.value}")

        if action.kind == ActionKind.CHI and not (opportunity.mask & self.rules.pm.REACT_OPT_CHI):
            raise ValueError("chi is not legal")
        if action.kind == ActionKind.PON and not (opportunity.mask & self.rules.pm.REACT_OPT_PON):
            raise ValueError("pon is not legal")
        if action.kind == ActionKind.MINKAN and not (opportunity.mask & self.rules.pm.REACT_OPT_MINKAN):
            raise ValueError("minkan is not legal")

        self._apply_call(opportunity.seat, action, opportunity)
        self.state.pending_call_action = action
        self.state.pending_call_seat = opportunity.seat
        self.state.reaction_queue.clear()
        if action.kind in {ActionKind.CHI, ActionKind.PON}:
            self.state.phase = Phase.SELF_DECISION
        else:
            self._prepare_rinshan_draw(opportunity.seat)
        info["call"] = action.kind.value
        return {"reward": reward, "terminated": terminated, "truncated": truncated, "info": info}

    def _prepare_rinshan_draw(self, seat: Seat) -> None:
        self.state.pending_draw_from_dead_wall = True
        self.state.current_seat = seat
        self.state.phase = Phase.DRAW
        self.state.last_draw_was_gangzimo = True

    def _apply_call(self, seat: Seat, action: Action, opportunity: ReactionOpportunity) -> None:
        player = self.state.players[seat]
        tile = opportunity.tile
        if action.kind == ActionKind.CHI:
            if action.bias is None:
                raise ValueError("chi requires a bias")
            needed = self._chi_tiles_for_bias(tile, action.bias)
            for needed_tile in needed:
                self._remove_first_tile_by_value(player, needed_tile)
            player.melds.append(Meld(ActionKind.CHI, tile=min((tile,) + needed), source_seat=opportunity.source_seat, bias=action.bias))
            player.open_meld_count += 1
        elif action.kind == ActionKind.PON:
            for _ in range(2):
                self._remove_first_tile_by_value(player, tile)
            player.melds.append(Meld(ActionKind.PON, tile=tile, source_seat=opportunity.source_seat))
            player.open_meld_count += 1
            player.open_pon_tiles.append(tile)
        elif action.kind == ActionKind.MINKAN:
            for _ in range(3):
                self._remove_first_tile_by_value(player, tile)
            player.melds.append(Meld(ActionKind.MINKAN, tile=tile, source_seat=opportunity.source_seat))
            player.open_meld_count += 1
        else:
            raise ValueError(f"unsupported call action: {action.kind.value}")

        player.first_turn_open_calls_seen = True
        self.state.first_turn_open_calls_seen = True
        player.drawn_tile = None
        player.has_drawn_this_turn = False
        player.is_menzen_locked = False

    def _resolve_pending_call_without_reactions(self) -> None:
        action = self.state.pending_call_action
        seat = self.state.pending_call_seat
        if action is None or seat is None:
            self.state.pending_call_action = None
            self.state.pending_call_seat = None
            return
        if action.kind in {ActionKind.CHI, ActionKind.PON}:
            self.state.current_seat = seat
            self.state.phase = Phase.SELF_DECISION
        elif action.kind == ActionKind.MINKAN:
            self._prepare_rinshan_draw(seat)
        self.state.pending_call_action = None
        self.state.pending_call_seat = None


__all__ = ["MahjongSelfPlayEnv", "StepResult", "TableState"]
