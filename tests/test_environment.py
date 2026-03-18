from qwen_mj import Action, ActionKind, MahjongSelfPlayEnv
from qwen_mj.environment import TableState
from qwen_mj.types import Phase, PlayerState, TileInstance


def _make_tiles(tile_ids: list[int]) -> list[TileInstance]:
    return [TileInstance(tile=tile) for tile in tile_ids]


def _make_empty_players() -> list[PlayerState]:
    return [PlayerState() for _ in range(4)]


def test_reset_clears_history_and_sets_dealer():
    env = MahjongSelfPlayEnv(seed=0)
    env.history.append(object())  # type: ignore[arg-type]

    obs = env.reset(dealer=2)

    assert obs["dealer"] == 2
    assert obs["current_seat"] == 2
    assert obs["phase"] == "self_decision"
    assert obs["self"]["hand_size"] == 14
    assert env.history == []


def test_discard_advances_to_next_draw_when_no_reactions():
    env = MahjongSelfPlayEnv(seed=0)
    players = _make_empty_players()
    players[0].tiles = _make_tiles([1, 2, 3, 10, 11, 12, 19, 20, 21, 3, 4, 5, 14, 14])
    players[0].drawn_tile = players[0].tiles[-1]
    players[0].has_drawn_this_turn = True
    env.state = TableState(
        dealer=0,
        current_seat=0,
        phase=Phase.SELF_DECISION,
        players=players,
        live_wall=_make_tiles([6, 7, 8]),
        dead_wall=_make_tiles([9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]),
    )

    discard = next(action for action in env.legal_actions() if action.kind == ActionKind.DISCARD)
    result = env.step(discard)

    assert result.observation["phase"] == "draw"
    assert env.state.phase == Phase.DRAW
    assert env.state.current_seat == 1
    assert env.history[-1].seat == 0
    assert env.history[-1].info["actor_seat"] == 0


def test_riichi_deducts_stick_cost_and_records_pot():
    env = MahjongSelfPlayEnv(seed=0)
    players = _make_empty_players()
    players[0].tiles = _make_tiles([1, 2, 3, 10, 11, 12, 19, 20, 21, 3, 4, 5, 14, 0])
    players[0].drawn_tile = players[0].tiles[-1]
    players[0].has_drawn_this_turn = True
    env.state = TableState(
        dealer=0,
        current_seat=0,
        phase=Phase.SELF_DECISION,
        players=players,
        live_wall=_make_tiles([6, 7, 8]),
        dead_wall=_make_tiles([9] * 14),
    )

    result = env.step(Action(ActionKind.RIICHI, tile=0, meta={"tile_index": 13, "red": False}))

    assert result.info["tile"].tile == 0
    assert env.state.scores == [24000, 25000, 25000, 25000]
    assert env.state.riichi_sticks == 1
    assert env.state.players[0].riichi is True


def test_tsumo_terminates_hand_for_known_winning_hand():
    env = MahjongSelfPlayEnv(seed=0)
    players = _make_empty_players()
    winning_tiles = [1, 2, 3, 10, 11, 12, 19, 20, 21, 3, 4, 5, 14, 14]
    players[0].tiles = _make_tiles(winning_tiles)
    players[0].drawn_tile = players[0].tiles[-1]
    players[0].has_drawn_this_turn = True
    env.state = TableState(
        dealer=0,
        current_seat=0,
        phase=Phase.SELF_DECISION,
        players=players,
        live_wall=_make_tiles([6, 7, 8]),
        dead_wall=_make_tiles([9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]),
    )

    actions = env.legal_actions()
    assert any(action.kind == ActionKind.TSUMO for action in actions)

    result = env.step(next(action for action in actions if action.kind == ActionKind.TSUMO))

    assert result.reward == 1.0
    assert result.terminated is True
    assert result.observation["phase"] == "hand_end"
    assert result.info["settlement"]["score_deltas"] == [11700, -3900, -3900, -3900]
    assert env.state.scores == [36700, 21100, 21100, 21100]
    assert result.info["win"]["fanshu"] == 4
    assert result.info["win"]["fu"] == 30


def test_ron_blocks_lower_priority_calls_on_same_discard():
    env = MahjongSelfPlayEnv(seed=0)
    players = _make_empty_players()
    players[0].tiles = _make_tiles([14, 0, 6, 7, 8, 9, 16, 17, 18, 22, 23, 24, 25, 26])
    players[0].drawn_tile = players[0].tiles[0]
    players[0].has_drawn_this_turn = True
    players[1].tiles = _make_tiles([1, 2, 3, 10, 11, 12, 19, 20, 21, 3, 4, 5, 14])
    players[1].riichi = True
    players[1].riichi_declared_turn = 0
    players[2].tiles = _make_tiles([13, 15, 27, 28, 29, 30, 31, 32, 33, 0, 6, 7, 8])
    env.state = TableState(
        dealer=0,
        current_seat=0,
        phase=Phase.SELF_DECISION,
        players=players,
        live_wall=_make_tiles([9, 9, 9]),
        dead_wall=_make_tiles([9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]),
    )

    discard = next(action for action in env.legal_actions() if action.kind == ActionKind.DISCARD and action.tile == 14)
    env.step(discard)

    assert env.state.phase == Phase.REACTION
    assert len(env.state.reaction_queue) == 1
    assert env.state.reaction_queue[0].seat == 1
    assert env.state.reaction_queue[0].kind == ActionKind.RON


def test_ron_settles_points_for_nondealer_winner():
    env = MahjongSelfPlayEnv(seed=0)
    players = _make_empty_players()
    players[1].tiles = _make_tiles([1, 2, 3, 10, 11, 12, 19, 20, 21, 3, 4, 5, 14])
    players[0].tiles = _make_tiles([14, 0, 6, 7, 8, 9, 16, 17, 18, 22, 23, 24, 25, 26])
    players[0].drawn_tile = players[0].tiles[0]
    players[0].has_drawn_this_turn = True
    env.state = TableState(
        dealer=0,
        current_seat=0,
        phase=Phase.SELF_DECISION,
        players=players,
        live_wall=_make_tiles([9, 9, 9]),
        dead_wall=_make_tiles([9] * 14),
    )

    discard = next(action for action in env.legal_actions() if action.kind == ActionKind.DISCARD and action.tile == 14)
    env.step(discard)
    result = env.step(Action(ActionKind.RON, tile=14))

    assert result.terminated is True
    assert result.info["settlement"]["score_deltas"] == [-5200, 5200, 0, 0]
    assert env.state.scores == [19800, 30200, 25000, 25000]
    assert result.info["win"]["fanshu"] == 3
    assert result.info["win"]["fu"] == 40


def test_exhaustive_draw_settles_tenpai_payments():
    env = MahjongSelfPlayEnv(seed=0)
    players = _make_empty_players()
    players[0].tiles = _make_tiles([1, 2, 3, 10, 11, 12, 19, 20, 21, 3, 4, 5, 14])
    env.state = TableState(
        dealer=0,
        current_seat=0,
        phase=Phase.DRAW,
        players=players,
        live_wall=[],
        dead_wall=_make_tiles([9] * 14),
    )

    result = env.step(Action(ActionKind.PASS))

    assert result.truncated is True
    assert result.info["settlement"]["tenpai_seats"] == [0]
    assert result.info["settlement"]["score_deltas"] == [3000, -1000, -1000, -1000]
    assert env.state.scores == [28000, 24000, 24000, 24000]
