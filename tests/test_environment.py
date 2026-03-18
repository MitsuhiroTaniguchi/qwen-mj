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
