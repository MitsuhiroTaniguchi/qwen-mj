import json

from qwen_mj import Action, ActionKind, MahjongMatchEnv, MahjongSelfPlayEnv
from qwen_mj import FirstLegalPolicy, JsonlRolloutLogger, ObservationEncoder, play_hand, play_match
from qwen_mj import evaluate_against_baseline, run_self_play_experiment
from qwen_mj.environment import TableState
from qwen_mj.match import MatchState
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


def test_match_advances_to_next_hand_after_non_dealer_ron():
    match = MahjongMatchEnv(seed=0, max_round_wind=2)
    match.reset(dealer=3)

    players = _make_empty_players()
    players[1].tiles = _make_tiles([1, 2, 3, 10, 11, 12, 19, 20, 21, 3, 4, 5, 14])
    players[0].tiles = _make_tiles([14, 0, 6, 7, 8, 9, 16, 17, 18, 22, 23, 24, 25, 26])
    players[0].drawn_tile = players[0].tiles[0]
    players[0].has_drawn_this_turn = True
    match.hand_env.state = TableState(
        dealer=3,
        current_seat=0,
        phase=Phase.SELF_DECISION,
        players=players,
        live_wall=_make_tiles([9, 9, 9]),
        dead_wall=_make_tiles([9] * 14),
    )
    match.state = MatchState(
        dealer=3,
        round_wind=0,
        honba=0,
        riichi_sticks=0,
        scores=[25000, 25000, 25000, 25000],
        hand_index=0,
        max_round_wind=2,
        phase=Phase.SELF_DECISION,
    )

    discard = next(action for action in match.legal_actions() if action.kind == ActionKind.DISCARD and action.tile == 14)
    match.step(discard)
    result = match.step(Action(ActionKind.RON, tile=14))
    assert result.terminated is True

    next_observation = match.advance_hand()

    assert next_observation["match"]["dealer"] == 0
    assert next_observation["match"]["round_wind"] == 1
    assert next_observation["match"]["honba"] == 0
    assert next_observation["match"]["hand_index"] == 1
    assert next_observation["hand"]["phase"] == "self_decision"


def test_match_keeps_dealer_and_honba_after_dealer_tsumo():
    match = MahjongMatchEnv(seed=0, max_round_wind=2)
    match.reset(dealer=0)

    players = _make_empty_players()
    winning_tiles = [1, 2, 3, 10, 11, 12, 19, 20, 21, 3, 4, 5, 14, 14]
    players[0].tiles = _make_tiles(winning_tiles)
    players[0].drawn_tile = players[0].tiles[-1]
    players[0].has_drawn_this_turn = True
    match.hand_env.state = TableState(
        dealer=0,
        current_seat=0,
        phase=Phase.SELF_DECISION,
        players=players,
        live_wall=_make_tiles([6, 7, 8]),
        dead_wall=_make_tiles([9] * 14),
    )
    match.state = MatchState(
        dealer=0,
        round_wind=0,
        honba=0,
        riichi_sticks=0,
        scores=[25000, 25000, 25000, 25000],
        hand_index=0,
        max_round_wind=2,
        phase=Phase.SELF_DECISION,
    )

    result = match.step(next(action for action in match.legal_actions() if action.kind == ActionKind.TSUMO))
    assert result.terminated is True

    next_observation = match.advance_hand()

    assert next_observation["match"]["dealer"] == 0
    assert next_observation["match"]["round_wind"] == 0
    assert next_observation["match"]["honba"] == 1
    assert next_observation["match"]["hand_index"] == 1
    assert next_observation["match"]["scores"] == [36700, 21100, 21100, 21100]


def test_observation_encoder_produces_stable_shapes():
    env = MahjongSelfPlayEnv(seed=0)
    observation = env.reset()
    encoder = ObservationEncoder()

    encoded = encoder.encode(observation)
    text = encoder.render_text(observation, env.legal_actions())

    assert encoded["self_hand_counts"].shape == (34,)
    assert encoded["scores"].shape == (4,)
    assert encoded["hand_sizes"].shape == (4,)
    assert encoded["table"].shape == (8,)
    assert encoded["phase"].shape == (1,)
    assert "phase=self_decision" in text
    assert "legal_actions=" in encoder.encode_with_text(observation, env.legal_actions()).text


def test_jsonl_rollout_logger_writes_records(tmp_path):
    env = MahjongSelfPlayEnv(seed=0)
    encoder = ObservationEncoder()
    path = tmp_path / "rollout.jsonl"

    with JsonlRolloutLogger(path) as logger:
        result = play_hand(env, FirstLegalPolicy(), reset_kwargs={"seed": 0}, logger=logger, encoder=encoder, max_steps=1)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["episode_id"] == 0
    assert record["step_index"] == 0
    assert record["action"]["kind"] == "discard"
    assert "encoded" in record
    assert "text" in record
    assert result["records"]


def test_play_match_returns_match_observation(tmp_path):
    env = MahjongMatchEnv(seed=0, max_round_wind=1)
    encoder = ObservationEncoder()
    path = tmp_path / "match_rollout.jsonl"

    with JsonlRolloutLogger(path) as logger:
        result = play_match(env, FirstLegalPolicy(), reset_kwargs={"seed": 0}, logger=logger, encoder=encoder, max_steps=1)

    assert "match" in result["final_observation"]
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_experiment_helpers_return_aggregates():
    summary = run_self_play_experiment(episodes=2, seed=0, max_steps=1)
    baseline = evaluate_against_baseline(episodes=2, policy=FirstLegalPolicy(), seed=0, max_steps=1)

    assert summary.num_episodes == 2
    assert len(summary.mean_final_scores) == 4
    assert len(summary.mean_score_deltas) == 4
    assert baseline.num_episodes == 2
    assert baseline.mean_rank >= 1.0
