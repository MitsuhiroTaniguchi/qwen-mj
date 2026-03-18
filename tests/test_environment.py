from qwen_mj import Action, ActionKind, MahjongSelfPlayEnv


def test_reset_clears_history_and_sets_dealer():
    env = MahjongSelfPlayEnv()
    env.history.append(object())  # type: ignore[arg-type]

    obs = env.reset(dealer=2)

    assert obs["dealer"] == 2
    assert obs["current_seat"] == 2
    assert obs["phase"] == "draw"
    assert env.history == []


def test_discard_advances_to_reaction_and_records_actor():
    env = MahjongSelfPlayEnv()
    env.reset(dealer=1)

    result = env.step(Action(ActionKind.DISCARD, tile=5))

    assert result.observation["phase"] == "reaction"
    assert env.history[-1].seat == 1
    assert env.history[-1].info["actor_seat"] == 1
    assert env.state.last_discard == (1, 5)


def test_tsumo_terminates_hand():
    env = MahjongSelfPlayEnv()
    env.reset()

    result = env.step(Action(ActionKind.TSUMO))

    assert result.reward == 1.0
    assert result.terminated is True
    assert result.observation["phase"] == "hand_end"

