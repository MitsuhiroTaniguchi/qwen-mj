from qwen_mj import Action, ActionKind, MahjongSelfPlayEnv


def test_import_and_reset():
    env = MahjongSelfPlayEnv()
    obs = env.reset()
    assert obs["phase"] == "self_decision"
    assert len(obs["scores"]) == 4
    assert obs["self"]["hand_size"] == 14


def test_legal_actions_exist():
    env = MahjongSelfPlayEnv()
    env.reset()
    actions = env.legal_actions()
    assert any(action.kind == ActionKind.DISCARD for action in actions)
