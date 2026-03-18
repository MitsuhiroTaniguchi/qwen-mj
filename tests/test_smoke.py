from qwen_mj import Action, ActionKind, MahjongSelfPlayEnv


def test_import_and_reset():
    env = MahjongSelfPlayEnv()
    obs = env.reset()
    assert obs["phase"] == "draw"
    assert len(obs["scores"]) == 4


def test_legal_actions_exist():
    env = MahjongSelfPlayEnv()
    env.reset()
    actions = env.legal_actions()
    assert any(action.kind == ActionKind.TSUMO for action in actions)
    assert Action(ActionKind.PASS) in actions

