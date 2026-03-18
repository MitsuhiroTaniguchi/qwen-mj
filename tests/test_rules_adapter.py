from qwen_mj.rules import PyMahjongRulesAdapter


def test_rules_adapter_imports_pymahjong():
    adapter = PyMahjongRulesAdapter()

    pm = adapter.pm

    assert hasattr(pm, "compute_self_option_mask")
    assert hasattr(pm, "compute_reaction_option_masks")
    assert hasattr(pm, "wait_mask")

