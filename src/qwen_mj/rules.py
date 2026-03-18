from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any


@dataclass(slots=True)
class PyMahjongRulesAdapter:
    """Thin adapter around `pymahjong`.

    The adapter is intentionally narrow. The table-flow layer should depend on
    this interface, not on pymahjong internals directly.
    """

    three_player: bool = False

    @cached_property
    def pm(self):
        try:
            import pymahjong as pm  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "pymahjong is required for rule evaluation. Install the optional "
                "`rules` dependency or add pymahjong to the environment."
            ) from exc
        return pm

    def compute_self_options(self, *args: Any, **kwargs: Any):
        return self.pm.compute_self_option_mask(*args, **kwargs)

    def compute_reaction_options(self, *args: Any, **kwargs: Any):
        return self.pm.compute_reaction_option_masks(*args, **kwargs)

    def compute_rob_kan_options(self, *args: Any, **kwargs: Any):
        return self.pm.compute_rob_kan_option_masks(*args, **kwargs)

    def wait_mask(self, *args: Any, **kwargs: Any):
        return self.pm.wait_mask(*args, **kwargs)

    def has_hupai(self, *args: Any, **kwargs: Any):
        return self.pm.has_hupai(*args, **kwargs)

    def evaluate_draw(self, *args: Any, **kwargs: Any):
        return self.pm.evaluate_draw(*args, **kwargs)

