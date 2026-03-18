from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .types import Action, ActionKind

PLAYER_COUNT = 4
NUM_TILES = 34
PHASE_TO_ID = {
    "dealing": 0,
    "draw": 1,
    "self_decision": 2,
    "reaction": 3,
    "hand_end": 4,
    "round_end": 5,
    "game_end": 6,
}

ACTION_KIND_TO_ID = {kind.value: index for index, kind in enumerate(ActionKind)}

SUIT_NAMES = ("m", "p", "s")
HONOR_NAMES = ("east", "south", "west", "north", "white", "green", "red")


@dataclass(slots=True)
class EncodedObservation:
    arrays: dict[str, np.ndarray]
    text: str


class ObservationEncoder:
    """Encode table observations into stable numeric and text views."""

    def encode(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        self_view = observation["self"]
        players = observation["players"]
        arrays = {
            "self_hand_counts": np.asarray(self_view["hand_counts"], dtype=np.int16),
            "scores": np.asarray(observation["scores"], dtype=np.int32),
            "hand_sizes": np.asarray([player["hand_size"] for player in players], dtype=np.int16),
            "riichi": np.asarray([1 if player["riichi"] else 0 for player in players], dtype=np.int8),
            "is_menzen": np.asarray([1 if player["is_menzen"] else 0 for player in players], dtype=np.int8),
            "is_furiten": np.asarray([1 if player["is_furiten"] else 0 for player in players], dtype=np.int8),
            "discard_counts": np.asarray(
                [self._tile_counts(player["discards"]) for player in players], dtype=np.int16
            ),
            "meld_counts": np.asarray(
                [self._meld_counts(player["melds"]) for player in players], dtype=np.int16
            ),
            "reaction_queue": np.asarray(
                [self._reaction_row(item) for item in observation["reaction_queue"]] or [[-1, -1, -1, -1, -1]],
                dtype=np.int16,
            ),
            "pending_wins": np.asarray(
                [self._pending_win_row(item) for item in observation["pending_wins"]] or [[-1, -1, -1, -1, -1]],
                dtype=np.int16,
            ),
            "table": np.asarray(
                [
                    observation["current_seat"],
                    observation["dealer"],
                    observation["round_wind"],
                    observation["honba"],
                    observation["riichi_sticks"],
                    observation["wall_remaining"],
                    observation["dead_wall_remaining"],
                    observation["turn_index"],
                ],
                dtype=np.int32,
            ),
            "phase": np.asarray([PHASE_TO_ID[observation["phase"]]], dtype=np.int8),
            "terminated": np.asarray([1 if observation["terminated"] else 0], dtype=np.int8),
            "truncated": np.asarray([1 if observation["truncated"] else 0], dtype=np.int8),
            "last_discard": np.asarray(self._tile_entry(observation["last_discard"]), dtype=np.int16),
        }
        return arrays

    def encode_with_text(self, observation: dict[str, Any], legal_actions: Sequence[Action] | None = None) -> EncodedObservation:
        return EncodedObservation(arrays=self.encode(observation), text=self.render_text(observation, legal_actions))

    def render_text(self, observation: dict[str, Any], legal_actions: Sequence[Action] | None = None) -> str:
        lines: list[str] = []
        lines.append(
            " | ".join(
                [
                    f"phase={observation['phase']}",
                    f"seat={observation['seat']}",
                    f"current={observation['current_seat']}",
                    f"dealer={observation['dealer']}",
                    f"round_wind={observation['round_wind']}",
                    f"honba={observation['honba']}",
                    f"riichi_sticks={observation['riichi_sticks']}",
                    f"wall={observation['wall_remaining']}",
                    f"dead_wall={observation['dead_wall_remaining']}",
                ]
            )
        )
        lines.append(f"scores={self._format_scores(observation['scores'])}")
        lines.append(f"hand={self._format_tiles(self._counts_to_tiles(observation['self']['hand_counts']))}")
        if observation["last_discard"] is not None:
            lines.append(f"last_discard={self._format_tile_entry(observation['last_discard'])}")
        if observation["reaction_queue"]:
            lines.append(
                "reaction_queue="
                + ", ".join(
                    f"{item['seat']}:{item['kind']}:{item['tile']}" for item in observation["reaction_queue"]
                )
            )
        if observation["pending_wins"]:
            lines.append(
                "pending_wins="
                + ", ".join(
                    f"{item['seat']}:{item['kind']}:{item['tile']}" for item in observation["pending_wins"]
                )
            )
        for player in observation["players"]:
            lines.append(
                "player "
                + str(player["seat"])
                + ": "
                + f"size={player['hand_size']} riichi={int(player['riichi'])} "
                + f"menzen={int(player['is_menzen'])} furiten={int(player['is_furiten'])} "
                + f"discards={self._format_tiles(self._tile_entries_to_tile_ids(player['discards']))} "
                + f"melds={self._format_melds(player['melds'])}"
            )
        if legal_actions is not None:
            lines.append(
                "legal_actions="
                + ", ".join(self.action_to_text(action) for action in legal_actions)
            )
        return "\n".join(lines)

    def action_to_text(self, action: Action) -> str:
        parts = [action.kind.value]
        if action.tile is not None:
            parts.append(self._format_tile_index(action.tile))
        if action.bias is not None:
            parts.append(f"bias={action.bias}")
        if action.source_seat is not None:
            parts.append(f"from={action.source_seat}")
        if action.meta:
            parts.append(f"meta={action.meta}")
        return ":".join(parts)

    def _reaction_row(self, item: dict[str, Any]) -> list[int]:
        return [
            int(item["seat"]),
            int(ACTION_KIND_TO_ID[item["kind"]]) if item["kind"] in ACTION_KIND_TO_ID else -1,
            int(item["tile"]),
            int(item["source_seat"]) if item["source_seat"] is not None else -1,
            int(item["bias"]) if item["bias"] is not None else -1,
        ]

    def _pending_win_row(self, item: dict[str, Any]) -> list[int]:
        return [
            int(item["seat"]),
            int(ACTION_KIND_TO_ID[item["kind"]]) if item["kind"] in ACTION_KIND_TO_ID else -1,
            int(item["tile"]),
            int(item["source_seat"]) if item["source_seat"] is not None else -1,
            int(item["has_hupai"]),
        ]

    def _tile_entry(self, tile: dict[str, Any] | None) -> list[int]:
        if tile is None:
            return [-1, -1]
        return [int(tile["tile"]), 1 if tile["red"] else 0]

    def _tile_counts(self, tiles: Sequence[dict[str, Any]]) -> list[int]:
        counts = [0] * NUM_TILES
        for tile in tiles:
            counts[int(tile["tile"])] += 1
        return counts

    def _meld_counts(self, melds: Sequence[dict[str, Any]]) -> list[int]:
        counts = [0, 0, 0, 0, 0]
        for meld in melds:
            kind = meld["kind"]
            if kind == ActionKind.CHI.value:
                counts[0] += 1
            elif kind == ActionKind.PON.value:
                counts[1] += 1
            elif kind == ActionKind.MINKAN.value:
                counts[2] += 1
            elif kind == ActionKind.ANKAN.value:
                counts[3] += 1
            elif kind == ActionKind.KAKAN.value:
                counts[4] += 1
        return counts

    def _tile_entries_to_tile_ids(self, tiles: Sequence[dict[str, Any]]) -> list[int]:
        return [int(tile["tile"]) for tile in tiles]

    def _counts_to_tiles(self, counts: Sequence[int]) -> list[int]:
        tiles: list[int] = []
        for tile, count in enumerate(counts):
            tiles.extend([tile] * int(count))
        return tiles

    def _format_scores(self, scores: Sequence[int]) -> str:
        return "/".join(str(int(score)) for score in scores)

    def _format_tile_entry(self, tile: dict[str, Any]) -> str:
        return f"{self._format_tile_index(int(tile['tile']))}{'r' if tile['red'] else ''}"

    def _format_melds(self, melds: Sequence[dict[str, Any]]) -> str:
        if not melds:
            return "-"
        return ",".join(f"{meld['kind']}@{self._format_tile_index(int(meld['tile']))}" for meld in melds)

    def _format_tiles(self, tiles: Sequence[int]) -> str:
        if not tiles:
            return "-"
        return " ".join(self._format_tile_index(tile) for tile in tiles)

    def _format_tile_index(self, tile: int) -> str:
        if tile < 0 or tile >= NUM_TILES:
            return "?"
        if tile < 9:
            return f"{tile + 1}m"
        if tile < 18:
            return f"{tile - 8}p"
        if tile < 27:
            return f"{tile - 17}s"
        honor = HONOR_NAMES[tile - 27]
        return honor


__all__ = ["EncodedObservation", "ObservationEncoder"]
