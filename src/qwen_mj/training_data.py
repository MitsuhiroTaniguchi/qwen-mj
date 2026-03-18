from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from .encoding import ObservationEncoder
from .types import Action, ActionKind

SYSTEM_PROMPT = (
    "You are a Japanese mahjong policy model.\n"
    "Choose exactly one legal action.\n"
    "Return only a single canonical action line.\n"
    "Do not explain your choice."
)


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(slots=True)
class SFTExample:
    messages: list[ChatMessage]
    prompt: str
    completion: str
    state_text: str
    legal_actions: list[str]
    action: dict[str, Any]


class CanonicalActionCodec:
    def __init__(self, encoder: ObservationEncoder | None = None):
        self.encoder = encoder or ObservationEncoder()

    def encode(self, action: Action) -> str:
        kind = action.kind
        if kind == ActionKind.DISCARD:
            return self._discard(action)
        if kind == ActionKind.RIICHI:
            return f"RIICHI DISCARD {self._tile_with_red(action)}"
        if kind in {ActionKind.TSUMO, ActionKind.RON, ActionKind.PASS, ActionKind.KYUSHUKYUHAI, ActionKind.PENUKI}:
            return kind.value.upper()
        if kind in {ActionKind.ANKAN, ActionKind.KAKAN, ActionKind.CHI, ActionKind.PON, ActionKind.MINKAN}:
            return self._call(action)
        raise ValueError(f"unsupported action kind: {kind.value}")

    def format_legal_actions(self, actions: Sequence[Action]) -> list[str]:
        encoded: list[str] = []
        seen: set[str] = set()
        for action in actions:
            text = self.encode(action)
            if text in seen:
                continue
            seen.add(text)
            encoded.append(text)
        return encoded

    def _discard(self, action: Action) -> str:
        return f"DISCARD {self._tile_with_red(action)}"

    def _call(self, action: Action) -> str:
        tile = self._tile_with_red(action)
        if action.kind == ActionKind.CHI:
            if action.bias is None:
                raise ValueError("chi action missing bias")
            return f"CHI {tile} BIAS={action.bias}"
        return f"{action.kind.value.upper()} {tile}"

    def _tile_with_red(self, action: Action) -> str:
        if action.tile is None:
            raise ValueError(f"action {action.kind.value} is missing tile")
        tile = self.encoder.format_tile(action.tile)
        if action.meta.get("red"):
            return f"{tile} RED"
        return tile


class PromptBuilder:
    def __init__(self, encoder: ObservationEncoder | None = None, codec: CanonicalActionCodec | None = None):
        self.encoder = encoder or ObservationEncoder()
        self.codec = codec or CanonicalActionCodec(self.encoder)

    def build_state_text(self, observation: dict[str, Any], legal_actions: Sequence[Action]) -> str:
        state = self.encoder.render_text(observation, legal_actions)
        legal_lines = "\n".join(f"- {text}" for text in self.codec.format_legal_actions(legal_actions))
        return (
            f"{state}\n\n"
            "Output format:\n"
            "One line only.\n"
            "Use one of the legal action strings below exactly.\n"
            f"{legal_lines}"
        )

    def build_prompt(self, observation: dict[str, Any], legal_actions: Sequence[Action]) -> str:
        return SYSTEM_PROMPT + "\n\n" + self.build_state_text(observation, legal_actions)

    def build_messages(self, observation: dict[str, Any], legal_actions: Sequence[Action]) -> list[ChatMessage]:
        return [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=self.build_state_text(observation, legal_actions)),
        ]

    def build_example(self, observation: dict[str, Any], legal_actions: Sequence[Action], action: Action) -> SFTExample:
        prompt = self.build_prompt(observation, legal_actions)
        completion = self.codec.encode(action)
        return SFTExample(
            messages=self.build_messages(observation, legal_actions),
            prompt=prompt,
            completion=completion,
            state_text=self.build_state_text(observation, legal_actions),
            legal_actions=self.codec.format_legal_actions(legal_actions),
            action={
                "kind": action.kind.value,
                "tile": action.tile,
                "source_seat": action.source_seat,
                "bias": action.bias,
                "meta": dict(action.meta),
            },
        )

    def record_to_example(self, record: dict[str, Any]) -> SFTExample:
        observation = record["observation_before"]
        legal_actions = [self._action_from_dict(item) for item in record["legal_actions"]]
        action = self._action_from_dict(record["action"])
        return self.build_example(observation, legal_actions, action)

    def _action_from_dict(self, payload: dict[str, Any]) -> Action:
        return Action(
            kind=ActionKind(payload["kind"]),
            tile=payload.get("tile"),
            source_seat=payload.get("source_seat"),
            bias=payload.get("bias"),
            meta=dict(payload.get("meta", {})),
        )


def example_to_dict(example: SFTExample) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    for message in example.messages:
        if isinstance(message, ChatMessage):
            messages.append({"role": message.role, "content": message.content})
        else:
            messages.append({"role": message["role"], "content": message["content"]})
    return {
        "messages": messages,
        "prompt": example.prompt,
        "completion": example.completion,
        "state_text": example.state_text,
        "legal_actions": list(example.legal_actions),
        "action": dict(example.action),
    }


def write_sft_jsonl(records: Sequence[dict[str, Any]], path: str | Path) -> int:
    output_path = Path(path)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = record.get("sft_example")
            if payload is None:
                continue
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


__all__ = [
    "CanonicalActionCodec",
    "ChatMessage",
    "PromptBuilder",
    "SFTExample",
    "SYSTEM_PROMPT",
    "example_to_dict",
    "write_sft_jsonl",
]
