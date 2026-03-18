from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from .training_data import SYSTEM_PROMPT


@dataclass(slots=True)
class DatasetValidationError:
    line_index: int
    message: str


@dataclass(slots=True)
class DatasetValidationReport:
    path: str
    num_records: int
    num_valid: int
    num_invalid: int
    errors: list[DatasetValidationError]

    @property
    def is_valid(self) -> bool:
        return self.num_invalid == 0


def load_jsonl(path: str | Path) -> Iterable[tuple[int, dict[str, Any]]]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_index}: JSON value must be an object")
            yield line_index, payload


def _unwrap_example(payload: dict[str, Any]) -> dict[str, Any]:
    if "sft_example" in payload:
        nested = payload["sft_example"]
        if not isinstance(nested, dict):
            raise ValueError("sft_example must be an object")
        return nested
    return payload


def _validate_message(message: Any, index: int) -> str | None:
    if not isinstance(message, dict):
        return f"message {index} must be an object"
    if "role" not in message or "content" not in message:
        return f"message {index} must contain role and content"
    if not isinstance(message["role"], str) or not isinstance(message["content"], str):
        return f"message {index} role/content must be strings"
    return None


def validate_sft_example(example: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    messages = example.get("messages")
    completion = example.get("completion")
    prompt = example.get("prompt")
    state_text = example.get("state_text")
    legal_actions = example.get("legal_actions")

    if not isinstance(messages, list) or len(messages) < 2:
        errors.append("messages must be a list with at least system and user messages")
    else:
        for index, message in enumerate(messages):
            error = _validate_message(message, index)
            if error is not None:
                errors.append(error)
                continue
        if isinstance(messages[0], dict):
            if messages[0].get("role") != "system":
                errors.append("first message must be system")
            if messages[0].get("content") != SYSTEM_PROMPT:
                errors.append("system message content does not match expected prompt")
        if len(messages) >= 2 and isinstance(messages[1], dict) and messages[1].get("role") != "user":
            errors.append("second message must be user")

    if not isinstance(completion, str) or not completion.strip():
        errors.append("completion must be a non-empty string")

    if not isinstance(prompt, str) or not prompt.strip():
        errors.append("prompt must be a non-empty string")
    elif not prompt.startswith(SYSTEM_PROMPT):
        errors.append("prompt must start with the system prompt")

    if not isinstance(state_text, str) or not state_text.strip():
        errors.append("state_text must be a non-empty string")
    elif isinstance(prompt, str) and prompt.strip() and prompt != SYSTEM_PROMPT + "\n\n" + state_text:
        errors.append("prompt and state_text are inconsistent")

    if not isinstance(legal_actions, list) or not legal_actions:
        errors.append("legal_actions must be a non-empty list")
    else:
        if any(not isinstance(item, str) or not item.strip() for item in legal_actions):
            errors.append("legal_actions entries must be non-empty strings")
        if len(set(legal_actions)) != len(legal_actions):
            errors.append("legal_actions must not contain duplicates")
        if isinstance(completion, str) and completion not in legal_actions:
            errors.append("completion must match one of the legal_actions")

    action = example.get("action")
    if action is not None and not isinstance(action, dict):
        errors.append("action must be an object when present")

    return errors


def validate_sft_jsonl(path: str | Path, max_errors: int = 20) -> DatasetValidationReport:
    input_path = Path(path)
    errors: list[DatasetValidationError] = []
    num_records = 0
    num_valid = 0
    num_invalid = 0

    with input_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            num_records += 1
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("JSON value must be an object")
                example = _unwrap_example(payload)
                example_errors = validate_sft_example(example)
            except Exception as exc:  # pragma: no cover - defensive
                example_errors = [str(exc)]

            if example_errors:
                num_invalid += 1
                if len(errors) < max_errors:
                    errors.extend(
                        DatasetValidationError(line_index=line_index, message=message) for message in example_errors
                    )
            else:
                num_valid += 1

    return DatasetValidationReport(
        path=str(input_path),
        num_records=num_records,
        num_valid=num_valid,
        num_invalid=num_invalid,
        errors=errors,
    )


__all__ = [
    "DatasetValidationError",
    "DatasetValidationReport",
    "load_jsonl",
    "validate_sft_example",
    "validate_sft_jsonl",
]
