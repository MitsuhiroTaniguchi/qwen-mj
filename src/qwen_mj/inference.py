from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .training_data import CanonicalActionCodec, PromptBuilder
from .types import Action


@dataclass(slots=True)
class InferenceConfig:
    model_path: str | Path
    adapter_path: str | Path | None = None
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass(slots=True)
class ModelPolicy:
    model: Any
    tokenizer: Any
    prompt_builder: PromptBuilder = field(default_factory=PromptBuilder)
    config: InferenceConfig | None = None

    def select_action(self, observation: dict[str, Any], legal_actions: Sequence[Action]) -> Action:
        return select_action(
            model=self.model,
            tokenizer=self.tokenizer,
            observation=observation,
            legal_actions=legal_actions,
            prompt_builder=self.prompt_builder,
            config=self.config,
        )


def _import_transformers():  # pragma: no cover - optional dependency
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("transformers is required for inference") from exc
    return AutoModelForCausalLM, AutoTokenizer


def load_model(config: InferenceConfig):  # pragma: no cover - optional dependency
    AutoModelForCausalLM, AutoTokenizer = _import_transformers()
    tokenizer = AutoTokenizer.from_pretrained(str(config.model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(config.model_path), trust_remote_code=True)
    if config.adapter_path is not None:
        from peft import PeftModel  # pragma: no cover - optional dependency

        model = PeftModel.from_pretrained(model, str(config.adapter_path))
    return model, tokenizer


def normalize_completion(text: str) -> str:
    line = text.strip().splitlines()[0].strip()
    if not line:
        return ""
    return " ".join(line.split())


def completion_to_action(
    completion: str,
    legal_actions: Sequence[Action],
    codec: CanonicalActionCodec | None = None,
) -> Action:
    codec = codec or CanonicalActionCodec()
    normalized = normalize_completion(completion)
    legal_map = {codec.encode(action): action for action in legal_actions}
    if normalized in legal_map:
        return legal_map[normalized]
    raise ValueError(f"completion does not match any legal action: {normalized}")


def generate_completion(
    model: Any,
    tokenizer: Any,
    observation: dict[str, Any],
    legal_actions: Sequence[Action],
    prompt_builder: PromptBuilder | None = None,
    config: InferenceConfig | None = None,
) -> str:  # pragma: no cover - depends on optional GPU stack
    prompt_builder = prompt_builder or PromptBuilder()
    config = config or InferenceConfig(model_path="")
    messages = prompt_builder.build_messages(observation, legal_actions)
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    generation = model.generate(
        inputs,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.temperature > 0.0,
        **({"temperature": config.temperature, "top_p": config.top_p} if config.temperature > 0.0 else {}),
    )
    prompt_length = inputs.shape[-1]
    decoded = tokenizer.decode(generation[0][prompt_length:], skip_special_tokens=True)
    return normalize_completion(decoded)


def select_action(
    model: Any,
    tokenizer: Any,
    observation: dict[str, Any],
    legal_actions: Sequence[Action],
    prompt_builder: PromptBuilder | None = None,
    config: InferenceConfig | None = None,
) -> Action:  # pragma: no cover - depends on optional GPU stack
    completion = generate_completion(
        model=model,
        tokenizer=tokenizer,
        observation=observation,
        legal_actions=legal_actions,
        prompt_builder=prompt_builder,
        config=config,
    )
    return completion_to_action(completion, legal_actions, codec=(prompt_builder.codec if prompt_builder else None))


__all__ = [
    "InferenceConfig",
    "ModelPolicy",
    "completion_to_action",
    "generate_completion",
    "load_model",
    "normalize_completion",
    "select_action",
]
