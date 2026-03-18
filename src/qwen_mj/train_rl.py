from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .environment import MahjongMatchEnv
from .experiment import summarize_episode
from .inference import InferenceConfig, ModelPolicy
from .rollout import FirstLegalPolicy, RandomPolicy, play_match
from .train_sft import example_to_training_text
from .training_data import PromptBuilder


@dataclass(slots=True)
class RLTrainConfig:
    output_dir: Path
    model_name: str = "Qwen/Qwen3.5-4B-Instruct"
    max_seq_length: int = 4096
    iterations: int = 1
    episodes_per_iteration: int = 8
    max_steps: int = 10000
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    reward_scale: float = 1000.0
    center_advantages: bool = True
    reward_normalization: bool = True
    seed: int = 0
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    load_in_4bit: bool = True
    save_method: str = "lora"
    baseline: str = "random"
    temperature: float = 0.8
    top_p: float = 1.0
    max_new_tokens: int = 32
    save_every: int = 1
    clip_grad_norm: float = 1.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass(slots=True)
class RLExperience:
    episode_id: int
    step_index: int
    seat: int
    reward: float
    advantage: float
    sft_example: dict[str, Any]


@dataclass(slots=True)
class RLIterationSummary:
    iteration: int
    num_episodes: int
    num_samples: int
    mean_reward: float
    mean_advantage: float
    mean_loss: float
    mean_score_deltas: list[float]
    episode_summaries: list[dict[str, Any]]


def _import_unsloth():  # pragma: no cover - optional dependency
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("unsloth is required for RL training; install the training extra and try again") from exc

    return FastLanguageModel


def _load_rl_model(config: RLTrainConfig):  # pragma: no cover - optional dependency
    FastLanguageModel = _import_unsloth()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=list(config.target_modules),
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )
    return model, tokenizer


def load_baseline_policy(name: str, seed: int) -> FirstLegalPolicy | RandomPolicy:
    if name == "random":
        return RandomPolicy(seed=seed)
    return FirstLegalPolicy()


def _reward_from_summary(summary: dict[str, Any], seat: int, reward_scale: float) -> float:
    final_scores = list(summary["final_scores"])
    baseline = 25000
    return (final_scores[seat] - baseline) / reward_scale


def _normalize_advantages(rewards: Sequence[float], center_advantages: bool, reward_normalization: bool) -> list[float]:
    values = list(rewards)
    if reward_normalization and len(values) > 1:
        scale = pstdev(values)
        if scale > 0.0:
            values = [value / scale for value in values]
    if center_advantages and values:
        offset = mean(values)
        values = [value - offset for value in values]
    return values


def _sequence_logprob(model: Any, tokenizer: Any, example: dict[str, Any], prompt_builder: PromptBuilder) -> torch.Tensor:
    messages = list(example["messages"])
    prompt_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    full_text = example_to_training_text(example, tokenizer)
    full_inputs = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    prompt_len = int(prompt_inputs.shape[-1])
    input_ids = full_inputs["input_ids"].to(model.device)
    attention_mask = full_inputs["attention_mask"].to(model.device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    token_log_probs = F.log_softmax(shift_logits, dim=-1).gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    completion_start = max(prompt_len - 1, 0)
    completion_log_probs = token_log_probs[:, completion_start:]
    return completion_log_probs.sum(dim=-1)


def _select_policy(model: Any, tokenizer: Any, config: RLTrainConfig) -> ModelPolicy:
    inference_config = InferenceConfig(
        model_path=config.model_name,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
    )
    return ModelPolicy(model=model, tokenizer=tokenizer, config=inference_config)


def collect_rl_experiences(
    model: Any,
    tokenizer: Any,
    config: RLTrainConfig,
    iteration: int,
    encoder: Any | None = None,
) -> tuple[list[RLExperience], list[dict[str, Any]]]:  # pragma: no cover - depends on optional GPU stack
    prompt_builder = PromptBuilder(encoder=encoder)
    experiences: list[RLExperience] = []
    episode_summaries: list[dict[str, Any]] = []

    for episode_id in range(config.episodes_per_iteration):
        reset_seed = config.seed + iteration * config.episodes_per_iteration + episode_id
        target_seat = episode_id % 4
        seat_policies: list[Any] = [load_baseline_policy(config.baseline, config.seed + seat) for seat in range(4)]
        seat_policies[target_seat] = _select_policy(model, tokenizer, config)
        env = MahjongMatchEnv(seed=reset_seed)
        result = play_match(
            env,
            seat_policies,
            reset_kwargs={"seed": reset_seed},
            encoder=encoder,
            episode_id=episode_id,
            max_steps=config.max_steps,
        )
        summary = summarize_episode(episode_id, target_seat, result)
        summary_payload = asdict(summary)
        episode_summaries.append(summary_payload)
        raw_rewards = [
            _reward_from_summary(summary_payload, seat=record["seat"], reward_scale=config.reward_scale)
            for record in result["records"]
            if record["seat"] == target_seat
        ]
        advantages = _normalize_advantages(
            raw_rewards,
            center_advantages=config.center_advantages,
            reward_normalization=config.reward_normalization,
        )
        for record, reward, advantage in zip(
            (record for record in result["records"] if record["seat"] == target_seat),
            raw_rewards,
            advantages,
            strict=True,
        ):
            experiences.append(
                RLExperience(
                    episode_id=episode_id,
                    step_index=record["step_index"],
                    seat=record["seat"],
                    reward=reward,
                    advantage=advantage,
                    sft_example=record["sft_example"],
                )
            )

    return experiences, episode_summaries


def train_rl(config: RLTrainConfig) -> dict[str, Any]:  # pragma: no cover - depends on optional GPU stack
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if config.episodes_per_iteration < 1:
        raise ValueError("episodes_per_iteration must be at least 1")
    if config.iterations < 1:
        raise ValueError("iterations must be at least 1")

    model, tokenizer = _load_rl_model(config)
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=config.learning_rate)
    prompt_builder = PromptBuilder()
    model.train()

    history: list[RLIterationSummary] = []
    total_samples = 0
    total_episodes = 0

    for iteration in range(config.iterations):
        experiences, episode_summaries = collect_rl_experiences(
            model=model,
            tokenizer=tokenizer,
            config=config,
            iteration=iteration,
            encoder=prompt_builder.encoder,
        )
        if not experiences:
            continue

        rewards = [item.reward for item in experiences]
        advantages = [item.advantage for item in experiences]
        loss_values: list[float] = []
        score_deltas = [summary["score_deltas"] for summary in episode_summaries]
        mean_score_deltas = [
            float(mean(values))
            for values in zip(*score_deltas, strict=True)
        ] if score_deltas else [0.0, 0.0, 0.0, 0.0]

        optimizer.zero_grad(set_to_none=True)
        for index, experience in enumerate(experiences):
            logprob = _sequence_logprob(model, tokenizer, experience.sft_example, prompt_builder)
            loss = -(logprob * torch.tensor(experience.advantage, device=logprob.device, dtype=logprob.dtype))
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            loss_values.append(float(loss.detach().cpu()))
            if (index + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if len(experiences) % config.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        iteration_summary = RLIterationSummary(
            iteration=iteration,
            num_episodes=len(episode_summaries),
            num_samples=len(experiences),
            mean_reward=float(mean(rewards)),
            mean_advantage=float(mean(advantages)),
            mean_loss=float(mean(loss_values)),
            mean_score_deltas=mean_score_deltas,
            episode_summaries=episode_summaries,
        )
        history.append(iteration_summary)
        total_samples += len(experiences)
        total_episodes += len(episode_summaries)

        config.output_dir.mkdir(parents=True, exist_ok=True)
        if (iteration + 1) % config.save_every == 0:
            adapter_dir = config.output_dir / f"adapter_iter_{iteration}"
            model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))

    final_adapter_dir = config.output_dir / "adapter"
    model.save_pretrained(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    if config.save_method == "merged_16bit":
        merged_dir = config.output_dir / "merged"
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    return {
        "iterations": config.iterations,
        "episodes_per_iteration": config.episodes_per_iteration,
        "total_episodes": total_episodes,
        "total_samples": total_samples,
        "history": [asdict(item) for item in history],
        "output_dir": str(config.output_dir),
        "model_name": config.model_name,
        "baseline": config.baseline,
    }


__all__ = [
    "RLExperience",
    "RLIterationSummary",
    "RLTrainConfig",
    "collect_rl_experiences",
    "load_baseline_policy",
    "train_rl",
]
