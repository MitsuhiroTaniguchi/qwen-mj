from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import torch
import torch.nn as nn
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
    value_learning_rate: float | None = None
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
    gamma: float = 1.0
    gae_lambda: float = 0.95
    ppo_epochs: int = 2
    minibatch_size: int = 8
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    target_kl: float | None = 0.05
    log_jsonl: bool = True
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
    done: bool
    value: float
    old_logprob: float
    advantage: float
    returns: float
    sft_example: dict[str, Any]


@dataclass(slots=True)
class RLIterationSummary:
    iteration: int
    num_episodes: int
    num_samples: int
    mean_reward: float
    mean_advantage: float
    mean_return: float
    mean_value: float
    mean_policy_loss: float
    mean_value_loss: float
    mean_entropy: float
    mean_total_loss: float
    mean_kl: float
    mean_clip_fraction: float
    mean_score_deltas: list[float]
    episode_summaries: list[dict[str, Any]]


@dataclass(slots=True)
class PPOBatchMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    approx_kl: float
    clip_fraction: float


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.value(hidden_states).squeeze(-1)


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
    hidden_size = int(model.config.hidden_size)
    value_head = ValueHead(hidden_size).to(model.device)
    return model, tokenizer, value_head


def load_baseline_policy(name: str, seed: int) -> FirstLegalPolicy | RandomPolicy:
    if name == "random":
        return RandomPolicy(seed=seed)
    return FirstLegalPolicy()


def _reward_from_summary(summary: dict[str, Any], seat: int, reward_scale: float) -> float:
    final_scores = list(summary["final_scores"])
    baseline = 25000
    return (final_scores[seat] - baseline) / reward_scale


def _sequence_metrics(
    model: Any,
    tokenizer: Any,
    value_head: ValueHead,
    example: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    logits = outputs.logits
    hidden_states = outputs.hidden_states[-1]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    token_log_probs = F.log_softmax(shift_logits, dim=-1).gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_entropy = -(F.softmax(shift_logits, dim=-1) * F.log_softmax(shift_logits, dim=-1)).sum(dim=-1)
    completion_start = max(prompt_len - 1, 0)
    completion_log_probs = token_log_probs[:, completion_start:]
    completion_entropy = token_entropy[:, completion_start:]
    if completion_log_probs.shape[-1] == 0:
        raise ValueError("completion segment is empty; cannot compute RL loss")
    sequence_logprob = completion_log_probs.sum(dim=-1)
    sequence_entropy = completion_entropy.mean(dim=-1)
    state_value = value_head(hidden_states[:, completion_start, :])
    return sequence_logprob, state_value, sequence_entropy


def _select_policy(model: Any, tokenizer: Any, config: RLTrainConfig) -> ModelPolicy:
    inference_config = InferenceConfig(
        model_path=config.model_name,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
    )
    return ModelPolicy(model=model, tokenizer=tokenizer, config=inference_config)


def _compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    if not (len(rewards) == len(values) == len(dones)):
        raise ValueError("rewards, values, and dones must have the same length")
    advantages = [0.0] * len(rewards)
    returns = [0.0] * len(rewards)
    next_advantage = 0.0
    next_value = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        mask = 0.0 if dones[index] else 1.0
        delta = rewards[index] + gamma * next_value * mask - values[index]
        next_advantage = delta + gamma * gae_lambda * mask * next_advantage
        advantages[index] = next_advantage
        returns[index] = advantages[index] + values[index]
        next_value = values[index]
    return advantages, returns


def _normalize_advantages(values: Sequence[float], center_advantages: bool, reward_normalization: bool) -> list[float]:
    normalized = list(values)
    if reward_normalization and len(normalized) > 1:
        scale = pstdev(normalized)
        if scale > 0.0:
            normalized = [value / scale for value in normalized]
    if center_advantages and normalized:
        offset = mean(normalized)
        normalized = [value - offset for value in normalized]
    return normalized


def _attach_advantages(
    records: Sequence[RLExperience],
    config: RLTrainConfig,
) -> list[RLExperience]:
    by_episode: dict[int, list[RLExperience]] = defaultdict(list)
    for record in records:
        by_episode[record.episode_id].append(record)

    completed: list[RLExperience] = []
    for episode_id in sorted(by_episode):
        episode_records = sorted(by_episode[episode_id], key=lambda item: item.step_index)
        rewards = [item.reward for item in episode_records]
        values = [item.value for item in episode_records]
        dones = [item.done for item in episode_records]
        advantages, returns = _compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        advantages = _normalize_advantages(
            advantages,
            center_advantages=config.center_advantages,
            reward_normalization=config.reward_normalization,
        )
        for item, advantage, returns_value in zip(episode_records, advantages, returns, strict=True):
            completed.append(
                RLExperience(
                    episode_id=item.episode_id,
                    step_index=item.step_index,
                    seat=item.seat,
                    reward=item.reward,
                    done=item.done,
                    value=item.value,
                    old_logprob=item.old_logprob,
                    advantage=advantage,
                    returns=returns_value,
                    sft_example=item.sft_example,
                )
            )
    return completed


def collect_rl_experiences(
    model: Any,
    tokenizer: Any,
    value_head: ValueHead,
    config: RLTrainConfig,
    iteration: int,
    encoder: Any | None = None,
) -> tuple[list[RLExperience], list[dict[str, Any]]]:  # pragma: no cover - depends on optional GPU stack
    raw_experiences: list[RLExperience] = []
    episode_summaries: list[dict[str, Any]] = []

    model.eval()
    value_head.eval()

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

        target_records = [record for record in result["records"] if record["seat"] == target_seat]
        final_reward = _reward_from_summary(summary_payload, seat=target_seat, reward_scale=config.reward_scale)
        with torch.no_grad():
            for index, record in enumerate(target_records):
                logprob, value, _ = _sequence_metrics(
                    model=model,
                    tokenizer=tokenizer,
                    value_head=value_head,
                    example=record["sft_example"],
                )
                raw_experiences.append(
                    RLExperience(
                        episode_id=episode_id,
                        step_index=record["step_index"],
                        seat=record["seat"],
                        reward=final_reward if index == len(target_records) - 1 else 0.0,
                        done=index == len(target_records) - 1,
                        value=float(value.detach().cpu().item()),
                        old_logprob=float(logprob.detach().cpu().item()),
                        advantage=0.0,
                        returns=0.0,
                        sft_example=record["sft_example"],
                    )
                )

    model.train()
    value_head.train()
    return _attach_advantages(raw_experiences, config), episode_summaries


def _iter_minibatches(
    experiences: Sequence[RLExperience],
    minibatch_size: int,
    generator: torch.Generator,
) -> list[list[RLExperience]]:
    permutation = torch.randperm(len(experiences), generator=generator).tolist()
    ordered = [experiences[index] for index in permutation]
    return [
        ordered[start : start + minibatch_size]
        for start in range(0, len(ordered), minibatch_size)
    ]


def _ppo_batch_loss(
    model: Any,
    tokenizer: Any,
    value_head: ValueHead,
    batch: Sequence[RLExperience],
    config: RLTrainConfig,
) -> tuple[torch.Tensor, PPOBatchMetrics]:
    logprobs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    for experience in batch:
        logprob, value, entropy = _sequence_metrics(
            model=model,
            tokenizer=tokenizer,
            value_head=value_head,
            example=experience.sft_example,
        )
        logprobs.append(logprob.squeeze(0))
        values.append(value.squeeze(0))
        entropies.append(entropy.squeeze(0))

    current_logprobs = torch.stack(logprobs)
    current_values = torch.stack(values)
    current_entropies = torch.stack(entropies)
    old_logprobs = torch.tensor([item.old_logprob for item in batch], device=current_logprobs.device, dtype=current_logprobs.dtype)
    old_values = torch.tensor([item.value for item in batch], device=current_values.device, dtype=current_values.dtype)
    advantages = torch.tensor([item.advantage for item in batch], device=current_logprobs.device, dtype=current_logprobs.dtype)
    returns = torch.tensor([item.returns for item in batch], device=current_values.device, dtype=current_values.dtype)

    ratios = torch.exp(current_logprobs - old_logprobs)
    clipped_ratios = torch.clamp(ratios, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
    policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

    clipped_values = old_values + torch.clamp(current_values - old_values, -config.value_clip_epsilon, config.value_clip_epsilon)
    unclipped_value_loss = (current_values - returns).pow(2)
    clipped_value_loss = (clipped_values - returns).pow(2)
    value_loss = 0.5 * torch.max(unclipped_value_loss, clipped_value_loss).mean()

    entropy = current_entropies.mean()
    total_loss = policy_loss + config.value_loss_coef * value_loss - config.entropy_coef * entropy

    approx_kl = (old_logprobs - current_logprobs).mean()
    clip_fraction = ((ratios - 1.0).abs() > config.clip_epsilon).float().mean()
    metrics = PPOBatchMetrics(
        policy_loss=float(policy_loss.detach().cpu()),
        value_loss=float(value_loss.detach().cpu()),
        entropy=float(entropy.detach().cpu()),
        total_loss=float(total_loss.detach().cpu()),
        approx_kl=float(approx_kl.detach().cpu()),
        clip_fraction=float(clip_fraction.detach().cpu()),
    )
    return total_loss, metrics


def _write_rl_history(history: Sequence[RLIterationSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in history:
            handle.write(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def train_rl(config: RLTrainConfig) -> dict[str, Any]:  # pragma: no cover - depends on optional GPU stack
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if config.episodes_per_iteration < 1:
        raise ValueError("episodes_per_iteration must be at least 1")
    if config.iterations < 1:
        raise ValueError("iterations must be at least 1")
    if not 0.0 <= config.gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= config.gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    if config.ppo_epochs < 1:
        raise ValueError("ppo_epochs must be at least 1")
    if config.minibatch_size < 1:
        raise ValueError("minibatch_size must be at least 1")

    model, tokenizer, value_head = _load_rl_model(config)
    base_lr = config.learning_rate
    value_lr = config.value_learning_rate if config.value_learning_rate is not None else config.learning_rate
    optimizer = torch.optim.AdamW(
        [
            {"params": [param for param in model.parameters() if param.requires_grad], "lr": base_lr},
            {"params": list(value_head.parameters()), "lr": value_lr},
        ]
    )
    prompt_builder = PromptBuilder()
    model.train()
    value_head.train()

    history: list[RLIterationSummary] = []
    total_samples = 0
    total_episodes = 0
    rng = torch.Generator()
    rng.manual_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "rl_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    for iteration in range(config.iterations):
        experiences, episode_summaries = collect_rl_experiences(
            model=model,
            tokenizer=tokenizer,
            value_head=value_head,
            config=config,
            iteration=iteration,
            encoder=prompt_builder.encoder,
        )
        if not experiences:
            continue

        rewards = [item.reward for item in experiences]
        advantages = [item.advantage for item in experiences]
        returns = [item.returns for item in experiences]
        values = [item.value for item in experiences]
        score_deltas = [summary["score_deltas"] for summary in episode_summaries]
        mean_score_deltas = [
            float(mean(items))
            for items in zip(*score_deltas, strict=True)
        ] if score_deltas else [0.0, 0.0, 0.0, 0.0]

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        total_losses: list[float] = []
        approx_kls: list[float] = []
        clip_fractions: list[float] = []

        stop_early = False
        for _epoch in range(config.ppo_epochs):
            minibatches = _iter_minibatches(experiences, config.minibatch_size, rng)
            optimizer.zero_grad(set_to_none=True)
            updates_in_window = 0
            for batch in minibatches:
                loss, metrics = _ppo_batch_loss(
                    model=model,
                    tokenizer=tokenizer,
                    value_head=value_head,
                    batch=batch,
                    config=config,
                )
                scaled_loss = loss / config.gradient_accumulation_steps
                scaled_loss.backward()
                updates_in_window += 1
                policy_losses.append(metrics.policy_loss)
                value_losses.append(metrics.value_loss)
                entropies.append(metrics.entropy)
                total_losses.append(metrics.total_loss)
                approx_kls.append(metrics.approx_kl)
                clip_fractions.append(metrics.clip_fraction)

                if updates_in_window % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(value_head.parameters()),
                        config.clip_grad_norm,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            if updates_in_window % config.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(value_head.parameters()),
                    config.clip_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if config.target_kl is not None and approx_kls and abs(mean(approx_kls)) > config.target_kl:
                stop_early = True
            if stop_early:
                break

        iteration_summary = RLIterationSummary(
            iteration=iteration,
            num_episodes=len(episode_summaries),
            num_samples=len(experiences),
            mean_reward=float(mean(rewards)),
            mean_advantage=float(mean(advantages)),
            mean_return=float(mean(returns)),
            mean_value=float(mean(values)),
            mean_policy_loss=float(mean(policy_losses)),
            mean_value_loss=float(mean(value_losses)),
            mean_entropy=float(mean(entropies)),
            mean_total_loss=float(mean(total_losses)),
            mean_kl=float(mean(approx_kls)),
            mean_clip_fraction=float(mean(clip_fractions)),
            mean_score_deltas=mean_score_deltas,
            episode_summaries=episode_summaries,
        )
        history.append(iteration_summary)
        total_samples += len(experiences)
        total_episodes += len(episode_summaries)

        if config.log_jsonl:
            _write_rl_history(history, config.output_dir / "rl_history.jsonl")
        if (iteration + 1) % config.save_every == 0:
            adapter_dir = config.output_dir / f"adapter_iter_{iteration}"
            model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))
            torch.save(value_head.state_dict(), adapter_dir / "value_head.pt")

    final_adapter_dir = config.output_dir / "adapter"
    model.save_pretrained(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    torch.save(value_head.state_dict(), final_adapter_dir / "value_head.pt")
    if config.save_method == "merged_16bit":
        merged_dir = config.output_dir / "merged"
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        torch.save(value_head.state_dict(), merged_dir / "value_head.pt")

    return {
        "iterations": config.iterations,
        "episodes_per_iteration": config.episodes_per_iteration,
        "total_episodes": total_episodes,
        "total_samples": total_samples,
        "history": [asdict(item) for item in history],
        "output_dir": str(config.output_dir),
        "model_name": config.model_name,
        "baseline": config.baseline,
        "ppo_epochs": config.ppo_epochs,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "entropy_coef": config.entropy_coef,
        "value_loss_coef": config.value_loss_coef,
        "clip_epsilon": config.clip_epsilon,
    }


__all__ = [
    "PPOBatchMetrics",
    "RLExperience",
    "RLIterationSummary",
    "RLTrainConfig",
    "ValueHead",
    "collect_rl_experiences",
    "load_baseline_policy",
    "train_rl",
]
