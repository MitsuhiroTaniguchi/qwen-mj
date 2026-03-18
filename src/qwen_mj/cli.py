from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .encoding import ObservationEncoder
from .experiment import evaluate_against_baseline, run_self_play_experiment
from .rollout import FirstLegalPolicy, JsonlRolloutLogger, RandomPolicy, play_hand, play_match
from .environment import MahjongSelfPlayEnv
from .dataset_validation import validate_sft_jsonl
from .benchmark import evaluate_model_paths, model_benchmark_result_to_dict, write_model_benchmark_jsonl
from .experiment import write_experiment_jsonl
from .inference import InferenceConfig, ModelPolicy, load_model
from .match import MahjongMatchEnv
from .training_data import write_sft_jsonl
from .train_sft import SFTTrainConfig, train_sft


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen-mj")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rollout_parser = subparsers.add_parser("rollout", help="generate a single rollout")
    rollout_parser.add_argument("--mode", choices=["hand", "match"], default="match")
    rollout_parser.add_argument("--seed", type=int, default=0)
    rollout_parser.add_argument("--max-steps", type=int, default=1000)
    rollout_parser.add_argument("--output", type=Path, default=None)
    rollout_parser.add_argument("--policy", choices=["first-legal", "random"], default="first-legal")

    experiment_parser = subparsers.add_parser("experiment", help="run repeated self-play experiments")
    experiment_parser.add_argument("--episodes", type=int, default=10)
    experiment_parser.add_argument("--seed", type=int, default=0)
    experiment_parser.add_argument("--max-steps", type=int, default=10000)
    experiment_parser.add_argument("--policy", choices=["first-legal", "random"], default="first-legal")

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate a policy against a baseline")
    evaluate_parser.add_argument("--episodes", type=int, default=10)
    evaluate_parser.add_argument("--seed", type=int, default=0)
    evaluate_parser.add_argument("--max-steps", type=int, default=10000)
    evaluate_parser.add_argument("--baseline", choices=["first-legal", "random"], default="random")
    evaluate_parser.add_argument("--policy", choices=["first-legal", "random"], default="first-legal")
    evaluate_parser.add_argument("--output", type=Path, default=None)

    dataset_parser = subparsers.add_parser("dataset", help="generate SFT JSONL dataset from rollouts")
    dataset_parser.add_argument("--mode", choices=["hand", "match"], default="match")
    dataset_parser.add_argument("--episodes", type=int, default=10)
    dataset_parser.add_argument("--seed", type=int, default=0)
    dataset_parser.add_argument("--max-steps", type=int, default=10000)
    dataset_parser.add_argument("--policy", choices=["first-legal", "random"], default="first-legal")
    dataset_parser.add_argument("--output", type=Path, required=True)

    train_parser = subparsers.add_parser("train-sft", help="fine-tune a Qwen model with Unsloth")
    train_parser.add_argument("--dataset", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B-Instruct")
    train_parser.add_argument("--max-seq-length", type=int, default=4096)
    train_parser.add_argument("--max-steps", type=int, default=1000)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--grad-accumulation", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--warmup-steps", type=int, default=50)
    train_parser.add_argument("--logging-steps", type=int, default=10)
    train_parser.add_argument("--save-steps", type=int, default=200)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--lora-r", type=int, default=16)
    train_parser.add_argument("--lora-alpha", type=int, default=16)
    train_parser.add_argument("--lora-dropout", type=float, default=0.0)
    train_parser.add_argument("--save-method", choices=["lora", "merged_16bit"], default="lora")

    validate_parser = subparsers.add_parser("validate-dataset", help="validate an SFT JSONL dataset")
    validate_parser.add_argument("--dataset", type=Path, required=True)
    validate_parser.add_argument("--max-errors", type=int, default=20)

    play_model_parser = subparsers.add_parser("play-model", help="run a model-backed self-play rollout")
    play_model_parser.add_argument("--mode", choices=["hand", "match"], default="match")
    play_model_parser.add_argument("--seed", type=int, default=0)
    play_model_parser.add_argument("--max-steps", type=int, default=1000)
    play_model_parser.add_argument("--output", type=Path, default=None)
    play_model_parser.add_argument("--model-path", type=Path, required=True)
    play_model_parser.add_argument("--adapter-path", type=Path, default=None)
    play_model_parser.add_argument("--max-new-tokens", type=int, default=32)
    play_model_parser.add_argument("--temperature", type=float, default=0.0)
    play_model_parser.add_argument("--top-p", type=float, default=1.0)

    evaluate_model_parser = subparsers.add_parser("evaluate-model", help="evaluate a model-backed policy against a baseline")
    evaluate_model_parser.add_argument("--episodes", type=int, default=10)
    evaluate_model_parser.add_argument("--seed", type=int, default=0)
    evaluate_model_parser.add_argument("--max-steps", type=int, default=10000)
    evaluate_model_parser.add_argument("--baseline", choices=["first-legal", "random"], default="random")
    evaluate_model_parser.add_argument("--model-path", type=Path, required=True)
    evaluate_model_parser.add_argument("--adapter-path", type=Path, default=None)
    evaluate_model_parser.add_argument("--max-new-tokens", type=int, default=32)
    evaluate_model_parser.add_argument("--temperature", type=float, default=0.0)
    evaluate_model_parser.add_argument("--top-p", type=float, default=1.0)
    evaluate_model_parser.add_argument("--output", type=Path, default=None)

    benchmark_parser = subparsers.add_parser("benchmark-models", help="evaluate multiple model checkpoints")
    benchmark_parser.add_argument("--model-paths", type=Path, nargs="+", required=True)
    benchmark_parser.add_argument("--episodes", type=int, default=10)
    benchmark_parser.add_argument("--seed", type=int, default=0)
    benchmark_parser.add_argument("--max-steps", type=int, default=10000)
    benchmark_parser.add_argument("--baseline", choices=["first-legal", "random"], default="random")
    benchmark_parser.add_argument("--adapter-path", type=Path, default=None)
    benchmark_parser.add_argument("--max-new-tokens", type=int, default=32)
    benchmark_parser.add_argument("--temperature", type=float, default=0.0)
    benchmark_parser.add_argument("--top-p", type=float, default=1.0)
    benchmark_parser.add_argument("--output", type=Path, default=None)

    return parser


def _policy_from_name(name: str, seed: int):
    if name == "random":
        return RandomPolicy(seed=seed)
    return FirstLegalPolicy()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    encoder = ObservationEncoder()

    if args.command == "rollout":
        policy = _policy_from_name(args.policy, args.seed)
        logger = JsonlRolloutLogger(args.output) if args.output is not None else None
        try:
            if args.mode == "hand":
                env = MahjongSelfPlayEnv(seed=args.seed)
                result = play_hand(env, policy, reset_kwargs={"seed": args.seed}, logger=logger, encoder=encoder, max_steps=args.max_steps)
            else:
                env = MahjongMatchEnv(seed=args.seed)
                result = play_match(env, policy, reset_kwargs={"seed": args.seed}, logger=logger, encoder=encoder, max_steps=args.max_steps)
        finally:
            if logger is not None:
                logger.close()
        print(json.dumps(result["final_observation"], ensure_ascii=False, indent=2, sort_keys=True, default=str))
        return 0

    if args.command == "experiment":
        policy = _policy_from_name(args.policy, args.seed)
        summary = run_self_play_experiment(
            episodes=args.episodes,
            seed=args.seed,
            encoder=encoder,
            max_steps=args.max_steps,
            policies=policy,
        )
        print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        return 0

    if args.command == "evaluate":
        policy = _policy_from_name(args.policy, args.seed)
        baseline = _policy_from_name(args.baseline, args.seed)
        summary = evaluate_against_baseline(
            episodes=args.episodes,
            policy=policy,
            baseline=baseline,
            seed=args.seed,
            encoder=encoder,
            max_steps=args.max_steps,
        )
        if args.output is not None:
            write_experiment_jsonl(summary, args.output)
        print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        return 0

    if args.command == "dataset":
        policy = _policy_from_name(args.policy, args.seed)
        output_records: list[dict[str, Any]] = []
        for episode_id in range(args.episodes):
            reset_seed = args.seed + episode_id
            if args.mode == "hand":
                env = MahjongSelfPlayEnv(seed=reset_seed)
                result = play_hand(env, policy, reset_kwargs={"seed": reset_seed}, encoder=encoder, episode_id=episode_id, max_steps=args.max_steps)
            else:
                env = MahjongMatchEnv(seed=reset_seed)
                result = play_match(env, policy, reset_kwargs={"seed": reset_seed}, encoder=encoder, episode_id=episode_id, max_steps=args.max_steps)
            output_records.extend(result["records"])
        count = write_sft_jsonl(output_records, args.output)
        print(json.dumps({"records_written": count, "output": str(args.output)}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "train-sft":
        config = SFTTrainConfig(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accumulation,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            seed=args.seed,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            save_method=args.save_method,
        )
        summary = train_sft(config)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "validate-dataset":
        report = validate_sft_jsonl(args.dataset, max_errors=args.max_errors)
        payload = {
            "path": report.path,
            "num_records": report.num_records,
            "num_valid": report.num_valid,
            "num_invalid": report.num_invalid,
            "is_valid": report.is_valid,
            "errors": [asdict(error) for error in report.errors],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report.is_valid else 1

    if args.command == "play-model":
        config = InferenceConfig(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        model, tokenizer = load_model(config)
        policy = ModelPolicy(model=model, tokenizer=tokenizer, config=config)
        logger = JsonlRolloutLogger(args.output) if args.output is not None else None
        try:
            if args.mode == "hand":
                env = MahjongSelfPlayEnv(seed=args.seed)
                result = play_hand(
                    env,
                    policy,
                    reset_kwargs={"seed": args.seed},
                    logger=logger,
                    encoder=encoder,
                    max_steps=args.max_steps,
                )
            else:
                env = MahjongMatchEnv(seed=args.seed)
                result = play_match(
                    env,
                    policy,
                    reset_kwargs={"seed": args.seed},
                    logger=logger,
                    encoder=encoder,
                    max_steps=args.max_steps,
                )
        finally:
            if logger is not None:
                logger.close()
        print(json.dumps(result["final_observation"], ensure_ascii=False, indent=2, sort_keys=True, default=str))
        return 0

    if args.command == "evaluate-model":
        config = InferenceConfig(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        model, tokenizer = load_model(config)
        policy = ModelPolicy(model=model, tokenizer=tokenizer, config=config)
        baseline = _policy_from_name(args.baseline, args.seed)
        summary = evaluate_against_baseline(
            episodes=args.episodes,
            policy=policy,
            baseline=baseline,
            seed=args.seed,
            encoder=encoder,
            max_steps=args.max_steps,
        )
        if args.output is not None:
            write_experiment_jsonl(summary, args.output)
        print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        return 0

    if args.command == "benchmark-models":
        results = evaluate_model_paths(
            model_paths=args.model_paths,
            episodes=args.episodes,
            baseline=args.baseline,
            seed=args.seed,
            adapter_path=args.adapter_path,
            max_steps=args.max_steps,
            encoder=encoder,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        if args.output is not None:
            write_model_benchmark_jsonl(results, args.output)
        print(
            json.dumps(
                [model_benchmark_result_to_dict(result) for result in results],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
