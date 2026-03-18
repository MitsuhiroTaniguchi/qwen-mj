from __future__ import annotations

import argparse
import json
from pathlib import Path

from .encoding import ObservationEncoder
from .experiment import evaluate_against_baseline, run_self_play_experiment
from .rollout import FirstLegalPolicy, JsonlRolloutLogger, RandomPolicy, play_hand, play_match
from .environment import MahjongSelfPlayEnv
from .match import MahjongMatchEnv


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
        print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        return 0

    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
