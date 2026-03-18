# Environment Design

This repository builds a Tenhou-rule self-play environment for Qwen3.5-4B.

## What We Reuse

- `pymahjong` for low-level hand logic, shanten, legal-action queries, and win evaluation.
- `MahjongLM` for tokenization ideas, multiview state serialization, and dataset logging structure.

## What Lives Here

- A table-flow layer that manages draw, discard, call, win, and exhaustive-draw transitions.
- An observation layer that can emit both structured tensors and text/token views.
- A reward layer for self-play RL experiments.
- A rollout logger for SFT and preference/RL datasets.

## Early Constraints

- The environment must separate rules logic from model I/O.
- The environment must support both legal-action masking and replay logging.
- Evaluation should use fixed baselines, not self-play only.

## Implementation Order

1. Define state, action, and transition types.
2. Wrap `pymahjong` behind a narrow adapter.
3. Implement table progression.
4. Add observation encoders.
5. Add rollout exporters and benchmarks.

## Current Coverage

- Four-player tile generation with red fives.
- Initial deal and dealer extra tile handling.
- Self-decision phase for discard, riichi, tsumo, kyushukyuhai, and kan actions.
- Reaction queues for ron and prioritized calls.
- Win settlement with fu/fan-based scoring, honba, and riichi stick payout.
- Exhaustive draw settlement for tenpai / noten payment.
- Match wrapper for dealer rotation, honba carryover, and round-wind advancement.
- Observation encoder for numeric tensors and deterministic text views.
- JSONL rollout writer and simple baseline policies for data generation.
- Self-play experiment aggregation and baseline evaluation helpers.
- SFT dataset validation for canonical prompt/completion consistency.
- Inference helpers that map canonical completions back to legal actions.
- CLI entrypoints for model-backed self-play, evaluation, and dataset validation.
- Benchmark helper for sweeping multiple checkpoints against the same baseline.
- Benchmark re-summarizer for saved JSONL results.
- Benchmark CSV and table renderers for quick inspection.
- Regression tests for reset, discard progression, and a known winning hand.

## Conservative Areas

- Kan-related robbing and some late-round edge cases are handled conservatively.
- Physical tile identity is preserved in state so riichi discard constraints can be tightened further.
