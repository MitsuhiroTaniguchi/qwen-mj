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

