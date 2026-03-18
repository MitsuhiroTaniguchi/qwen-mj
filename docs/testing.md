# Testing Strategy

Mahjong rules are dense, stateful, and easy to break with small refactors.
This repository treats tests as a first-class defense against silent regressions.

## Principles

- Add failing tests before widening behavior.
- Prefer narrow unit tests around state transitions and rule-adapter boundaries.
- Keep smoke tests for importability and basic environment flow.
- Preserve historical bug cases as regression tests.

## What To Test First

1. Reset and initialization invariants.
2. Legal action masks per phase.
3. Discard, call, and turn-advance transitions.
4. Win / draw termination paths.
5. Observation serialization stability.
6. Rule-adapter behavior around `pymahjong` calls.

## Failure Discipline

- Every bug fix should add at least one regression test.
- If a rule cannot be modeled completely yet, codify the partial contract explicitly.
- Avoid large untested rewrites in the table-flow layer.

## Notes From MahjongLM

`MahjongLM` is a useful reference because it treats serialization and validation carefully.
The same discipline should apply here: explicit encodings, narrow interfaces, and repeated checks against known-good logs.

