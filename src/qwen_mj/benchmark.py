from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

from .encoding import ObservationEncoder
from .experiment import ExperimentSummary, evaluate_against_baseline, experiment_summary_to_dict
from .inference import InferenceConfig, ModelPolicy, load_model
from .rollout import FirstLegalPolicy, Policy, RandomPolicy


@dataclass(slots=True)
class ModelBenchmarkResult:
    model_path: str
    adapter_path: str | None
    baseline: str
    summary: ExperimentSummary
    metadata: dict[str, Any] = field(default_factory=dict)


def _baseline_from_name(name: str, seed: int) -> Policy:
    if name == "random":
        return RandomPolicy(seed=seed)
    return FirstLegalPolicy()


def evaluate_model_paths(
    model_paths: Sequence[str | Path],
    episodes: int,
    baseline: str = "random",
    seed: int | None = None,
    adapter_path: str | Path | None = None,
    max_steps: int = 10000,
    encoder: ObservationEncoder | None = None,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> list[ModelBenchmarkResult]:  # pragma: no cover - depends on optional GPU stack
    encoder = encoder or ObservationEncoder()
    results: list[ModelBenchmarkResult] = []

    for index, model_path in enumerate(model_paths):
        baseline_policy = _baseline_from_name(baseline, seed or 0)
        config = InferenceConfig(
            model_path=model_path,
            adapter_path=adapter_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        model, tokenizer = load_model(config)
        policy = ModelPolicy(model=model, tokenizer=tokenizer, config=config)
        summary = evaluate_against_baseline(
            episodes=episodes,
            policy=policy,
            baseline=baseline_policy,
            seed=None if seed is None else seed + index * episodes,
            encoder=encoder,
            max_steps=max_steps,
        )
        results.append(
            ModelBenchmarkResult(
                model_path=str(model_path),
                adapter_path=str(adapter_path) if adapter_path is not None else None,
                baseline=baseline,
                summary=summary,
                metadata={
                    "index": index,
                    "episodes": episodes,
                    "seed": seed,
                    "max_steps": max_steps,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            )
        )

    return results


def model_benchmark_result_to_dict(result: ModelBenchmarkResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["summary"] = experiment_summary_to_dict(result.summary)
    return payload


def write_model_benchmark_jsonl(results: Sequence[ModelBenchmarkResult], path: str | Path) -> int:
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps({"kind": "model_benchmark", "result": model_benchmark_result_to_dict(result)}, ensure_ascii=False, sort_keys=True) + "\n")
    return len(results)


__all__ = [
    "ModelBenchmarkResult",
    "evaluate_model_paths",
    "model_benchmark_result_to_dict",
    "write_model_benchmark_jsonl",
]
