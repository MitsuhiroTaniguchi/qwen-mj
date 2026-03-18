from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
from io import StringIO
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


@dataclass(slots=True)
class BenchmarkSummary:
    num_results: int
    best_model_path: str | None
    mean_rank: float
    mean_final_scores: list[float]
    mean_score_deltas: list[float]
    results: list[dict[str, Any]]


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


def load_model_benchmark_jsonl(path: str | Path) -> list[ModelBenchmarkResult]:
    input_path = Path(path)
    results: list[ModelBenchmarkResult] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("kind") != "model_benchmark":
                continue
            result = payload["result"]
            summary = result["summary"]
            results.append(
                ModelBenchmarkResult(
                    model_path=result["model_path"],
                    adapter_path=result.get("adapter_path"),
                    baseline=result["baseline"],
                    summary=ExperimentSummary(
                        num_episodes=summary["num_episodes"],
                        mean_steps=summary["mean_steps"],
                        terminated_episodes=summary["terminated_episodes"],
                        truncated_episodes=summary["truncated_episodes"],
                        mean_final_scores=list(summary["mean_final_scores"]),
                        mean_score_deltas=list(summary["mean_score_deltas"]),
                        top_seat_counts=list(summary["top_seat_counts"]),
                        mean_rank=summary["mean_rank"],
                        episode_summaries=[],
                    ),
                    metadata=dict(result.get("metadata", {})),
                )
            )
    return results


def summarize_model_benchmarks(results: Sequence[ModelBenchmarkResult]) -> BenchmarkSummary:
    if not results:
        return BenchmarkSummary(
            num_results=0,
            best_model_path=None,
            mean_rank=0.0,
            mean_final_scores=[0.0, 0.0, 0.0, 0.0],
            mean_score_deltas=[0.0, 0.0, 0.0, 0.0],
            results=[],
        )

    best = min(results, key=lambda item: item.summary.mean_rank)
    mean_rank = sum(result.summary.mean_rank for result in results) / len(results)
    mean_final_scores = [0.0, 0.0, 0.0, 0.0]
    mean_score_deltas = [0.0, 0.0, 0.0, 0.0]
    for result in results:
        for seat in range(4):
            mean_final_scores[seat] += result.summary.mean_final_scores[seat]
            mean_score_deltas[seat] += result.summary.mean_score_deltas[seat]
    count = float(len(results))
    return BenchmarkSummary(
        num_results=len(results),
        best_model_path=best.model_path,
        mean_rank=mean_rank,
        mean_final_scores=[value / count for value in mean_final_scores],
        mean_score_deltas=[value / count for value in mean_score_deltas],
        results=[model_benchmark_result_to_dict(result) for result in results],
    )


def benchmark_result_row(result: ModelBenchmarkResult) -> dict[str, Any]:
    return {
        "model_path": result.model_path,
        "adapter_path": result.adapter_path,
        "baseline": result.baseline,
        "num_episodes": result.summary.num_episodes,
        "mean_rank": result.summary.mean_rank,
        "mean_steps": result.summary.mean_steps,
        "mean_final_scores": list(result.summary.mean_final_scores),
        "mean_score_deltas": list(result.summary.mean_score_deltas),
    }


def benchmark_results_to_csv_text(results: Sequence[ModelBenchmarkResult]) -> str:
    rows = sorted((benchmark_result_row(result) for result in results), key=lambda row: row["mean_rank"])
    if not rows:
        return ""
    headers = [
        "model_path",
        "adapter_path",
        "baseline",
        "num_episodes",
        "mean_rank",
        "mean_steps",
        "mean_final_scores",
        "mean_score_deltas",
    ]
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    for row in rows:
        writer.writerow(
            [
                row["model_path"],
                row["adapter_path"],
                row["baseline"],
                row["num_episodes"],
                f"{row['mean_rank']:.3f}",
                f"{row['mean_steps']:.3f}",
                json.dumps(row["mean_final_scores"], ensure_ascii=False),
                json.dumps(row["mean_score_deltas"], ensure_ascii=False),
            ]
        )
    return buffer.getvalue()


def render_benchmark_table(results: Sequence[ModelBenchmarkResult]) -> str:
    rows = sorted((benchmark_result_row(result) for result in results), key=lambda row: row["mean_rank"])
    if not rows:
        return ""
    headers = ["model_path", "mean_rank", "mean_steps", "mean_final_scores", "mean_score_deltas"]
    table_rows = [headers]
    for row in rows:
        table_rows.append(
            [
                str(row["model_path"]),
                f"{row['mean_rank']:.3f}",
                f"{row['mean_steps']:.3f}",
                json.dumps(row["mean_final_scores"], ensure_ascii=False),
                json.dumps(row["mean_score_deltas"], ensure_ascii=False),
            ]
        )
    widths = [max(len(item[i]) for item in table_rows) for i in range(len(headers))]
    lines: list[str] = []
    for index, row in enumerate(table_rows):
        lines.append(" | ".join(item.ljust(widths[i]) for i, item in enumerate(row)))
        if index == 0:
            lines.append("-+-".join("-" * width for width in widths))
    return "\n".join(lines) + "\n"


__all__ = [
    "BenchmarkSummary",
    "ModelBenchmarkResult",
    "benchmark_result_row",
    "benchmark_results_to_csv_text",
    "evaluate_model_paths",
    "load_model_benchmark_jsonl",
    "model_benchmark_result_to_dict",
    "render_benchmark_table",
    "summarize_model_benchmarks",
    "write_model_benchmark_jsonl",
]
