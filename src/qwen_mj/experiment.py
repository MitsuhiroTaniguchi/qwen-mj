from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from .encoding import ObservationEncoder
from .match import MahjongMatchEnv
from .rollout import FirstLegalPolicy, Policy, RandomPolicy, play_match

PLAYER_COUNT = 4
STARTING_SCORE = 25000


@dataclass(slots=True)
class EpisodeSummary:
    episode_id: int
    seat: int
    steps: int
    terminated: bool
    truncated: bool
    final_scores: list[int]
    score_deltas: list[int]
    top_seat: int
    top_score: int
    seat_rank: int
    dealer: int
    round_wind: int
    honba: int


@dataclass(slots=True)
class ExperimentSummary:
    num_episodes: int
    mean_steps: float
    terminated_episodes: int
    truncated_episodes: int
    mean_final_scores: list[float]
    mean_score_deltas: list[float]
    top_seat_counts: list[int]
    mean_rank: float
    episode_summaries: list[EpisodeSummary] = field(default_factory=list)


def _normalize_policies(policies: Policy | Sequence[Policy]) -> list[Policy]:
    if isinstance(policies, Sequence):
        if len(policies) != PLAYER_COUNT:
            raise ValueError("policy sequence must contain four seat policies")
        return list(policies)
    return [policies for _ in range(PLAYER_COUNT)]


def _score_rank(scores: Sequence[int], seat: int) -> int:
    sorted_scores = sorted(((score, idx) for idx, score in enumerate(scores)), reverse=True)
    for rank, (_, idx) in enumerate(sorted_scores, start=1):
        if idx == seat:
            return rank
    raise ValueError("seat not found in scores")


def summarize_episode(
    episode_id: int,
    seat: int,
    result: dict[str, Any],
    initial_scores: Sequence[int] | None = None,
) -> EpisodeSummary:
    initial_scores = list(initial_scores) if initial_scores is not None else [STARTING_SCORE for _ in range(PLAYER_COUNT)]
    match_obs = result["final_observation"]["match"]
    final_scores = list(match_obs["scores"])
    score_deltas = [final - start for final, start in zip(final_scores, initial_scores, strict=True)]
    top_score = max(final_scores)
    top_seat = final_scores.index(top_score)
    return EpisodeSummary(
        episode_id=episode_id,
        seat=seat,
        steps=len(result["records"]),
        terminated=result["final_observation"]["match"]["terminated"],
        truncated=result["final_observation"]["match"]["truncated"],
        final_scores=final_scores,
        score_deltas=score_deltas,
        top_seat=top_seat,
        top_score=top_score,
        seat_rank=_score_rank(final_scores, seat),
        dealer=match_obs["dealer"],
        round_wind=match_obs["round_wind"],
        honba=match_obs["honba"],
    )


def aggregate_experiment(episode_summaries: Sequence[EpisodeSummary]) -> ExperimentSummary:
    if not episode_summaries:
        return ExperimentSummary(
            num_episodes=0,
            mean_steps=0.0,
            terminated_episodes=0,
            truncated_episodes=0,
            mean_final_scores=[0.0 for _ in range(PLAYER_COUNT)],
            mean_score_deltas=[0.0 for _ in range(PLAYER_COUNT)],
            top_seat_counts=[0 for _ in range(PLAYER_COUNT)],
            mean_rank=0.0,
            episode_summaries=[],
        )

    mean_final_scores = [0.0 for _ in range(PLAYER_COUNT)]
    mean_score_deltas = [0.0 for _ in range(PLAYER_COUNT)]
    top_seat_counts = [0 for _ in range(PLAYER_COUNT)]
    terminated_episodes = 0
    truncated_episodes = 0
    total_steps = 0
    total_rank = 0
    for summary in episode_summaries:
        total_steps += summary.steps
        total_rank += summary.seat_rank
        if summary.terminated:
            terminated_episodes += 1
        if summary.truncated:
            truncated_episodes += 1
        top_seat_counts[summary.top_seat] += 1
        for seat in range(PLAYER_COUNT):
            mean_final_scores[seat] += summary.final_scores[seat]
            mean_score_deltas[seat] += summary.score_deltas[seat]

    count = float(len(episode_summaries))
    return ExperimentSummary(
        num_episodes=len(episode_summaries),
        mean_steps=total_steps / count,
        terminated_episodes=terminated_episodes,
        truncated_episodes=truncated_episodes,
        mean_final_scores=[value / count for value in mean_final_scores],
        mean_score_deltas=[value / count for value in mean_score_deltas],
        top_seat_counts=top_seat_counts,
        mean_rank=total_rank / count,
        episode_summaries=list(episode_summaries),
    )


def experiment_summary_to_dict(summary: ExperimentSummary) -> dict[str, Any]:
    payload = asdict(summary)
    payload["episode_summaries"] = [asdict(item) for item in summary.episode_summaries]
    return payload


def write_experiment_jsonl(summary: ExperimentSummary, path: str | Path) -> int:
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as handle:
        for episode in summary.episode_summaries:
            handle.write(
                json.dumps(
                    {
                        "kind": "episode_summary",
                        "episode_summary": asdict(episode),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        handle.write(
            json.dumps(
                {
                    "kind": "experiment_summary",
                    "experiment_summary": experiment_summary_to_dict(summary),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
    return len(summary.episode_summaries) + 1


def run_self_play_experiment(
    episodes: int,
    env_factory: Callable[[], MahjongMatchEnv] | None = None,
    policies: Policy | Sequence[Policy] | None = None,
    seed: int | None = None,
    encoder: ObservationEncoder | None = None,
    max_steps: int = 10000,
) -> ExperimentSummary:
    env_factory = env_factory or (lambda: MahjongMatchEnv(seed=seed))
    policies = policies or FirstLegalPolicy()
    encoder = encoder or ObservationEncoder()
    episode_summaries: list[EpisodeSummary] = []

    for episode_id in range(episodes):
        env = env_factory()
        reset_kwargs = {"seed": None if seed is None else seed + episode_id}
        result = play_match(env, policies, reset_kwargs=reset_kwargs, encoder=encoder, episode_id=episode_id, max_steps=max_steps)
        seat = episode_id % PLAYER_COUNT
        episode_summaries.append(summarize_episode(episode_id, seat, result))

    return aggregate_experiment(episode_summaries)


def evaluate_against_baseline(
    episodes: int,
    policy: Policy | Sequence[Policy],
    baseline: Policy | Sequence[Policy] | None = None,
    env_factory: Callable[[], MahjongMatchEnv] | None = None,
    seed: int | None = None,
    encoder: ObservationEncoder | None = None,
    max_steps: int = 10000,
) -> ExperimentSummary:
    env_factory = env_factory or (lambda: MahjongMatchEnv(seed=seed))
    baseline = baseline or RandomPolicy(seed=seed)
    encoder = encoder or ObservationEncoder()
    episode_summaries: list[EpisodeSummary] = []

    for episode_id in range(episodes):
        env = env_factory()
        seat = episode_id % PLAYER_COUNT
        seat_policies = _normalize_policies(baseline)
        if isinstance(policy, Sequence):
            seat_policies = list(policy)
        else:
            seat_policies[seat] = policy
        reset_kwargs = {"seed": None if seed is None else seed + episode_id}
        result = play_match(env, seat_policies, reset_kwargs=reset_kwargs, encoder=encoder, episode_id=episode_id, max_steps=max_steps)
        episode_summaries.append(summarize_episode(episode_id, seat, result))

    return aggregate_experiment(episode_summaries)


__all__ = [
    "EpisodeSummary",
    "ExperimentSummary",
    "aggregate_experiment",
    "evaluate_against_baseline",
    "experiment_summary_to_dict",
    "run_self_play_experiment",
    "summarize_episode",
    "write_experiment_jsonl",
]
