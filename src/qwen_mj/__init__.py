from .environment import MahjongSelfPlayEnv, TableState
from .benchmark import BenchmarkSummary, ModelBenchmarkResult, benchmark_results_to_csv_text, evaluate_model_paths, load_model_benchmark_jsonl, model_benchmark_result_to_dict, render_benchmark_table, summarize_model_benchmarks, write_model_benchmark_jsonl
from .encoding import EncodedObservation, ObservationEncoder
from .experiment import EpisodeSummary, ExperimentSummary, aggregate_experiment, evaluate_against_baseline, run_self_play_experiment, summarize_episode, write_experiment_jsonl
from .dataset_validation import DatasetValidationError, DatasetValidationReport, load_jsonl, validate_sft_example, validate_sft_jsonl
from .inference import InferenceConfig, ModelPolicy, completion_to_action, generate_completion, load_model, normalize_completion, select_action
from .match import MahjongMatchEnv, MatchState
from .rollout import FirstLegalPolicy, JsonlRolloutLogger, Policy, RandomPolicy, play_hand, play_match
from .training_data import CanonicalActionCodec, PromptBuilder, SFTExample, SYSTEM_PROMPT, example_to_dict, write_sft_jsonl
from .train_sft import SFTTrainConfig, build_training_dataset, example_to_training_text, load_sft_examples, train_sft

try:  # pragma: no cover - optional dependency surface
    from .train_rl import PPOBatchMetrics, RLExperience, RLIterationSummary, RLTrainConfig, ValueHead, collect_rl_experiences, load_baseline_policy, train_rl
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency surface
    if exc.name != "torch":
        raise
    PPOBatchMetrics = None  # type: ignore[assignment]
    RLExperience = None  # type: ignore[assignment]
    RLIterationSummary = None  # type: ignore[assignment]
    RLTrainConfig = None  # type: ignore[assignment]
    ValueHead = None  # type: ignore[assignment]
    collect_rl_experiences = None  # type: ignore[assignment]
    load_baseline_policy = None  # type: ignore[assignment]
    train_rl = None  # type: ignore[assignment]
from .types import Action, ActionKind, Meld, Phase, PlayerState, ReactionOpportunity, Seat, StepResult, Tile, TileInstance, Transition, WinEvent
from .rules import PyMahjongRulesAdapter

__all__ = [
    "Action",
    "ActionKind",
    "EncodedObservation",
    "EpisodeSummary",
    "CanonicalActionCodec",
    "DatasetValidationError",
    "DatasetValidationReport",
    "BenchmarkSummary",
    "benchmark_results_to_csv_text",
    "ModelBenchmarkResult",
    "InferenceConfig",
    "ModelPolicy",
    "FirstLegalPolicy",
    "JsonlRolloutLogger",
    "MahjongSelfPlayEnv",
    "MahjongMatchEnv",
    "Meld",
    "Phase",
    "PyMahjongRulesAdapter",
    "ObservationEncoder",
    "ExperimentSummary",
    "PlayerState",
    "MatchState",
    "PromptBuilder",
    "load_jsonl",
    "SFTTrainConfig",
    "PPOBatchMetrics",
    "RLExperience",
    "RLIterationSummary",
    "RLTrainConfig",
    "aggregate_experiment",
    "evaluate_against_baseline",
    "evaluate_model_paths",
    "load_model_benchmark_jsonl",
    "build_training_dataset",
    "collect_rl_experiences",
    "Policy",
    "ReactionOpportunity",
    "RandomPolicy",
    "SFTExample",
    "example_to_training_text",
    "completion_to_action",
    "generate_completion",
    "load_model",
    "normalize_completion",
    "validate_sft_example",
    "validate_sft_jsonl",
    "select_action",
    "model_benchmark_result_to_dict",
    "render_benchmark_table",
    "summarize_model_benchmarks",
    "Seat",
    "StepResult",
    "SYSTEM_PROMPT",
    "example_to_dict",
    "load_sft_examples",
    "load_baseline_policy",
    "run_self_play_experiment",
    "summarize_episode",
    "train_rl",
    "train_sft",
    "write_experiment_jsonl",
    "write_model_benchmark_jsonl",
    "write_sft_jsonl",
    "play_hand",
    "play_match",
    "TableState",
    "Tile",
    "TileInstance",
    "Transition",
    "ValueHead",
    "WinEvent",
]
