from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class SFTTrainConfig:
    dataset_path: Path
    output_dir: Path
    model_name: str = "Qwen/Qwen3.5-4B-Instruct"
    max_seq_length: int = 4096
    max_steps: int = 1000
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 50
    logging_steps: int = 10
    save_steps: int = 200
    seed: int = 0
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    load_in_4bit: bool = True
    save_method: str = "lora"


def load_sft_examples(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "sft_example" in payload:
                payload = payload["sft_example"]
            records.append(payload)
    return records


def example_to_training_text(example: dict[str, Any], tokenizer: Any) -> str:
    messages = list(example["messages"])
    if not messages or messages[0]["role"] != "system":
        raise ValueError("SFT example is missing the system message")
    chat = messages + [{"role": "assistant", "content": example["completion"]}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
        text += tokenizer.eos_token
    return text


def build_training_dataset(records: Iterable[dict[str, Any]], tokenizer: Any):
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("datasets is required to build a training dataset") from exc

    texts = [{"text": example_to_training_text(record, tokenizer)} for record in records]
    return Dataset.from_list(texts)


def _import_unsloth():  # pragma: no cover - optional dependency
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "unsloth is required for training; install the training extra and try again"
        ) from exc

    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("trl is required for training; install the training extra and try again") from exc

    return FastLanguageModel, SFTConfig, SFTTrainer


def train_sft(config: SFTTrainConfig) -> dict[str, Any]:  # pragma: no cover - depends on optional GPU stack
    FastLanguageModel, SFTConfig, SFTTrainer = _import_unsloth()

    records = load_sft_examples(config.dataset_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )

    dataset = build_training_dataset(records, tokenizer)
    training_args = SFTConfig(
        output_dir=str(config.output_dir),
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        seed=config.seed,
        dataset_num_proc=1,
        packing=False,
        dataset_text_field="text",
        report_to="none",
        ddp_find_unused_parameters=False,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    trainer.train()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = config.output_dir / "adapter"
    merged_dir = config.output_dir / "merged"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    if config.save_method == "merged_16bit":
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    return {
        "records": len(records),
        "output_dir": str(config.output_dir),
        "model_name": config.model_name,
        "save_method": config.save_method,
    }


__all__ = [
    "example_to_training_text",
    "SFTTrainConfig",
    "build_training_dataset",
    "load_sft_examples",
    "train_sft",
]
