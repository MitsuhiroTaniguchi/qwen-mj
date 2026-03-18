# qwen-mj

Qwen3.5-4B を麻雀の自己対戦環境で学習・評価するための実験リポジトリ。

## 目的

- Unsloth を使った LoRA / RL 学習の検証
- 自己対戦による方策改善の確認
- 固定ベースラインに対する強さの測定

## 予定

- 環境定義
- データ生成
- 学習スクリプト
- 評価スクリプト
- 実験ログ

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

`pymahjong` を使うルール検証まで入れる場合は、`requirements-rules.txt` を追加で入れる。
Unsloth で学習まで回す場合は、`requirements-training.txt` を追加で入れる。
`pip install -e .[training]` でも学習用依存を入れられる。

## 学習

SFT データを作る:

```bash
qwen-mj dataset --mode match --episodes 100 --output data/sft.jsonl
```

Unsloth で SFT を回す:

```bash
qwen-mj train-sft --dataset data/sft.jsonl --output-dir runs/sft
```

自己対戦 RL を回す:

```bash
qwen-mj train-rl --output-dir runs/rl --iterations 1 --episodes-per-iteration 8
```

学習前にデータを検査する:

```bash
qwen-mj validate-dataset --dataset data/sft.jsonl
```

推論は `qwen_mj.select_action(...)` または `qwen_mj.completion_to_action(...)` を使う。

学習済みモデルで self-play を回す:

```bash
qwen-mj play-model --mode match --model-path runs/sft/merged
```

学習済みモデルを固定ベースラインと比較する:

```bash
qwen-mj evaluate-model --episodes 20 --model-path runs/sft/merged --baseline random
```

結果を JSONL に保存する:

```bash
qwen-mj evaluate-model --episodes 20 --model-path runs/sft/merged --baseline random --output runs/eval.jsonl
```

複数 checkpoint をまとめて比較する:

```bash
qwen-mj benchmark-models --model-paths runs/sft/step-100 runs/sft/step-200 --baseline random --output runs/benchmark.jsonl
```

保存済み benchmark を再集計する:

```bash
qwen-mj summarize-benchmark --input runs/benchmark.jsonl --output runs/benchmark-summary.json
```

CSV や表形式でも出せる:

```bash
qwen-mj summarize-benchmark --input runs/benchmark.jsonl --format table
qwen-mj benchmark-models --model-paths runs/sft/step-100 runs/sft/step-200 --format csv
```

## テスト方針

- まず壊れ方をテストで固定する
- ルール分岐の変更には必ず回帰テストを追加する
- 自己対戦の大規模実験より前に、状態遷移の単体テストで潰す
