# NVIDIA Nemotron Model Reasoning Challenge - Competition Overview

## 基本情報

| 項目 | 内容 |
|------|------|
| コンペ名 | NVIDIA Nemotron Model Reasoning Challenge |
| Kaggle スラッグ | nvidia-nemotron-model-reasoning-challenge |
| 期間 | 2026-03-16 〜 2026-06-15 |
| 賞金総額 | $106,388 |
| 参加チーム数 | ~1,916 チーム（2026-04-11 時点） |
| カテゴリ | Featured（メダル付与あり） |

## タスク概要

「Alice's Wonderland」という設定で、**隠れたルールを few-shot 例から推論**して正解を出すタスク。

LLM（Nemotron 3 Nano）が few-shot 例を見て、隠されたパターン・変換ルールを逆算して答える能力を評価する。

## データセット

### 問題カテゴリ（6種類）

| カテゴリ | 問題数（train） | 答えの形式 |
|----------|-----------------|------------|
| bit_manipulation | 1,602 | 8ビット2進数（固定8文字） |
| encryption（text cipher） | 1,576 | 英単語列（13〜39文字） |
| equation_transform | 1,555 | 記号列（1〜4文字） |
| gravity（物理）| 1,597 | 数値（小数、3〜6文字） |
| unit_conversion | 1,594 | 数値（小数） |
| numeral_system | 1,576 | 数字/ローマ数字など |
| **合計** | **9,500** | |

### ファイル構成

- `train.csv`: 9,500 行（id, prompt, answer）
- `test.csv`: 3 行（公開テストは極めて少ない）

### 問題形式の例

```
In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers.
The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT...

Here are some examples of input -> output:
01010001 -> 11011101
00001001 -> 01101101
...
Now, determine the output for: 00110100
```

- Few-shot 例（8〜9個）が与えられる
- 1問ごとに独立したルール（同一 id なら同じルール）
- train と test は同じ id を持つ（train で答えあり → test は答えなし）

## 評価指標

- **pass@1 (maj@64)**: 64回生成した結果の多数決で最終予測を決定し、正解かどうかを判定
- 評価は Kaggle ノートブックとして提出（コードコンペ形式）

## コンペ形式・制約

- **コードコンペ（ノートブック提出）**
- 推論時間制限：**約5時間**（AIMO-2 の先例から）
- GPU：Google Cloud の G4 VM（NVIDIA L4 GPU × 4）
- インターネット接続：不可
- 使用モデル：**NVIDIA Nemotron シリーズのオープンモデルのみ**

## 主な許可されるアプローチ

1. **プロンプトエンジニアリング**（few-shot 構成の工夫）
2. **合成データ生成**
3. **データキュレーション（フィルタリング）**
4. **強化学習（RL）**
5. **軽量ファインチューニング（LoRA など）**

## 使用対象モデル（公式推奨）

| モデル | パラメータ | 特徴 |
|--------|-----------|------|
| **Nemotron-3-Nano-30B-A3B** | 30B total / 3B active | Hybrid Mamba-2+Transformer MoE。コンペ主役モデル |
| **NVIDIA-Nemotron-3-Nano-4B-BF16** | 4B | 小型・エッジ向け。Reasoning On/Off モード |
| Nemotron-3-Super / Ultra | 120B+ | H1 2026 以降リリース予定 |

## 背景・経緯

- NVIDIA はすでに AIMO-2（AI Mathematical Olympiad Progress Prize 2）に優勝（NemoSkills チーム）
- AIMO-2 優勝手法（Qwen2.5-14B + CoT ファインチューニング + code execution + FP8 量子化 + speculative decoding）が Nemotron Ultra モデルに組み込まれた
- 本コンペは NVIDIA の Nemotron Nano（オープンモデル）を使って同様の推論能力向上を競う
