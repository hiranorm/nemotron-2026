# 受賞解法分析 — tonghuikang/nemotron (Progress Prize)

調査日: 2026-05-27 / ソース: https://github.com/tonghuikang/nemotron （master ブランチの実コード）

Tong Hui Kang = 当方が既に使っている CoT データの作者。Progress Prize 受賞解法のコードが**全公開**されている。

## LoRA / 学習レシピ（train_sft.py）
- base model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- **lora_rank = 32**（コード default、コメント "# 32"）→ **rank ≤ 32 が制約と判断**。当方の r=64（child-exp002 OOM / child-exp005）は**無効・徒労だった公算大**。要 rules 最終確認。
- lr = **2e-4**、schedule = **StepLinearDecayLRSchedule**（`lr * (1 - step/total_steps)` で epoch 内を 0 まで線形減衰）。当方は cosine → ここが差分。
- **num_epochs = 1**、batch_size = 64、micro_batch_size = 16、max_length = 8192
- **train_mlp=True / train_attn=True / train_unembed=True**（MLP+Attention+unembed(lm_head) を学習）
- backend = tinker（Thinking Machines の学習API）or modal。当方は Unsloth ローカル（RTX Pro 6000）。レシピは移植可能。
- epoch 0 で logprob を保存し epoch 1+ で ref として使う仕組み（多epoch実験用、default は1）。loss_config に advantages あり（加重損失の余地、未深掘り）。

## 勝ち筋＝データエンジニアリング（本命）
6カテゴリは**決定論的ルールパズル**（bit_manipulation / cipher / equation_numeric / cryptarithm / gravity / numeral / unit_conversion）。

- **reasoning.py**: 各カテゴリの**ソルバを自作**（`reasoners/<category>.py`）し、解法手順を**自然な CoT に変換**して `reasoning/<problem_id>.txt` を生成。「rule_found」な問題のみ採用、`cryptarithm_guess` はスキップ。**正解保証の合成 CoT**が肝。
- **augmentation.py**: `spelling / concatenation / splitting / matching / lstrip` の augmenter でトークナイズ/書式ロバスト性を付与（`augmentations/<id>.txt`、`[category]/[prompt]/[completion]` 形式）。
- investigations を持ち、`--delete-investigations`（正解時に調査ファイル削除）で品質管理。

→ 当方の現状（Tong の merged CoT CSV を流用）より、**ソルバ由来の正解 CoT + augmentation** が 0.85→0.87 の本命差分。コードが公開なので再利用/再生成が可能。

## 当方への適用（軽量・~6/8 まで）
1. **rank は 32 に固定**。r=64 路線は破棄（無効）。→ [[GUARDRAILS 相当]]。
2. **lr schedule を step-linear-decay→0 に**（cheap な config 変更）。
3. **target を MLP+attn+unembed に**揃える（現状 lm_head は入っている）。
4. **受賞者の公開データ（reasoning/ + augmentations/）を採用**して 1 epoch 学習・提出（最大レバレッジの軽量手）。**ライセンス要確認**。
5. **カテゴリ別 maj@64 診断**で弱カテゴリ特定 → データ補強の的を絞る。

## 注意
- ライセンス: 再利用前に repo の LICENSE / コンペの再配布規約を確認。
- 提出で制御できるのは **adapter のみ**（maj@64 のデコードは不可侵）。
