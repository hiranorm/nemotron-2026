# 過去の優勝アプローチ・有効手法

## 直接の前身コンペ：AIMO-2（AI Mathematical Olympiad Progress Prize 2）

### NemoSkills チーム（NVIDIA）- 1位

**使用モデル：** Qwen2.5-14B-Base（OpenMath-Nemotron-14B-Kaggle として公開）

**主な手法：**

1. **合成データでのファインチューニング**
   - DeepSeek-R1 と QwQ-32B で生成した何百万もの合成解答データで Qwen2.5-14B をファインチューニング
   - OpenMathReasoning データセットのサブセットを使用
   - Code execution ベースの推論（コードを書かせて実行）

2. **並列 long-thinking（多数決）**
   - 複数の長い推論トレースを並列生成してから多数決
   - Early-stopping 技術で効率化

3. **推論最適化**
   - **FP8 量子化**：1.5× スループット向上
   - **ReDrafter speculative decoding**：1.8× スループット向上
   - TensorRT-LLM 使用

4. **データキュレーション**
   - コンペの形式・難易度に近い問題を重点的に学習

**結果：** 50 問中 34 問正解（5時間制限、NVIDIA L4 GPU × 4）

**モデル性能（OpenMath-Nemotron-14B-Kaggle）：**
| Benchmark | pass@1 | maj@64 |
|-----------|--------|--------|
| AIME24 | 73.7 | 86.7 |
| AIME25 | 57.9 | 73.3 |
| HMMT-24-25 | 50.5 | 64.8 |

## 本コンペへの示唆

本コンペは **AIMO-2 の後継**として、数学的推論ではなく **パターン推論（few-shot rule inference）** に特化している。

### 有効と考えられるアプローチ

| アプローチ | 理由 |
|-----------|------|
| Chain-of-Thought プロンプト | 推論トレースを生成させるほうが精度が高い |
| Few-shot 例の順序・フォーマット最適化 | 例の見せ方でモデルの理解が変わる |
| 多数決（maj@k） | 評価指標が maj@64 なので多様な生成が重要 |
| LoRA ファインチューニング | 各問題タイプに特化した学習 |
| temperature=1.0, top_p=0.95 | Reasoning-On モードの推奨パラメータ |

### 各問題タイプ別のアプローチ推定

| タイプ | 想定アプローチ |
|--------|---------------|
| bit_manipulation | ルール帰納 → ビット演算のコード実行 |
| encryption | 文字置換ルール推定 → アルファベットシフト計算 |
| equation_transform | 記号変換ルール帰納 |
| gravity | 定数フィッティング（g の値を推定） |
| unit_conversion | 変換係数推定 |
| numeral_system | 記数法の推定（ローマ数字、16進数など） |

## 関連リソース

- [OpenMath-Nemotron-14B-Kaggle（HuggingFace）](https://huggingface.co/nvidia/OpenMath-Nemotron-14B-Kaggle)
- [NeMo-Skills（GitHub）](https://github.com/NVIDIA/NeMo-Skills)
- [OpenMathReasoning データセット](https://huggingface.co/datasets/nvidia/OpenMathReasoning)
- [論文：arXiv:2504.16891](https://arxiv.org/abs/2504.16891)
- Kaggle ノートブック：[Structured Reasoning for NVIDIA NeMoTron](https://www.kaggle.com/code/barkataliarbab/structured-reasoning-for-nvidia-nemotron)
- Kaggle ノートブック：[NVIDIA Nemotron Submission Demo](https://www.kaggle.com/code/ryanholbrook/nvidia-nemotron-submission-demo)
- Kaggle ノートブック：[Nemotron SFT LoRA with CoT](https://www.kaggle.com/code/konbu17/nemotron-sft-lora-with-cot-v2-prep-now-plz-wait)
