# SOTA モデル・論文リファレンス

## コンペで使用できるモデル（Nemotron シリーズ）

### Nemotron-3-Nano-30B-A3B（コンペのメインモデル）

| 項目 | 詳細 |
|------|------|
| アーキテクチャ | Hybrid Mamba-2 + Transformer MoE |
| 総パラメータ | 30B |
| アクティブパラメータ | 3B（推論時） |
| コンテキスト長 | 1M トークン |
| 推論モード | Reasoning On/Off（`enable_thinking` パラメータ） |
| ライセンス | NVIDIA Open Model License（商用利用可） |
| Kaggle モデル | `metric/nemotron-3-nano-30b-a3b-bf16` |

**推奨推論パラメータ（Reasoning-On）：**
```python
tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    enable_thinking=True,
    add_generation_prompt=True,
    return_tensors="pt"
)
# temperature=1.0, top_p=0.95
```

### NVIDIA-Nemotron-3-Nano-4B-BF16

| 項目 | 詳細 |
|------|------|
| アーキテクチャ | Hybrid Mamba-2 + Transformer |
| パラメータ | 3.97B（4B） |
| コンテキスト長 | 262K トークン |
| 特徴 | Attention 層は4つのみ、主に Mamba-2 + MLP |
| 用途 | エッジデバイス・軽量推論 |

**MATH500 ベンチマーク：**
- Reasoning-Off: 95.4
- Reasoning-On: 95.4（AIME25 は 78.5）

### Nemotron-3-Super（120B-A12B）/ Ultra（Coming H1 2026）

- Latent MoE（4× experts at same inference cost）
- Multi-token prediction (MTP)
- NVFP4 4-bit training
- コンペ制約上使用できない可能性あり（要確認）

## 関連技術・手法

### Chain-of-Thought × Reasoning Trace

- Nemotron モデルは `enable_thinking=True` で推論トレースを生成
- Reasoning Budget: 1,024 tokens が sweet spot（256〜16,384 の実験より）
- 複雑なタスクでは reasoning trace ありのほうが精度向上

### vLLM での高速推論

```bash
pip install -U "vllm>=0.15.1"

# カスタム reasoning parser が必要
wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/resolve/main/nano_v3_reasoning_parser.py

vllm serve nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --served-model-name nemotron3-nano-4B-BF16 \
  --max-model-len 262144 \
  --trust-remote-code \
  --reasoning-parser-plugin nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3
```

### LoRA ファインチューニング

- コミュニティノートブック [Nemotron SFT LoRA with CoT](https://www.kaggle.com/code/konbu17/nemotron-sft-lora-with-cot-v2-prep-now-plz-wait) が存在
- Train データ（9,500 問）を使って問題タイプ別に LoRA 学習できる

## 参考論文

| 論文 | 内容 |
|------|------|
| [arXiv:2504.16891](https://arxiv.org/abs/2504.16891) | AIMO-2 優勝解法・OpenMathReasoning データセット |
| [Nemotron-3 Technical Blog](https://developer.nvidia.com/blog/inside-nvidia-nemotron-3-techniques-tools-and-data-that-make-it-efficient-and-accurate/) | Nemotron-3 アーキテクチャ詳細 |

## Kaggle Models（利用可能）

- `metric/nemotron-3-nano-30b-a3b-bf16` — コンペ主役モデル（30B MoE）
- `konstantinboyko/nvidia-nemotron-3-nano-30b-a3b` — コミュニティ提供
- `seyominaoto/nvidia-nemotron-repo` — 関連リポジトリ

## 注意事項

- Nemotron-3-Nano-30B-A3B は vLLM 0.15.1+ が必要
- `trust_remote_code=True` が必要
- BF16 での推論が標準（NVFP4/FP8 は Super/Ultra のみ）
- Kaggle のオフライン環境では事前にモデルをダウンロード済みのデータセットを参照する必要あり
