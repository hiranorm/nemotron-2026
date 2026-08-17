# EXP002: Unsloth SFT LoRA (ノートブック as EXP)

**注意**: このEXPは `train.py` の代わりに `train.ipynb` を使う例外運用。
mamba_ssm / triton の繊細な wheel 依存を 0.85 公開ノートブック環境に乗せることで解決済み。

## 出典

[fork-of-training-with-unsloth-to-achieve-0-85-lb-6.ipynb](../../research/reference_notebooks/fork-of-training-with-unsloth-to-achieve-0-85-lb-6.ipynb)  
（Tong Hui Kang データ + Unsloth で LB 0.85 を達成した公開ノート）

## 変更点（元ノートから）

- 先頭に `CHILD_EXP = "child-exp000"` セル追加 → `config/{CHILD_EXP}.yaml` でハイパラ切り替え
- LoRA / SFTConfig の直書き値を `_get([...], default)` に置き換え（デフォルトは元ノートと同一）
- `warmup_steps=0` → `warmup_ratio` に統一（等価な変更）

## 実行方法（Kaggle Notebooks）

1. `train.ipynb` をアップロード（または Kaggle Dataset 経由）
2. 先頭セルの `CHILD_EXP = "child-exp000"` を実行したい child に書き換え
3. `config/child-expN.yaml` も同じ Dataset に含めること
4. Save & Run All（5h 以内に完了すること）
5. `infer.ipynb` で `submission.zip` を生成して提出

## child-exp 一覧

| child-exp | epochs | scheduler | grad_norm | r / α | 目的 |
|---|---|---|---|---|---|
| child-exp000 | 1 | linear | 1e9 (OFF) | 32/32 | 元ノート再現（基準） |
| child-exp001 | 2 | cosine + warmup=0.03 | 1.0 | 32/32 | A1+A2+A3 同時投入 |
| child-exp002 | 2 | cosine + warmup=0.03 | 1.0 | 64/64 | r=64 ablation |
