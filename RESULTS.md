# nvidia-nemotron-model-reasoning-challenge — 実験結果履歴

実験のスコアと考察の一元管理。**事実の記録**に徹する（次アクションは backlog、未着手アイデアは IDEAS.md）。

> このコンペは旧 `EXP/EXP_SUMMARY.md` + `MEMORY.md` 運用から **template スタイル（RESULTS/GUARDRAILS/IDEAS + backlog）に移行**（2026-05-27）。
> 実験命名は Kaggle Dataset slug と紐づくため **旧来の `EXP002` / `child-exp00N` のまま**（exp001-001 形式へは改名しない）。
> コンペ概要は `research/competition_overview.md`、ダメパターンは `GUARDRAILS.md`、タスクは `backlog task list --plain`。

## 運用ルール（AI への指示）
- ユーザーが LB を報告したら即座にこのファイルを更新（該当 child に LB、考察、比較テーブル、新ベストなら Competition Status）。
- LB を下げた原因が判明したら `GUARDRAILS.md` に転記。新アイデアは `IDEAS.md`（着手するなら backlog task 化）。

---

## Competition Status

| 項目 | 値 |
|------|-----|
| **Current Best LB** | 0.85 (EXP002/child-exp001, 2026-04-30) |
| **Current Best CV** | — |
| **Target** | 0.87（金圏。public に多数密集） |
| **Deadline** | 2026-06-15 |
| **Top LB** | 0.87（多数チーム密集、2026-05-27） |
| **Submission format** | LoRA adapter zip（adapter_config.json + adapter_model.safetensors）。**rank ≤ 32**。制御できるのは adapter のみ（maj@64 は不可侵）|
| **位置づけ** | **2026-06-02 撤退（軽量打ち切り、decision-001）**。最終提出は child-exp001（LB 0.85）放置。主戦場は ROGII |

## Model Comparison Table

| Exp | Child | Approach | LB | Notes |
|-----|-------|----------|----|-------|
| EXP000 | child-exp000 | Untrained LoRA baseline, r=32, α=16 | **0.53** | 素の base model の基準点 |
| EXP001 | child-exp000 | CoT-SFT (konbu17 dataset), r=32, α=32 | — | mamba-ssm wheel 問題で未提出・廃棄 |
| EXP002 | child-exp000 | Unsloth SFT, Tong-CoT, r=32, lr=2e-4, epochs=1 | **0.84** | 再現確認用。Train+Infer 分離フロー健全性確認済み |
| EXP002 | child-exp001 | Unsloth SFT + cosine + epochs=2 + clip=1.0 | **0.85** | A1+A2+A3 同時投入。child-exp000 比 +0.01。**現ベスト** |
| EXP002 | child-exp002 | child-exp001 + r=64, α=64 | **OOM** | bs=2+r=64 が VRAM 超過（rank≤32 違反で破棄）|
| EXP002 | child-exp003 | child-exp001 + Tong+konbu17混合 (~14k rows) | **0.85** | データ追加の効果なし（品質律速の可能性）|
| EXP002 | child-exp004 | child-exp002 + bs=1, grad_accum=32 | **TO/破棄** | r=64 路線。rank≤32 違反で破棄 |
| EXP002 | child-exp005 | child-exp004 + epochs=1 | **未提出/破棄** | r=64。5/2 放置・未提出。rank≤32 違反で破棄 |
| EXP002 | child-exp006 | child-exp001 + lr_scheduler linear（warmup 0）| **0.85** | 受賞 LR schedule 単変数。child-exp001 と同等・改善なし（2026-05-30）|
| EXP002 | child-exp007 | 受賞データ(ソルバ由来CoT 9500) + rank32 + linear + 1ep | **0.84** | **本命だったが -0.01 悪化**。受賞データ単体採用では効かず（2026-05-30）→ GUARDRAILS |

Kaggle private Dataset: `hiranorm/nemotron-winner-cot-v1`（c007 データ）/ `nemotron-exp002-child-exp006` / `-child-exp007`（config）。run notebook: `notebooks/run-EXP002-child-exp00[67].ipynb`。

## lr スイープ結果（参照ノート上で手動 fork）

| lr | LB | 備考 |
|---|---|---|
| 5e-5 | 0.65-0.66 | 低すぎ（→ GUARDRAILS）|
| 1e-4 | 0.82 | |
| 2e-4 | 0.83-0.84 | **スイートスポット**。seed 依存ほぼなし |
| 3e-4 | 0.84 | 2e-4 と同等 |

→ **2e-4 固定で他 axis を探索**。

---

## 主要な設計決定（旧 MEMORY.md より移行）

| 決定 | 理由 |
|------|------|
| 提出形式は LoRA アダプタ zip | メトリクスノートブックが LLM 推論全体を担う仕組み |
| SFT 回答に `</think>\boxed{answer}` を付与、`<think>` 自体は入れない | chat template が `enable_thinking=True` で `<think>` を自動付与するため |
| EXP002 はノートブック as EXP | mamba_ssm wheel 依存が 0.85 公開ノート環境で解決済み |
| lr=2e-4 固定で他 axis を探索 | lr sweep（5e-5〜3e-4）で 2e-4 がスイートスポット |
| データは Tong Hui Kang CoT | サンプル数が多く LB 0.85 達成実績。**ただし受賞解法はソルバ由来CoTが鍵（research/winner_tonghuikang_analysis.md）** |
| target_modules にリスト指定（lm_head 含む）| 0.85 ノート準拠。受賞解法も MLP+attn+unembed |
| rank=32 固定 | **rank ≤ 32 がコンペ制約**（2026-05-27 確定）|

---

## Experiment Log

### EXP000: Untrained LoRA Baseline — LB 0.53
学習なしの素の Nemotron-3-Nano-30B-A3B の基準値（2026-04-22）。

### EXP001: CoT-SFT（廃棄）
L4×4 offline で mamba_ssm wheel 問題が未解決 → EXP002 の 0.85 ノート環境へ乗り換え。

### EXP002: Unsloth SFT LoRA（ノートブック as EXP）
出典: 公開 0.85 ノート（fork-of-training-with-unsloth-to-achieve-0-85-lb）。
- **child-exp000 (0.84)**: 元ノート再現。Train+Infer 分離フロー健全性確認。
- **child-exp001 (0.85)**: epochs=2 + cosine + warmup=0.03 + max_grad_norm=1.0。child-exp000 比 +0.01。現ベスト（個別 ablation 未実施）。
- **child-exp002 (OOM)**: r=64,α=64 で VRAM 超過。
- **child-exp003 (0.85)**: Tong+konbu17 14,388 rows。**改善なし** → データ量より品質が律速の可能性。
- **child-exp004 (TO) / child-exp005 (未提出)**: r=64 路線。rank≤32 違反で破棄。
- **child-exp006 (0.85, 2026-05-30)**: 受賞 lr schedule（linear-decay, warmup 0）に寄せた単変数。child-exp001 と同等で改善なし → lr schedule 単独では現状を超えられない。
- **child-exp007 (0.84, 2026-05-30)**: 受賞データ（ソルバ由来 CoT 9500）+ rank32 + linear + 1ep。**-0.01 悪化**。本命だったが効かず → GUARDRAILS。

### 受賞解法の知見（tonghuikang, Progress Prize）
詳細 `research/winner_tonghuikang_analysis.md`。rank=32 / lr=2e-4 linear-decay→0 / 1 epoch / MLP+attn+unembed。
**勝ち筋＝データ**（6カテゴリは決定論パズル→ソルバ自作で正解保証 CoT 9500件）と仮説していたが、child-exp007 では LB -0.01。
仮説: (a) 受賞 CoT 9500 件は Tong CoT より少量で量律速、(b) 受賞レシピ＋受賞データの組み合わせ前提（単独移植では効かない）、(c) ライセンス由来の私的版 winner_cot_v1 の品質劣化。

### TASK-004 カテゴリ別 maj@N 診断（2026-05-30 進行中）
本命 2 本（c006/c007）が外れたため、どのカテゴリが弱いかをまず見る診断軸を整備。
- カテゴリ分割: train.csv 9500 行を 6 カテゴリ（bit 1602 / gravity 1597 / unit 1594 / cipher 1576 / numeral 1576 / equation 1555）に prompt prefix 完全一致で分類（`scripts/categorize_train.py`、`data/inputs/train_categorized.csv`）。
- 診断 subset: 各カテゴリ 60 行 × 6 = 360 行を stratified サンプル（seed=42、`data/inputs/diagnostic_subset.csv`）。
- 推論 notebook: `notebooks/diagnose-per-category.ipynb`（既存 0.85 notebook の wheel/Unsloth セットアップを流用、SFTTrainer の代わりに PeftModel.from_pretrained で adapter ロード → maj@8 × 30 問/カテゴリ = 180 問 → per-category accuracy を保存）。
- Kaggle private dataset: `hiranorm/nemotron-diagnostic-subset`（subset CSV 113KB、push 済）。
- 診断対象 adapter: 公開 0.85 reference `dgxchen/trained-adapter`（child-exp001 と同系統で代表性十分・追加学習不要）。
- 残: Kaggle で notebook を作成 → 3 つの dataset（adapter / subset / mayukh18 wheel パッケージ）と utility script を attach → GPU runtime で実行。
- 期待アウトプット: `/kaggle/working/diagnostic_report.csv`（category / correct / total / accuracy）と `diagnostic_per_sample.csv`（投票内訳）。
