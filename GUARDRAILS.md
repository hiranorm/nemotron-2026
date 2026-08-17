# nvidia-nemotron-model-reasoning-challenge — ダメだったこと（LB を下げる/効かない/無効パターン）

新規実験の前に必ず読む。フォーマット: 問題 → 指標インパクト → 仮説 → 次回の指針。
重大度: Minor(±0.003) / Notable(±0.01〜0.02) / Major(±0.05+) / Invalid(提出無効)。

### [Invalid] LoRA rank > 32
- **問題**: rank=64（child-exp002/004/005）を試行。
- **インパクト**: **提出無効**（コンペ rules で rank ≤ 32、2026-05-27 確定）。加えて bs=2+r=64 は OOM、bs=1 は学習タイムアウト。
- **指針**: **rank は 32 固定**。受賞解法も 32。r 倍増には投資しない。

### [Major] lr=5e-5（低すぎる学習率）
- **問題**: lr sweep で 5e-5 を試行。
- **インパクト**: LB 0.65-0.66（2e-4 比 -0.19）。
- **指針**: **lr=2e-4 がスイートスポット**（seed 依存ほぼなし）。1e-4 で 0.82、3e-4 で 0.84。2e-4 を基準に。

### [Notable] micro batch 構成の変更（per_device_bs=2 / grad_accum=16）
- **問題**: micro batch サイズを変えると精度低下傾向（公開ノート著者の注記）。
- **指針**: effective batch=32 を維持しつつ micro batch は元ノート構成に従う。

### [Notable] データ量を増やしても品質が伴わないと無効
- **問題**: Tong CoT + konbu17 CoT を結合（14,388 rows、~1.8倍）したが LB 0.85 で**改善なし**（child-exp003）。
- **仮説**: konbu17 の CoT 品質が Tong に劣る。**データ量より品質が律速**。
- **指針**: 量ではなく**正解保証の高品質 CoT**（受賞解法のソルバ由来 CoT）を優先。→ child-exp007。

### [Notable] 受賞データ（ソルバ由来 CoT 9500）への単体差し替えは効かない
- **問題**: child-exp007（受賞データ + 受賞 lr schedule linear + rank32 + 1ep）で LB **0.84**（child-exp001=0.85 比 **-0.01**、現行 Tong CoT 路線比悪化）。
- **仮説**: (a) 9500件は Tong CoT 比で少量・量律速の側面、(b) 受賞解法は「データ＋レシピ＋他の細部」のセットで効くもので単独移植では再現しない、(c) 私的版 winner_cot_v1 の品質劣化。
- **指針**: 受賞データ単独 swap は推さない。続けるなら **Tong CoT に受賞データを mix（重みづけ）**、または受賞解法の他の細部（augmentation 等）と組み合わせて検証。深追い厳禁の方針上、~6/8 までに動かなければ確定終了。

### [Notable] lr schedule（cosine → linear-decay）単独変更は効かない
- **問題**: child-exp006（child-exp001 と同データのまま lr_scheduler を受賞流の linear に変更）で LB **0.85**（同等、改善なし）。
- **指針**: lr schedule を受賞流に寄せるだけでは現状打破にならない。schedule 単独変更には追加投資しない。

### [Infra] Kaggle のデータセット mount パス規約 + config silent fallback の罠
- **問題**: child-exp007 で `FileNotFoundError: /kaggle/input/nemotron-winner-cot-v1/winner_cot_v1.csv`。
- **原因**: この Kaggle 環境はデータセットを **`/kaggle/input/datasets/{owner}/{slug}/`** にマウントする（notebook の既定 `/kaggle/input/datasets/dgxchen/...` が証拠）。`/kaggle/input/{slug}/` は誤り。
- **二次的罠**: config 読み込みが `... if _cfg_path.exists() else {}` で、パス不一致時に**黙って空 config→デフォルトデータ**を使う（誤データで学習する事故）。
- **指針**: パスは `/kaggle/input/datasets/{owner}/{slug}/`。加えて notebook に **glob フォールバック**（`/kaggle/input/**/{basename}` を再帰探索）を入れ、config も同様に自動発見＋発見ログを出す（実装済み: train.ipynb / run-*.ipynb）。upload script の config パス生成も同規約に修正済み。

### [メモ] 制御できるのは adapter のみ
- maj@64 のデコード（サンプリング/多数決）はメトリクスノートブック側で固定 → **デコード調整は不可侵**。改善レバーは「学習する adapter（データ・LoRA 設定）」に限られる。
