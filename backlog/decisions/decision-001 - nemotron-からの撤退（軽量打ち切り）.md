---
id: decision-001
title: nemotron からの撤退（軽量打ち切り）
date: '2026-06-02'
status: accepted
---
## Context

- Best LB = 0.85（EXP002/child-exp001）で頭打ち。本命 2 本がともに不発：
  - child-exp006（lr linear-decay 単独）→ 0.85 横ばい
  - child-exp007（受賞ソルバ CoT 9500 単独 swap）→ 0.84（-0.01 悪化）
- 制御可能なのは LoRA アダプターのみ（maj@64 デコードは不可侵）、GPU 依存で試行レバレッジが低い。
- public と金（0.87）が密集しモデル選定不可。
- Kaggle は単一フォーカス運用で、主戦場は ROGII。残時間（締切 2026-06-15）と実装コストが見合わない。
- 撤退判断時点で TASK-004（per-category 診断）は Kaggle 実行待ちのまま。

## Decision

2026-06-02 に nemotron から撤退（軽量打ち切り）。これ以上の新規実験・提出は行わない。
TASK-004（診断）も実行せず Frozen。最終提出は既存ベスト child-exp001（LB 0.85）のまま放置。

## Consequences

- リソースを単一主戦場 ROGII に集中。
- 診断は未実施のまま終了（弱カテゴリ特定の信号は得られず）。準備済みリソース
  （diagnostic notebook / dataset push 済）は再開時に流用可能だが再開予定なし。
- GUARDRAILS（lr 単独変更不可・受賞データ単独 swap 不可・konbu17 無効・rank>32 違反）は
  今後 LLM 推論系コンペで参照する知見として保持。
