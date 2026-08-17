---
id: TASK-003
title: 受賞者の公開データ(ソルバ由来CoT + augmentation)を採用
status: Done
assignee: []
created_date: '2026-05-27 06:39'
updated_date: '2026-05-30 11:45'
labels:
  - data
  - exp002
dependencies: []
priority: high
ordinal: 3000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tonghuikang/nemotron の reasoning/(ソルバ由来CoT)と augmentations/ を採用(ライセンス確認後)。正解保証CoTが0.85->0.87の本命差分。rank=32/1epoch/上記lrで学習・提出。
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 repo の LICENSE と再配布規約を確認
- [x] #2 公開データで学習用 corpus を構築
- [x] #3 学習・提出して LB を確認
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
完了。child-exp007 LB=0.84（child-exp001=0.85 比 -0.01 悪化）。受賞データ単独 swap は効かないと判明 → GUARDRAILS 転記済。RESULTS.md 更新済。
<!-- SECTION:NOTES:END -->
