---
id: TASK-002
title: 学習レシピを受賞解法に寄せる (lr schedule + target modules)
status: Done
assignee: []
created_date: '2026-05-27 06:39'
updated_date: '2026-05-30 11:45'
labels:
  - exp002
dependencies: []
priority: high
ordinal: 2000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
rank=32 維持のまま、lr schedule を cosine -> step-linear-decay(2e-4 -> 0)、target を MLP+attn+unembed に揃える。1 epoch。既存 child-exp001(0.85) からの差分1軸ずつ。
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 step-linear-decay lr で 1 本学習・提出
- [x] #2 target modules を MLP+attn+unembed に揃えて比較
- [x] #3 LB を child-exp001(0.85) と比較し RESULTS/EXP_SUMMARY 更新
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
完了。child-exp006 LB=0.85（child-exp001 と同等、改善なし）。lr schedule linear 単独変更は効かないと判明 → GUARDRAILS 転記済。RESULTS.md 更新済。
<!-- SECTION:NOTES:END -->
