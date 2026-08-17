---
id: TASK-001
title: rank<=32 制約の確定と r=64 路線の破棄
status: Done
assignee: []
created_date: '2026-05-27 06:39'
updated_date: '2026-05-27 06:54'
labels:
  - research
dependencies: []
priority: high
ordinal: 1000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
受賞解法(tonghuikang)が lora_rank=32 固定。コンペ rules で rank<=32 を最終確認し、r=64(child-exp002 OOM / child-exp005 未提出)を無効として破棄する。
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 コンペ rules で LoRA rank 上限を確認
- [x] #2 r=64 路線を Frozen 化し EXP_SUMMARY に記録
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
コンペ rules で LoRA rank<=32 を確認(2026-05-27)。受賞解法も rank=32。r=64 路線(child-exp002/004/005)は提出無効=徒労として破棄、EXP_SUMMARY ガードレールに記録。
<!-- SECTION:NOTES:END -->
