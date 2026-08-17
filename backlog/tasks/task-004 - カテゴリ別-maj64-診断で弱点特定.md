---
id: TASK-004
title: カテゴリ別 maj@64 診断で弱点特定
status: Frozen
assignee: []
created_date: '2026-05-27 06:39'
updated_date: '2026-06-02 06:54'
labels:
  - research
dependencies: []
priority: medium
ordinal: 4000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
6カテゴリ(bit/cipher/equation/gravity/unit/numeral)別に現行アダプタの正答率を測り、弱カテゴリにデータ補強の的を絞る。
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ローカルでカテゴリ別正答率を算出
- [ ] #2 弱カテゴリを特定し data タスクに反映
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ローカル下ごしらえ完了 (2026-05-30):
- scripts/categorize_train.py で 9500 行を 6 カテゴリに分類: bit/gravity/unit/cipher/numeral/equation (各 ~1500-1600 行)
- 360 行 stratified subset (60/cat): data/inputs/diagnostic_subset.csv
- Kaggle private dataset push 済: hiranorm/nemotron-diagnostic-subset (113KB)
- notebooks/diagnose-per-category.ipynb 生成 (既存 0.85 notebook の wheel/Unsloth セットアップ流用、SFTTrainer→PeftModel.from_pretrained に置換、maj@8×30問/cat=180問の推論ループ追加)
- 診断対象 adapter: 公開 0.85 reference dgxchen/trained-adapter (代表性十分・追加学習不要)
- 残: Kaggle で notebook 作成 → adapter dataset + diagnostic-subset + mayukh18/nemotron-packages を attach → GPU runtime で実行 (~4-6h 見込み)。
- 出力: /kaggle/working/diagnostic_report.csv (per-category accuracy) と diagnostic_per_sample.csv (vote breakdown)。
<!-- SECTION:NOTES:END -->
