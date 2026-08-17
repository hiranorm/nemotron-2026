# nvidia-nemotron-model-reasoning-challenge — 実験アイデア

着手前の brainstorm。着手段階になったら `backlog task create` で task 化（着手中は TASK-002/003 が実行中）。
**位置づけ: 軽量継続・深追い厳禁（~6/8 stop）。** 投資はここまでで、ここから先は ROI を見て絞る。

## アイデア

- [ ] **augmentation 行の統合** — 受賞解法の `augmentations/`（spelling/concat/split/match/lstrip, 8463件）で書式ロバスト性を付与。要 train 側整形分岐（answer 分離不可、`source=='thk_augmentation'` は completion をそのまま使い boxed 再付与しない）。`build_winner_cot.py --include-augmentation`。
- [ ] **カテゴリ別 maj@64 診断**（TASK-004）— 6カテゴリ別の正答率で弱点特定 → データ補強の的を絞る。
- [ ] **ソルバ由来 CoT の自前再生成** — ライセンスを避けるなら受賞 `reasoners/` を当方の problems に対し実行して CoT を再生成（公開データ流用の代替）。
- [ ] **受賞レシピの完全再現**（c007 が効いた場合）— max_len/データ件数/loss masking など細部を寄せる。loss_config の advantages（加重損失）は未深掘り。
- [ ] **seed アンサンブルは不可**（提出は単一 adapter）→ データ・レシピ側で勝負する。

## 判断メモ
- 0.87 に public 多数密集（金との差が小さい）。c006/c007 で 0.85 から動かなければ ~6/8 で確定終了。
