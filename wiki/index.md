# Wiki — nvidia-nemotron-model-reasoning-challenge（終了後アーカイブ）

**このコンペは終了済み（2026-06-15 締切）。この wiki は「終了後に分かったこと」だけを足す場所。**
コンペ期間中の記録は旧レイアウトのまま残す（この wiki に転記しない）:
[../RESULTS.md](../RESULTS.md) / [../GUARDRAILS.md](../GUARDRAILS.md) / [../IDEAS.md](../IDEAS.md) / [../research/](../research/)。

## 現在地（2026-08-17）

- フェーズ: **終了後の上位解法の収集と、次コンペへの転用可能物の抽出。**
- 直近の作業: 1st〜18th の Solution Writeup を収集し [research/top-solutions](research/top-solutions.md) に集約。
  最終順位と自陣の位置を [research/final-leaderboard](research/final-leaderboard.md) に確定。
  上位に共通する手法を [method/learnable-trace](method/learnable-trace.md) / [method/memorize-vs-compute](method/memorize-vs-compute.md) に分解。
- 直近の完了: **[postmortem](postmortem.md) の転用可能物を [shared_knowledge](../../shared_knowledge/index.md) へ昇格した**
  （新規 `ops/proxy-metrics` と `modeling/teacher-signal-learnability`、既存 `decision/cv-vs-lb` へ追記）。
- 次のアクション:
  1. 未収集の writeup（8th・11th・12th・14〜17th 等）が出ていないか、半年後にもう一度だけ棚卸しする。
  2. 次に LLM の SFT / 蒸留を含むコンペに入るとき、`ops/proxy-metrics` の 4 層を**最初の週に**組む。
- ブロッカー: なし（実験は再開しない。**再開するなら別コンペで、この wiki は読み物として使う**）。

## 結果（確定）

| 項目 | 値 |
|---|---|
| 自陣 | **private 0.852 → 401位 / 4,182チーム = 🥉銅メダル**（public 0.856 / 1672位 → **+1271 の shake-up**） |
| 自陣の最終提出 | EXP002/child-exp001 系（2026-05-01 提出、以後放置。6/2 に撤退決定） |
| 優勝 | NullSira（public 0.912 / **private 0.920**） |
| 銀圏（209位）との差 | **+1問**（0.856 = private 177〜347位） / 金圏（16位）は 0.868 = +4問 |

詳細は [research/final-leaderboard](research/final-leaderboard.md)。

## ページ一覧

### research/ — 終了後に外から得た事実

| ページ | 内容 |
|---|---|
| [top-solutions](research/top-solutions.md) | 1st〜18th の解法要点（順位別）と、上位に共通する構造 |
| [final-leaderboard](research/final-leaderboard.md) | 最終 LB・メダル境界・自陣の位置・**public 0.86 の壁と private の乖離** |

### method/ — 上位解法の手法解説（次コンペで使う道具として）

| ページ | 内容 |
|---|---|
| [learnable-trace](method/learnable-trace.md) | **「解けるトレース」と「学習できるトレース」は別物**。可学習性の設計原則と計測方法 |
| [memorize-vs-compute](method/memorize-vs-compute.md) | 探索を「重みに記憶させる部分」と「トレース内で計算する部分」に割る設計 |

### 総括

- [postmortem](postmortem.md) — **自陣 0.85 と金圏 0.92 の間に何があったか。**撤退判断の事後評価と、持ち帰る規則。

## 参考リンク

- コンペ: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
- 1st 解法リポジトリ: https://github.com/xrwr/kaggle-nvidia-nemotron-model-reasoning-challenge-1st-place
- 2nd 解法リポジトリ: https://github.com/livctr/nvidia-nemotron
- 全解法の起点（Open Progress Prize, @huikang）: https://github.com/tonghuikang/nemotron
