# 上位解法（終了後に公開された Solution Writeup）

> 関連: [index](../index.md) · [final-leaderboard](final-leaderboard.md) · [postmortem](../postmortem.md) · [method/learnable-trace](../method/learnable-trace.md) · [method/memorize-vs-compute](../method/memorize-vs-compute.md)

収集日 2026-08-17。出典はすべて Kaggle の Solution Writeup（`/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/{slug}`）。
**数値は各チームの自己申告**（オラクル解答率・トークン数・学習時間）。private LB は Kaggle 集計値。

## まず結論: 全チームが同じ骨格を使っている

上位 18 位までの解法は**例外なく同型**だった。

```
train.csv を解析して出題生成器を逆算
  → カテゴリごとに決定論ソルバ（Python）を書く
  → ソルバの実行過程を CoT トレースとして書き出す（答えは見ない）
  → 合成問題を大量生成して同じトレースを付ける
  → rank32 LoRA で 1 epoch SFT（＝トレースの模倣を学習させる）
```

差がついたのは**次の 3 点だけ**で、学習ハイパラではなかった。

1. **ソルバの仮説クラスが真の出題規則に一致しているか**（bit manipulation で 85% → 99% の差になった）
2. **トレースが「学習可能」か**（正しいトレースでも模倣できない形がある → [learnable-trace](../method/learnable-trace.md)）
3. **7,680 トークンの生成予算にどう収めるか**（HEX 圧縮・探索の分割 → [memorize-vs-compute](../method/memorize-vs-compute.md)）

> **自陣（LB 0.85）はこの 3 点をどれも触っていない。**公開 0.85 ノートのレシピ側（lr schedule / epochs / データ差し替え）だけを動かしていた。→ [postmortem](../postmortem.md)

### カテゴリ別の難易度（ソルバのオラクル解答率、複数チームの申告値）

| カテゴリ | 上位のオラクル解答率 | 一言 |
|---|---|---|
| gravity / unit conversion / numeral / cipher | 100%（cipher は 99.5〜100%） | 公開 baseline のまま。**ここは差がつかない** |
| equation numeric (deduce) | 93〜96% | 24 種の演算インベントリ＋「同一問題内で演算族は重複しない」制約 |
| equation numeric (guess) | 39〜57% | 未観測演算子の推定。族の排他性＋頻度事前分布で稼ぐ |
| **bit manipulation** | 85%（公開 baseline）→ **98.8〜99.4%** | **最大の得点源。ここを解いたチームが上位** |
| **cryptarithm (symbol)** | 8%（公開 baseline）→ 最大 **42.9%**（1st） | **金圏と銀圏を分けた領域。**触っていないチームも多い |

## 1st — NullSira（public 0.912 / **private 0.920**、best unselected 0.932）

`writeups/1st-place-solution` / コード: https://github.com/xrwr/kaggle-nvidia-nemotron-model-reasoning-challenge-1st-place

**核心は「何を重みに記憶させ、何をトレース内で計算するか」の切り分け。**

- **cryptarithm を signature catalog で攻略（唯一の 40% 超）。** 1 式を `AB op CD = 出力` の記号反復パターン（signature）に正規化し、
  100×100 の被演算子 × 22 演算を全列挙して **4,205 個の signature → 候補数・候補桁列** の表を事前計算。
  この表を**SFT で暗記させ**、推論時は「表から引いた候補」から DFS を始めて残り式との整合だけ検証する。
  素の探索空間 10! × 24³ ≒ 5e10 をトレースに書き下すのは不可能なので、**探索を「記憶」と「検証」に割った**のが勝ち筋。
- **bit manipulation は @huikang のソルバを 3 点改良。** (a) 4 文字以上のビット列を HEX 化（生成トークン中央値 6,771 → 4,888、**-27.8%**）、
  (b) majority / 3入力パリティ / 条件選択 / 3入力合成を規則に追加、(c) **repair**: 選ばれた 8 規則列を、
  事前計算した 5,238 個の妥当な規則列カタログの中からハミング距離が近いものへ射影し直す。
- repair の着想は**開発中の事故**から出た。「トレースはソルバ出力、最終 `\boxed{}` だけ正解に差し替え」という壊れたサンプルを誤って作ったところ、
  モデルが「ソルバが間違えたときに最後だけ直す」形式を学習した。
- equation numeric: **24 規則インベントリ**を確定。観測 2 つが強力 —
  (1) 同一プロンプト内の非 join 演算子はすべて同じ mode（normal / flip）、(2) 非 join 演算子は同じ族を再利用しない。
- 学習: r=32 / α=32 / lr 2e-4（cosine）/ 1 epoch / eff. batch 16 / target に `lm_head` を含む / Unsloth /
  **RTX PRO 6000 Blackwell 1枚で main 約 119h + 追加 51h**、主学習 22 万サンプル・8.9 億トークン。
- ソルバ 93.16% に対しモデル 92.00%（＝**トレースが正しければモデルはほぼ再現できる**）。
- 「コードは 1 行も自分で書いていない（Codex）。ただしスコアを上げた発想・分析・トレース設計は Codex からは出てこなかった。」

## 2nd — vli（public 0.884 / **private 0.908**）

`writeups/2nd-place-solution` / コード: https://github.com/livctr/nvidia-nemotron

- **bit manipulation を出題生成器の文法として逆算し 99.06%（LoRA 出力 97.32%）。**
  規則を「no-op / 1-op / 2-op(3形) / Majority / Choice」の文法に整理し、
  各 shift 量が 1..7 であることから **bit 0 と bit 7 では必ず片方の shift が死ぬ**＝端の領域は 2 変数関数に限られると証明。
  端（corner）の関数対を列挙して中央領域の形を決める、という探索順にした。
- **cryptarithm は手を付けていない（9.6%）。** それでも private 0.908。
- **複雑な学習スキームはすべて負けた。** focal loss / トークン損失の再重み付け / 多段学習（データ mix と lr を段で変える）は
  いずれも素の cross-entropy LoRA SFT に勝てなかった。
- **インフラの罠が最大の落とし穴だった。** Tinker → Megatron-Bridge に替えただけで bit manipulation 0.81 → 0.89。
  理由は **Tinker の LoRA を PEFT 形式に変換する際に SVD が入り lossy** だから（Megatron-Bridge は厳密対応）。
- `output_layer`（lm_head）の学習が**極めて重要**。特に語彙の裾にあるトークン（非 ASCII の区切り `│` など）を使う場合。
- **80M トークン以降は収穫逓減。**「80M トークンの LoRA でも 2 位は取れたと思う。トレースの設計のほうが重要だった。」
- 学習: 57,600 例 / 1.59 億トークン / 4×A100 / 11h。EP=4 で 128 experts を 4 分割した結果、
  **LoRA が 32 experts 分しか学習されずに tile された**（容量 4 分の 1）ことに後から気付いた。それでも影響は小さかった。

## 3rd — YS-L（**private 0.900**）

`writeups/3rd-place-solution`

- **二段学習（暗記 → 実行）が効いた。**
  第 1 段: `<RECALL_DOMAINS>` / `<RECALL_ROWS>` タグ付きの**ドリルだけ**で学習し、
  signature → 可能な桁集合・候補行を暗記させる（1 サンプルに 16 リクエストを packing、44,136 例 / 1.59 億トークン / 25.2h）。
  第 2 段: その LoRA を初期値として実タスクのトレースで学習（72,377 例 / 2.99 億トークン / 32.7h）。
  - 12K cryptarithm での比較: **直接学習 0/11 solved、二段学習 7/11 solved。**
  - 終了後の ablation（同一データ mix、40K+ cryptarithm）: **直接 12/17 → private 0.888、二段 16/17 → private 0.900。**
  - 副産物として「ソルバやトレース形式を変えても第 1 段は再利用できる」というモジュール性が得られる。
- cryptarithm は乗算パターン表 → 制約伝播 → バックトラッキング。**予算内 24.2%**（deduce 27.8% / guess 9.8%）。
- bit manipulation は HEX 化で約 25% トークン削減し、その余裕で MAJ 探索を追加。
  規則順序を `OR-NOT` を素の `OR`/`XOR` より先に試す並びへ変えて +0.6%。
- **検証は 10% ホールドアウト（954 行、カテゴリ層化、augment 由来のリークも除去）。これが private をよく追随した。**
  「**public LB には 0.86 の壁がはっきり見える。自分の public 0.85 の提出のうち 1 本は private 0.90 だった。**」
- 追加指標として **perfect trace rate**（トレースが期待どおり完全一致か）を追い、bit >85% / cipher・numeric >95% を維持。
  「たまたま答えが合っているがトレースを間違えている」を先に検知するための先行指標。
- eff. batch は大きいと学習不足になる。16 → 8 → **4** に落として決着。MoE tie weights を切ると良化。

## 4th — Dipam Chakraborty（**private 0.896**）

`writeups/4th-place-solution`

- **bit manipulation 99.4%（1593/1602）。** 1op/2op で解けない問題は **5 テンプレート**に収まると発見
  （Majority / Choice / M5 / C3 / C4、いずれも SHLa・SHRb・ROTc の 3 葉の合成）。
  ただし 7,500 トークンでは総当たりできないので、**bit 7（左シフトが死ぬ）と bit 0（右シフトが死ぬ）の 2 本のアンカー表**から
  シフト量を「数えずに読み取る」トレースにした。
- **「可学習トレース」の原則を明文化した唯一のチーム**（→ [learnable-trace](../method/learnable-trace.md) にまとめた）。
  隠れ計算の禁止 / 局所性 / 因果整合 / 参照整合性 / 規則は少なく一様に。
  **原則が機械的なので、コードでテストできる**（参照整合性監査・局所性監査を全トレースに対して回した）。
- 計測は **first token divergence**（学習時に greedy で次トークンが正解と一致するかを見る）。
  「一致率 80% ≒ 生成一致 95%」（間違いの多くは結果を変えない）。
- 生成結果と正解トレースを並べて差分を色分けする web UI を作り、**その出力を Claude に直接見せて形式の修正案を出させた**
  （train → diff → classify → reformat → retrain を約 1 時間で 1 周）。
- **総額約 $120**（Colab Pro G4 + Kaggle 週次コンピュート）。学習データは原問題 9,500 + 合成 15,000（bit 偏重）。
  MoE experts untied、batch 16 / micro 4（小さいほうが速く収束）。
- 「bit の部分解（93%）だけでも金は取れた」。cryptarithm は時間切れで未着手。

## 5th — Domdolus Tolus（**private 0.896**）「Jack of Trades, Loser of Cryptarithm」

`writeups/domdolus-tolus-solution`

- **train.csv を一切学習に使わず、無限合成生成器＋train.csv を検証専用にした。**
  「@huikang 方式（train で学習し train で検証）は CV が高くても LB が不安定になる」という仮説に基づく判断。
  結果、**検証 90.4% に対し private 90.0%** と素直に対応した。
- bit manipulation はバイト単位変換を 4 段（unary / pairwise / 簡約 ternary / 真の ternary）に整理し、
  **8 ビット分の「継続（continuation）」約 140 種・固有列約 7,500 をモデルに暗記させて一手で構成させる**
  （@huikang の「1 ビットずつ stride を繋ぐ」の逆）。
- **LoRA soup（model soup）を段階的に適用。** 学習前後の 2 モデルを平均していく phase 2、
  同一チェックポイントから別データで 3 本学習して平均する phase 3。
  「**スコアの安定性が明確に上がった**（public の振れが 1% 未満に収まった）」。ただし
  「CoT 改良と soup を同時に入れたので、どちらの寄与かは切り分けられていない」と自己申告。
- 失敗の記録が有用: pseudo-GRPO（DPO 代替）は 5 反復でも簡単な bit を学べず。
  **Unsloth + 4bit 量子化 + packing は Nemotron を壊す**（10 万例回しても学習しない）→ @huikang の素の学習ループで解決。
  **DoRA は vLLM が非対応で提出エラー**（LoRA への変換は lossy なので断念）。
- 短い CoT（結論だけ）は 25 万例でも失敗。「**論理の足場がない状態で LLM は推測できない。**」
- 最終提出は public 最良を選び、**CV 最良（private 0.900）は選べなかった**。public 88.8 / private 89.6 の安定モデルを選択。

## 6th — Alehandreus & Yurnero（**public 0.908（2位）/ private 0.896**）

`writeups/2nd-public-6th-private-place-solution`

- **ドメイン別 LB プロービングという珍しい測り方。** 対象ドメインだけ本物の CoT を学習させ、
  **他の全ドメインは `I will ignore this problem\n\boxed{DUMMY}` を出すように学習**させて提出。
  これで public LB から**特定ドメインだけの純粋な信号**を取り出した。
  併せて「test は 500 問（public/private 250/250）＝ LB 0.01 は約 3 問」と推定し、提出スコアを -0/-1/-2 表記で管理。
- 検証は 10% ホールドアウトでは足りない（学習サンプルすら覚えきれていない）と判断し、
  **合成データで学習して train.csv 全件を検証に回した**（検証集合を 10 倍に）。
- 「**単一ドメインの合成データ追加は、そのドメインだけ改善し他はほぼ壊さない。ただし bit manipulation だけは例外**で、
  他ドメインの構成比が数 % 動くだけで劣化する」→ 最終データの 66〜69% が bit のトークン。
- **最終選択のミスを本人が明記。** 0.5B トークン版（cryptarithm を 150 → 241 問に伸ばした）は
  bit の CV 低下が想定より大きく、0.25B 版の 2 seed を選んだ。「これが 2〜3 位を失った原因」。

## 7th — Prateek 他（**private 0.888**）

`writeups/7th-place-solution-solving-bitmanipulation-via-st`

- **bit manipulation 単騎で >96%（このカテゴリで参加チーム最高と自称）。** 算術的な論理演算の推論を捨て、
  出力間の**文字列類似（最小ビット反転）**で primitive（基底）を切り出し、真理値表を復元する定式化。
- **単一ビットのトークン化を強制**（`10100011` が `[1010][00][11]` に割れるのを防ぐ）＋**動的損失マスク**で
  「仮説 → 自己評価 → バックトラック」をモデルにネイティブに学習させた。arXiv preprint あり。
- 残り 10 日で参加したため **symbolic equation（cryptarithm）は 1 問も学習していない**。それで 7 位。

## 9th — Fate（**private 0.884**）

`writeups/9th-place-solution-human-auditable-cot`

- **合成問題を 1 問も増やさず、CoT ターゲットの書き直しだけで単独金。**
  検証済み CoT ターゲット数: bit 1364 → **1447** / equation 561 → **633** / cryptarithm 65 → 73。
- 「**人間が監査できるトレース**」を基準にした。読めるので誤りの原因（ターゲットが悪いのか、模倣が失敗したのか）を切り分けられる。
  実際に「ソースの CoT は正しいのにモデルの予測が誤り」の件数を別集計している（bit 8 件など）。
- gravity / unit conversion の不安定は**タスクの論理ではなく数値の書式**だった。
  10 進分解でゼロ位が消える（`3.592*0.0700` は出るが `0.0000` は出ない）と、数値によってトレースの見た目が変わる。
  ゼロ位を明示的に残し、**上位桁から先に展開する**順に変更して解消。
- 上振れしたのは upsampling の設計: bit を 5x、cipher と equation を 2x。numeral は 400 例だけで 100%。
- 「**CoT ターゲットの品質がすべて。**形式が一貫して読めて模倣しやすければ、合成データを増やさなくても LoRA SFT は十分学ぶ。」

## 10th — Rick / neilus / eikichi（public 0.860 / **private 0.880**）

`writeups/10th-place-solution`

- **bit manipulation のソルバを「ビットごとの論理演算探索」から「rotate/shift + キー表」に作り替えて 85.14% → 98.75%。**
  真の規則が `out[i] = F(rotl(x,a)[i], shl(x,b)[i], shr(x,c)[i])`（F は全ビット共通の 3 入力ブール関数）だと見抜き、
  **F に名前を付けず、例から 3 ビットキー → 出力ビットの表として読み取る。**
  (a,b,c) は境界ビット（bit 0 では shr=0、bit 7 では shl=0）から復元する。
- 残る失敗 20/1602 はすべて**仮説クラス内**。うち 11 件は原理的に不定（query が例に現れない 3 ビットキーを要求する）。
- 「ソルバのオラクル精度が上限で、モデルはそこにほぼ張り付く」を前提に、**モデル精度とオラクル精度の差**を監視して
  「ソルバが悪い」と「模倣が失敗している」を分離した。低 logprob 問題を priority sample として 2x 複製。
- `lora_alpha / r = 4`（α=128, r=32）にすると決定論 CoT への追従が強まり CV 改善。
- **効かなかったもの（貴重）**: カリキュラム学習（変化なし）/ 二段学習（勝ち負けの再配分で純増しない）/
  priority 3x（過学習）/ **失敗トレースの除外（精度が下がった＝失敗パターンにも信号がある）**/
  **LoRA merge（SVD merge も seed soup も単体最良に勝てない）**。
- **public 0.872 / private 0.852 の提出より、public 0.860 / private 0.880 の提出のほうが CV が高かった。CV で選んで正解だった。**

## 13th — Haraguchi-T（**private 0.872**）

`writeups/13th-place-solution`

- 教師 CoT の解答率を 87.7% → **89.4%** に上げるヒューリスティクス群（equation の反転規則を多数決で全体決定、
  演算族の排他性、guess の族優先順位 `sub > add > mul > concat` など）。
- **3 段 SFT（lr を 2e-4 → 2e-5 → 5e-6 と落としていく）。** 最終段で bit と cryptarithm を upsample して残存誤りを潰す。
- 「教師が解けた/解けなかったタスクを 1:1 に再バランス」＋ equation_numeric を種に cryptarithm を約 3 倍に増量。

## 18th — MOONMOON 他（**private 0.868**、未選択の expv006 が private 0.880）

`writeups/18th-solution-from-deterministic-solvers-to-learn`

**最大の教訓を自分から書いているので、自陣にとっていちばん近い記録。**

- 「**最大の失敗はローカル検証集合を早く作らなかったこと。これが最後の 2 か月で実質的な進展がなかった根本原因。**」
  - public 0.87 に達した後、探索が発散し、どの方向も深追いされなかった。
  - **合成データ方向（expv003）は public が下がったので 1 版で打ち切ったが、private では意味のある改善だった。**
  - private 最良の提出を選べなかった（expv006: public 0.864 / private 0.880 は未選択）。
- 実験ごとの public / private が並んでいる表が有用: **public と private の順序が入れ替わる例が多数**。
  | 実験 | public | private |
  |---|---|---|
  | expv001（安定 baseline） | 0.872 | 0.864 |
  | expv003（合成データ） | 0.868 | 0.872 |
  | expv006（保守的な GT 補助） | 0.864 | **0.880** |
  | expv001_002（LoRA fusion、選択） | 0.876 | 0.868 |
- **LoRA fusion**: 2 つの rank32 adapter を α/r で真のスケールに戻して線形結合し、truncated SVD で rank 制約内に射影。
  TIES / DARE より自作 SVD fusion が良かった（ただし 10th は同じ手法で単体に勝てていない）。
- 「CoT トレースは**ソルバが動いた記録ではなく、学習可能な軌跡**であるべき」と明記。

## 関連する終了後の議論

- **「なぜ『より良い』データセットはスコアを下げたのか」**（`discussion/697491`、TAHA, 619位）
  合成データの解答率を 87.7% → 95.8% に上げたのに LB は 0.82〜0.84 に落ちた、という報告。
  診断: (a) **ソルバの正しさ ≠ トレースの可学習性**（base model の logprob で難しさを測ると cryptarithm_deduce が突出）、
  (b) 難カテゴリを 14x oversample した結果、基礎カテゴリの知識を上書き（LB 0.73）。
  コメント欄で 15 位の MAJ0RT0M が「guess カテゴリで deduce と同等の精度が出るのは原理的にありえない、
  正解を混入していないか」と指摘し、投稿者が自分のソルバの誤りを認めている。
  **→ 自陣の child-exp007（受賞データ単体差し替えで -0.01）と同じ罠の可能性。**[postmortem](../postmortem.md) で扱う。
