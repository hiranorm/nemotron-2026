"""受賞解法 (tonghuikang/nemotron, Progress Prize) の公開データを当方の SFT スキーマに変換する。

出力: data/winner-cot-v1/winner_cot_v1.csv
カラム: id, prompt, answer, type, generated_cot, source
  （merged_cot_v1.csv と同一スキーマ。train.ipynb はこのまま読める。
    train 側で \\boxed{} は除去→answer で再付与・<think> は chat template が付与する。）

データ源（GitHub https://github.com/tonghuikang/nemotron, master）:
  - train.csv            : id, prompt, answer（コンペ問題＝当方も保有）
  - problems.jsonl       : id, category, status(rule_found 等)
  - reasoning/<id>.txt   : ソルバ生成の決定論的 CoT（rule_found のみ存在）
  - augmentations/<id>.txt: 書式ロバスト性データ（[category]/[prompt]/[completion] 形式）

ライセンス注意（重要）:
  - 当該 repo に LICENSE が無い。prompt/answer はコンペ問題由来（当方も正規保有）、
    CoT は公開コードによる生成物。**個人の学習用途**に限定し、Kaggle へ上げる場合は
    **private Dataset** にすること（公開再配布はしない）。
  - より安全な代替: repo の reasoners/ を当方の problems に対して自前で実行し CoT を再生成する
    （本スクリプトは公開済み生成物を流用する簡便版）。

使い方:
  # repo を取得（未取得なら）
  git clone https://github.com/tonghuikang/nemotron /tmp/thk_nemotron
  # 変換
  uv run python scripts/build_winner_cot.py --repo /tmp/thk_nemotron
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

OUT_REL = "data/winner-cot-v1/winner_cot_v1.csv"
COLUMNS = ["id", "prompt", "answer", "type", "generated_cot", "source"]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_augmentation(text: str) -> dict[str, str] | None:
    """[category]\\n..\\n[prompt]\\n..\\n[completion]\\n.. を dict に分解。"""
    parts: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.strip() in ("[category]", "[prompt]", "[completion]"):
            if current is not None:
                parts[current] = "\n".join(buf).strip()
            current = line.strip().strip("[]")
            buf = []
        else:
            buf.append(line)
    if current is not None:
        parts[current] = "\n".join(buf).strip()
    if "prompt" in parts and "completion" in parts:
        return parts
    return None


def build(repo: Path, include_augmentation: bool = False) -> list[dict]:
    train_csv = repo / "train.csv"
    problems_jsonl = repo / "problems.jsonl"
    reasoning_dir = repo / "reasoning"
    aug_dir = repo / "augmentations"

    for p in (train_csv, problems_jsonl, reasoning_dir):
        if not p.exists():
            sys.exit(f"見つからない: {p} （--repo のパス/clone を確認）")

    # id -> category
    cat_by_id = {r["id"]: r.get("category", "") for r in load_jsonl(problems_jsonl)}

    # id -> (prompt, answer)
    pa_by_id: dict[str, tuple[str, str]] = {}
    with open(train_csv, newline="") as f:
        for r in csv.DictReader(f):
            pa_by_id[r["id"]] = (r["prompt"], r["answer"])

    rows: list[dict] = []

    # 1) ソルバ生成 CoT（reasoning/<id>.txt があるものだけ）
    n_reason = 0
    for txt in sorted(reasoning_dir.glob("*.txt")):
        pid = txt.stem
        if pid not in pa_by_id:
            continue
        prompt, answer = pa_by_id[pid]
        cot = txt.read_text().strip()
        if len(cot) < 5:
            continue
        rows.append({
            "id": pid,
            "prompt": prompt,
            "answer": answer,
            "type": cat_by_id.get(pid, ""),
            "generated_cot": cot,
            "source": "thk_reasoning",
        })
        n_reason += 1

    # 2) augmentation データ（書式ロバスト性）
    # 注意: aug 行は answer を分離できない（completion が自己完結）。
    #   当方 train.ipynb は (cot, answer) 前提で \\boxed{answer} を再付与するため、
    #   既定では除外する。使うなら train 側で source=='thk_augmentation' を
    #   「completion をそのまま使い boxed 再付与しない」分岐にすること。
    n_aug = 0
    if include_augmentation and aug_dir.exists():
        for txt in sorted(aug_dir.glob("*.txt")):
            parsed = parse_augmentation(txt.read_text())
            if not parsed:
                continue
            rows.append({
                "id": f"aug_{txt.stem}",
                "prompt": parsed["prompt"],
                "answer": "",  # augmentation は completion 内に解が含まれる想定
                "type": parsed.get("category", "augmentation"),
                "generated_cot": parsed["completion"],
                "source": "thk_augmentation",
            })
            n_aug += 1

    print(f"reasoning rows: {n_reason} / augmentation rows: {n_aug} / total: {len(rows)}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="tonghuikang/nemotron の clone パス")
    ap.add_argument("--out", default=None, help=f"出力 CSV（既定: {OUT_REL}）")
    ap.add_argument("--include-augmentation", action="store_true",
                    help="augmentation 行も含める（train 側の整形分岐が必要、既定は除外）")
    args = ap.parse_args()

    repo = Path(args.repo).expanduser()
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = build(repo, include_augmentation=args.include_augmentation)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
