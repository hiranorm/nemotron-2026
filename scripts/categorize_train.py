"""Categorize train.csv into 6 task categories by prompt prefix.

Outputs:
  - data/inputs/train_categorized.csv : full 9500 rows + `category` column
  - data/inputs/diagnostic_subset.csv  : stratified sample (default 60/category = 360)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CATEGORY_PREFIXES = {
    "bit":      "a secret bit manipulation rule",
    "gravity":  "the gravitational constant has been secretly changed",
    "unit":     "a secret unit conversion is applied to measurements",
    "cipher":   "secret encryption rules are used on text",
    "numeral":  "numbers are secretly converted into a different numeral",
    "equation": "a secret set of transformation rules is applied to equations",
}


def categorize(prompt: str) -> str:
    for cat, marker in CATEGORY_PREFIXES.items():
        if marker in prompt:
            return cat
    return "unknown"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/inputs/train.csv")
    p.add_argument("--out-full", default="data/inputs/train_categorized.csv")
    p.add_argument("--out-subset", default="data/inputs/diagnostic_subset.csv")
    p.add_argument("--per-cat", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.train)
    df["category"] = df["prompt"].apply(categorize)

    counts = df["category"].value_counts()
    print("Category counts:")
    for cat, n in counts.items():
        print(f"  {cat:10s}: {n}")
    if (df["category"] == "unknown").any():
        n_unk = int((df["category"] == "unknown").sum())
        raise RuntimeError(f"{n_unk} prompts unmatched — update CATEGORY_PREFIXES")

    Path(args.out_full).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_full, index=False)
    print(f"\nWrote {args.out_full}  ({len(df)} rows)")

    subset = (
        df.groupby("category", group_keys=False)
          .apply(lambda g: g.sample(n=min(args.per_cat, len(g)), random_state=args.seed))
          .reset_index(drop=True)
    )
    subset.to_csv(args.out_subset, index=False)
    print(f"Wrote {args.out_subset} ({len(subset)} rows, ~{args.per_cat}/category)")


if __name__ == "__main__":
    main()
