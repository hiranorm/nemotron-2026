"""
Build merged CoT dataset: Tong (problem_ids_matched) + konbu17 (train_split_with_cot)
Output: data/merged-cot-v1/merged_cot_v1.csv
"""
import pathlib, random
import csv

BASE = pathlib.Path(__file__).parent.parent / "data"
SRC_TONG    = BASE / "nemotron-cot-tong" / "problem_ids_matched.csv"
SRC_KONBU   = BASE / "nemotron-sft-lora-cot-selection" / "train_split_with_cot.csv"
OUT_DIR     = BASE / "merged-cot-v1"
OUT_CSV     = OUT_DIR / "merged_cot_v1.csv"

# Unified type labels (6 competition categories, snake_case)
TONG_MAP = {
    "bit_manipulation":        "bit_manipulation",
    "cipher":                  "cipher",
    "gravity":                 "gravity",
    "unit_conversion":         "unit_conversion",
    "numeral":                 "numeral",
    "equation_numeric_deduce": "equation",
    "equation_numeric_guess":  "equation",
    "cryptarithm_deduce":      "equation",
    "cryptarithm_guess":       "equation",
}

KONBU_MAP = {
    "Bit Manipulation":     "bit_manipulation",
    "Text Encryption":      "cipher",
    "Gravitational Constant": "gravity",
    "Unit Conversion":      "unit_conversion",
    "Numeral Conversion":   "numeral",
    "Equation Transformation": "equation",
}

SEED = 42
COLUMNS = ["id", "prompt", "answer", "type", "generated_cot", "source"]


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def normalize(rows, type_map, source_tag):
    out = []
    skipped = 0
    for row in rows:
        raw_type = row.get("type", "")
        mapped = type_map.get(raw_type)
        if mapped is None:
            skipped += 1
            continue
        cot = row.get("generated_cot", "")
        if not cot or cot == "nan" or len(cot.strip()) < 5:
            skipped += 1
            continue
        out.append({
            "id":            row["id"],
            "prompt":        row["prompt"],
            "answer":        row["answer"],
            "type":          mapped,
            "generated_cot": cot,
            "source":        source_tag,
        })
    return out, skipped


def main():
    print("Loading sources...")
    tong_rows  = read_csv(SRC_TONG)
    konbu_rows = read_csv(SRC_KONBU)
    print(f"  Tong  raw: {len(tong_rows)}")
    print(f"  konbu raw: {len(konbu_rows)}")

    tong_clean,  s1 = normalize(tong_rows,  TONG_MAP,  "tong")
    konbu_clean, s2 = normalize(konbu_rows, KONBU_MAP, "konbu17")
    print(f"  Tong  clean: {len(tong_clean)}  (skipped {s1})")
    print(f"  konbu clean: {len(konbu_clean)}  (skipped {s2})")

    merged = tong_clean + konbu_clean
    random.seed(SEED)
    random.shuffle(merged)
    print(f"\nMerged total: {len(merged)} rows")

    from collections import Counter
    type_counts = Counter(r["type"] for r in merged)
    src_counts  = Counter(r["source"] for r in merged)
    print("\nType breakdown:")
    for t, n in sorted(type_counts.items()):
        print(f"  {t}: {n}")
    print("\nSource breakdown:")
    for s, n in sorted(src_counts.items()):
        print(f"  {s}: {n}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(merged)

    size_mb = OUT_CSV.stat().st_size / 1024 / 1024
    print(f"\nSaved: {OUT_CSV}  ({size_mb:.1f} MB, {len(merged)} rows)")


if __name__ == "__main__":
    main()
