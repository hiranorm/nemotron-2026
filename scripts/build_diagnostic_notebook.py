"""Build notebooks/diagnose-per-category.ipynb by adapting run-EXP002-child-exp001.ipynb.

The diagnostic notebook:
  1. reuses the wheel / Unsloth setup cells verbatim,
  2. skips training and instead loads a pretrained adapter from a Kaggle dataset,
  3. runs (greedy or maj@N) inference on data/inputs/diagnostic_subset.csv,
  4. saves per-category accuracy to /kaggle/working/diagnostic_report.csv.

Run from the competition root:
  uv run --with nbformat python scripts/build_diagnostic_notebook.py
"""
from __future__ import annotations

import copy
from pathlib import Path

import nbformat

SRC = "notebooks/run-EXP002-child-exp001.ipynb"
DST = "notebooks/diagnose-per-category.ipynb"


CFG_CELL = '''\
# ============================================================
# DIAGNOSTIC CONFIG — fill these in before running on Kaggle
# ============================================================
# Pretrained LoRA adapter (must be a Kaggle dataset you attach to this notebook).
# Path layout: /kaggle/input/datasets/{owner}/{slug}/ containing adapter_config.json + adapter_model.safetensors
PRETRAINED_ADAPTER_DATASET_PATH = "/kaggle/input/datasets/dgxchen/trained-adapter"  # TODO: replace with your best adapter

# Diagnostic subset CSV (3-column: prompt, answer, category) attached as a Kaggle dataset.
DIAGNOSTIC_SUBSET_PATH = "/kaggle/input/datasets/hiranorm/nemotron-diagnostic-subset/diagnostic_subset.csv"

BASE_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

# Inference cost knobs. Default = greedy (maj@1) to keep the run tractable on Kaggle GPU budgets.
# Bump MAJ_N to 8 (or higher) if you want a closer proxy of the maj@64 eval, but expect proportional time.
MAJ_N = 8              # 1 = greedy; >1 = sample MAJ_N completions and majority-vote the boxed answer
MAX_PROBLEMS_PER_CAT = 30  # subset-of-subset cap (the input subset has ~60/category)
MAX_NEW_TOKENS = 1536
TEMPERATURE = 0.7      # only used when MAJ_N > 1
TOP_P = 0.9

import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")

import random, numpy as np, torch
GLOBAL_SEED = 777
random.seed(GLOBAL_SEED); np.random.seed(GLOBAL_SEED); torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)

# We always run in inference mode; reuse upstream flag names so the shared setup cells branch correctly.
TRAIN_ON_KAGGLE = 1   # keep the wheel/Unsloth setup cells active
USE_PRETRAINED = 0    # we load the adapter manually in the diagnostic cells below

print({"adapter": PRETRAINED_ADAPTER_DATASET_PATH, "subset": DIAGNOSTIC_SUBSET_PATH,
       "MAJ_N": MAJ_N, "MAX_PROBLEMS_PER_CAT": MAX_PROBLEMS_PER_CAT})
'''


LOAD_ADAPTER_CELL = '''\
# Load the pretrained LoRA adapter onto the base model (replaces the SFTTrainer training step).
import os, glob, json
from peft import PeftModel

ADAPTER_DIR = PRETRAINED_ADAPTER_DATASET_PATH
if not os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json")):
    # fallback: search anywhere under /kaggle/input
    candidates = glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
    if not candidates:
        raise FileNotFoundError(f"adapter_config.json not found at {ADAPTER_DIR} nor under /kaggle/input")
    ADAPTER_DIR = os.path.dirname(candidates[0])
    print("Auto-discovered adapter dir:", ADAPTER_DIR)

print("Loading adapter from:", ADAPTER_DIR)
for fname in ["adapter_config.json", "adapter_model.safetensors"]:
    fp = os.path.join(ADAPTER_DIR, fname)
    print(f"  {fname}: {os.path.getsize(fp)/1024/1024:.1f} MB")

# `model` from the upstream setup cell is the bare base. Wrap it with the adapter.
model = PeftModel.from_pretrained(model, ADAPTER_DIR, is_trainable=False)
from unsloth import FastLanguageModel
FastLanguageModel.for_inference(model)
model.eval()
print("Adapter loaded and model switched to inference mode.")
'''


INFERENCE_CELL = '''\
# Per-category diagnostic inference.
import os, re, time, json, glob, pathlib
from collections import Counter, defaultdict

import pandas as pd
import torch

PROMPT_SUFFIX = "\\nPlease put your final answer inside `\\\\boxed{}`. For example: `\\\\boxed{your answer}`"
BOXED_RE = re.compile(r"\\\\boxed\\{([^{}]*)\\}")

subset_path = DIAGNOSTIC_SUBSET_PATH
if not os.path.exists(subset_path):
    cands = sorted(glob.glob("/kaggle/input/**/diagnostic_subset.csv", recursive=True))
    if not cands:
        raise FileNotFoundError(f"diagnostic_subset.csv not found at {subset_path} nor under /kaggle/input")
    subset_path = cands[0]
    print("Auto-discovered subset:", subset_path)

df = pd.read_csv(subset_path)
assert {"prompt", "answer", "category"}.issubset(df.columns), df.columns
print("Loaded subset:", len(df), "rows. Per-category:")
print(df["category"].value_counts().to_dict())

# Cap per category.
df = df.groupby("category", group_keys=False).apply(
    lambda g: g.head(MAX_PROBLEMS_PER_CAT)
).reset_index(drop=True)
print("After cap:", len(df), "rows")


def extract_boxed(text: str) -> str:
    matches = BOXED_RE.findall(text or "")
    return matches[-1].strip() if matches else ""


def render_prompt(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt + PROMPT_SUFFIX}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate_one(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    do_sample = MAJ_N > 1
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=do_sample,
            temperature=TEMPERATURE if do_sample else 1.0,
            top_p=TOP_P if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    new = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True)


def majority_predict(prompt: str) -> tuple[str, list[str]]:
    text = render_prompt(prompt)
    votes = []
    for _ in range(MAJ_N):
        gen = generate_one(text)
        votes.append(extract_boxed(gen))
    nonempty = [v for v in votes if v]
    if not nonempty:
        return "", votes
    pred, _ = Counter(nonempty).most_common(1)[0]
    return pred, votes


t0 = time.time()
per_cat_correct = defaultdict(int)
per_cat_total = defaultdict(int)
rows = []
for i, r in df.iterrows():
    cat = r["category"]
    gold = str(r["answer"]).strip()
    pred, votes = majority_predict(str(r["prompt"]))
    ok = (pred == gold)
    per_cat_total[cat] += 1
    per_cat_correct[cat] += int(ok)
    rows.append({"category": cat, "gold": gold, "pred": pred, "ok": int(ok), "votes": json.dumps(votes)})
    if (i + 1) % 10 == 0:
        elapsed = time.time() - t0
        print(f"[{i+1}/{len(df)}] cat={cat} ok={ok} elapsed={elapsed:.1f}s")

print("\\n=== Per-category accuracy ===")
report_rows = []
for cat in sorted(per_cat_total):
    tot = per_cat_total[cat]
    cor = per_cat_correct[cat]
    acc = cor / tot if tot else 0.0
    print(f"  {cat:10s}: {cor:3d}/{tot:3d} = {acc:.3f}")
    report_rows.append({"category": cat, "correct": cor, "total": tot, "accuracy": acc})

OUT_DIR = "/kaggle/working"
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "diagnostic_per_sample.csv"), index=False)
pd.DataFrame(report_rows).to_csv(os.path.join(OUT_DIR, "diagnostic_report.csv"), index=False)
print("\\nSaved:")
print("  /kaggle/working/diagnostic_per_sample.csv")
print("  /kaggle/working/diagnostic_report.csv")
print(f"Total elapsed: {time.time()-t0:.1f}s, MAJ_N={MAJ_N}, samples={len(df)}")
'''


def make_code_cell(src: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source=src)


def make_md_cell(src: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source=src)


def main() -> None:
    src_nb = nbformat.read(SRC, as_version=4)
    cells = src_nb.cells

    def find_idx_starting_with(prefix: str) -> int:
        for i, c in enumerate(cells):
            if c.cell_type == "code" and c.source.lstrip().startswith(prefix):
                return i
        raise ValueError(prefix)

    # Reuse the wheel/triton/mamba/unsloth/base-model/LoRA-wrapper setup cells (indices 5..10 in src).
    triton_idx = find_idx_starting_with("import os, glob, sys, subprocess, site\n\ncandidates = glob.glob")
    ptxas_idx = find_idx_starting_with("if TRAIN_ON_KAGGLE:\n    import sys, os, shutil, stat")
    pip_skip_idx = find_idx_starting_with("# trl installation is handled")
    pkg_idx = find_idx_starting_with("if TRAIN_ON_KAGGLE:\n    import glob\n    import os")
    base_load_idx = find_idx_starting_with("if TRAIN_ON_KAGGLE:\n    import torch\n    import kagglehub")

    setup_cells = [
        copy.deepcopy(cells[triton_idx]),
        copy.deepcopy(cells[ptxas_idx]),
        copy.deepcopy(cells[pip_skip_idx]),
        copy.deepcopy(cells[pkg_idx]),
        copy.deepcopy(cells[base_load_idx]),
    ]
    # The "Create LoRA peft wrapper" cell from src builds a *fresh* LoRA — we DON'T need it because
    # PeftModel.from_pretrained will attach the trained adapter directly to the base model.

    new_cells = [
        make_md_cell("# Per-category Diagnostic — Nemotron LoRA adapter\n\n"
                     "Loads a pretrained adapter and runs maj@N inference on a stratified subset of "
                     "`train.csv` to find weak categories. Output: `/kaggle/working/diagnostic_report.csv`."),
        make_code_cell(CFG_CELL),
        make_md_cell("## Environment setup (reused from run-EXP002-child-exp001.ipynb)"),
        *setup_cells,
        make_md_cell("## Load pretrained adapter"),
        make_code_cell(LOAD_ADAPTER_CELL),
        make_md_cell("## Inference and per-category accuracy"),
        make_code_cell(INFERENCE_CELL),
    ]

    out_nb = nbformat.v4.new_notebook(cells=new_cells, metadata=src_nb.metadata)
    Path(DST).parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(out_nb, DST)
    print(f"Wrote {DST} ({len(new_cells)} cells)")


if __name__ == "__main__":
    main()
