"""
学習済み LoRA アダプターを Kaggle Dataset としてアップロードするスクリプト。

Usage:
    # 初回（新規データセット作成）:
    python scripts/upload_adapter.py \\
        --adapter_dir data/outputs/2026-04-21_EXP000_child-exp000/adapter \\
        --exp_no EXP000 --child_no child-exp000

    # 更新（既存データセットに新バージョン追加）:
    python scripts/upload_adapter.py \\
        --adapter_dir data/outputs/2026-04-21_EXP000_child-exp000/adapter \\
        --exp_no EXP000 --child_no child-exp000 \\
        --update

必要:
    ~/.kaggle/kaggle.json が設定済みであること
    kaggle CLI が利用可能: ~/.local/bin/kaggle
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path


KAGGLE_CLI = os.path.expanduser("~/.local/bin/kaggle")
KAGGLE_JSON = os.path.expanduser("~/.kaggle/kaggle.json")


def get_username() -> str:
    with open(KAGGLE_JSON) as f:
        return json.load(f)["username"]


def make_dataset_slug(exp_no: str, child_no: str) -> str:
    """Kaggle Dataset のスラッグを生成（小文字英数字とハイフンのみ）。"""
    tag = f"nemotron-{exp_no}-{child_no}".lower().replace("_", "-")
    return tag  # e.g. "nemotron-exp000-child-exp000"


def build_dataset_metadata(username: str, slug: str, exp_no: str, child_no: str) -> dict:
    return {
        "title": f"Nemotron LoRA Adapter - {exp_no}/{child_no}",
        "id": f"{username}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    return result


def upload(adapter_dir: Path, exp_no: str, child_no: str, update: bool) -> None:
    # 必要ファイルの確認
    required = ["adapter_config.json", "adapter_model.safetensors"]
    for fname in required:
        p = adapter_dir / fname
        if not p.exists():
            # bin 形式でも可
            bin_p = adapter_dir / fname.replace(".safetensors", ".bin")
            if fname.endswith(".safetensors") and bin_p.exists():
                print(f"Note: Using .bin format: {bin_p}")
                continue
            sys.exit(f"ERROR: Required file not found: {p}")
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {fname}: {size_mb:.1f} MB")

    username = get_username()
    slug = make_dataset_slug(exp_no, child_no)
    dataset_id = f"{username}/{slug}"
    print(f"\nDataset: {dataset_id}")

    # 一時ディレクトリにアダプターファイルをコピー
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # アダプターファイルをコピー
        for fname in required:
            src = adapter_dir / fname
            if not src.exists():
                # bin fallback
                src = adapter_dir / fname.replace(".safetensors", ".bin")
            if src.exists():
                shutil.copy2(src, tmp_path / fname)
                print(f"Copied: {src.name}")

        # dataset-metadata.json を作成
        metadata = build_dataset_metadata(username, slug, exp_no, child_no)
        meta_path = tmp_path / "dataset-metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Created: dataset-metadata.json")

        # アップロード
        if update:
            print(f"\nUpdating dataset (new version): {dataset_id}")
            run(f"{KAGGLE_CLI} datasets version -p {tmpdir} -m '{exp_no}/{child_no} update'")
        else:
            # 既存チェック
            result = subprocess.run(
                f"{KAGGLE_CLI} datasets list --mine --csv",
                shell=True, capture_output=True, text=True
            )
            if slug in result.stdout:
                print(f"\nDataset '{slug}' already exists. Use --update to add a new version.")
                sys.exit(1)

            print(f"\nCreating new dataset: {dataset_id}")
            run(f"{KAGGLE_CLI} datasets create -p {tmpdir}")

    print(f"\nUpload complete!")
    print(f"Dataset: https://www.kaggle.com/datasets/{dataset_id}")
    print(f"\nKaggle Notebooks での参照パス:")
    print(f"  /kaggle/input/{slug}/")
    print(f"\ninfer.ipynb の ADAPTER_DATASET_PATH を以下に設定してください:")
    print(f"  ADAPTER_DATASET_PATH = \"/kaggle/input/{slug}\"")


def main():
    parser = argparse.ArgumentParser(description="Upload LoRA adapter to Kaggle Dataset")
    parser.add_argument(
        "--adapter_dir",
        required=True,
        help="Path to adapter directory (containing adapter_config.json)",
    )
    parser.add_argument("--exp_no",   default="EXP000",      help="Experiment number (e.g. EXP000)")
    parser.add_argument("--child_no", default="child-exp000", help="Child exp number (e.g. child-exp000)")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Add a new version to existing dataset instead of creating new",
    )
    parser.add_argument(
        "--base_dir",
        default=None,
        help="Base dir of competition (adapter_dir is relative to this if set)",
    )
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    if args.base_dir:
        adapter_dir = Path(args.base_dir) / adapter_dir
    adapter_dir = adapter_dir.resolve()

    if not adapter_dir.exists():
        sys.exit(f"ERROR: adapter_dir not found: {adapter_dir}")

    upload(adapter_dir, args.exp_no, args.child_no, args.update)


if __name__ == "__main__":
    main()
