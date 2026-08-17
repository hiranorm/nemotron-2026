"""
EXP000: Package LoRA adapter as submission.zip.

Usage:
    python infer.py [--config config/child-exp000.yaml]

Reads adapter from the path specified in config, patches adapter_config.json,
and creates submission.zip ready for competition submission.
"""

import argparse
import json
import zipfile
from pathlib import Path

import yaml

BASE_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
REQUIRED_FILES = ["adapter_config.json", "adapter_model.safetensors"]


def patch_adapter_config(config_path: Path) -> None:
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["base_model_name_or_path"] = BASE_MODEL_NAME
    cfg["inference_mode"] = True
    cfg["lora_dropout"] = 0.0
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Patched adapter_config.json (base_model={BASE_MODEL_NAME})")


def run(adapter_dir: str, submission_zip: str) -> None:
    src = Path(adapter_dir)
    for fname in REQUIRED_FILES:
        if not (src / fname).exists():
            raise FileNotFoundError(f"Missing: {src / fname}")

    patch_adapter_config(src / "adapter_config.json")

    zip_path = Path(submission_zip)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in REQUIRED_FILES:
            fpath = src / fname
            zf.write(fpath, fname)
            print(f"  Added {fname} ({fpath.stat().st_size / 1024 / 1024:.1f} MB)")

    print(f"\nsubmission.zip: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("Ready to submit!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/child-exp000.yaml")
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--submission_zip", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    adapter_dir = args.adapter_dir or cfg["paths"]["adapter_dir"]
    submission_zip = args.submission_zip or cfg["paths"]["submission_zip"]
    run(adapter_dir, submission_zip)


if __name__ == "__main__":
    main()
