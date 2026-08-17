"""
EXP000: Untrained LoRA Baseline

Loads Nemotron-3-Nano-30B-A3B, attaches an untrained LoRA adapter, and saves it.
This establishes the base model's raw score with no fine-tuning.

Usage:
    python train.py --config config/child-exp000.yaml [--adapter_dir /path/to/output]

Runs on Kaggle Notebooks (internet=OFF, model attached as Dataset).
"""

import argparse
import os
import site
from pathlib import Path

import yaml

_CUTLASS_PATH = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages/"
if os.path.exists(_CUTLASS_PATH):
    site.addsitedir(_CUTLASS_PATH)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter_dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import kagglehub

    model_cfg = cfg["model"]
    adapter_dir = Path(args.adapter_dir or cfg["paths"]["adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model: {model_cfg['kagglehub_slug']}")
    model_path = kagglehub.model_download(model_cfg["kagglehub_slug"])
    print(f"Model path: {model_path}")

    dtype = getattr(torch, model_cfg["torch_dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True, torch_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded.")

    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules_regex"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- YOUR TRAINING CODE HERE ---

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Adapter saved to: {adapter_dir}")


if __name__ == "__main__":
    main()
