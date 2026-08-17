"""
EXP001: CoT-SFT with verified Chain-of-Thought training data.

Fine-tunes Nemotron-3-Nano-30B-A3B with LoRA using konbu17's
verified-correct CoT dataset (2,907 samples, type-balanced).

Reference: research/reference_notebooks/nemotron-sft-lora-with-cot.ipynb

Usage:
    python train.py --config config/child-exp000.yaml [--adapter_dir /path/to/output]

Runs on Kaggle Notebooks (internet=OFF, model + CoT dataset attached as Datasets).
"""

import argparse
import os
import re
import site
import sys
from pathlib import Path

import yaml

_CUTLASS_PATH = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages/"
if os.path.exists(_CUTLASS_PATH):
    site.addsitedir(_CUTLASS_PATH)


def setup_kaggle_environment() -> None:
    """Install and configure Kaggle-specific dependencies.

    Must be called before importing torch/transformers to avoid import-time CUDA issues.
    Handles: Triton wheel, ptxas-blackwell binary, mamba_ssm stubs, trl/datasets.
    """
    import glob
    import importlib.util
    import shutil
    import stat
    import subprocess
    import types

    def sh(cmd: str, check: bool = True) -> None:
        print("+", cmd)
        subprocess.run(cmd, shell=True, check=check)

    def is_installed(name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    # --- 1. Triton wheel ---
    triton_wheels = glob.glob("/kaggle/input/**/*triton*.whl", recursive=True)
    print("Found Triton wheels:", triton_wheels)
    if triton_wheels:
        wheel = triton_wheels[0]
        target = "/kaggle/working/pydeps"
        os.makedirs(target, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "--no-deps", "--target", target, "--upgrade", "--ignore-installed", wheel],
            check=True,
        )
        if target not in sys.path:
            sys.path.insert(0, target)
        site.addsitedir(target)
        print("Triton installed to:", target)

    # --- 2. ptxas-blackwell ---
    ptxas_src = (
        "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script"
        "/triton/backends/nvidia/bin/ptxas-blackwell"
    )
    ptxas_dst = "/tmp/ptxas-blackwell"
    if os.path.exists(ptxas_src) and not os.path.exists(ptxas_dst):
        shutil.copy2(ptxas_src, ptxas_dst)
        os.chmod(ptxas_dst, os.stat(ptxas_dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        src_bin = os.path.dirname(ptxas_src)
        dst_bin = "/tmp/triton_nvidia_bin"
        shutil.copytree(src_bin, dst_bin, dirs_exist_ok=True)
        for f in os.listdir(dst_bin):
            fp = os.path.join(dst_bin, f)
            if os.path.isfile(fp):
                os.chmod(fp, os.stat(fp).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = ptxas_dst
        os.environ["TRITON_PTXAS_PATH"] = ptxas_dst
        try:
            import triton.backends.nvidia as nv_backend
            nv_backend.__file__ = os.path.join(dst_bin, "..", "__init__.py")
        except ImportError:
            pass
    try:
        import triton.backends.nvidia.compiler as nv_compiler
        nv_compiler.get_ptxas_version = lambda arch: "12.0"
        print("ptxas monkey-patch applied.")
    except ImportError:
        pass

    # --- 3. Wheel discovery helpers ---
    import torch
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    abi_tag = "cxx11abiTRUE" if torch.compiled_with_cxx11_abi() else "cxx11abiFALSE"
    print(f"Wheel selector: py={py_tag}, torch={torch_mm}, abi={abi_tag}")

    def find_wheels(pattern: str):
        return sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))

    def pick_best(wheels):
        exact = [w for w in wheels if py_tag in w and f"torch{torch_mm}" in w and abi_tag in w]
        if exact:
            return exact[-1]
        py_only = [w for w in wheels if py_tag in w]
        return py_only[-1] if py_only else None

    def install_wheel(pattern: str, pkg_name: str, check: bool = True) -> None:
        if is_installed(pkg_name):
            return
        w = pick_best(find_wheels(pattern))
        if w:
            sh(f'{sys.executable} -m pip install --no-index --no-deps "{w}"', check=check)

    # --- 4. datasets / trl / helper packages ---
    install_wheel("datasets-*.whl", "datasets")
    install_wheel("trl-*.whl", "trl")
    install_wheel("multiprocess-*.whl", "multiprocess", check=False)
    install_wheel("dill-*.whl", "dill", check=False)
    install_wheel("xxhash-*.whl", "xxhash", check=False)

    if not is_installed("trl"):
        offline_dirs = [
            "/kaggle/input/datasets/dennisfong/nvidia-nemotron-offline-packages/offline_packages/",
            # Add more dataset paths here if you upload a trl wheel dataset
        ]
        installed = False
        for offline in offline_dirs:
            if os.path.exists(offline):
                sh(f"pip install --no-index --find-links={offline} trl")
                installed = True
                break
        if not installed:
            raise RuntimeError(
                "trl is not installed and no offline wheel was found.\n"
                "Fix: Upload a trl wheel to a Kaggle dataset and attach it.\n"
                "  1. pip download trl --dest ./trl_wheels\n"
                "  2. kaggle datasets create / update with that folder\n"
                "  3. Attach the dataset to this notebook and add its path to offline_dirs above.\n"
                "Searched paths:\n" + "\n".join(f"  {d}" for d in offline_dirs)
            )

    # --- 5. mamba_ssm + causal_conv1d ---
    if not is_installed("mamba_ssm"):
        causal_wheel = pick_best(find_wheels("causal*conv1d*.whl"))
        mamba_wheel = pick_best(find_wheels("mamba_ssm-*.whl"))
        print("causal_conv1d wheel:", causal_wheel)
        print("mamba_ssm wheel:", mamba_wheel)
        if causal_wheel:
            sh(f'{sys.executable} -m pip install --no-index --no-deps "{causal_wheel}"')
        if mamba_wheel:
            sh(f'{sys.executable} -m pip install --no-index --no-deps "{mamba_wheel}"')
        else:
            raise FileNotFoundError(
                f"No compatible mamba_ssm wheel found for py={py_tag}, torch={torch_mm}, abi={abi_tag}"
            )

    # --- 6. mamba_ssm stub modules (required by Nemotron model code) ---
    for mod_name in [
        "mamba_ssm.modules.mamba3",
        "mamba_ssm.ops.cute",
        "mamba_ssm.ops.cute.mamba3",
        "mamba_ssm.ops.cute.mamba3.mamba3_step_fn",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    sys.modules["mamba_ssm.modules.mamba3"].Mamba3 = None

    import datasets as _ds
    import trl as _trl
    import mamba_ssm as _ms
    print(f"datasets: {_ds.__version__}  trl: {_trl.__version__}  mamba_ssm: {_ms.__version__}")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_cot_dataset(cfg: dict, tokenizer) -> "datasets.Dataset":
    """Build SFT dataset from the verified-correct CoT CSV.

    Format per record:
      user:      prompt + PROMPT_SUFFIX
      assistant: cot_cleaned + "\\n</think>\\n\\boxed{answer}"
    Note: <think> tag is auto-prepended by the chat template at inference time.
    """
    import pandas as pd
    from datasets import Dataset as HFDataset

    data_cfg = cfg["data"]
    csv_path = data_cfg["cot_csv_path"]
    prompt_suffix = data_cfg["prompt_suffix"]
    type_samples = data_cfg["type_samples"]
    seed = cfg["training"].get("seed", 123)

    df = pd.read_csv(csv_path)
    print(f"Loaded CoT CSV: {len(df)} rows from {csv_path}")
    print(df["type"].value_counts().sort_index().to_string())

    # Type-based sampling
    sampled = []
    for ptype, n_max in type_samples.items():
        subset = df[df["type"] == ptype]
        n = min(n_max, len(subset))
        sampled.append(subset.sample(n=n, random_state=seed) if n < len(subset) else subset)
        print(f"  {ptype}: {len(subset)} -> {n}")

    train_df = pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"Training samples after sampling: {len(train_df)}")

    records = []
    for _, row in train_df.iterrows():
        prompt = str(row["prompt"])
        answer = str(row["answer"])
        cot = str(row["generated_cot"])
        if not cot or cot == "nan" or len(cot.strip()) < 5:
            continue
        # Strip any boxed answers from the CoT body; the correct answer is appended at the end
        cot_cleaned = re.sub(r"\\boxed\{[^}]*\}", "", cot).rstrip()
        user_content = prompt + prompt_suffix
        assistant_content = cot_cleaned + f"\n</think>\n\\boxed{{{answer}}}"
        records.append({"messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]})

    print(f"SFT records built: {len(records)}")
    return HFDataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter_dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    setup_kaggle_environment()

    # Deferred imports (after environment setup)
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    import kagglehub

    model_cfg = cfg["model"]
    adapter_dir = Path(args.adapter_dir or cfg["paths"]["adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model: {model_cfg['kagglehub_slug']}")
    model_path = kagglehub.model_download(model_cfg["kagglehub_slug"])
    print(f"Model path: {model_path}")

    dtype = getattr(torch, model_cfg["torch_dtype"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True, torch_dtype=dtype,
    )
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

    dataset = load_cot_dataset(cfg, tokenizer)

    train_cfg = cfg["training"]
    training_args = SFTConfig(
        output_dir=cfg["paths"].get("sft_output_dir", "/kaggle/working/sft_output"),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        max_length=train_cfg["max_length"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=train_cfg.get("seed", 123),
        report_to="none",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    import time
    print("Starting SFT training...")
    t0 = time.time()
    trainer.train()
    print(f"Training done in {(time.time() - t0) / 60:.1f} min")

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Adapter saved to: {adapter_dir}")


if __name__ == "__main__":
    main()
