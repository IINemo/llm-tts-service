#!/usr/bin/env python3
"""ClearML wrapper for run_tts_eval.py that bypasses ClearML's Hydra binding.

This wrapper runs run_tts_eval.py in a clean subprocess (without ClearML Hydra patches).
It also bootstraps lm-polygraph into ClearML's auto-created venv without breaking vLLM.
"""

import importlib
import json
import os
import subprocess
import sys


def _pip(*args: str) -> None:
    cmd = [sys.executable, "-m", "pip", *args]
    print("[bootstrap]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def ensure_lm_polygraph_installed() -> None:
    # Fast path
    try:
        importlib.import_module("lm_polygraph")
        print("[bootstrap] lm_polygraph already installed", flush=True)
        return
    except Exception as e:
        print(f"[bootstrap] lm_polygraph not importable: {e}", flush=True)

    # 1) Keep critical pins compatible with vLLM 0.12.0 + transformers>=4.56
    #    (These should already be installed since vllm imports worked, but we enforce anyway.)
    _pip("install", "--force-reinstall", "huggingface-hub>=0.34.0,<1.0")
    _pip("install", "--force-reinstall", "tokenizers==0.22.2")
    _pip("install", "--force-reinstall", "--no-deps", "transformers>=4.57.0,<5.0.0")

    # 2) ANTLR pin (OmegaConf grammar compatibility)
    _pip("install", "--force-reinstall", "--no-deps", "antlr4-python3-runtime==4.9.3")

    # 3) Install lm-polygraph WITHOUT deps so it won't downgrade transformers/tokenizers
    _pip(
        "install",
        "--no-deps",
        "lm-polygraph @ git+https://github.com/IINemo/lm-polygraph.git@dev",
    )

    # 4) Install lm-polygraph runtime deps explicitly (safe versions)
    _pip(
        "install",
        "bert-score>=0.3.13",
        "bitsandbytes",
        "bs4",
        "fastchat",
        "flask>=2.3.2",
        "fschat>=0.2.3",
        "hf-lfs>=0.0.3",
        "matplotlib>=3.6",
        "nlpaug>=1.1.10",
        "nltk>=3.7,<4",
        "pytest>=4.4.1",
        "pytreebank>=0.2.7",
        "rouge-score>=0.0.4",
        "unbabel-comet==2.2.1",
        "wget",
        "spacy>=3.4.0,<3.8.0",
        "sentence-transformers",
        "sacrebleu>=1.5.0",
        "sentencepiece>=0.1.97",
        "evaluate>=0.4.2",
        "datasets>=2.19.0,<4.0.0",
        "fsspec>=2023.1.0,<=2024.6.1",
    )

    # 5) latex2sympy2 must be installed without deps (antlr conflict)
    _pip("install", "--no-deps", "latex2sympy2")

    # 6) Final re-pin in case any dependency tried to move the stack
    _pip("install", "--force-reinstall", "huggingface-hub>=0.34.0,<1.0")
    _pip("install", "--force-reinstall", "tokenizers==0.22.2")
    _pip("install", "--force-reinstall", "--no-deps", "transformers>=4.57.0,<5.0.0")
    _pip("install", "--force-reinstall", "--no-deps", "antlr4-python3-runtime==4.9.3")

    # Sanity check
    importlib.import_module("lm_polygraph")
    print("[bootstrap] lm_polygraph OK", flush=True)


# Bootstrap once in the ClearML venv
ensure_lm_polygraph_installed()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_script = os.path.join(script_dir, "run_tts_eval.py")

    # Read Hydra args from ClearML task parameters
    hydra_args = []
    try:
        from clearml import Task

        task = Task.current_task()
        if task:
            params = task.get_parameters()
            config_path = params.get("HydraArgs/config_path", "")
            config_name = params.get("HydraArgs/config_name", "")
            overrides_json = params.get("HydraArgs/overrides", "[]")

            if config_path:
                hydra_args.extend(["--config-path", config_path])
            if config_name:
                hydra_args.extend(["--config-name", config_name])

            try:
                overrides = json.loads(overrides_json)
                hydra_args.extend(overrides)
            except json.JSONDecodeError:
                pass

            print(
                f"[ClearML wrapper] Read HydraArgs from task: {hydra_args}", flush=True
            )
    except Exception as e:
        print(f"[ClearML wrapper] Could not read ClearML params: {e}", flush=True)

    # Build command
    cmd = [sys.executable, target_script] + hydra_args + sys.argv[1:]

    # Clean environment (avoid ClearML auto init inside subprocess)
    env = os.environ.copy()
    for key in list(env.keys()):
        if key.startswith("CLEARML"):
            del env[key]

    print(f"[ClearML wrapper] Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, env=env, stderr=subprocess.STDOUT)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
