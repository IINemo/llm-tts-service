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
    _pip("install", "--no-deps", "git+https://github.com/IINemo/lm-polygraph.git@dev")

    # 4) Install sentencepiece binary first (avoids cmake build issues)
    _pip("install", "--only-binary=:all:", "sentencepiece>=0.1.97")

    # 5) Install lm-polygraph runtime deps explicitly (safe versions)
    _pip(
        "install",
        "accelerate>=0.32.1",  # Required by lm_polygraph
        "diskcache>=5.6.3",  # Required by lm_polygraph
        "einops",  # Required by lm_polygraph
        "hydra-core>=1.3.2",  # Required by lm_polygraph
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
        "evaluate>=0.4.2",
        "datasets>=2.19.0,<4.0.0",
        "fsspec>=2023.1.0,<=2024.6.1",
    )

    # 6) Install openai without deps to avoid huggingface-hub upgrade
    _pip("install", "--no-deps", "openai")
    _pip(
        "install", "httpx", "pydantic", "sniffio", "tqdm", "jiter"
    )  # openai deps without hf-hub

    # 7) latex2sympy2 must be installed without deps (antlr conflict)
    _pip("install", "--no-deps", "latex2sympy2")

    # 8) Final re-pin - CRITICAL: force exact versions
    _pip("install", "--force-reinstall", "huggingface-hub==0.30.2")
    _pip("install", "--force-reinstall", "tokenizers==0.22.2")
    _pip("install", "--force-reinstall", "--no-deps", "transformers>=4.57.0,<5.0.0")
    _pip("install", "--force-reinstall", "--no-deps", "antlr4-python3-runtime==4.9.3")

    # Sanity check
    importlib.import_module("lm_polygraph")
    print("[bootstrap] lm_polygraph OK", flush=True)


def ensure_essential_deps_installed() -> None:
    """Install essential deps for run_tts_eval.py including lm_polygraph.

    Note: Do NOT install vllm/torch here - use system's pre-installed versions
    to avoid CUDA library conflicts.
    """
    print("[bootstrap] Installing essential dependencies...", flush=True)

    # All lm_polygraph dependencies (from requirements.txt)
    # Note: torch/vllm excluded - use system's pre-installed versions
    _pip(
        "install",
        "hydra-core>=1.3.2",
        "omegaconf",
        "transformers",
        "datasets",
        "python-dotenv",
        "sympy",
        "openai",
        "accelerate",
        "einops",
        "diskcache",
        "scipy",
        "scikit-learn",
        "nltk",
        "sentencepiece",
        "rouge-score",
        "bert-score",
        "sacrebleu",
        "evaluate",
        "sentence-transformers",
        "spacy",
        "nlpaug",
        "matplotlib",
        "pandas",
        "bs4",
        "hf-lfs",
        "pytest",
        "pytreebank",
        "numpy",
        "dill",
        "flask",
        "protobuf",
        "fschat",
        "bitsandbytes",
        "wget",
        "unbabel-comet",
        "fastchat",
    )

    # lm_polygraph without deps (to avoid version conflicts)
    _pip("install", "--no-deps", "git+https://github.com/IINemo/lm-polygraph.git@dev")

    # latex2sympy2 with specific antlr version
    _pip("install", "--no-deps", "latex2sympy2")
    _pip("install", "antlr4-python3-runtime==4.9.3")

    print("[bootstrap] Essential deps installed", flush=True)


# Bootstrap dependencies
# Set SKIP_LM_POLYGRAPH=1 to skip lm_polygraph and only install essentials
if os.environ.get("SKIP_LM_POLYGRAPH", "0") != "1":
    try:
        import lm_polygraph  # noqa: F401

        print(
            "[bootstrap] lm_polygraph already available, skipping bootstrap", flush=True
        )
    except ImportError:
        ensure_lm_polygraph_installed()
else:
    print(
        "[bootstrap] Skipping lm_polygraph installation (SKIP_LM_POLYGRAPH=1)",
        flush=True,
    )
    ensure_essential_deps_installed()


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
