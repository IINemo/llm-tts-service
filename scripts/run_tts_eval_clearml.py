#!/usr/bin/env python3
"""ClearML wrapper for run_tts_eval.py that bypasses ClearML's Hydra binding.

ClearML's PatchHydra intercepts @hydra.main and modifies config_sources,
which breaks code that reads HydraConfig.get().runtime.config_sources.
This wrapper runs the actual script in a clean subprocess without ClearML's
Hydra patches.

Hydra arguments are read from the ClearML task's 'HydraArgs/' parameter section:
  - HydraArgs/config_path: Hydra --config-path value
  - HydraArgs/config_name: Hydra --config-name value
  - HydraArgs/overrides: JSON list of Hydra overrides (e.g. ["++key=val", ...])
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
    # Only check module importability here (no WhiteboxModel import => no F811/F401 issues)
    try:
        importlib.import_module("lm_polygraph")
        print("[bootstrap] lm_polygraph already installed", flush=True)
        return
    except Exception as e:
        print(f"[bootstrap] lm_polygraph not importable: {e}", flush=True)

    # Upgrade tooling (pip on workers was old; VCS installs/builds behave better after upgrade)
    try:
        _pip("install", "-U", "pip", "setuptools", "wheel")
    except Exception as e:
        print(f"[bootstrap] tooling upgrade failed (continuing): {e}", flush=True)

    # Install lm-polygraph into this venv (non-editable => avoids PEP660 build_editable issue)
    _pip("install", "lm-polygraph @ git+https://github.com/IINemo/lm-polygraph.git@dev")

    # Fix the specific observed breakage:
    # transformers 4.57.x requires tokenizers>=0.22.0,<=0.23.0 but env had 0.21.4
    _pip("install", "--force-reinstall", "tokenizers==0.23.0")

    # Re-pin known conflict points (belt-and-suspenders)
    _pip("install", "--force-reinstall", "--no-deps", "antlr4-python3-runtime==4.9.3")
    _pip("install", "--force-reinstall", "--no-deps", "transformers>=4.57.0,<5.0.0")

    # Sanity check: module import only
    importlib.import_module("lm_polygraph")
    print("[bootstrap] lm_polygraph OK", flush=True)


ensure_lm_polygraph_installed()
# --- end bootstrap ---


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_script = os.path.join(script_dir, "run_tts_eval.py")

    # Try to read arguments from ClearML task parameters
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
    cmd = [sys.executable, target_script]
    cmd.extend(hydra_args)
    # Also pass any CLI arguments (fallback if not using ClearML params)
    cmd.extend(sys.argv[1:])

    # Clean environment: remove ClearML env vars to prevent auto-initialization
    # in the subprocess. This ensures @hydra.main runs without ClearML patches.
    env = os.environ.copy()
    for key in list(env.keys()):
        if key.startswith("CLEARML"):
            del env[key]

    # GPU diagnostics
    print("[ClearML wrapper] === GPU Diagnostics ===", flush=True)
    subprocess.run(["nvidia-smi"], env=env)
    print("[ClearML wrapper] === nvidia-smi -L ===", flush=True)
    subprocess.run(["nvidia-smi", "-L"], env=env)
    print("[ClearML wrapper] === Environment ===", flush=True)
    print(
        f"  CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '(not set)')}",
        flush=True,
    )
    print(
        f"  CUDA_MODULE_LOADING={env.get('CUDA_MODULE_LOADING', '(not set)')}",
        flush=True,
    )
    print(f"  VLLM_USE_V1={env.get('VLLM_USE_V1', '(not set)')}", flush=True)
    print(
        f"  VLLM_ATTENTION_BACKEND={env.get('VLLM_ATTENTION_BACKEND', '(not set)')}",
        flush=True,
    )

    # Auto-detect MIG UUID and set CUDA_VISIBLE_DEVICES
    import re as _re

    try:
        result_smi = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            env=env,
        )
        mig_uuids = _re.findall(r"(MIG-[0-9a-f-]+)", result_smi.stdout)
        if mig_uuids:
            mig_uuid = mig_uuids[0]
            print(f"[ClearML wrapper] Detected MIG UUID: {mig_uuid}", flush=True)
            env["CUDA_VISIBLE_DEVICES"] = mig_uuid
            print(
                f"[ClearML wrapper] Set CUDA_VISIBLE_DEVICES={mig_uuid}",
                flush=True,
            )
        else:
            print("[ClearML wrapper] No MIG UUID found in nvidia-smi -L", flush=True)
    except Exception as e:
        print(f"[ClearML wrapper] MIG UUID detection failed: {e}", flush=True)

    # Granular import diagnostic to isolate std::bad_alloc
    diag_code = """
import sys, traceback, os
print(f'[diag] Python: {sys.executable}', flush=True)
print(f'[diag] CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")}', flush=True)
print(f'[diag] VLLM_ATTENTION_BACKEND={os.environ.get("VLLM_ATTENTION_BACKEND", "(not set)")}', flush=True)

# Step 1: basic torch + CUDA init
print('[diag] Step 1: importing torch and initializing CUDA...', flush=True)
try:
    import torch
    print(f'[diag] torch {torch.__version__} imported OK', flush=True)
    torch.cuda.init()
    print(f'[diag] torch.cuda.init() OK', flush=True)
    print(f'[diag] CUDA available: {torch.cuda.is_available()}', flush=True)
    if torch.cuda.is_available():
        print(f'[diag] device name: {torch.cuda.get_device_name(0)}', flush=True)
        print(f'[diag] device count: {torch.cuda.device_count()}', flush=True)
        free, total = torch.cuda.mem_get_info()
        print(f'[diag] free GiB: {free/1024**3:.1f}, total GiB: {total/1024**3:.1f}', flush=True)
except Exception:
    traceback.print_exc()
    sys.stdout.flush()

# Step 2: vllm import (lazy)
print('[diag] Step 2: importing vllm...', flush=True)
try:
    import vllm
    print(f'[diag] vllm {vllm.__version__} imported OK', flush=True)
except Exception:
    traceback.print_exc()
    sys.stdout.flush()

# Step 3: vllm.config (triggers CUDA ops loading)
print('[diag] Step 3: importing vllm.config...', flush=True)
try:
    import vllm.config
    print('[diag] vllm.config imported OK', flush=True)
except Exception:
    traceback.print_exc()
    sys.stdout.flush()

# Step 4: LLM class
print('[diag] Step 4: importing LLM, SamplingParams...', flush=True)
try:
    from vllm import LLM, SamplingParams
    print('[diag] LLM, SamplingParams imported OK', flush=True)
except Exception:
    traceback.print_exc()
    sys.stdout.flush()

print('[diag] All imports OK, exiting cleanly', flush=True)
os._exit(0)  # force exit to avoid cleanup crashes
"""
    print("[ClearML wrapper] Running import diagnostic...", flush=True)
    subprocess.run(
        [sys.executable, "-c", diag_code],
        env=env,
        cwd=os.path.dirname(script_dir),
        stderr=subprocess.STDOUT,
    )

    print(f"[ClearML wrapper] Running: {' '.join(cmd)}", flush=True)
    # Merge stderr into stdout so ClearML captures errors
    # (run_tts_eval.py redirects sys.stderr to a file, hiding tracebacks)
    result = subprocess.run(cmd, env=env, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print(
            f"[ClearML wrapper] Process exited with code {result.returncode}",
            flush=True,
        )
        # Try to print the stderr.log from the output directory
        import glob

        for stderr_log in glob.glob(
            os.path.join(os.path.dirname(script_dir), "outputs", "**", "stderr.log"),
            recursive=True,
        ):
            print(f"[ClearML wrapper] Contents of {stderr_log}:", flush=True)
            with open(stderr_log) as f:
                print(f.read(), flush=True)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
