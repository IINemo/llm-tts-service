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

import json
import os
import subprocess
import sys


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
    print("[ClearML wrapper] === CUDA_VISIBLE_DEVICES ===", flush=True)
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

    # Granular import diagnostic to isolate std::bad_alloc
    diag_code = """
import sys, traceback, os
print(f'[diag] Python: {sys.executable}', flush=True)
print(f'[diag] CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")}', flush=True)

# Step 1: basic torch
try:
    import torch
    print(f'[diag] torch {torch.__version__} imported OK', flush=True)
    print(f'[diag] CUDA available: {torch.cuda.is_available()}', flush=True)
    if torch.cuda.is_available():
        print(f'[diag] CUDA device count: {torch.cuda.device_count()}', flush=True)
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_mem / 1024**3
            print(f'[diag] GPU {i}: {props.name}, {mem_gb:.1f} GB', flush=True)
except Exception:
    traceback.print_exc()

# Step 2: vllm import
try:
    import vllm
    print(f'[diag] vllm {vllm.__version__} imported OK', flush=True)
except Exception:
    traceback.print_exc()

# Step 3: scorers import
try:
    from llm_tts.scorers import StepScorerPRM
    print('[diag] StepScorerPRM imported OK', flush=True)
except Exception:
    traceback.print_exc()
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
