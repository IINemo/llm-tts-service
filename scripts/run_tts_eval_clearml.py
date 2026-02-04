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

    # Run import diagnostic first to catch detailed traceback
    diag_cmd = [
        sys.executable,
        "-c",
        "import traceback\n"
        "try:\n"
        "    from llm_tts.scorers import StepScorerPRM\n"
        "    print('[diag] StepScorerPRM imported OK')\n"
        "except Exception:\n"
        "    traceback.print_exc()\n",
    ]
    print("[ClearML wrapper] Running import diagnostic...", flush=True)
    subprocess.run(diag_cmd, env=env, cwd=os.path.dirname(script_dir))

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
