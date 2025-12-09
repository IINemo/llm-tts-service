#!/usr/bin/env python
"""
Subprocess worker for CUDA kernel verification.

This script is called by the parallel verifier to verify a single kernel
in a separate process with clean CUDA state.

Usage:
    python cuda_verify_worker.py --cuda_file <path> --task_file <path> --result_file <path> [options]
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[Worker] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def verify_kernel(args):
    """Run verification on a single kernel."""
    import torch
    from torch.utils.cpp_extension import load

    log.info(f"Starting verification on GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")

    # Load task from file
    with open(args.task_file, "r") as f:
        task = json.load(f)

    # Load CUDA code from file
    with open(args.cuda_file, "r") as f:
        cuda_code = f.read()

    result = {
        "compiled": False,
        "compile_error": "",
        "correct": False,
        "max_diff": float("inf"),
        "num_correct_trials": 0,
        "total_trials": args.num_trials,
        "cuda_time_ms": float("inf"),
        "torch_time_ms": float("inf"),
        "speedup": 0.0,
        "task_id": task.get("task_id", "unknown"),
    }

    # Step 1: Compile the CUDA kernel
    log.info("Compiling kernel...")
    try:
        ext_dir = os.environ.get(
            "TORCH_EXTENSIONS_DIR",
            os.path.expanduser("~/.cache/torch_extensions/cudabench")
        )
        os.makedirs(ext_dir, exist_ok=True)

        # Find CUDA include paths
        cuda_include_paths = []
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix:
            cuda_target_include = os.path.join(conda_prefix, "targets/x86_64-linux/include")
            if os.path.exists(cuda_target_include):
                cuda_include_paths.append(cuda_target_include)

        module = load(
            name=f"cuda_worker_{hash(cuda_code) % 10000}",
            sources=[args.cuda_file],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_include_paths=cuda_include_paths if cuda_include_paths else None,
            with_cuda=True,
            verbose=False,
            build_directory=ext_dir,
        )

        if not hasattr(module, "forward"):
            result["compile_error"] = "Module has no 'forward' function"
            return result

        cuda_fn = module.forward
        result["compiled"] = True
        log.info("Compilation successful")

    except Exception as e:
        result["compile_error"] = str(e)
        log.error(f"Compilation failed: {e}")
        return result

    # Step 2: Load PyTorch reference
    log.info("Loading PyTorch reference...")
    try:
        pytorch_code = task["pytorch_code"]
        namespace = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional}
        exec(pytorch_code, namespace)

        model_cls = namespace.get("Model")
        forward_fn = namespace.get("forward_fn")
        get_inputs = namespace.get("get_inputs")

        if model_cls is None or forward_fn is None or get_inputs is None:
            result["compile_error"] = "Missing Model/forward_fn/get_inputs"
            return result

    except Exception as e:
        result["compile_error"] = f"Failed to load PyTorch reference: {e}"
        return result

    # Step 3: Get settings
    import inspect

    # Get default settings from get_inputs signature
    sig = inspect.signature(get_inputs)
    input_settings = {}
    for name, param in sig.parameters.items():
        if param.default is not inspect.Parameter.empty:
            input_settings[name] = param.default

    # Get model params
    model_sig = inspect.signature(model_cls.__init__)
    model_params = set(model_sig.parameters.keys()) - {"self"}
    init_settings = {k: v for k, v in input_settings.items() if k in model_params}

    # Use small inputs for faster testing
    if args.use_small_inputs:
        small_settings = {"batch_size": 2, "num_features": 8, "dim1": 16, "dim2": 16}
        input_settings.update(small_settings)
        init_settings.update({k: v for k, v in small_settings.items() if k in model_params})

    device = "cuda:0"

    # Step 4: Check correctness
    log.info("Checking correctness...")
    try:
        num_correct = 0
        max_diff = 0.0

        with torch.no_grad():
            for trial in range(args.num_trials):
                torch.manual_seed(42 + trial)
                inputs = get_inputs(**input_settings)
                inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

                torch.manual_seed(42 + trial)
                model_ref = model_cls(**init_settings).to(device)
                torch.manual_seed(42 + trial)
                model_cuda = model_cls(**init_settings).to(device)

                output_ref = model_ref(*inputs, fn=forward_fn)

                try:
                    output_cuda = model_cuda(*inputs, fn=cuda_fn)
                except Exception as e:
                    log.error(f"CUDA kernel execution failed: {e}")
                    result["compile_error"] = f"Execution error: {e}"
                    return result

                if torch.allclose(output_ref, output_cuda, rtol=args.rtol, atol=args.atol):
                    num_correct += 1
                else:
                    diff = float(torch.max(torch.abs(output_ref - output_cuda)))
                    max_diff = max(max_diff, diff)

                del model_ref, model_cuda, inputs, output_ref, output_cuda
                torch.cuda.empty_cache()

        result["num_correct_trials"] = num_correct
        result["max_diff"] = max_diff
        result["correct"] = num_correct == args.num_trials

        if result["correct"]:
            log.info(f"Correctness: PASS ({num_correct}/{args.num_trials})")
        else:
            log.info(f"Correctness: FAIL ({num_correct}/{args.num_trials}, max_diff={max_diff:.6f})")

    except Exception as e:
        log.error(f"Correctness check failed: {e}")
        traceback.print_exc()
        return result

    # Step 5: Benchmark (only if correct)
    if result["correct"]:
        log.info("Benchmarking...")
        try:
            torch.manual_seed(42)
            inputs = get_inputs(**input_settings)
            inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

            torch.manual_seed(42)
            model = model_cls(**init_settings).to(device)

            # Warmup
            with torch.no_grad():
                for _ in range(args.warmup_iters):
                    model(*inputs, fn=forward_fn)
                    model(*inputs, fn=cuda_fn)
                    torch.cuda.synchronize()

            # Benchmark PyTorch
            torch_times = []
            with torch.no_grad():
                for _ in range(args.benchmark_iters):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    model(*inputs, fn=forward_fn)
                    end.record()
                    torch.cuda.synchronize()
                    torch_times.append(start.elapsed_time(end))

            # Benchmark CUDA
            cuda_times = []
            with torch.no_grad():
                for _ in range(args.benchmark_iters):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    model(*inputs, fn=cuda_fn)
                    end.record()
                    torch.cuda.synchronize()
                    cuda_times.append(start.elapsed_time(end))

            import numpy as np
            result["cuda_time_ms"] = float(np.median(cuda_times))
            result["torch_time_ms"] = float(np.median(torch_times))
            result["speedup"] = result["torch_time_ms"] / result["cuda_time_ms"] if result["cuda_time_ms"] > 0 else 0.0

            log.info(f"Speedup: {result['speedup']:.2f}x (cuda={result['cuda_time_ms']:.3f}ms, torch={result['torch_time_ms']:.3f}ms)")

            del model, inputs
            torch.cuda.empty_cache()

        except Exception as e:
            log.error(f"Benchmarking failed: {e}")
            traceback.print_exc()

    return result


def main():
    parser = argparse.ArgumentParser(description="CUDA kernel verification worker")
    parser.add_argument("--cuda_file", required=True, help="Path to CUDA kernel file")
    parser.add_argument("--task_file", required=True, help="Path to task JSON file")
    parser.add_argument("--result_file", required=True, help="Path to write results")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance")
    parser.add_argument("--num_trials", type=int, default=5, help="Correctness trials")
    parser.add_argument("--warmup_iters", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--benchmark_iters", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--use_small_inputs", action="store_true", help="Use small inputs")

    args = parser.parse_args()

    try:
        result = verify_kernel(args)
    except Exception as e:
        log.error(f"Verification failed: {e}")
        traceback.print_exc()
        result = {
            "compiled": False,
            "compile_error": str(e),
            "correct": False,
            "max_diff": float("inf"),
            "speedup": 0.0,
        }

    # Write results
    with open(args.result_file, "w") as f:
        json.dump(result, f, indent=2)

    log.info(f"Results written to {args.result_file}")


if __name__ == "__main__":
    main()
