"""
CUDA kernel verifier for CUDABench evaluation.

This module provides verification and benchmarking of LLM-generated CUDA kernels
against PyTorch reference implementations.

Uses robust-kbench evaluation infrastructure when available, with fallback
to standalone verification.

Supports parallel verification using multiprocessing for faster evaluation
of multiple kernel candidates.
"""

import json
import logging
import multiprocessing as mp
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

log = logging.getLogger(__name__)


def _verify_kernel_worker(args: Tuple) -> Dict[str, Any]:
    """
    Worker function for parallel kernel verification.

    Runs in a separate process to avoid CUDA state conflicts.
    Must be a top-level function for pickling.

    Args:
        args: Tuple of (cuda_code, task, verifier_config, gpu_id, kernel_idx)

    Returns:
        Dictionary with verification results
    """
    cuda_code, task, config, gpu_id, kernel_idx = args

    # Set GPU for this worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TORCH_EXTENSIONS_DIR"] = os.path.expanduser(
        f"~/.cache/torch_extensions/cudabench_worker_{gpu_id}_{uuid.uuid4().hex[:8]}"
    )

    # Setup logging for this worker
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Worker-{kernel_idx}/GPU-{gpu_id}] %(message)s",
        stream=sys.stdout,
        force=True,
    )
    worker_log = logging.getLogger(__name__)

    worker_log.info(f"Starting verification for kernel {kernel_idx}")

    # Import torch after setting CUDA_VISIBLE_DEVICES
    import torch

    try:
        worker_log.info(f"Creating verifier (GPU {gpu_id})...")

        # Create verifier with config
        verifier = CUDAVerifier(
            rtol=config.get("rtol", 1e-5),
            atol=config.get("atol", 1e-5),
            num_trials=config.get("num_trials", 5),
            warmup_iters=config.get("warmup_iters", 25),
            benchmark_iters=config.get("benchmark_iters", 100),
            timeout=config.get("timeout", 60.0),
            use_small_inputs=config.get("use_small_inputs", True),
        )

        worker_log.info(f"Running verification...")

        # Run verification
        result = verifier.verify(cuda_code, task, device="cuda:0")

        status = []
        if result.compiled:
            status.append("compiled")
        if result.correct:
            status.append("correct")
        if result.speedup > 0:
            status.append(f"speedup={result.speedup:.2f}x")
        worker_log.info(f"Kernel {kernel_idx}: {' | '.join(status) if status else 'failed'}")

        return result.to_dict()

    except Exception as e:
        worker_log.error(f"Kernel {kernel_idx} verification error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "compiled": False,
            "compile_error": str(e),
            "correct": False,
            "max_diff": float("inf"),
            "speedup": 0.0,
            "task_id": task.get("task_id", "unknown"),
        }
    finally:
        # Cleanup GPU memory
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass


@dataclass
class VerificationResult:
    """Result of CUDA kernel verification."""

    # Compilation
    compiled: bool
    compile_error: str = ""

    # Correctness
    correct: bool = False
    max_diff: float = float("inf")
    num_correct_trials: int = 0
    total_trials: int = 5

    # Performance
    cuda_time_ms: float = float("inf")
    torch_time_ms: float = float("inf")
    speedup: float = 0.0

    # Metadata
    task_id: str = ""
    kernel_code: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compiled": self.compiled,
            "compile_error": self.compile_error,
            "correct": self.correct,
            "max_diff": self.max_diff,
            "num_correct_trials": self.num_correct_trials,
            "total_trials": self.total_trials,
            "cuda_time_ms": self.cuda_time_ms,
            "torch_time_ms": self.torch_time_ms,
            "speedup": self.speedup,
            "task_id": self.task_id,
        }

    @property
    def passed(self) -> bool:
        """Kernel passes if it compiles and is correct."""
        return self.compiled and self.correct

    @property
    def score(self) -> float:
        """Score for ranking kernels (higher is better)."""
        if not self.passed:
            return 0.0
        return self.speedup


class CUDAVerifier:
    """
    Verifier for CUDA kernels against PyTorch reference.

    Compatible with the evaluator interface used by run_tts_eval.py:
    - __call__(questions, generated_answers, gold_answers) -> annotations, responses

    For CUDA evaluation:
    - question = PyTorch reference code
    - generated_answer = CUDA kernel code
    - gold_answer = not used (evaluated by execution)

    Supports two modes:
    1. robust-kbench mode: Uses the full robust-kbench infrastructure
    2. standalone mode: Direct compilation and verification
    """

    def __init__(
        self,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        num_trials: int = 5,
        warmup_iters: int = 25,
        benchmark_iters: int = 100,
        timeout: float = 60.0,
        use_robust_kbench: bool = True,
        use_small_inputs: bool = True,
    ):
        """
        Initialize verifier.

        Args:
            rtol: Relative tolerance for correctness check
            atol: Absolute tolerance for correctness check
            num_trials: Number of correctness trials
            warmup_iters: Warmup iterations before benchmarking
            benchmark_iters: Benchmark iterations for timing
            timeout: Timeout in seconds for compilation/execution
            use_robust_kbench: Try to use robust-kbench infrastructure
            use_small_inputs: Use smaller input sizes for faster testing
        """
        self.rtol = rtol
        self.atol = atol
        self.num_trials = num_trials
        self.warmup_iters = warmup_iters
        self.benchmark_iters = benchmark_iters
        self.timeout = timeout
        self.use_robust_kbench = use_robust_kbench
        self.use_small_inputs = use_small_inputs

        # Small input overrides for faster testing
        self.small_input_settings = {
            "batch_size": 2,
            "num_features": 8,
            "dim1": 16,
            "dim2": 16,
        }

        # Cache for compiled modules
        self._compile_cache: Dict[str, Any] = {}

        # Extension directory for torch cpp_extension
        self.ext_dir = os.path.expanduser("~/.cache/torch_extensions/cudabench")
        os.makedirs(self.ext_dir, exist_ok=True)

    def __call__(
        self,
        questions: List[str],
        generated_answers: List[str],
        gold_answers: List[str],
    ) -> Tuple[List[int], List[Dict]]:
        """
        Evaluator interface for run_tts_eval.py compatibility.

        For CUDA evaluation:
        - questions = PyTorch reference code (used to build task)
        - generated_answers = CUDA kernel code
        - gold_answers = not used

        Returns:
            Tuple of (annotations, responses) where:
            - annotations: List of 1 (correct) or 0 (incorrect)
            - responses: List of verification result dicts
        """
        annotations = []
        responses = []

        for i, (pytorch_code, cuda_code) in enumerate(zip(questions, generated_answers)):
            # Build task dict from pytorch_code
            task = {
                "task_id": f"eval_task_{i}",
                "task_name": f"Evaluation Task {i}",
                "pytorch_code": pytorch_code,
                "config": {},
            }

            # Verify the kernel
            result = self.verify(cuda_code, task)

            # Convert to evaluator format
            annotations.append(1 if result.passed else 0)
            responses.append(result.to_dict())

        return annotations, responses

    def verify(
        self,
        cuda_code: str,
        task: Dict,
        device: str = "cuda",
    ) -> VerificationResult:
        """
        Verify a CUDA kernel against task's PyTorch reference.

        Args:
            cuda_code: Generated CUDA kernel code
            task: Task dictionary from CUDABench dataset
            device: Device to run verification on

        Returns:
            VerificationResult with compilation, correctness, and performance data
        """
        result = VerificationResult(
            compiled=False,
            task_id=task.get("task_id", "unknown"),
            kernel_code=cuda_code,
        )

        # Step 1: Compile the CUDA kernel
        try:
            cuda_fn, compile_error = self._compile_kernel(cuda_code, task)
            if cuda_fn is None:
                result.compile_error = compile_error
                return result
            result.compiled = True
        except Exception as e:
            result.compile_error = str(e)
            log.warning(f"Compilation failed: {e}")
            return result

        # Step 2: Load PyTorch reference
        try:
            model_cls, forward_fn, get_inputs = self._load_pytorch_reference(task)
        except Exception as e:
            result.compile_error = f"Failed to load PyTorch reference: {e}"
            log.warning(result.compile_error)
            return result

        # Step 3: Check correctness
        try:
            correct, num_correct, max_diff = self._check_correctness(
                cuda_fn, model_cls, forward_fn, get_inputs, task, device
            )
            result.correct = correct
            result.num_correct_trials = num_correct
            result.total_trials = self.num_trials
            result.max_diff = max_diff
        except Exception as e:
            log.warning(f"Correctness check failed: {e}")
            result.correct = False
            result.max_diff = float("inf")

        # Step 4: Benchmark (only if correct)
        if result.correct:
            try:
                cuda_time, torch_time = self._benchmark(
                    cuda_fn, model_cls, forward_fn, get_inputs, task, device
                )
                result.cuda_time_ms = cuda_time
                result.torch_time_ms = torch_time
                result.speedup = torch_time / cuda_time if cuda_time > 0 else 0.0
            except Exception as e:
                log.warning(f"Benchmarking failed: {e}")

        return result

    def verify_batch_parallel(
        self,
        cuda_codes: List[str],
        task: Dict,
        max_workers: Optional[int] = None,
    ) -> List[VerificationResult]:
        """
        Verify multiple CUDA kernels in parallel using multiprocessing.

        Uses separate processes to avoid CUDA state conflicts.
        Follows robust-kbench approach: spawn context + round-robin GPU assignment.

        Args:
            cuda_codes: List of CUDA kernel codes to verify
            task: Task dictionary (same for all kernels)
            max_workers: Max parallel workers (default: min(num_codes, num_gpus))

        Returns:
            List of VerificationResult objects
        """
        if not cuda_codes:
            return []

        # Determine number of GPUs and workers
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            log.warning("No GPUs available, falling back to sequential verification")
            return [self.verify(code, task) for code in cuda_codes]

        num_codes = len(cuda_codes)
        if max_workers is None:
            max_workers = min(num_codes, num_gpus)

        log.info(f"Parallel verification: {num_codes} kernels on {num_gpus} GPUs with {max_workers} workers")

        # Build verifier config for workers
        config = {
            "rtol": self.rtol,
            "atol": self.atol,
            "num_trials": self.num_trials,
            "warmup_iters": self.warmup_iters,
            "benchmark_iters": self.benchmark_iters,
            "timeout": self.timeout,
            "use_small_inputs": self.use_small_inputs,
        }

        # Prepare worker arguments with round-robin GPU assignment
        worker_args = []
        for i, code in enumerate(cuda_codes):
            gpu_id = i % num_gpus
            worker_args.append((code, task, config, gpu_id, i))  # Added kernel_idx

        # Use subprocess-based parallelization (like robust-kbench)
        # This avoids multiprocessing pickling issues and ensures clean CUDA state
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import subprocess
        import signal

        # Get path to worker script
        worker_script = os.path.join(
            os.path.dirname(__file__), "cuda_verify_worker.py"
        )

        # Create temp directory for this batch
        batch_dir = tempfile.mkdtemp(prefix="cuda_verify_batch_")

        def run_worker(args):
            """Run verification in subprocess."""
            code, gpu_id, kernel_idx = args

            # Write CUDA code and task to temp files
            cuda_file = os.path.join(batch_dir, f"kernel_{kernel_idx}.cu")
            task_file = os.path.join(batch_dir, f"task_{kernel_idx}.json")
            result_file = os.path.join(batch_dir, f"result_{kernel_idx}.json")

            with open(cuda_file, "w") as f:
                f.write(code)
            with open(task_file, "w") as f:
                json.dump(task, f)

            # Setup environment
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["TORCH_EXTENSIONS_DIR"] = os.path.expanduser(
                f"~/.cache/torch_extensions/cudabench_gpu{gpu_id}"
            )

            # Build command
            cmd = [
                "python", worker_script,
                "--cuda_file", cuda_file,
                "--task_file", task_file,
                "--result_file", result_file,
                "--rtol", str(self.rtol),
                "--atol", str(self.atol),
                "--num_trials", str(self.num_trials),
                "--warmup_iters", str(self.warmup_iters),
                "--benchmark_iters", str(self.benchmark_iters),
            ]
            if self.use_small_inputs:
                cmd.append("--use_small_inputs")

            log.info(f"  Starting kernel {kernel_idx} on GPU {gpu_id}...")

            try:
                # Run subprocess with timeout
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    universal_newlines=True,
                    start_new_session=True,
                )

                # Wait with timeout (5 min - compilation can take 30-60s per kernel)
                try:
                    stdout, _ = process.communicate(timeout=300)  # 5 min timeout
                    # Print worker output
                    for line in stdout.strip().split("\n"):
                        if line:
                            log.info(f"    [K{kernel_idx}] {line}")
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    log.error(f"  Kernel {kernel_idx} timed out")
                    return kernel_idx, {
                        "compiled": False,
                        "compile_error": "Verification timed out",
                        "correct": False,
                        "max_diff": float("inf"),
                        "speedup": 0.0,
                    }

                # Read results
                if os.path.exists(result_file):
                    with open(result_file, "r") as f:
                        result = json.load(f)
                    return kernel_idx, result
                else:
                    return kernel_idx, {
                        "compiled": False,
                        "compile_error": "No result file produced",
                        "correct": False,
                        "max_diff": float("inf"),
                        "speedup": 0.0,
                    }

            except Exception as e:
                log.error(f"  Kernel {kernel_idx} subprocess error: {e}")
                return kernel_idx, {
                    "compiled": False,
                    "compile_error": str(e),
                    "correct": False,
                    "max_diff": float("inf"),
                    "speedup": 0.0,
                }

        # Prepare worker arguments: (code, gpu_id, kernel_idx)
        subprocess_args = [
            (code, i % num_gpus, i)
            for i, (code, _, _, _, _) in enumerate(worker_args)
        ]

        # Run parallel verification using ThreadPoolExecutor
        # (threads for I/O, actual work in subprocesses)
        results_dicts = [None] * len(cuda_codes)
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(run_worker, args) for args in subprocess_args]

                completed = 0
                for future in as_completed(futures):
                    idx, result = future.result()
                    results_dicts[idx] = result
                    completed += 1

                    status = []
                    if result.get("compiled"):
                        status.append("compiled")
                    if result.get("correct"):
                        status.append("correct")
                    if result.get("speedup", 0) > 0:
                        status.append(f"speedup={result['speedup']:.2f}x")
                    log.info(f"  Kernel {idx}: {' | '.join(status) if status else 'failed'} ({completed}/{len(cuda_codes)})")

        except Exception as e:
            log.error(f"Parallel verification failed: {e}")
            import traceback
            traceback.print_exc()
            log.info("Falling back to sequential verification")
            # Cleanup
            import shutil
            shutil.rmtree(batch_dir, ignore_errors=True)
            return [self.verify(code, task) for code in cuda_codes]
        finally:
            # Cleanup temp directory
            import shutil
            shutil.rmtree(batch_dir, ignore_errors=True)

        # Convert dicts back to VerificationResult objects
        results = []
        for i, result_dict in enumerate(results_dicts):
            result = VerificationResult(
                compiled=result_dict.get("compiled", False),
                compile_error=result_dict.get("compile_error", ""),
                correct=result_dict.get("correct", False),
                max_diff=result_dict.get("max_diff", float("inf")),
                num_correct_trials=result_dict.get("num_correct_trials", 0),
                total_trials=result_dict.get("total_trials", self.num_trials),
                cuda_time_ms=result_dict.get("cuda_time_ms", float("inf")),
                torch_time_ms=result_dict.get("torch_time_ms", float("inf")),
                speedup=result_dict.get("speedup", 0.0),
                task_id=result_dict.get("task_id", ""),
                kernel_code=cuda_codes[i],
            )
            results.append(result)

        return results

    def _compile_kernel(
        self, cuda_code: str, task: Dict
    ) -> Tuple[Optional[Callable], str]:
        """Compile CUDA code to a callable function."""
        from torch.utils.cpp_extension import load

        # Write code to temp file
        task_id = task.get("task_id", "kernel")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cu", delete=False, dir=self.ext_dir
        ) as f:
            f.write(cuda_code)
            cuda_file = f.name

        try:
            # Find CUDA include paths (for cusparse.h, etc.)
            import sys
            cuda_include_paths = []
            # Check conda environment
            conda_prefix = os.environ.get("CONDA_PREFIX", "")
            if conda_prefix:
                # CUDA toolkit headers in conda
                cuda_target_include = os.path.join(conda_prefix, "targets/x86_64-linux/include")
                if os.path.exists(cuda_target_include):
                    cuda_include_paths.append(cuda_target_include)
                # nvidia package headers
                nvidia_cusparse = os.path.join(
                    conda_prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}",
                    "site-packages/nvidia/cusparse/include"
                )
                if os.path.exists(nvidia_cusparse):
                    cuda_include_paths.append(nvidia_cusparse)

            # Compile with torch cpp_extension
            module = load(
                name=f"cuda_{task_id}_{hash(cuda_code) % 10000}",
                sources=[cuda_file],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                extra_include_paths=cuda_include_paths if cuda_include_paths else None,
                with_cuda=True,
                verbose=False,
                build_directory=self.ext_dir,
            )

            # Get forward function
            if hasattr(module, "forward"):
                return module.forward, ""
            else:
                return None, "Compiled module has no 'forward' function"

        except Exception as e:
            return None, str(e)
        finally:
            # Cleanup temp file
            try:
                os.unlink(cuda_file)
            except Exception:
                pass

    def _load_pytorch_reference(
        self, task: Dict
    ) -> Tuple[type, Callable, Callable]:
        """Load PyTorch reference implementation from task."""
        pytorch_code = task["pytorch_code"]

        # Execute the code to get Model, forward_fn, get_inputs
        namespace = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional}
        exec(pytorch_code, namespace)

        model_cls = namespace.get("Model")
        forward_fn = namespace.get("forward_fn")
        get_inputs = namespace.get("get_inputs")

        if model_cls is None:
            raise ValueError("PyTorch code missing 'Model' class")
        if forward_fn is None:
            raise ValueError("PyTorch code missing 'forward_fn' function")
        if get_inputs is None:
            raise ValueError("PyTorch code missing 'get_inputs' function")

        return model_cls, forward_fn, get_inputs

    def _get_config_settings(self, task: Dict) -> Tuple[Dict, Dict]:
        """Get input and init settings from task config."""
        config = task.get("config", {})
        if isinstance(config, str):
            config = json.loads(config)

        # Use single settings for verification
        input_settings = {}
        init_settings = {}

        if "single_input_configs" in config and config["single_input_configs"]:
            input_settings = config["single_input_configs"][0]
        if "single_init_configs" in config and config["single_init_configs"]:
            init_settings = config["single_init_configs"][0]
        if "single_shared_configs" in config and config["single_shared_configs"]:
            # Merge shared configs into both
            shared = config["single_shared_configs"][0]
            input_settings = {**shared, **input_settings}
            init_settings = {**shared, **init_settings}

        return input_settings, init_settings

    def _get_default_settings_from_signature(self, get_inputs: Callable) -> Dict:
        """Extract default settings from get_inputs function signature."""
        import inspect
        sig = inspect.signature(get_inputs)
        defaults = {}
        for name, param in sig.parameters.items():
            if param.default is not inspect.Parameter.empty:
                defaults[name] = param.default
        return defaults

    def _check_correctness(
        self,
        cuda_fn: Callable,
        model_cls: type,
        forward_fn: Callable,
        get_inputs: Callable,
        task: Dict,
        device: str,
    ) -> Tuple[bool, int, float]:
        """Check correctness of CUDA kernel against PyTorch reference."""
        input_settings, init_settings = self._get_config_settings(task)

        # If no init settings, try to get defaults from get_inputs signature
        if not init_settings:
            defaults = self._get_default_settings_from_signature(get_inputs)
            # Filter to only params that Model.__init__ accepts
            import inspect
            model_sig = inspect.signature(model_cls.__init__)
            model_params = set(model_sig.parameters.keys()) - {"self"}
            init_settings = {k: v for k, v in defaults.items() if k in model_params}

        # Override with small inputs for faster testing
        if self.use_small_inputs:
            input_settings = {**input_settings, **self.small_input_settings}
            init_settings = {**init_settings, **{k: v for k, v in self.small_input_settings.items() if k in model_params}}

        num_correct = 0
        max_diff = 0.0

        with torch.no_grad():
            for trial in range(self.num_trials):
                # Set seed for reproducibility
                torch.manual_seed(42 + trial)

                # Create inputs
                inputs = get_inputs(**input_settings)
                inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

                # Create model instances
                torch.manual_seed(42 + trial)
                model_ref = model_cls(**init_settings).to(device)
                torch.manual_seed(42 + trial)
                model_cuda = model_cls(**init_settings).to(device)

                # Run PyTorch reference
                output_ref = model_ref(*inputs, fn=forward_fn)

                # Run CUDA kernel
                try:
                    output_cuda = model_cuda(*inputs, fn=cuda_fn)
                except Exception as e:
                    log.warning(f"CUDA kernel execution failed: {e}")
                    return False, 0, float("inf")

                # Compare outputs
                if torch.allclose(output_ref, output_cuda, rtol=self.rtol, atol=self.atol):
                    num_correct += 1
                else:
                    diff = float(torch.max(torch.abs(output_ref - output_cuda)))
                    max_diff = max(max_diff, diff)

                # Cleanup
                del model_ref, model_cuda, inputs, output_ref, output_cuda
                torch.cuda.empty_cache()

        correct = num_correct == self.num_trials
        return correct, num_correct, max_diff

    def _benchmark(
        self,
        cuda_fn: Callable,
        model_cls: type,
        forward_fn: Callable,
        get_inputs: Callable,
        task: Dict,
        device: str,
    ) -> Tuple[float, float]:
        """Benchmark CUDA kernel vs PyTorch reference."""
        input_settings, init_settings = self._get_config_settings(task)

        # If no init settings, try to get defaults from get_inputs signature
        if not init_settings:
            defaults = self._get_default_settings_from_signature(get_inputs)
            import inspect
            model_sig = inspect.signature(model_cls.__init__)
            model_params = set(model_sig.parameters.keys()) - {"self"}
            init_settings = {k: v for k, v in defaults.items() if k in model_params}

        # Override with small inputs for faster testing
        if self.use_small_inputs:
            input_settings = {**input_settings, **self.small_input_settings}
            init_settings = {**init_settings, **{k: v for k, v in self.small_input_settings.items() if k in model_params}}

        # Setup
        torch.manual_seed(42)
        inputs = get_inputs(**input_settings)
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

        torch.manual_seed(42)
        model = model_cls(**init_settings).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_iters):
                model(*inputs, fn=forward_fn)
                model(*inputs, fn=cuda_fn)
                torch.cuda.synchronize()

        # Benchmark PyTorch
        torch_times = []
        with torch.no_grad():
            for _ in range(self.benchmark_iters):
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
            for _ in range(self.benchmark_iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                model(*inputs, fn=cuda_fn)
                end.record()
                torch.cuda.synchronize()

                cuda_times.append(start.elapsed_time(end))

        # Cleanup
        del model, inputs
        torch.cuda.empty_cache()

        # Return median times
        import numpy as np
        return float(np.median(cuda_times)), float(np.median(torch_times))


def verify_kernel(
    cuda_code: str,
    task: Dict,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> VerificationResult:
    """
    Convenience function to verify a single kernel.

    Args:
        cuda_code: Generated CUDA kernel code
        task: Task dictionary from CUDABench
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        VerificationResult
    """
    verifier = CUDAVerifier(rtol=rtol, atol=atol)
    return verifier.verify(cuda_code, task)


if __name__ == "__main__":
    # Test the verifier
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from llm_tts.datasets.cudabench import load_cudabench

    logging.basicConfig(level=logging.INFO)

    # Load a task
    tasks = load_cudabench(source="local", category="robust_kbench", subset_size=1)
    task = tasks[0]

    print(f"\nTask: {task['task_id']}")
    print(f"Name: {task['task_name']}")

    # Test with a simple (likely incorrect) kernel
    test_kernel = '''
#include <torch/extension.h>

torch::Tensor forward(torch::Tensor x) {
    return x;  // Identity - will fail correctness
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Forward pass");
}
'''

    print("\nVerifying test kernel...")
    result = verify_kernel(test_kernel, task)

    print(f"\nResults:")
    print(f"  Compiled: {result.compiled}")
    print(f"  Compile error: {result.compile_error[:100] if result.compile_error else 'None'}")
    print(f"  Correct: {result.correct}")
    print(f"  Max diff: {result.max_diff}")
    print(f"  Speedup: {result.speedup:.2f}x")
