"""
Best-of-N CUDA kernel selection strategy.

Generates N kernel candidates and selects the best based on:
1. Correctness (must compile and produce correct output)
2. Speedup (faster is better)

This is the proper approach for CUDA kernel generation as described
in the CUDABench/robust-kbench paper.
"""

import logging
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from llm_tts.evaluation.cuda_verifier import CUDAVerifier
from llm_tts.strategies.strategy_self_consistency import StrategySelfConsistency

log = logging.getLogger(__name__)


def extract_cuda_code(response: str) -> str:
    """Extract CUDA code from LLM response."""
    # Try ```cuda ... ```
    match = re.search(r"```cuda\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try ```c++ ... ```
    match = re.search(r"```c\+\+\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try ```cpp ... ```
    match = re.search(r"```cpp\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try ``` ... ``` (generic code block)
    match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        # Check if it looks like CUDA/C++ code
        if "#include" in code or "__global__" in code:
            return code

    return ""


class BestOfNCUDAStrategy(StrategySelfConsistency):
    """
    Best-of-N strategy for CUDA kernel generation.

    Inherits generation logic from SelfConsistencyStrategy but uses
    CUDA verification for selection instead of majority voting.
    """

    def __init__(
        self,
        model: Any,
        num_paths: int = 8,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        generation_batch_size: Optional[int] = None,
        n_threads: int = 1,
        # CUDA verifier settings
        rtol: float = 1e-5,
        atol: float = 1e-5,
        num_trials: int = 1,
        warmup_iters: int = 5,
        benchmark_iters: int = 10,
        use_small_inputs: bool = True,
        parallel_verification: bool = True,  # Enable parallel verification
        # Task context (set per-sample)
        pytorch_code: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Best-of-N CUDA strategy.

        Args:
            model: Language model for generation
            num_paths: Number of kernel candidates to generate
            max_new_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            generation_batch_size: Batch size for generation
            n_threads: Number of threads for parallel generation
            rtol: Relative tolerance for correctness
            atol: Absolute tolerance for correctness
            num_trials: Number of correctness trials
            warmup_iters: Warmup iterations for benchmarking
            benchmark_iters: Benchmark iterations
            use_small_inputs: Use small tensors for fast testing
            parallel_verification: Use multiprocessing for verification
            pytorch_code: PyTorch reference code (set per-sample)
        """
        # Initialize parent with a dummy scorer (we'll override selection)
        super().__init__(
            model=model,
            num_paths=num_paths,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            generation_batch_size=generation_batch_size,
            n_threads=n_threads,
            **kwargs,
        )

        # CUDA verifier for scoring kernels
        self.verifier = CUDAVerifier(
            rtol=rtol,
            atol=atol,
            num_trials=num_trials,
            warmup_iters=warmup_iters,
            benchmark_iters=benchmark_iters,
            use_small_inputs=use_small_inputs,
        )

        # Parallel verification flag
        self.parallel_verification = parallel_verification

        # Current task context (set before generation)
        self.pytorch_code = pytorch_code

    def set_task_context(self, pytorch_code: str):
        """Set the PyTorch reference code for the current task."""
        self.pytorch_code = pytorch_code

    def select_best_answer(self, reasoning_paths: List) -> Dict[str, Any]:
        """
        Select the best kernel using CUDA verification.

        Instead of majority voting on extracted answers, we:
        1. Extract CUDA code from each path
        2. Verify each kernel (compile + correctness)
        3. Select the best based on correctness and speedup

        Args:
            reasoning_paths: List of generated responses

        Returns:
            Dictionary with best kernel and verification results
        """
        if not reasoning_paths:
            return {
                "best_path": "",
                "best_answer": "no_answer",
                "consensus_score": 0.0,
                "all_answers": [],
                "answer_distribution": {},
                "all_traces": [],
            }

        # Handle both string and dict formats
        path_texts = []
        path_tokens = []
        for p in reasoning_paths:
            if isinstance(p, dict):
                path_texts.append(p.get("text", ""))
                path_tokens.append(p.get("num_tokens", 0))
            else:
                path_texts.append(p)
                path_tokens.append(0)

        # Extract CUDA code from each path
        cuda_codes = [extract_cuda_code(text) for text in path_texts]

        # Build task for verification
        task = {
            "task_id": "best_of_n_task",
            "task_name": "Best-of-N Selection",
            "pytorch_code": self.pytorch_code or "",
            "config": {},
        }

        # Verify kernels (parallel or sequential)
        valid_codes = [(i, code) for i, code in enumerate(cuda_codes) if code]
        log.info(f"Verifying {len(valid_codes)} kernel candidates (of {len(cuda_codes)} total)...")

        verification_results = [None] * len(cuda_codes)
        scores = [0.0] * len(cuda_codes)

        if self.parallel_verification and len(valid_codes) > 1:
            # Parallel verification using multiprocessing
            log.info("Using parallel verification...")
            valid_indices, valid_code_list = zip(*valid_codes) if valid_codes else ([], [])

            parallel_results = self.verifier.verify_batch_parallel(
                list(valid_code_list), task
            )

            # Map results back to original indices
            for idx, result in zip(valid_indices, parallel_results):
                verification_results[idx] = result
                score = result.speedup if result.passed else 0.0
                scores[idx] = score

                status = []
                if result.compiled:
                    status.append("compiled")
                if result.correct:
                    status.append("correct")
                if result.speedup > 0:
                    status.append(f"speedup={result.speedup:.2f}x")
                log.info(f"  Kernel {idx}: {' | '.join(status) if status else 'failed'}")
        else:
            # Sequential verification
            for i, code in enumerate(cuda_codes):
                if not code:
                    log.info(f"  Kernel {i}: no code extracted")
                    continue

                try:
                    result = self.verifier.verify(code, task)
                    verification_results[i] = result

                    # Score: 0 if failed, speedup if passed
                    score = result.speedup if result.passed else 0.0
                    scores[i] = score

                    status = []
                    if result.compiled:
                        status.append("compiled")
                    if result.correct:
                        status.append("correct")
                    if result.speedup > 0:
                        status.append(f"speedup={result.speedup:.2f}x")

                    log.info(f"  Kernel {i}: {' | '.join(status) if status else 'failed'}")

                except Exception as e:
                    log.warning(f"  Kernel {i}: verification error - {e}")

        # Select best kernel
        scores_array = np.array(scores)
        best_idx = int(np.argmax(scores_array))
        best_score = scores[best_idx]

        # Count passed kernels
        num_passed = sum(1 for s in scores if s > 0)
        num_compiled = sum(
            1 for r in verification_results
            if r is not None and r.compiled
        )
        num_correct = sum(
            1 for r in verification_results
            if r is not None and r.correct
        )

        log.info(f"Selection results:")
        log.info(f"  Compiled: {num_compiled}/{len(cuda_codes)}")
        log.info(f"  Correct: {num_correct}/{len(cuda_codes)}")
        log.info(f"  Best kernel: {best_idx}")
        log.info(f"  Best speedup: {best_score:.2f}x {'(faster than PyTorch)' if best_score > 1 else '(slower than PyTorch)'}")

        # Build answer distribution (by correctness status)
        status_counts = Counter()
        for r in verification_results:
            if r is None:
                status_counts["no_code"] += 1
            elif not r.compiled:
                status_counts["compile_failed"] += 1
            elif not r.correct:
                status_counts["incorrect"] += 1
            else:
                status_counts["correct"] += 1

        # Build all_traces with verification info
        all_traces = []
        for i, (text, tokens, code, result) in enumerate(
            zip(path_texts, path_tokens, cuda_codes, verification_results)
        ):
            trace_info = {
                "text": text,
                "num_tokens": tokens,
                "cuda_code": code[:500] + "..." if len(code) > 500 else code,
                "score": float(scores[i]),
                "selected": i == best_idx,
            }
            if result is not None:
                trace_info["compiled"] = result.compiled
                trace_info["correct"] = result.correct
                trace_info["speedup"] = result.speedup
            all_traces.append(trace_info)

        total_tokens = sum(path_tokens)

        return {
            "best_path": path_texts[best_idx],
            "best_answer": cuda_codes[best_idx] if best_score > 0 else "no_valid_kernel",
            "consensus_score": best_score,  # Parent class expects this; for CUDA it's speedup (see best_speedup)
            "all_answers": cuda_codes,
            "answer_distribution": dict(status_counts),
            "all_paths": path_texts,
            "all_scores": list(scores),
            "all_traces": all_traces,
            "total_tokens": total_tokens,
            # CUDA-specific metrics
            "num_compiled": num_compiled,
            "num_correct": num_correct,
            "num_passed": num_passed,
            "best_speedup": best_score,
        }


def create_best_of_n_cuda_strategy(
    model: Any,
    config: Dict[str, Any],
) -> BestOfNCUDAStrategy:
    """
    Factory function to create BestOfNCUDAStrategy from config.

    Args:
        model: Language model
        config: Strategy configuration

    Returns:
        Configured BestOfNCUDAStrategy
    """
    return BestOfNCUDAStrategy(
        model=model,
        num_paths=config.get("num_paths", 8),
        max_new_tokens=config.get("max_new_tokens", 4096),
        temperature=config.get("temperature", 0.7),
        generation_batch_size=config.get("generation_batch_size"),
        n_threads=config.get("n_threads", 1),
        rtol=config.get("rtol", 1e-5),
        atol=config.get("atol", 1e-5),
        num_trials=config.get("num_trials", 1),
        warmup_iters=config.get("warmup_iters", 5),
        benchmark_iters=config.get("benchmark_iters", 10),
        use_small_inputs=config.get("use_small_inputs", True),
        parallel_verification=config.get("parallel_verification", True),
    )
