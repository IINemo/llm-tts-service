"""
CUDA Speedup Scorer for test-time scaling strategies.

Scores CUDA kernels based on verified speedup over PyTorch reference.
"""

import logging
from typing import Any, Dict, List, Optional

from llm_tts.evaluation.cuda_verifier import CUDAVerifier, VerificationResult

log = logging.getLogger(__name__)


class CUDASpeedupScorer:
    """
    Scorer that evaluates CUDA kernels by speedup over PyTorch reference.

    Returns:
        - 0.0 for kernels that fail compilation or correctness
        - speedup ratio for correct kernels (e.g., 1.5 means 1.5x faster)
    """

    def __init__(
        self,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        num_trials: int = 5,
        benchmark_iters: int = 100,
    ):
        """
        Initialize scorer.

        Args:
            rtol: Relative tolerance for correctness
            atol: Absolute tolerance for correctness
            num_trials: Number of correctness trials
            benchmark_iters: Iterations for benchmarking
        """
        self.verifier = CUDAVerifier(
            rtol=rtol,
            atol=atol,
            num_trials=num_trials,
            benchmark_iters=benchmark_iters,
        )

    def score(
        self,
        cuda_code: str,
        task: Dict,
        **kwargs,
    ) -> float:
        """
        Score a CUDA kernel.

        Args:
            cuda_code: Generated CUDA kernel code
            task: Task dictionary from CUDABench

        Returns:
            Speedup ratio (0.0 if failed)
        """
        result = self.verifier.verify(cuda_code, task)
        return result.score

    def score_batch(
        self,
        cuda_codes: List[str],
        task: Dict,
        **kwargs,
    ) -> List[float]:
        """
        Score multiple CUDA kernels for the same task.

        Args:
            cuda_codes: List of generated CUDA kernel codes
            task: Task dictionary

        Returns:
            List of speedup ratios
        """
        return [self.score(code, task) for code in cuda_codes]

    def verify_and_score(
        self,
        cuda_code: str,
        task: Dict,
        **kwargs,
    ) -> VerificationResult:
        """
        Full verification with detailed results.

        Args:
            cuda_code: Generated CUDA kernel code
            task: Task dictionary

        Returns:
            VerificationResult with all details
        """
        return self.verifier.verify(cuda_code, task)

    def select_best(
        self,
        cuda_codes: List[str],
        task: Dict,
        return_all: bool = False,
    ) -> Dict[str, Any]:
        """
        Select the best kernel from a list of candidates.

        Args:
            cuda_codes: List of CUDA kernel candidates
            task: Task dictionary
            return_all: If True, return all results

        Returns:
            Dict with best kernel and optionally all results
        """
        results = []
        for i, code in enumerate(cuda_codes):
            result = self.verifier.verify(code, task)
            results.append({
                "index": i,
                "result": result,
                "score": result.score,
            })

        # Sort by score (speedup)
        results.sort(key=lambda x: x["score"], reverse=True)

        # Best kernel
        best = results[0] if results else None

        output = {
            "best_index": best["index"] if best else -1,
            "best_score": best["score"] if best else 0.0,
            "best_kernel": cuda_codes[best["index"]] if best and best["score"] > 0 else None,
            "best_result": best["result"].to_dict() if best else None,
            "num_passed": sum(1 for r in results if r["score"] > 0),
            "num_candidates": len(cuda_codes),
        }

        if return_all:
            output["all_results"] = [
                {"index": r["index"], "score": r["score"], "passed": r["result"].passed}
                for r in results
            ]

        return output


class CUDABinaryScorer:
    """
    Simple binary scorer - returns 1.0 if kernel passes, 0.0 otherwise.

    Useful for self-consistency voting where we just need correct/incorrect.
    """

    def __init__(
        self,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        num_trials: int = 5,
    ):
        self.verifier = CUDAVerifier(
            rtol=rtol,
            atol=atol,
            num_trials=num_trials,
            benchmark_iters=10,  # Minimal benchmarking
        )

    def score(self, cuda_code: str, task: Dict) -> float:
        """Returns 1.0 if correct, 0.0 otherwise."""
        result = self.verifier.verify(cuda_code, task)
        return 1.0 if result.passed else 0.0
