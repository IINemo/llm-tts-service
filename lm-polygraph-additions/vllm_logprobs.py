"""
Stat calculator for extracting logprobs from vLLM output.

Converts vLLM logprobs format to lm-polygraph format for use with
Perplexity, MeanTokenEntropy, and PDGap estimators.

Reference implementation (HuggingFace):
    lm_polygraph/stat_calculators/infer_causal_lm_calculator.py
    https://github.com/IINemo/lm-polygraph/blob/main/src/lm_polygraph/stat_calculators/infer_causal_lm_calculator.py
"""

from typing import Dict, List

import numpy as np

from .stat_calculator import StatCalculator


class VLLMLogprobsCalculator(StatCalculator):
    """
    Extracts greedy_log_likelihoods and greedy_log_probs from vLLM output.

    Args:
        output_matrix: If True, output greedy_log_probs as 2D matrix [T, K]
                      for PDGap estimator. If False (default), output as
                      list of 1D arrays for EntropyCalculator.

    Usage:
        stat_calculators = [VLLMLogprobsCalculator()]  # For Perplexity
        stat_calculators = [VLLMLogprobsCalculator(), EntropyCalculator()]  # For MeanTokenEntropy
        stat_calculators = [VLLMLogprobsCalculator(output_matrix=True)]  # For PDGap
    """

    def __init__(self, output_matrix: bool = False):
        super().__init__()
        self.output_matrix = output_matrix

    @staticmethod
    def meta_info():
        return ["greedy_log_likelihoods", "greedy_log_probs", "greedy_tokens"], ["vllm_output"]

    def __call__(self, dependencies: Dict, **kwargs) -> Dict[str, np.ndarray]:
        """
        Extract logprobs from vLLM output.

        Args:
            dependencies: Dict containing 'vllm_output' (vLLM CompletionOutput)

        Returns:
            Dict with:
            - greedy_log_likelihoods: [[log_likelihood per token]]
            - greedy_log_probs: format depends on output_matrix:
                - False: [[[log_probs per position]]] for EntropyCalculator
                - True: [2D array of shape [T, K]] for PDGap
            - greedy_tokens: [[token_ids]] - generated token IDs
        """
        output = dependencies["vllm_output"]
        token_ids = output.token_ids
        logprobs = output.logprobs

        if not logprobs or not token_ids:
            return {
                "greedy_log_likelihoods": [[]],
                "greedy_log_probs": [np.array([[]]) if self.output_matrix else []],
                "greedy_tokens": [[]],
            }

        # Extract log-likelihood for each chosen token
        log_likelihoods = []
        for token_id, logprob_dict in zip(token_ids, logprobs):
            if token_id in logprob_dict:
                log_likelihoods.append(logprob_dict[token_id].logprob)
            else:
                log_likelihoods.append(-100.0)

        if self.output_matrix:
            # Output as 2D matrix [T, K] for PDGap
            # K = number of logprobs per position (top_k or vocab_size)
            k = len(logprobs[0]) if logprobs else 0
            matrix = np.full((len(logprobs), k), -np.inf)
            for t, logprob_dict in enumerate(logprobs):
                for i, info in enumerate(logprob_dict.values()):
                    matrix[t, i] = info.logprob
            greedy_log_probs = [matrix]
        else:
            # Output as list of 1D arrays for EntropyCalculator
            greedy_log_probs = []
            for logprob_dict in logprobs:
                position_logprobs = np.array([info.logprob for info in logprob_dict.values()])
                greedy_log_probs.append(position_logprobs)
            greedy_log_probs = [greedy_log_probs]

        return {
            "greedy_log_likelihoods": [log_likelihoods],
            "greedy_log_probs": greedy_log_probs,
            "greedy_tokens": [list(token_ids)],
        }
