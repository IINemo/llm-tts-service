"""
Metadata builder for consistent logging across all TTS strategies.

This builder ensures all strategies log common metrics in a consistent format,
while allowing each strategy to add its own specific metrics.
"""

import logging
from typing import Any, Dict, List

import numpy as np


class StrategyMetadataBuilder:
    """
    Builder for constructing consistent metadata across all TTS strategies.

    Usage:
        builder = StrategyMetadataBuilder("self_consistency")
        builder.add_config(temperature=0.7, num_paths=5)
        builder.add_results(selected_answer="42", consensus_score=0.8)
        builder.add_generation_details(all_paths=[...], all_scores=[...])

        metadata = builder.build()
        builder.log_summary(logger)
    """

    def __init__(self, strategy_name: str):
        """
        Initialize builder with strategy name.

        Args:
            strategy_name: Name of the strategy (e.g., "self_consistency", "deepconf")
        """
        self.strategy_name = strategy_name
        self.config = {}
        self.results = {}
        self.generation_details = {}
        self.strategy_specific = {}

    def add_config(self, **kwargs) -> "StrategyMetadataBuilder":
        """
        Add configuration parameters (for reproducibility).

        Common examples:
            - temperature: Sampling temperature
            - max_new_tokens: Maximum tokens to generate
            - n_threads: Number of parallel threads
            - budget: Total computation budget

        Args:
            **kwargs: Configuration key-value pairs

        Returns:
            Self for method chaining
        """
        self.config.update(kwargs)
        return self

    def add_results(self, **kwargs) -> "StrategyMetadataBuilder":
        """
        Add high-level results (for quick analysis).

        Common examples:
            - selected_answer: The final answer selected by the strategy
            - confidence_score / consensus_score: Quality metric
            - answer_distribution: Distribution of answers across samples

        Args:
            **kwargs: Result key-value pairs

        Returns:
            Self for method chaining
        """
        self.results.update(kwargs)
        return self

    def add_generation_details(self, **kwargs) -> "StrategyMetadataBuilder":
        """
        Add detailed generation data (for deep analysis).

        Common examples:
            - all_paths / all_traces: Full text of all generated samples
            - all_scores: Score for each generated sample
            - all_answers: Extracted answer from each sample
            - path_summaries / trace_summaries: Summary stats per sample

        Args:
            **kwargs: Generation detail key-value pairs

        Returns:
            Self for method chaining
        """
        # Convert numpy arrays to lists for JSON serialization
        processed = {}
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                processed[key] = value.tolist()
            elif isinstance(value, list) and value and hasattr(value[0], "tolist"):
                processed[key] = [
                    v.tolist() if hasattr(v, "tolist") else v for v in value
                ]
            else:
                processed[key] = value

        self.generation_details.update(processed)
        return self

    def add_strategy_specific(self, **kwargs) -> "StrategyMetadataBuilder":
        """
        Add strategy-specific metrics not common to all strategies.

        Examples:
            - DeepConf: min_conf, mean_conf, token_data, filtered_traces
            - Best-of-N: step_scores, prm_scores
            - Self-Consistency: consensus_scores

        Args:
            **kwargs: Strategy-specific key-value pairs

        Returns:
            Self for method chaining
        """
        self.strategy_specific.update(kwargs)
        return self

    def build(self) -> Dict[str, Any]:
        """
        Build the final metadata dictionary with consistent structure.

        Returns:
            Structured metadata dictionary
        """
        metadata = {
            "strategy": self.strategy_name,
        }

        # Add sections that have data
        if self.config:
            metadata["config"] = self.config

        if self.results:
            metadata["results"] = self.results

        if self.generation_details:
            metadata["generation_details"] = self.generation_details

        if self.strategy_specific:
            metadata["strategy_specific"] = self.strategy_specific

        return metadata

    def log_summary(
        self, logger: logging.Logger, include_details: bool = False
    ) -> None:
        """
        Log a summary to console for real-time monitoring.

        Args:
            logger: Logger instance to use
            include_details: Whether to log detailed generation info
        """
        logger.info(f"{self.strategy_name} summary:")

        # Log configuration
        if self.config:
            logger.info("  Configuration:")
            for key, value in self.config.items():
                logger.info(f"    {key}: {value}")

        # Log results
        if self.results:
            logger.info("  Results:")
            for key, value in self.results.items():
                # Format based on type
                if isinstance(value, float):
                    logger.info(f"    {key}: {value:.3f}")
                elif isinstance(value, dict):
                    logger.info(f"    {key}: {value}")
                else:
                    logger.info(f"    {key}: {value}")

        # Optionally log generation details (can be verbose)
        if include_details and self.generation_details:
            logger.info("  Generation details:")
            for key, value in self.generation_details.items():
                if isinstance(value, list):
                    logger.info(f"    {key}: {len(value)} items")
                else:
                    logger.info(f"    {key}: {value}")

    def create_path_summaries(
        self,
        paths: List[str],
        scores: List[float],
        answers: List[str],
        selected_index: int,
    ) -> List[Dict[str, Any]]:
        """
        Helper to create consistent path/trace summaries.

        Args:
            paths: List of generated paths/traces
            scores: Score for each path
            answers: Extracted answer from each path
            selected_index: Index of the selected path

        Returns:
            List of summary dictionaries for each path
        """
        summaries = []
        for i, (path, score, answer) in enumerate(zip(paths, scores, answers)):
            summaries.append(
                {
                    "index": i,
                    "score": float(score),
                    "answer": answer,
                    "selected": i == selected_index,
                    "text_length": len(path),
                }
            )
        return summaries
