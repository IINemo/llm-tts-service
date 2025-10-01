"""
Deep Think with Confidence (DeepConf) strategy implementation.

Based on the paper "Deep Think with Confidence" - implements confidence-based
filtering and weighted voting for improved reasoning efficiency and accuracy.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ..confidence_scoring import (
    create_confidence_scorer,
    confidence_weighted_voting,
    HybridConfidenceScorer,
    score_trace_from_token_data
)

log = logging.getLogger(__name__)


class DeepThinkConfidenceStrategy:
    """
    DeepConf strategy that uses confidence signals to filter reasoning traces
    and perform weighted voting for improved accuracy and efficiency.
    """

    def __init__(
        self,
        model,
        num_paths: int = 10,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        confidence_threshold: float = 0.3,
        filtering_percentage: float = 0.7,  # Keep top 70% of traces (η parameter)
        confidence_metric: str = "avg_confidence",
        early_termination: bool = False,
        early_termination_threshold: float = 0.2,
        mode: str = "offline",  # "offline" or "online"
        warmup_traces: int = 16,  # Ninit parameter from paper
        use_warmup_threshold: bool = True
    ):
        """
        Initialize DeepConf strategy.

        Args:
            model: Language model for generation
            num_paths: Number of reasoning paths to generate
            max_new_tokens: Maximum tokens per reasoning path
            temperature: Sampling temperature for diversity
            confidence_threshold: Minimum confidence for trace acceptance
            filtering_percentage: Percentage of top traces to keep (η)
            confidence_metric: Which confidence metric to use for filtering
            early_termination: Whether to use early termination during generation
            early_termination_threshold: Confidence threshold for early termination
            mode: "offline" (filter after generation) or "online" (filter during)
            warmup_traces: Number of initial traces for threshold calibration (Ninit)
            use_warmup_threshold: Whether to use warmup-based threshold calibration
        """
        self.model = model
        self.num_paths = num_paths
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.filtering_percentage = filtering_percentage
        self.confidence_metric = confidence_metric
        self.early_termination = early_termination
        self.early_termination_threshold = early_termination_threshold
        self.mode = mode
        self.warmup_traces = warmup_traces
        self.use_warmup_threshold = use_warmup_threshold

        # Check if model supports real logprobs (OpenRouter with compatible models)
        self.supports_logprobs = hasattr(model, 'supports_logprobs') and model.supports_logprobs()
        self.supports_confidence_generation = hasattr(model, 'generate_with_confidence')

        if self.supports_logprobs and self.supports_confidence_generation:
            log.info("✅ Model supports logprobs - using real token probabilities for DeepConf")
            # For models with logprobs, we'll compute confidence from token data directly
            self.confidence_scorer = None  # Will use score_trace_from_token_data instead
        elif hasattr(model, 'api_model') or model.device == "api":
            log.info("⚠️  Using API-based heuristic confidence scoring")
            self.confidence_scorer = create_confidence_scorer("api")
        else:
            log.info("Using hybrid confidence scoring")
            self.confidence_scorer = create_confidence_scorer("hybrid")

    def warmup_phase(self, prompt: str) -> float:
        """
        Warmup phase to calibrate confidence threshold.

        From DeepConf paper: Generate Ninit = 16 traces to establish confidence threshold
        s = Percentile_{100-η}({Ct : t ∈ T_warmup})

        Args:
            prompt: Input prompt for reasoning

        Returns:
            Calibrated confidence threshold
        """
        log.info(f"\n{'='*60}")
        log.info(f"🔥 WARMUP PHASE - Calibrating confidence threshold")
        log.info(f"{'='*60}")
        log.info(f"Generating {self.warmup_traces} warmup traces...")

        warmup_paths = []
        warmup_confidences = []

        for i in range(self.warmup_traces):
            log.info(f"🧠 WARMUP TRACE {i+1}/{self.warmup_traces}")

            # Generate single reasoning path with confidence data if available
            if self.supports_logprobs and self.supports_confidence_generation:
                # Use generate_with_confidence for models with logprobs
                response, token_data = self.model.generate_with_confidence(
                    prompt=prompt,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature
                )
                # Calculate confidence from real token probabilities
                confidence_scores = score_trace_from_token_data(token_data)
                trace_confidence = confidence_scores.get(self.confidence_metric, 0.5)
            elif hasattr(self.model, 'api_model') or self.model.device == "api":
                response = self.model.generate(
                    prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    num_return_sequences=1
                )[0]
                # Score using heuristic confidence scorer
                confidence_scores = self.confidence_scorer.score_trace(response)
                trace_confidence = confidence_scores.get(self.confidence_metric, 0.5)
            else:
                # For local models - simplified generation
                response = f"Warmup reasoning path {i+1}: solving step by step..."
                confidence_scores = self.confidence_scorer.score_trace(response)
                trace_confidence = confidence_scores.get(self.confidence_metric, 0.5)

            warmup_paths.append({
                "trace": response,
                "confidence_scores": confidence_scores,
                "confidence": trace_confidence
            })
            warmup_confidences.append(trace_confidence)

            log.info(f"  Trace confidence ({self.confidence_metric}): {trace_confidence:.3f}")

        # Calculate threshold using percentile method from paper
        # s = Percentile_{100-η}({Ct : t ∈ T_warmup})
        percentile = 100 - (self.filtering_percentage * 100)  # η=0.7 → 30th percentile
        calibrated_threshold = np.percentile(warmup_confidences, percentile)

        log.info(f"\n📊 Warmup calibration results:")
        log.info(f"  • Warmup traces generated: {len(warmup_paths)}")
        log.info(f"  • Confidence range: {min(warmup_confidences):.3f} - {max(warmup_confidences):.3f}")
        log.info(f"  • Average confidence: {np.mean(warmup_confidences):.3f}")
        log.info(f"  • Target percentile: {percentile:.1f}%")
        log.info(f"  • Calibrated threshold: {calibrated_threshold:.3f}")
        log.info(f"  • Original threshold: {self.confidence_threshold:.3f}")

        return calibrated_threshold

    def generate_reasoning_paths(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Generate reasoning paths with confidence scoring.

        Returns:
            List of dictionaries containing trace text and confidence scores
        """
        log.info(f"\n{'='*80}")
        log.info(f"🧠 DEEP THINK WITH CONFIDENCE - GENERATING {self.num_paths} PATHS")
        log.info(f"{'='*80}")
        log.info(f"🎛️  Configuration:")
        log.info(f"   • Temperature: {self.temperature}")
        log.info(f"   • Filtering: Keep top {self.filtering_percentage*100:.0f}% of traces")
        log.info(f"   • Confidence metric: {self.confidence_metric}")
        log.info(f"   • Mode: {self.mode}")

        all_paths_data = []

        for i in range(self.num_paths):
            log.info(f"\n{'-'*60}")
            log.info(f"🧠 REASONING PATH {i+1}/{self.num_paths}")
            log.info(f"{'-'*60}")

            try:
                # Generate reasoning path
                if hasattr(self.model, 'api_model') or self.model.device == "api":
                    path_data = self._generate_api_path(prompt, i+1)
                else:
                    path_data = self._generate_local_path(prompt, i+1)

                all_paths_data.append(path_data)

            except Exception as e:
                log.warning(f"❌ Failed to generate path {i+1}: {e}")
                # Add fallback path
                fallback_data = {
                    "trace": prompt + f" [Generation failed: {str(e)}]",
                    "confidence_scores": {"avg_confidence": 0.1, "reasoning_quality": 0.1},
                    "tokens_generated": 0,
                    "generation_successful": False
                }
                all_paths_data.append(fallback_data)

        log.info(f"\n✅ Generated {len(all_paths_data)} reasoning paths")
        return all_paths_data

    def _generate_api_path(self, prompt: str, path_num: int) -> Dict[str, Any]:
        """Generate a single path using API model"""
        # Check if model supports logprobs and use real token confidence
        if self.supports_logprobs and self.supports_confidence_generation:
            # Generate with token probabilities
            generated_text, token_data = self.model.generate_with_confidence(
                prompt=prompt,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature
            )
            full_trace = prompt + generated_text

            # Calculate confidence from real token probabilities
            confidence_scores = score_trace_from_token_data(token_data)

            # Log detailed reasoning
            self._log_path_details(path_num, generated_text, confidence_scores)

            return {
                "trace": full_trace,
                "confidence_scores": confidence_scores,
                "tokens_generated": len(token_data),  # Actual token count
                "generation_successful": True
            }
        else:
            # Fallback to regular generation without logprobs
            if hasattr(self.model, 'generate'):
                # Direct API model
                completions = self.model.generate(
                    prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    num_return_sequences=1
                )
                generated_text = completions[0] if completions else ""
            else:
                # API model wrapped in adapter
                completions = self.model.api_model.generate(
                    prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    num_return_sequences=1
                )
                generated_text = completions[0] if completions else ""

            full_trace = prompt + generated_text

            # Calculate confidence scores using heuristic scorer
            confidence_scores = self.confidence_scorer.score_trace(full_trace)

            # Log detailed reasoning
            self._log_path_details(path_num, generated_text, confidence_scores)

            return {
                "trace": full_trace,
                "confidence_scores": confidence_scores,
                "tokens_generated": len(generated_text.split()),  # Approximate
                "generation_successful": True
            }

    def _generate_local_path(self, prompt: str, path_num: int) -> Dict[str, Any]:
        """Generate a single path using local model with logits access"""
        # This would be implemented for local models with logits access
        # For now, fallback to API-style generation
        log.debug("Local model generation not fully implemented, using API-style")
        return self._generate_api_path(prompt, path_num)

    def _log_path_details(self, path_num: int, generated_text: str, confidence_scores: Dict[str, float]):
        """Log detailed information about the generated path"""
        log.info(f"📝 Generated reasoning:")
        reasoning_lines = generated_text.strip().split('\n')
        for line_num, line in enumerate(reasoning_lines, 1):  # Show all lines
            if line.strip():
                log.info(f"  {line_num:2d}: {line.strip()}")

        log.info(f"🎯 Confidence scores:")
        for metric, score in confidence_scores.items():
            if isinstance(score, (int, float)):
                log.info(f"  • {metric}: {score:.3f}")

    def filter_traces_by_confidence(self, paths_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter traces based on confidence scores.

        Args:
            paths_data: List of path dictionaries with traces and confidence scores

        Returns:
            Filtered list of high-confidence paths
        """
        log.info(f"\n{'='*60}")
        log.info(f"🔍 CONFIDENCE-BASED FILTERING")
        log.info(f"{'='*60}")

        if not paths_data:
            return []

        # Extract confidence scores for the chosen metric
        confidences = []
        for i, path_data in enumerate(paths_data):
            conf_score = path_data["confidence_scores"].get(self.confidence_metric, 0.5)
            confidences.append((i, conf_score))

        # Log all confidence scores
        log.info(f"📊 Confidence scores ({self.confidence_metric}):")
        for i, (path_idx, conf) in enumerate(confidences):
            log.info(f"  Path {path_idx+1}: {conf:.3f}")

        # Sort by confidence (highest first)
        confidences.sort(key=lambda x: x[1], reverse=True)

        # Apply filtering strategies
        filtered_indices = self._apply_filtering_strategy(confidences)

        # Select filtered paths
        filtered_paths = [paths_data[i] for i in filtered_indices]

        log.info(f"\n🎯 Filtering results:")
        log.info(f"  • Original paths: {len(paths_data)}")
        log.info(f"  • Filtered paths: {len(filtered_paths)}")
        log.info(f"  • Kept paths: {[i+1 for i in filtered_indices]}")

        return filtered_paths

    def _apply_filtering_strategy(self, confidences: List[Tuple[int, float]]) -> List[int]:
        """Apply the filtering strategy to select high-confidence traces"""
        # Strategy 1: Top percentage filtering (η parameter from paper)
        num_to_keep = max(1, int(len(confidences) * self.filtering_percentage))
        top_percentage_indices = [idx for idx, _ in confidences[:num_to_keep]]

        # Strategy 2: Absolute threshold filtering
        threshold_indices = [idx for idx, conf in confidences if conf >= self.confidence_threshold]

        # Use intersection of both strategies (more restrictive)
        if threshold_indices:
            filtered_indices = list(set(top_percentage_indices) & set(threshold_indices))
        else:
            # If no traces meet threshold, fall back to top percentage
            filtered_indices = top_percentage_indices

        # Ensure we keep at least one trace
        if not filtered_indices and confidences:
            filtered_indices = [confidences[0][0]]  # Keep the highest confidence trace

        return filtered_indices

    def confidence_weighted_voting(self, filtered_paths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform confidence-weighted majority voting on filtered traces.

        Args:
            filtered_paths: List of filtered path dictionaries

        Returns:
            Voting results with selected answer and metadata
        """
        log.info(f"\n{'='*60}")
        log.info(f"🗳️  CONFIDENCE-WEIGHTED VOTING")
        log.info(f"{'='*60}")

        if not filtered_paths:
            return {
                "selected_answer": "no_answer",
                "selected_trace": "",
                "confidence_score": 0.0,
                "vote_distribution": {},
                "traces_used": 0
            }

        # Extract traces and confidence scores
        traces = [path_data["trace"] for path_data in filtered_paths]
        confidence_scores = [path_data["confidence_scores"] for path_data in filtered_paths]

        # Perform weighted voting
        voting_result = confidence_weighted_voting(
            traces=traces,
            confidence_scores=confidence_scores,
            confidence_metric=self.confidence_metric,
            answer_extractor=self._extract_answer
        )

        # Find the trace that gave the winning answer
        selected_trace = ""
        for trace_info in voting_result["trace_info"]:
            if trace_info["answer"] == voting_result["selected_answer"]:
                selected_trace = traces[trace_info["trace_idx"]]
                break

        log.info(f"📊 Voting analysis:")
        log.info(f"  • Traces voted: {len(traces)}")
        log.info(f"  • Selected answer: '{voting_result['selected_answer']}'")
        log.info(f"  • Decision confidence: {voting_result['confidence_score']:.3f}")

        log.info(f"\n📈 Vote distribution:")
        for answer, weight in voting_result["vote_distribution"].items():
            percentage = (weight / voting_result["total_weight"]) * 100
            log.info(f"  • '{answer}': {weight:.2f} ({percentage:.1f}%)")

        return {
            "selected_answer": voting_result["selected_answer"],
            "selected_trace": selected_trace,
            "confidence_score": voting_result["confidence_score"],
            "vote_distribution": voting_result["vote_distribution"],
            "traces_used": len(traces),
            "voting_metadata": voting_result
        }

    def _extract_answer(self, trace: str) -> str:
        """Extract answer from trace, matching DeepConf paper format"""
        import re

        # Primary pattern from DeepConf paper
        match = re.search(r"the final answer is\s*(\d+)", trace.lower())
        if match:
            return match.group(1)

        # Fallback patterns
        patterns = [
            r"final answer:\s*(\d+)",
            r"answer:\s*(\d+)",
            r"therefore,?.*?(\d+)\s*\.?\s*$",
            r"(?:equals?|=)\s*(\d+)(?:\s*\.?\s*$)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, trace.lower())
            if matches:
                return matches[-1]

        # Last resort: extract last number
        numbers = re.findall(r'\d+', trace)
        if numbers:
            return numbers[-1]

        return "unknown"

    def generate_trajectory(self, prompt: str) -> Dict[str, Any]:
        """
        Main entry point for DeepConf reasoning.

        Args:
            prompt: Input prompt/question

        Returns:
            Dictionary with trajectory information compatible with evaluation framework
        """
        log.info(f"🚀 Starting DeepConf reasoning for prompt: {prompt[:100]}...")

        # Warmup phase for threshold calibration (if enabled)
        original_threshold = self.confidence_threshold
        if self.use_warmup_threshold:
            calibrated_threshold = self.warmup_phase(prompt)
            # Use calibrated threshold for filtering
            self.confidence_threshold = calibrated_threshold
            log.info(f"Using calibrated threshold: {calibrated_threshold:.3f}")
        else:
            log.info(f"Skipping warmup, using fixed threshold: {self.confidence_threshold:.3f}")

        # Generate multiple reasoning paths with confidence scoring
        paths_data = self.generate_reasoning_paths(prompt)

        # Filter paths based on confidence
        filtered_paths = self.filter_traces_by_confidence(paths_data)

        # Perform confidence-weighted voting
        voting_result = self.confidence_weighted_voting(filtered_paths)

        # Calculate efficiency metrics
        total_tokens_generated = sum(p["tokens_generated"] for p in paths_data)
        tokens_in_selected = len(voting_result["selected_trace"].split()) if voting_result["selected_trace"] else 0

        # Determine confidence level
        confidence_score = voting_result["confidence_score"]
        if confidence_score >= 0.8:
            confidence_level = "Very High"
        elif confidence_score >= 0.6:
            confidence_level = "High"
        elif confidence_score >= 0.4:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

        log.info(f"\n🏆 DEEPCONF FINAL RESULTS:")
        log.info(f"  • Selected answer: '{voting_result['selected_answer']}'")
        log.info(f"  • Decision confidence: {confidence_score:.3f} (proportion of votes)")
        log.info(f"  • Confidence level: {confidence_level}")
        log.info(f"  • Paths generated: {len(paths_data)}")
        log.info(f"  • Paths used for voting: {voting_result['traces_used']}")
        log.info(f"  • Efficiency: Used {voting_result['traces_used']}/{len(paths_data)} traces")

        # Restore original threshold
        if self.use_warmup_threshold:
            self.confidence_threshold = original_threshold

        return {
            "trajectory": voting_result["selected_trace"],
            "steps": [voting_result["selected_trace"]],
            "completed": True,
            "strategy": "deep_think_confidence",
            "metadata": {
                "num_paths_generated": len(paths_data),
                "num_paths_used": voting_result["traces_used"],
                "selected_answer": voting_result["selected_answer"],
                "confidence_score": confidence_score,
                "confidence_level": confidence_level,
                "vote_distribution": voting_result["vote_distribution"],
                "filtering_percentage": self.filtering_percentage,
                "confidence_metric": self.confidence_metric,
                "total_tokens_generated": total_tokens_generated,
                "efficiency_ratio": voting_result["traces_used"] / len(paths_data),
                "all_confidence_scores": [p["confidence_scores"] for p in paths_data],
                "warmup_enabled": self.use_warmup_threshold,
                "warmup_traces": self.warmup_traces,
                "original_threshold": original_threshold,
                "final_threshold": self.confidence_threshold
            }
        }

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self.model, 'cleanup'):
            self.model.cleanup()


# Factory function for easy creation
def create_deepconf_strategy(
    model,
    num_paths: int = 10,
    filtering_percentage: float = 0.7,
    confidence_metric: str = "avg_confidence",
    **kwargs
) -> DeepThinkConfidenceStrategy:
    """
    Factory function to create a DeepConf strategy.

    Args:
        model: Language model
        num_paths: Number of reasoning paths to generate
        filtering_percentage: Percentage of top traces to keep (0.7 = keep top 70%)
        confidence_metric: Confidence metric to use for filtering
        **kwargs: Additional parameters for DeepThinkConfidenceStrategy

    Returns:
        DeepThinkConfidenceStrategy instance
    """
    return DeepThinkConfidenceStrategy(
        model=model,
        num_paths=num_paths,
        filtering_percentage=filtering_percentage,
        confidence_metric=confidence_metric,
        **kwargs
    )