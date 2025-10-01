"""
Confidence scoring mechanisms for DeepConf implementation.
Supports both API-based approximate confidence and local model token-level confidence.
"""

import logging
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import math

log = logging.getLogger(__name__)


class ConfidenceScorer:
    """Base class for confidence scoring"""

    def __init__(self, name: str):
        self.name = name

    def score_trace(self, trace: str, **kwargs) -> Dict[str, float]:
        """
        Score a reasoning trace for confidence.

        Args:
            trace: Complete reasoning trace text
            **kwargs: Additional parameters (e.g., logits for local models)

        Returns:
            Dictionary with confidence metrics
        """
        raise NotImplementedError

    def score_traces(self, traces: List[str], **kwargs) -> List[Dict[str, float]]:
        """Score multiple traces"""
        return [self.score_trace(trace, **kwargs) for trace in traces]


class APIConfidenceScorer(ConfidenceScorer):
    """
    Approximate confidence scoring for API models.
    Uses text-based heuristics since we don't have access to token probabilities.
    """

    def __init__(self):
        super().__init__("api_confidence")

        # Patterns that indicate high confidence
        self.high_confidence_patterns = [
            r"\b(clearly|obviously|definitely|certainly|undoubtedly)\b",
            r"\b(the answer is|therefore|thus|hence)\b",
            r"\b(final answer|conclusion)\b",
            r"\d+\s*[+\-*/=]\s*\d+",  # Mathematical expressions
            r"\b\d+(\.\d+)?\b",  # Numbers (often indicate specific answers)
        ]

        # Patterns that indicate low confidence
        self.low_confidence_patterns = [
            r"\b(maybe|perhaps|possibly|might|could be|not sure|unclear)\b",
            r"\b(i think|i believe|seems like|appears)\b",
            r"\?\s*$",  # Ending with question
            r"\b(or|either)\b.*\b(or|either)\b",  # Multiple options
        ]

        # Quality indicators
        self.step_patterns = [
            r"step \d+",
            r"\d+\)",
            r"first|second|third|next|then|finally",
            r"let me|let's|we need to",
        ]

    def score_trace(self, trace: str, **kwargs) -> Dict[str, float]:
        """
        Score trace confidence using text-based heuristics.

        Returns confidence metrics:
        - avg_confidence: Overall confidence estimate
        - reasoning_quality: How well-structured the reasoning is
        - answer_confidence: Confidence in the final answer
        - length_penalty: Penalty for overly short/long responses
        """
        trace_lower = trace.lower()

        # 1. High confidence indicators (reduced weights)
        high_conf_score = 0
        for pattern in self.high_confidence_patterns:
            matches = len(re.findall(pattern, trace_lower, re.IGNORECASE))
            high_conf_score += matches * 0.03  # Reduced from 0.1 to 0.03

        # Cap high confidence contribution
        high_conf_score = min(high_conf_score, 0.2)

        # 2. Low confidence indicators (negative score)
        low_conf_score = 0
        for pattern in self.low_confidence_patterns:
            matches = len(re.findall(pattern, trace_lower, re.IGNORECASE))
            low_conf_score += matches * 0.08  # Reduced from 0.15 to 0.08

        # 3. Reasoning structure quality (reduced weights)
        step_score = 0
        for pattern in self.step_patterns:
            matches = len(re.findall(pattern, trace_lower, re.IGNORECASE))
            step_score += matches * 0.02  # Reduced from 0.05 to 0.02

        # Cap step score contribution
        step_score = min(step_score, 0.1)

        # 4. Mathematical content (reduced weight)
        math_score = len(re.findall(r"\d+\s*[+\-*/=]\s*\d+", trace)) * 0.05  # Reduced from 0.1 to 0.05
        math_score = min(math_score, 0.15)  # Cap math contribution

        # 5. Length-based confidence (moderate length is good)
        words = len(trace.split())
        if 20 <= words <= 200:
            length_score = 0.2
        elif 10 <= words <= 300:
            length_score = 0.1
        else:
            length_score = -0.1

        # 6. Answer extraction confidence
        answer_conf = self._assess_answer_confidence(trace)

        # Combine scores
        base_confidence = 0.4  # Lower starting point (was 0.5)
        confidence_adjustment = (
            high_conf_score +
            step_score +
            math_score +
            length_score +
            answer_conf -
            low_conf_score
        )

        # Add some variability based on content to avoid perfect scores
        raw_confidence = base_confidence + confidence_adjustment

        # Introduce subtle randomness based on content hash for consistency
        import hashlib
        content_hash = int(hashlib.md5(trace.encode()).hexdigest()[:8], 16)
        noise_factor = (content_hash % 100) / 1000.0  # 0.0 to 0.099 range

        # Scale down very high confidences
        if raw_confidence > 0.85:
            raw_confidence = 0.7 + (raw_confidence - 0.85) * 0.4  # Compress high values

        avg_confidence = max(0.1, min(0.95, raw_confidence + noise_factor - 0.05))

        # Quality score (how well-structured)
        reasoning_quality = max(0.0, min(1.0, 0.3 + step_score + math_score))

        return {
            "avg_confidence": avg_confidence,
            "reasoning_quality": reasoning_quality,
            "answer_confidence": answer_conf + 0.5,  # Separate answer confidence
            "length_penalty": -abs(length_score) if length_score < 0 else 0,
            "metadata": {
                "word_count": words,
                "high_conf_indicators": high_conf_score,
                "low_conf_indicators": low_conf_score,
                "step_indicators": step_score,
                "math_content": math_score
            }
        }

    def _assess_answer_confidence(self, trace: str) -> float:
        """Assess confidence in the final answer specifically"""
        # Look for clear answer indicators
        answer_patterns = [
            r"the answer is\s*(.+?)(?:\n|\.|\$|$)",
            r"therefore,?\s*(.+?)(?:\n|\.|\$|$)",
            r"final answer:\s*(.+?)(?:\n|\.|\$|$)",
            r"<answer>:\s*(.+?)(?:\n|$)",
        ]

        answer_confidence = 0.0
        for pattern in answer_patterns:
            matches = re.findall(pattern, trace.lower(), re.IGNORECASE)
            if matches:
                answer_text = matches[-1].strip()
                # More confident if answer is numeric/specific
                if re.search(r'\d+', answer_text):
                    answer_confidence += 0.3
                if len(answer_text.split()) <= 3:  # Concise answer
                    answer_confidence += 0.2
                answer_confidence += 0.1  # Base bonus for having clear answer

        return min(0.5, answer_confidence)  # Cap at 0.5 additional confidence


class TokenConfidenceScorer(ConfidenceScorer):
    """
    True confidence scoring using token probabilities from local models.
    Implements the exact DeepConf methodology.
    """

    def __init__(self, vocab_size: int = 50000):
        super().__init__("token_confidence")
        self.vocab_size = vocab_size

    def calculate_token_entropy(self, logits: np.ndarray) -> float:
        """Calculate entropy for a single token's logits"""
        # Convert logits to probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probs = exp_logits / np.sum(exp_logits)

        # Calculate entropy: Hi = −∑j Pi(j) log Pi(j)
        entropy = -np.sum(probs * np.log(probs + 1e-12))  # Add small epsilon

        # Convert to confidence (0 = max entropy, 1 = min entropy)
        max_entropy = np.log(self.vocab_size)
        confidence = 1.0 - (entropy / max_entropy)

        return max(0.0, min(1.0, confidence))

    def calculate_token_confidence_deepconf(self, logits: np.ndarray, k: int = 10) -> float:
        """
        Calculate token confidence as defined in DeepConf paper.

        Formula: Ci = −(1/k) ∑j=1^k log Pi(j)
        Negative average log-probability of top-k tokens.

        Args:
            logits: Token logits
            k: Number of top tokens to consider

        Returns:
            Token confidence score
        """
        # Convert logits to probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probs = exp_logits / np.sum(exp_logits)

        # Get top-k probabilities
        top_k_indices = np.argsort(probs)[-k:]
        top_k_probs = probs[top_k_indices]

        # Calculate token confidence: Ci = −(1/k) ∑j=1^k log Pi(j)
        log_probs = np.log(top_k_probs + 1e-12)
        confidence = -(1/k) * np.sum(log_probs)

        return float(confidence)

    def calculate_group_confidence(self, token_confidences: List[float], window_size: int = 2048) -> List[float]:
        """
        Calculate group confidence using sliding windows as defined in DeepConf paper.

        Group confidence is defined as the average token confidence over a sliding window of tokens.
        Each token is associated with a sliding window group Gi consisting of n previous tokens
        (e.g., n=1024 or 2048) with overlapping adjacent windows.

        Mathematical formula: CGi = (1/|Gi|) * ∑(t∈Gi) Ct
        Where:
        - CGi is the group confidence for group i
        - |Gi| is the number of tokens in the group (window_size)
        - Ct is the confidence of each token t in the group

        Args:
            token_confidences: List of individual token confidence scores
            window_size: Size of sliding window (typically 1024 or 2048)

        Returns:
            List of group confidence scores for each window
        """
        if len(token_confidences) == 0:
            return []

        if len(token_confidences) <= window_size:
            # If sequence is shorter than window, return single group confidence
            return [np.mean(token_confidences)]

        group_confidences = []

        # Create sliding windows
        for i in range(len(token_confidences) - window_size + 1):
            window = token_confidences[i:i + window_size]
            group_conf = np.mean(window)  # CGi = (1/|Gi|) * sum(Ct for t in Gi)
            group_confidences.append(group_conf)

        return group_confidences

    def _calculate_bottom_percent_group_confidence(self, group_confidences: List[float], percent: int, fallback: float) -> float:
        """Calculate bottom percent of group confidences"""
        if not group_confidences:
            return fallback

        bottom_count = max(1, len(group_confidences) * percent // 100)
        sorted_groups = sorted(group_confidences)
        return float(np.mean(sorted_groups[:bottom_count]))

    def _calculate_bottom_percent_confidence(self, token_confidences: List[float], percent: int) -> float:
        """Calculate bottom percent of token confidences"""
        if not token_confidences:
            return 0.5

        bottom_count = max(1, len(token_confidences) * percent // 100)
        sorted_confidences = sorted(token_confidences)
        return float(np.mean(sorted_confidences[:bottom_count]))

    def score_trace(self, trace: str, logits_sequence: Optional[List[np.ndarray]] = None, **kwargs) -> Dict[str, float]:
        """
        Score trace using token-level confidence from logits.

        Args:
            trace: Reasoning trace text
            logits_sequence: List of logits arrays for each generated token
        """
        if logits_sequence is None:
            # Fallback to API scoring if no logits available
            log.warning("No logits provided, falling back to API confidence scoring")
            api_scorer = APIConfidenceScorer()
            return api_scorer.score_trace(trace)

        # Calculate confidence for each token using DeepConf method
        token_confidences = []
        for logits in logits_sequence:
            # Use the DeepConf token confidence formula: Ci = −(1/k) ∑j=1^k log Pi(j)
            confidence = self.calculate_token_confidence_deepconf(logits, k=10)
            token_confidences.append(confidence)

        if not token_confidences:
            return {"avg_confidence": 0.5, "bottom_10_confidence": 0.5, "tail_confidence": 0.5}

        # Calculate DeepConf metrics
        avg_confidence = np.mean(token_confidences)

        # Bottom 10% confidence
        bottom_10_count = max(1, len(token_confidences) // 10)
        sorted_confidences = sorted(token_confidences)
        bottom_10_confidence = np.mean(sorted_confidences[:bottom_10_count])

        # Tail confidence (last 2048 tokens or all if shorter)
        tail_start = max(0, len(token_confidences) - 2048)
        tail_confidence = np.mean(token_confidences[tail_start:])

        # Group confidence: sliding window averages (DeepConf paper definition)
        group_confidences_512 = self.calculate_group_confidence(token_confidences, window_size=512)
        group_confidences_1024 = self.calculate_group_confidence(token_confidences, window_size=1024)
        group_confidences_2048 = self.calculate_group_confidence(token_confidences, window_size=2048)

        # Lowest group confidence (minimum among all groups) - Cleast from paper
        lowest_group_confidence_512 = min(group_confidences_512) if group_confidences_512 else avg_confidence
        lowest_group_confidence_1024 = min(group_confidences_1024) if group_confidences_1024 else avg_confidence
        lowest_group_confidence_2048 = min(group_confidences_2048) if group_confidences_2048 else avg_confidence

        # Bottom 10% group confidence for different window sizes
        bottom_10_group_confidence_512 = self._calculate_bottom_percent_group_confidence(group_confidences_512, 10, avg_confidence)
        bottom_10_group_confidence_1024 = self._calculate_bottom_percent_group_confidence(group_confidences_1024, 10, avg_confidence)
        bottom_10_group_confidence_2048 = self._calculate_bottom_percent_group_confidence(group_confidences_2048, 10, avg_confidence)

        # Head confidence (first portion of tokens)
        head_length = min(2048, len(token_confidences) // 2)  # First half or 2048 tokens
        head_confidence = np.mean(token_confidences[:head_length]) if head_length > 0 else avg_confidence

        # Additional percentile-based confidences
        bottom_5_confidence = self._calculate_bottom_percent_confidence(token_confidences, 5)
        bottom_20_confidence = self._calculate_bottom_percent_confidence(token_confidences, 20)

        return {
            # Basic token-level confidence metrics
            "avg_confidence": float(avg_confidence),  # Cavg: Average trace confidence
            "bottom_5_confidence": float(bottom_5_confidence),
            "bottom_10_confidence": float(bottom_10_confidence),
            "bottom_20_confidence": float(bottom_20_confidence),
            "tail_confidence": float(tail_confidence),  # Ctail: Last 2048 tokens
            "head_confidence": float(head_confidence),  # First portion confidence

            # Group confidence metrics (all window sizes from paper)
            "group_confidence": float(lowest_group_confidence_2048),  # Default to 2048 window
            "group_confidence_512": float(lowest_group_confidence_512),
            "group_confidence_1024": float(lowest_group_confidence_1024),
            "group_confidence_2048": float(lowest_group_confidence_2048),
            "lowest_group_confidence": float(lowest_group_confidence_2048),  # Cleast from paper

            # Bottom percent group confidences
            "bottom_10_group_confidence": float(bottom_10_group_confidence_2048),  # Default to 2048
            "bottom_10_group_confidence_512": float(bottom_10_group_confidence_512),
            "bottom_10_group_confidence_1024": float(bottom_10_group_confidence_1024),
            "bottom_10_group_confidence_2048": float(bottom_10_group_confidence_2048),

            # Token count and metadata
            "token_count": len(token_confidences),
            "metadata": {
                "min_confidence": float(np.min(token_confidences)),
                "max_confidence": float(np.max(token_confidences)),
                "std_confidence": float(np.std(token_confidences)),
                "num_groups_512": len(group_confidences_512),
                "num_groups_1024": len(group_confidences_1024),
                "num_groups_2048": len(group_confidences_2048),
                "head_length": head_length,
                "tail_length": len(token_confidences) - tail_start
            }
        }


class HybridConfidenceScorer(ConfidenceScorer):
    """
    Combines API-based and token-based confidence scoring.
    Uses the best available method based on what data is provided.
    """

    def __init__(self, vocab_size: int = 50000):
        super().__init__("hybrid_confidence")
        self.api_scorer = APIConfidenceScorer()
        self.token_scorer = TokenConfidenceScorer(vocab_size)

    def score_trace(self, trace: str, logits_sequence: Optional[List[np.ndarray]] = None, **kwargs) -> Dict[str, float]:
        """Score using the best available method"""

        if logits_sequence is not None:
            # Use token-based scoring (true DeepConf)
            log.debug("Using token-based confidence scoring")
            token_scores = self.token_scorer.score_trace(trace, logits_sequence, **kwargs)
            api_scores = self.api_scorer.score_trace(trace, **kwargs)

            # Combine both scores
            combined_scores = token_scores.copy()
            combined_scores["reasoning_quality"] = api_scores["reasoning_quality"]
            combined_scores["answer_confidence"] = api_scores["answer_confidence"]
            combined_scores["api_metadata"] = api_scores["metadata"]

            return combined_scores
        else:
            # Use API-based scoring
            log.debug("Using API-based confidence scoring")
            return self.api_scorer.score_trace(trace, **kwargs)


def calculate_token_confidence_from_logprobs(top_logprobs: List[Dict], k: int = 10) -> float:
    """
    Calculate token confidence using DeepConf formula from API logprobs.

    Formula from paper: C_i = −(1/k) ∑_{j=1}^k log P_i(j)

    Args:
        top_logprobs: List of {token, logprob} dicts for top-k tokens
        k: Number of top tokens to consider

    Returns:
        Token confidence score
    """
    k = min(k, len(top_logprobs))
    if k == 0:
        return 0.0

    # DeepConf formula: negative average of log probabilities
    # logprob is already log(P), so we just negate and average
    total = sum(lp['logprob'] for lp in top_logprobs[:k])
    confidence = -(total / k)

    return float(confidence)


def score_trace_from_token_data(token_confidence_data: List[Dict]) -> Dict[str, float]:
    """
    Calculate all DeepConf confidence metrics from token logprob data.
    Used for OpenRouter and other APIs that provide token probabilities.

    Args:
        token_confidence_data: List of token data with logprobs from API

    Returns:
        Dictionary with all DeepConf confidence metrics
    """
    # Calculate token-level confidences using DeepConf formula
    token_confidences = []
    for token_data in token_confidence_data:
        token_conf = calculate_token_confidence_from_logprobs(
            token_data['top_logprobs'],
            k=10
        )
        token_confidences.append(token_conf)

    if not token_confidences:
        return {
            'avg_confidence': 0.0,
            'min_confidence': 0.0,
            'bottom_10_confidence': 0.0,
            'bottom_20_confidence': 0.0,
            'num_tokens': 0
        }

    # Calculate aggregate confidence metrics
    avg_confidence = np.mean(token_confidences)
    min_confidence = np.min(token_confidences)
    bottom_10_conf = np.percentile(token_confidences, 10)
    bottom_20_conf = np.percentile(token_confidences, 20)

    # Calculate group confidences for different window sizes
    window_sizes = [16, 32, 64]  # Smaller windows for typical API responses
    group_results = {}

    for window_size in window_sizes:
        if len(token_confidences) >= window_size:
            # Calculate group confidence using sliding windows
            group_confidences = []
            for i in range(len(token_confidences) - window_size + 1):
                window = token_confidences[i:i + window_size]
                group_conf = np.mean(window)  # CG_i = (1/|G_i|) ∑ C_t
                group_confidences.append(float(group_conf))

            min_group_conf = min(group_confidences) if group_confidences else avg_confidence
            group_results[f'group_confidence_{window_size}'] = float(min_group_conf)
            group_results[f'avg_group_confidence_{window_size}'] = float(np.mean(group_confidences))

    return {
        'avg_confidence': float(avg_confidence),
        'min_confidence': float(min_confidence),
        'bottom_10_confidence': float(bottom_10_conf),
        'bottom_20_confidence': float(bottom_20_conf),
        'num_tokens': len(token_confidences),
        **group_results,
        'token_confidences': token_confidences  # Include for detailed analysis
    }


def create_confidence_scorer(model_type: str = "api", vocab_size: int = 50000) -> ConfidenceScorer:
    """
    Factory function to create appropriate confidence scorer.

    Args:
        model_type: "api", "token", or "hybrid"
        vocab_size: Vocabulary size for token-based scoring

    Returns:
        Appropriate ConfidenceScorer instance
    """
    if model_type == "api":
        return APIConfidenceScorer()
    elif model_type == "token":
        return TokenConfidenceScorer(vocab_size)
    elif model_type == "hybrid":
        return HybridConfidenceScorer(vocab_size)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def confidence_weighted_voting(
    traces: List[str],
    confidence_scores: List[Dict[str, float]],
    confidence_metric: str = "avg_confidence",
    answer_extractor = None
) -> Dict[str, Any]:
    """
    Perform confidence-weighted majority voting.

    Args:
        traces: List of reasoning traces
        confidence_scores: List of confidence score dictionaries
        confidence_metric: Which confidence metric to use for weighting
        answer_extractor: Function to extract answers from traces

    Returns:
        Voting results with weighted decision
    """
    if len(traces) != len(confidence_scores):
        raise ValueError("Number of traces must match number of confidence scores")

    # Extract answers and weights
    answer_weights = defaultdict(float)
    trace_info = []

    for i, (trace, conf_dict) in enumerate(zip(traces, confidence_scores)):
        # Extract answer (use provided extractor or simple fallback)
        if answer_extractor:
            answer = answer_extractor(trace)
        else:
            # Simple answer extraction
            answer = _simple_answer_extraction(trace)

        # Get confidence weight
        weight = conf_dict.get(confidence_metric, 0.5)

        # Accumulate weighted votes
        answer_weights[answer] += weight

        trace_info.append({
            "trace_idx": i,
            "answer": answer,
            "confidence": weight,
            "confidence_scores": conf_dict
        })

    # Find winning answer
    if not answer_weights:
        return {
            "selected_answer": "no_answer",
            "confidence_score": 0.0,
            "vote_distribution": {},
            "trace_info": trace_info
        }

    winning_answer = max(answer_weights.keys(), key=lambda k: answer_weights[k])
    winning_weight = answer_weights[winning_answer]
    total_weight = sum(answer_weights.values())

    # Calculate decision confidence (proportion of total votes for winning answer)
    # This represents how dominant the winning answer was in the voting
    decision_confidence = winning_weight / total_weight if total_weight > 0 else 0.0

    return {
        "selected_answer": winning_answer,
        "confidence_score": decision_confidence,
        "vote_distribution": dict(answer_weights),
        "total_weight": total_weight,
        "trace_info": trace_info,
        "metadata": {
            "num_traces": len(traces),
            "unique_answers": len(answer_weights),
            "confidence_metric_used": confidence_metric
        }
    }


def _simple_answer_extraction(trace: str) -> str:
    """Simple answer extraction for fallback"""
    import re

    patterns = [
        r"the answer is\s*(.+?)(?:\n|\.|\$|$)",
        r"therefore,?\s*(.+?)(?:\n|\.|\$|$)",
        r"final answer:\s*(.+?)(?:\n|\.|\$|$)",
        r"<answer>:\s*(.+?)(?:\n|$)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, trace.lower(), re.IGNORECASE)
        if matches:
            answer = matches[-1].strip()
            # Clean up answer
            answer = re.sub(r'[^\w\s\-\+\*\/\(\)\.\,]', '', answer)
            return answer.lower()

    # Fallback to number extraction
    numbers = re.findall(r'-?\d+\.?\d*', trace)
    if numbers:
        return numbers[-1]

    return "unknown"