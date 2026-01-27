"""
Chain-of-Embedding (CoE) scorer for ST-BoN strategy.

Based on paper "Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N
Sampling in Early Decoding" (https://arxiv.org/pdf/2503.01422).

CoE uses hidden states across transformer layers to measure the "curvature"
of the latent thinking path. Samples with lower CoE feature differences
to other samples are considered more promising.

For backends without hidden state access (like vLLM), this module provides
alternative similarity measures using semantic embeddings or string similarity.
"""

import logging
import math
from typing import Any, List, Optional

import numpy as np

log = logging.getLogger(__name__)


class CoEScorer:
    """
    Chain-of-Embedding (CoE) scorer for measuring early consistency between samples.

    CoE Feature Formula (from ST-BoN paper Eq. 4):
        F(H) = (1/L) * Σ[M(h_l, h_{l+1})/M(h_0, h_L) - A(h_l, h_{l+1})/A(h_0, h_L)]

    Where:
        - H = {h_0, h_1, ..., h_L} are sentence embeddings per layer
        - M(h_i, h_j) = ||h_i - h_j||_2 (Euclidean/Magnitude distance)
        - A(h_i, h_j) = arccos(h_i · h_j / (||h_i|| * ||h_j||)) (Angular distance)

    The CoE feature captures the "curvature" of the latent thinking path.
    Samples with similar CoE features are considered more consistent.
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize CoE scorer.

        Args:
            epsilon: Small constant to avoid division by zero
        """
        self.epsilon = epsilon

    def compute_magnitude_distance(self, h_i: np.ndarray, h_j: np.ndarray) -> float:
        """
        Compute Euclidean (magnitude) distance between two embeddings.

        M(h_i, h_j) = ||h_i - h_j||_2
        """
        return np.linalg.norm(h_i - h_j, ord=2)

    def compute_angular_distance(self, h_i: np.ndarray, h_j: np.ndarray) -> float:
        """
        Compute angular distance between two embeddings.

        A(h_i, h_j) = arccos(h_i · h_j / (||h_i||_2 * ||h_j||_2))
        """
        norm_i = np.linalg.norm(h_i, ord=2)
        norm_j = np.linalg.norm(h_j, ord=2)

        if norm_i < self.epsilon or norm_j < self.epsilon:
            return 0.0

        dot_product = np.dot(h_i, h_j)
        cosine_sim = dot_product / (norm_i * norm_j)

        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)

        return math.acos(cosine_sim)

    def compute_coe_feature(self, hidden_states_per_layer: List[np.ndarray]) -> float:
        """
        Compute CoE feature F(H) from per-layer hidden state embeddings.

        Args:
            hidden_states_per_layer: List of embeddings [h_0, h_1, ..., h_L],
                where each h_l is the mean hidden state for layer l.

        Returns:
            CoE feature value F(H)
        """
        if len(hidden_states_per_layer) < 2:
            return 0.0

        L = len(hidden_states_per_layer) - 1
        h_0 = hidden_states_per_layer[0]
        h_L = hidden_states_per_layer[-1]

        M_norm = self.compute_magnitude_distance(h_0, h_L)
        A_norm = self.compute_angular_distance(h_0, h_L)

        if M_norm < self.epsilon or A_norm < self.epsilon:
            return 0.0

        feature_sum = 0.0
        for layer in range(L):
            h_l = hidden_states_per_layer[layer]
            h_l_plus_1 = hidden_states_per_layer[layer + 1]

            M_local = self.compute_magnitude_distance(h_l, h_l_plus_1)
            A_local = self.compute_angular_distance(h_l, h_l_plus_1)

            feature_sum += (M_local / M_norm) - (A_local / A_norm)

        return feature_sum / L

    def compute_sample_distances(self, coe_features: List[float]) -> List[float]:
        """
        Compute average squared distance of each sample to all others.

        Distance D(Y_i, Y_j) = (F(H_i) - F(H_j))^2
        Score S(Y_i) = (1/(N-1)) * Σ_{j≠i} D(Y_i, Y_j)

        Lower score means higher consistency with other samples.

        Args:
            coe_features: List of CoE features for N samples

        Returns:
            List of consistency scores (lower is better)
        """
        N = len(coe_features)
        if N < 2:
            return [0.0] * N

        scores = []
        for i in range(N):
            total_distance = 0.0
            for j in range(N):
                if i != j:
                    distance = (coe_features[i] - coe_features[j]) ** 2
                    total_distance += distance
            scores.append(total_distance / (N - 1))

        return scores

    def select_best_sample(self, coe_features: List[float]) -> int:
        """
        Select the sample with lowest consistency score (most consistent).

        Args:
            coe_features: List of CoE features for N samples

        Returns:
            Index of the best sample
        """
        scores = self.compute_sample_distances(coe_features)
        return int(np.argmin(scores))

    def compute_features_from_hidden_states(
        self, samples_hidden_states: List[List[np.ndarray]]
    ) -> List[float]:
        """
        Compute CoE features for multiple samples from their hidden states.

        Args:
            samples_hidden_states: List of N samples, each containing
                per-layer hidden state embeddings [h_0, h_1, ..., h_L]

        Returns:
            List of CoE features for each sample
        """
        return [self.compute_coe_feature(hs) for hs in samples_hidden_states]


class SemanticSimilarityScorer:
    """
    Alternative scorer using semantic similarity for backends without hidden states.

    Uses sentence embeddings to measure consistency between sample prefixes.
    This is a fallback for vLLM or API-based models that don't expose hidden states.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize semantic similarity scorer.

        Args:
            model_name: Sentence transformer model for embeddings
            device: Device to run model on (auto-detected if None)
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of sentence transformer model."""
        if self._initialized:
            return

        try:
            import torch
            from sentence_transformers import SentenceTransformer

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._initialized = True
            log.info(f"Initialized semantic similarity model: {self.model_name}")
        except ImportError:
            log.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute sentence embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            Array of embeddings (N x embedding_dim)
        """
        self._ensure_initialized()
        return self._model.encode(texts, convert_to_numpy=True)

    def compute_sample_distances(self, embeddings: np.ndarray) -> List[float]:
        """
        Compute average squared Euclidean distance of each sample to others.

        Args:
            embeddings: Array of embeddings (N x dim)

        Returns:
            List of distance scores (lower is more consistent)
        """
        N = embeddings.shape[0]
        if N < 2:
            return [0.0] * N

        scores = []
        for i in range(N):
            total_distance = 0.0
            for j in range(N):
                if i != j:
                    distance = np.sum((embeddings[i] - embeddings[j]) ** 2)
                    total_distance += distance
            scores.append(total_distance / (N - 1))

        return scores

    def select_best_sample(self, texts: List[str]) -> int:
        """
        Select the sample most similar to others (lowest average distance).

        Args:
            texts: List of text samples

        Returns:
            Index of the best sample
        """
        if len(texts) < 2:
            return 0

        embeddings = self.compute_embeddings(texts)
        scores = self.compute_sample_distances(embeddings)
        return int(np.argmin(scores))

    def compute_consistency_scores(self, texts: List[str]) -> List[float]:
        """
        Compute consistency scores for samples (lower is more consistent).

        Args:
            texts: List of text samples

        Returns:
            List of consistency scores
        """
        if len(texts) < 2:
            return [0.0] * len(texts)

        embeddings = self.compute_embeddings(texts)
        return self.compute_sample_distances(embeddings)


class StringSimilarityScorer:
    """
    Alternative scorer using string similarity (Rouge-L) for backends without hidden states.

    Uses Rouge-L to measure consistency between sample prefixes.
    This is a lightweight fallback that doesn't require additional models.
    """

    def __init__(self):
        """Initialize string similarity scorer."""
        self._rouge = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of rouge scorer."""
        if self._initialized:
            return

        try:
            from rouge_score import rouge_scorer

            self._rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            self._initialized = True
            log.info("Initialized Rouge-L string similarity scorer")
        except ImportError:
            log.error(
                "rouge-score not installed. " "Install with: pip install rouge-score"
            )
            raise

    def compute_rouge_l(self, text_a: str, text_b: str) -> float:
        """
        Compute Rouge-L F1 score between two texts.

        Args:
            text_a: First text
            text_b: Second text

        Returns:
            Rouge-L F1 score (0-1, higher is more similar)
        """
        self._ensure_initialized()
        scores = self._rouge.score(text_a, text_b)
        return scores["rougeL"].fmeasure

    def compute_sample_distances(self, texts: List[str]) -> List[float]:
        """
        Compute average distance (1 - Rouge-L) of each sample to others.

        Args:
            texts: List of text samples

        Returns:
            List of distance scores (lower is more consistent)
        """
        N = len(texts)
        if N < 2:
            return [0.0] * N

        scores = []
        for i in range(N):
            total_distance = 0.0
            for j in range(N):
                if i != j:
                    rouge_score = self.compute_rouge_l(texts[i], texts[j])

                    # Convert similarity to distance
                    distance = 1.0 - rouge_score
                    total_distance += distance
            scores.append(total_distance / (N - 1))

        return scores

    def select_best_sample(self, texts: List[str]) -> int:
        """
        Select the sample most similar to others (lowest average distance).

        Args:
            texts: List of text samples

        Returns:
            Index of the best sample
        """
        if len(texts) < 2:
            return 0

        scores = self.compute_sample_distances(texts)
        return int(np.argmin(scores))

    def compute_consistency_scores(self, texts: List[str]) -> List[float]:
        """
        Compute consistency scores for samples (lower is more consistent).

        Args:
            texts: List of text samples

        Returns:
            List of consistency scores
        """
        return self.compute_sample_distances(texts)


class STBoNScorerFactory:
    """
    Factory for creating the appropriate scorer based on backend capabilities.
    """

    @staticmethod
    def create_scorer(
        backend: str = "auto",
        use_hidden_states: bool = True,
        semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> Any:
        """
        Create a scorer appropriate for the given backend.

        Args:
            backend: "huggingface", "vllm", or "auto"
            use_hidden_states: Whether to use hidden states (HuggingFace only)
            semantic_model: Model for semantic similarity (vLLM/API backends)

        Returns:
            Appropriate scorer instance
        """
        if backend == "huggingface" and use_hidden_states:
            log.info("Using CoE scorer with hidden states")
            return CoEScorer()
        elif backend == "vllm" or (backend == "auto" and not use_hidden_states):
            log.info("Using semantic similarity scorer (no hidden states)")
            return SemanticSimilarityScorer(model_name=semantic_model)
        else:
            log.info("Using string similarity scorer (lightweight fallback)")
            return StringSimilarityScorer()
