"""
FLOP calculation utilities for LLM inference.

Two approaches based on docs/evaluation/metrics.md:
1. Simple: FLOPs = 2 * num_parameters * tokens (Kaplan et al., 2020)
2. Precise: Architecture-aware calculation (PaLM, 2022)

References:
- Kaplan et al. (2020): https://arxiv.org/abs/2001.08361
- PaLM (2022): https://arxiv.org/abs/2204.02311
- Kipply (2022): https://kipp.ly/transformer-inference-arithmetic/
"""

import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class ModelArchitecture:
    """Model architecture parameters for FLOP calculation."""

    num_parameters: int  # Total parameters (non-embedding approximation)
    hidden_size: int  # d_model / hidden dimension
    num_hidden_layers: int  # Number of transformer layers
    num_attention_heads: int  # Number of attention heads
    intermediate_size: int  # FFN intermediate size (often 4*hidden_size)
    vocab_size: int  # Vocabulary size

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def calculate_flops_simple(
    num_tokens: int,
    num_parameters: int = 8_000_000_000,  # Default: 8B
) -> float:
    """
    Simple FLOP calculation: FLOPs = 2 * N * tokens

    Based on Kaplan et al. (2020) - inference requires ~2 FLOPs per parameter per token.

    Args:
        num_tokens: Number of generated tokens
        num_parameters: Model size in parameters (default: 8B for Qwen3-8B)

    Returns:
        Total FLOPs for inference
    """
    return 2 * num_parameters * num_tokens


def calculate_flops_precise(
    num_tokens: int,
    arch: ModelArchitecture,
    batch_size: int = 1,
    avg_sequence_length: Optional[int] = None,
) -> float:
    """
    Precise FLOP calculation using architecture details.

    Per-layer FLOPs with KV cache (PaLM, 2022):
        FLOPs_per_layer = 24 * b * s * h^2  +  4 * b * s * h
                          (FFN + projections)    (attention with KV cache)

    For autoregressive inference with KV cache, we generate one token at a time,
    so s=1 for the query, but attention spans previous tokens.

    Simplified per-token cost (dominated by dense compute):
        FLOPs_per_token ≈ 2 * num_parameters (when KV cache is used)

    Args:
        num_tokens: Number of generated tokens
        arch: Model architecture
        batch_size: Batch size (default: 1)
        avg_sequence_length: Average sequence length for attention cost.
                            If None, uses rough approximation.

    Returns:
        Total FLOPs for inference
    """
    h = arch.hidden_size
    L = arch.num_hidden_layers
    b = batch_size

    # Per-token costs (with KV cache, attention is O(s) not O(s^2))

    # 1. QKV projections: 3 * h * h = 3h^2 FLOPs per layer
    # 2. Output projection: h * h = h^2 FLOPs per layer
    # 3. FFN: 2 * h * intermediate = 2 * h * 4h = 8h^2 FLOPs per layer (up + down)
    # Total per layer: 3h^2 + h^2 + 8h^2 = 12h^2

    # With multiply-add = 2 ops: 24h^2 per layer per token
    flops_per_token_dense = 24 * h * h * L * b

    # Attention cost with KV cache: O(s * h) per layer
    # For each new token, we compute attention over all previous tokens
    # This adds 4 * s * h FLOPs per layer (Q @ K^T and softmax @ V)
    if avg_sequence_length is not None:
        s = avg_sequence_length
        flops_per_token_attention = 4 * b * s * h * L
    else:
        # Average case: assume generation at ~1/2 max sequence
        # For typical 32k context, avg ~16k
        s = 16000
        flops_per_token_attention = 4 * b * s * h * L

    flops_per_token = flops_per_token_dense + flops_per_token_attention

    return flops_per_token * num_tokens


def flops_to_tflops(flops: float) -> float:
    """Convert FLOPs to TFLOPs."""
    return flops / 1e12


class FLOPCalculator:
    """
    FLOP calculator with model-specific configuration.

    Usage:
        calc = FLOPCalculator("Qwen/Qwen3-8B")
        tflops = calc.compute_tflops(num_tokens=45000)
    """

    # Known model configurations
    KNOWN_MODELS = {
        "Qwen/Qwen3-8B": ModelArchitecture(
            num_parameters=8_000_000_000,
            hidden_size=4096,
            num_hidden_layers=36,
            num_attention_heads=32,
            intermediate_size=14336,
            vocab_size=151936,
        ),
        "Qwen/Qwen2.5-Math-7B-Instruct": ModelArchitecture(
            num_parameters=7_000_000_000,
            hidden_size=3584,
            num_hidden_layers=28,
            num_attention_heads=28,
            intermediate_size=18944,
            vocab_size=152064,
        ),
        "Qwen/Qwen2.5-Math-PRM-7B": ModelArchitecture(
            num_parameters=7_000_000_000,
            hidden_size=3584,
            num_hidden_layers=28,
            num_attention_heads=28,
            intermediate_size=18944,
            vocab_size=152064,
        ),
    }

    def __init__(
        self,
        model_name: str,
        method: str = "simple",  # "simple" or "precise"
    ):
        """
        Initialize FLOP calculator.

        Args:
            model_name: HuggingFace model name
            method: "simple" (2*N*tokens) or "precise" (architecture-aware)
        """
        self.model_name = model_name
        self.method = method

        # Look up architecture in known models
        self.arch = None
        for known_name, known_arch in self.KNOWN_MODELS.items():
            if known_name.lower() in model_name.lower():
                self.arch = known_arch
                log.info(f"Using known architecture for {known_name}")
                break

        if self.arch is None:
            raise ValueError(
                f"Unknown model '{model_name}'. Add it to FLOPCalculator.KNOWN_MODELS."
            )

    def compute_flops(
        self,
        num_tokens: int,
        batch_size: int = 1,
        avg_sequence_length: Optional[int] = None,
    ) -> float:
        """Compute FLOPs for given token count."""
        if self.method == "precise" and self.arch:
            return calculate_flops_precise(
                num_tokens=num_tokens,
                arch=self.arch,
                batch_size=batch_size,
                avg_sequence_length=avg_sequence_length,
            )
        else:
            return calculate_flops_simple(
                num_tokens=num_tokens,
                num_parameters=self.arch.num_parameters,
            )

    def compute_tflops(
        self,
        num_tokens: int,
        batch_size: int = 1,
        avg_sequence_length: Optional[int] = None,
    ) -> float:
        """Compute TFLOPs for given token count."""
        return flops_to_tflops(
            self.compute_flops(num_tokens, batch_size, avg_sequence_length)
        )

    @property
    def flops_per_token(self) -> float:
        """Get FLOPs per token (simple method)."""
        return 2 * self.arch.num_parameters

    @property
    def tflops_per_1k_tokens(self) -> float:
        """Get TFLOPs per 1000 tokens."""
        return flops_to_tflops(self.flops_per_token * 1000)
