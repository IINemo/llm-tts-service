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

        # Look up architecture: exact match first, then substring fallback
        self.arch = self.KNOWN_MODELS.get(model_name)
        if self.arch is not None:
            log.info(f"Using known architecture for {model_name}")
        else:
            for known_name, known_arch in self.KNOWN_MODELS.items():
                if known_name.lower() in model_name.lower():
                    self.arch = known_arch
                    log.info(
                        f"Using known architecture for {known_name} "
                        f"(substring match for '{model_name}')"
                    )
                    break

        # Auto-detect from HuggingFace config if not in KNOWN_MODELS
        if self.arch is None:
            self.arch = self._try_load_from_hf(model_name)

        if self.arch is None:
            raise ValueError(
                f"Unknown model '{model_name}'. Add it to FLOPCalculator.KNOWN_MODELS "
                f"or ensure it's available on HuggingFace."
            )

    @staticmethod
    def _try_load_from_hf(model_name: str) -> Optional[ModelArchitecture]:
        """Try to auto-detect model architecture from HuggingFace config."""
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

            h = config.hidden_size
            L = config.num_hidden_layers
            V = config.vocab_size
            n_heads = config.num_attention_heads
            n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
            ffn = getattr(config, "intermediate_size", 4 * h)
            head_dim = h // n_heads

            # Attention: Q proj (h * h) + K proj (h * n_kv * head_dim)
            #          + V proj (h * n_kv * head_dim) + O proj (h * h)
            attn_params = h * h + 2 * h * n_kv_heads * head_dim + h * h

            # FFN: SwiGLU has 3 matrices (gate, up, down), standard has 2
            # Heuristic: if hidden_act contains "silu"/"swiglu"/"gelu" with
            # intermediate_size != 4*h, likely gated (3 matrices)
            hidden_act = getattr(config, "hidden_act", "")
            if hidden_act in ("silu", "swiglu") or ffn != 4 * h:
                ffn_params = 3 * h * ffn  # gate_proj + up_proj + down_proj
            else:
                ffn_params = 2 * h * ffn

            # Layer norms: 2 per layer (pre-attn + pre-ffn), h params each
            norm_params = 2 * h

            # Per-layer total
            per_layer = attn_params + ffn_params + norm_params

            # Embeddings
            tie = getattr(config, "tie_word_embeddings", False)
            embed_params = V * h if tie else 2 * V * h

            num_params = L * per_layer + embed_params

            arch = ModelArchitecture(
                num_parameters=int(num_params),
                hidden_size=h,
                num_hidden_layers=L,
                num_attention_heads=n_heads,
                intermediate_size=ffn,
                vocab_size=V,
            )
            log.info(
                f"Auto-detected architecture for '{model_name}' from HuggingFace: "
                f"~{arch.num_parameters / 1e9:.1f}B params, "
                f"h={h}, L={L}, n_kv_heads={n_kv_heads}"
            )
            return arch
        except Exception as e:
            log.warning(f"Could not auto-detect architecture for '{model_name}': {e}")
            return None

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
