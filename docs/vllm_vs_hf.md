# vLLM vs HuggingFace for Uncertainty Quantification

This document compares vLLM and HuggingFace Transformers for LLM inference with uncertainty quantification (UQ), discusses their limitations, and proposes solutions.

## Overview

| Aspect | vLLM | HuggingFace |
|--------|------|-------------|
| **Primary Use** | High-throughput inference | Research & flexibility |
| **Memory Management** | PagedAttention (dynamic) | Static KV cache |
| **Generation Speed** | Fast (optimized CUDA kernels) | Slower (general purpose) |
| **Logprobs Access** | Top-k (configurable to full vocab) | Full distribution |
| **Hidden States** | Not exposed | Full access |
| **Attention Weights** | Not exposed | Full access |

## vLLM

### Advantages

1. **PagedAttention**: Memory-efficient attention that allocates KV cache dynamically in blocks, similar to OS virtual memory. This prevents OOM errors during long-context generation.

2. **Continuous Batching**: Dynamically batches requests, maximizing GPU utilization.

3. **Optimized Kernels**: Custom CUDA kernels for faster inference.

4. **Pre-allocated Memory**: Sets `gpu_memory_utilization` upfront, preventing fragmentation.

### Limitations for Uncertainty Quantification

```python
# vLLM generate() returns:
outputs = llm.generate(prompts, sampling_params)
# - Generated text
# - Token IDs
# - Log probabilities (top-k, configurable via logprobs parameter)

# NOT available:
# - Hidden states (needed for Mahalanobis, RDE, embeddings-based UQ)
# - Attention weights (needed for attention-based UQ)
```

**Note on logprobs:** vLLM's `SamplingParams(logprobs=k)` can be set to any value, including the full vocabulary size. However, returning full vocab logprobs (100k+ tokens) is memory-intensive and slow. For most UQ methods, top-k (e.g., k=100) is sufficient to compute entropy approximations.

```python
# Example: Get full vocabulary logprobs (expensive but possible)
from vllm import SamplingParams

# Top-100 logprobs (efficient, good approximation)
params = SamplingParams(logprobs=100, max_tokens=1024)

# Full vocabulary logprobs (expensive, exact entropy)
vocab_size = tokenizer.vocab_size  # e.g., 128256 for Llama
params = SamplingParams(logprobs=vocab_size, max_tokens=1024)
```

**Why hidden states/attention are not available:**
- vLLM is optimized for production inference, not research
- Exposing hidden states would require storing all intermediate activations (memory intensive)
- These would negate PagedAttention memory benefits

### Available UQ Methods with vLLM

Methods that work with **logprobs only**:

| Method | Description | Works with vLLM |
|--------|-------------|-----------------|
| Perplexity | exp(-mean(log_probs)) | Yes |
| Entropy | -sum(p * log(p)) | Yes (exact with full vocab logprobs) |
| MaxProb | max token probability | Yes |
| SelfCertainty | Final token confidence | Yes |

Methods requiring **multiple samples** (vLLM can generate these):

| Method | Description | Works with vLLM |
|--------|-------------|-----------------|
| SemanticEntropy | Cluster samples by meaning | Yes (text only) |
| LexicalSimilarity | ROUGE/BLEU between samples | Yes |
| NumSemSets | Count semantic clusters | Yes |
| Consistency | Agreement between samples | Yes |

## HuggingFace Transformers

### Advantages

1. **Full Access**: Complete access to logits, hidden states, attention weights.

2. **Flexibility**: Easy to modify generation, add hooks, extract internals.

3. **Research-Friendly**: Designed for experimentation and analysis.

### Limitations

1. **Memory Management**: Static KV cache allocation leads to OOM on long sequences.

2. **Slower Generation**: General-purpose implementation, not optimized for throughput.

3. **Batch Generation Memory**: Generating N candidates simultaneously requires N× KV cache memory.

### OOM Mitigation Strategies

#### 1. GPU Memory Limit

Set memory fraction before any CUDA operations:

```python
# At the START of main(), before any model loading
gpu_memory_utilization = 0.95  # Leave 5% headroom
for i in range(torch.cuda.device_count()):
    torch.cuda.set_per_process_memory_fraction(gpu_memory_utilization, i)
```

#### 2. Sequential Generation

Instead of generating N candidates in parallel:

```python
# BAD: Batch generation (N× memory for KV cache)
gen_params = {"num_return_sequences": 4}
outputs = model.generate(**inputs, **gen_params)  # OOM risk

# GOOD: Sequential generation (1× memory)
candidates = []
for _ in range(4):
    outputs = model.generate(**inputs, num_return_sequences=1)
    candidates.append(outputs)
    torch.cuda.empty_cache()  # Free memory between generations
```

#### 3. Memory Cache Clearing

Clear unused memory after each generation step:

```python
def generate_candidates(self, ...):
    candidates = []
    for _ in range(n_candidates):
        with torch.no_grad():
            output = self.model.generate(**inputs)
        candidates.append(output)

        # Clear CUDA cache after each candidate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return candidates
```

#### 4. Expandable Segments

Enable PyTorch memory allocator optimization:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python script.py
```

This reduces memory fragmentation by using expandable memory segments instead of fixed-size blocks.

#### 5. Flash Attention 2

Use memory-efficient attention implementation:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",  # Requires flash-attn package
)
```

Flash Attention 2:
- Reduces memory from O(n²) to O(n) for attention
- Fuses operations to reduce memory bandwidth
- Does NOT provide PagedAttention benefits (still static allocation)

### Available UQ Methods with HuggingFace

**All methods available**, including:

| Method | Required Data | HuggingFace Access |
|--------|--------------|-------------------|
| Perplexity | Log probabilities | `output.logits` |
| Entropy | Full logit distribution | `output.logits` |
| MahalanobisDistance | Hidden states | `output.hidden_states` |
| RDE | Hidden states | `output.hidden_states` |
| SemanticDensity | Embeddings | `output.hidden_states[-1]` |
| AttentionScore | Attention weights | `output.attentions` |
| EigenScore | Sample embeddings | `output.hidden_states` |

Enable with:
```python
outputs = model(
    **inputs,
    output_hidden_states=True,  # For embedding-based UQ
    output_attentions=True,     # For attention-based UQ
    return_dict=True,
)
```

## Hybrid Approach: vLLM Generate + HuggingFace Score

The optimal solution combines both backends:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     vLLM        │     │   HuggingFace   │     │   UQ Methods    │
│   (Generate)    │ ──► │    (Score)      │ ──► │   (Compute)     │
│                 │     │                 │     │                 │
│ - Fast          │     │ - One forward   │     │ - All methods   │
│ - No OOM        │     │   pass only     │     │   available     │
│ - N samples     │     │ - Hidden states │     │                 │
└─────────────────┘     │ - Attention     │     └─────────────────┘
                        └─────────────────┘
```

### Key Insight: Generation vs Scoring

**Generation (autoregressive):**
- For N tokens: ~N forward passes (sequential)
- Each step depends on previous output
- Cannot parallelize across tokens

**Scoring (teacher forcing):**
- For N tokens: 1 forward pass
- All tokens known in advance
- Fully parallelizable

| Operation | Forward Passes | Time (1000 tokens) |
|-----------|---------------|-------------------|
| HF Generate | ~1000 | ~40s |
| HF Score | 1 | ~0.04s |

### Implementation

```python
class HybridUncertaintyGenerator:
    """vLLM for generation, HuggingFace for uncertainty scoring."""

    def __init__(self, model_name: str):
        # vLLM for fast generation
        self.vllm = LLM(
            model=model_name,
            gpu_memory_utilization=0.45,  # Leave room for HF
        )

        # HuggingFace for scoring (can use quantization)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_with_uncertainty(
        self,
        prompt: str,
        n_samples: int = 4,
        uncertainty_methods: List[str] = ["entropy", "mahalanobis"],
    ) -> List[Dict]:
        """Generate samples with vLLM, score with HuggingFace."""

        # 1. Fast generation with vLLM
        sampling_params = SamplingParams(
            n=n_samples,
            temperature=0.7,
            max_tokens=1024,
        )
        vllm_outputs = self.vllm.generate([prompt], sampling_params)[0]

        # 2. Score each sample with HuggingFace
        results = []
        for output in vllm_outputs.outputs:
            generated_text = output.text

            # Single forward pass for all stats
            uncertainty = self._score_sequence(prompt, generated_text)

            results.append({
                "text": generated_text,
                "uncertainty": uncertainty,
            })

        return results

    def _score_sequence(self, prompt: str, generated: str) -> Dict:
        """Score a generated sequence with HuggingFace (single forward pass)."""
        full_text = prompt + generated
        inputs = self.tokenizer(full_text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.hf_model(
                **inputs,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )

        # Get stats for generated portion only
        prompt_tokens = len(self.tokenizer(prompt).input_ids)

        # Logits for generated tokens (shifted by 1 for next-token prediction)
        gen_logits = outputs.logits[0, prompt_tokens-1:-1]
        gen_tokens = inputs.input_ids[0, prompt_tokens:]

        # Hidden states for generated tokens
        gen_hidden = outputs.hidden_states[-1][0, prompt_tokens:]

        # Compute uncertainty metrics
        return {
            "entropy": self._compute_entropy(gen_logits),
            "perplexity": self._compute_perplexity(gen_logits, gen_tokens),
            "hidden_states": gen_hidden,  # For Mahalanobis, RDE, etc.
        }

    def _compute_entropy(self, logits: torch.Tensor) -> float:
        """Compute mean token entropy."""
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy.mean().item()

    def _compute_perplexity(
        self, logits: torch.Tensor, tokens: torch.Tensor
    ) -> float:
        """Compute perplexity of generated sequence."""
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        return torch.exp(-token_log_probs.mean()).item()
```

### Memory Considerations

| Configuration | GPU Memory | Speed |
|--------------|------------|-------|
| Both models loaded | ~2× model size | Fastest |
| HF quantized (int8) | ~1.5× model size | Fast |
| Sequential load/unload | ~1× model size | Slowest |

For 8B model on 48GB GPU:
- vLLM: ~16GB (with 0.45 utilization)
- HF fp16: ~16GB
- HF int8: ~8GB
- Total (vLLM + HF int8): ~24GB (fits comfortably)

## vLLM-Only Approach: Generation and Scoring

For simpler deployments, vLLM can handle both generation **and** uncertainty scoring without HuggingFace, using the `prompt_logprobs` feature for teacher-forcing scoring.

### How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     vLLM        │     │     vLLM        │     │  lm-polygraph   │
│   (Generate)    │ ──► │ (Score via      │ ──► │   Estimators    │
│                 │     │  prompt_logprobs)│    │                 │
│ - Fast          │     │ - Forward pass  │     │ - Perplexity    │
│ - No OOM        │     │   via prompt    │     │ - MeanTokenEntropy│
│ - N samples     │     │ - Full logprobs │     │ - MaxProb       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Key Insight: prompt_logprobs for Scoring

vLLM's `prompt_logprobs` parameter enables **teacher-forcing** (forward pass) scoring:
- Pass the full sequence (prompt + generated text) as input
- Set `max_tokens=1` to avoid additional generation
- vLLM returns logprobs for all **prompt** tokens (including generated portion)
- This is equivalent to a single forward pass through the model

```python
from vllm import SamplingParams

# Scoring parameters - use prompt_logprobs for teacher forcing
scoring_params = SamplingParams(
    max_tokens=1,           # Don't generate more
    temperature=0.0,        # Deterministic (doesn't matter for scoring)
    prompt_logprobs=20,     # Get top-20 logprobs for each prompt token
)

# Score the sequence: prompt + generated_text treated as "prompt"
full_sequence = prompt + generated_text
outputs = llm.generate([full_sequence], scoring_params)[0]

# Extract logprobs for generated portion
prompt_token_count = len(tokenizer.encode(prompt))
generated_logprobs = outputs.prompt_logprobs[prompt_token_count:]
```

### Implementation with lm-polygraph

The vLLM generator uses [lm-polygraph](https://github.com/IINemo/lm-polygraph) estimators for uncertainty computation:

```python
from lm_polygraph.estimators.perplexity import Perplexity
from lm_polygraph.estimators.token_entropy import MeanTokenEntropy

class VLLMStepGenerator:
    def __init__(self, model, ...):
        # Initialize estimators once (cached for performance)
        self._perplexity_estimator = Perplexity()
        self._entropy_estimator = MeanTokenEntropy()

    def _compute_uncertainty_metrics(
        self, token_ids: List[int], logprobs: List[Dict]
    ) -> Dict[str, float]:
        """Compute uncertainty using lm-polygraph estimators."""
        # Build log_probs array for selected tokens
        log_probs = np.array([
            lp.get(tid, float("-inf"))
            for tid, lp in zip(token_ids, logprobs)
        ])

        # Build full logit distributions for entropy
        all_log_probs = []
        for lp_dict in logprobs:
            logits = np.full(self.vocab_size, -100.0)
            for token_id, log_prob in lp_dict.items():
                logits[token_id] = log_prob
            all_log_probs.append(logits)

        # Use lm-polygraph estimators
        stats = {"greedy_log_probs": log_probs}
        perplexity = self._perplexity_estimator(stats)

        stats["greedy_log_likelihoods"] = np.array(all_log_probs)
        mean_entropy = self._entropy_estimator(stats)

        return {
            "perplexity": float(perplexity),
            "mean_entropy": float(mean_entropy),
            "uncertainty_score": float(mean_entropy),  # Use entropy as default
        }

    def score_truncated_sequence(
        self, prompt: str, truncated_text: str
    ) -> Dict[str, float]:
        """Re-score a truncated sequence using forward pass."""
        scoring_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            prompt_logprobs=20,
        )

        full_sequence = prompt + truncated_text
        outputs = self.model.generate([full_sequence], scoring_params)[0]

        # Extract logprobs for truncated portion only
        prompt_ids = self.tokenizer.encode(prompt)
        truncated_ids = self.tokenizer.encode(truncated_text)

        generated_logprobs = outputs.prompt_logprobs[len(prompt_ids):]
        return self._compute_uncertainty_metrics(truncated_ids, generated_logprobs)
```

### When to Use vLLM-Only

**Advantages:**
- Single model in memory (~1× model size vs 2× for hybrid)
- Simpler deployment (no HuggingFace dependency for inference)
- Sufficient for logprob-based UQ methods (entropy, perplexity, max_prob)

**Limitations:**
- **No hidden states access** - Cannot use embedding-based methods (Mahalanobis, RDE, SemanticDensity)
- **No attention weights** - Cannot use attention-based methods (AttentionScore, RAUQ)
- **UHead method unavailable** - Requires hidden states for uncertainty head

### Compatible UQ Methods

| Method | vLLM-Only | Notes |
|--------|-----------|-------|
| Perplexity | ✅ | Via logprobs |
| MeanTokenEntropy | ✅ | Via logprobs distribution |
| MaxProb | ✅ | Via logprobs |
| SelfCertainty | ✅ | Via final token logprobs |
| **MahalanobisDistance** | ❌ | Requires hidden states |
| **RDE** | ❌ | Requires hidden states |
| **UHead** | ❌ | Requires hidden states |
| **AttentionScore** | ❌ | Requires attention weights |

### Configuration

```yaml
generator:
  type: vllm
  gpu_memory_utilization: 0.9
  tensor_parallel_size: 1
  enable_prefix_caching: true

scorer:
  type: entropy  # or perplexity, maxprob
  # Note: mahalanobis, rde, uhead not available with vLLM-only
```

### Trade-off Summary

| Approach | Memory | Speed | UQ Methods |
|----------|--------|-------|------------|
| vLLM-only | 1× model | Fastest | Logprob-based only |
| Hybrid | 2× model | Fast | All methods |
| HF-only | 1× model | Slowest | All methods |

**Recommendation:** Use vLLM-only when logprob-based uncertainty (entropy, perplexity) is sufficient for your use case. Switch to hybrid approach only if you need hidden-state-based methods like UHead.

## UQ Methods Compatibility Summary

### By Data Requirement

| Data Required | vLLM | HF | Hybrid |
|--------------|------|-----|--------|
| Logprobs only | Yes | Yes | Yes |
| Full logits | Yes (configurable) | Yes | Yes |
| Hidden states | No | Yes | Yes |
| Attention | No | Yes | Yes |
| Multiple samples | Yes | Yes (slow) | Yes |

### By Method Category

**1. Logprob-based (vLLM compatible):**
- Perplexity, Entropy, MaxProb, SelfCertainty, FisherRao, RenyiNeg

**2. Sampling-based (vLLM generates, external scoring):**
- SemanticEntropy, LexicalSimilarity, NumSemSets, SAR, DegMat, Eccentricity

**3. Hidden state-based (HuggingFace only):**
- MahalanobisDistance, RelativeMahalanobis, RDE, SemanticDensity, EigenScore, KernelLanguageEntropy

**4. Attention-based (HuggingFace only):**
- AttentionScore, RAUQ

**5. Prompting-based (requires generation):**
- PTrue, Verbalized, Linguistic

## Recommendations

### For Production (Speed Priority)
Use vLLM with logprob-based methods:
```yaml
generator:
  type: vllm
  gpu_memory_utilization: 0.9
scorer:
  type: entropy  # or perplexity, maxprob
```

### For Research (Full UQ Access)
Use HuggingFace with memory optimizations:
```yaml
generator:
  type: huggingface
  gpu_memory_utilization: 0.95
  sequential_generation: true
  empty_cache_after_step: true
scorer:
  type: mahalanobis  # or any method
```

### For Best of Both Worlds
Use hybrid approach:
```yaml
generator:
  type: hybrid
  vllm:
    gpu_memory_utilization: 0.45
  huggingface:
    dtype: float16
    quantization: int8  # optional
scorer:
  type: [entropy, mahalanobis, semantic_entropy]
```

## References

- [vLLM Paper: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)
- [lm-polygraph: Uncertainty Quantification for LLMs](https://github.com/IINemo/lm-polygraph)
