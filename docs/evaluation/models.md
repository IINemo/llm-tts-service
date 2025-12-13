# Models

## Our Evaluation Models

| Model | Parameters | HuggingFace | Notes |
|-------|------------|-------------|-------|
| **Qwen2.5-7B** | 7B | [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) | Qwen 2.5 family |
| **Qwen3-8B** | 8B | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Qwen 3 family, non-thinking mode |
| **Qwen3-32B** | 32B | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | Larger model for scaling analysis |

---

## Models Used in Related Papers

Overview of models used across test-time compute scaling papers.

### Open-Source Models

| Model Family | Variants | Used By |
|--------------|----------|---------|
| **LLaMA-2** | 7B, 13B, 70B | Reasoning in Flux, Tree of Uncertain Thoughts |
| **LLaMA-3 / 3.1** | 8B, 70B | φ-Decoding |
| **Qwen2 / 2.5** | 3B, 7B | φ-Decoding, UnCert-CoT (Coder) |
| **Qwen3** | 1.7B, 4B, 8B, 32B | MUR, DeepConf |
| **DeepSeek** | 8B | DeepConf |
| **DeepSeek-Coder** | 6.7B | UnCert-CoT (Coder) |
| **CodeLlama** | 13B | UnCert-CoT (Coder) |
| **Mistral** | 7B, 8x7B | Reasoning in Flux, φ-Decoding |

### Proprietary Models

| Model | Used By |
|-------|---------|
| **GPT-4** | Let's Verify Step by Step, Tree of Thoughts, Tree of Uncertain Thoughts |
| **GPT-3.5** | Tree of Thoughts, Tree of Uncertain Thoughts |
| **Claude 2** | Tree of Uncertain Thoughts |
| **PaLM 2** | Tree of Uncertain Thoughts |
| **Cohere** | Tree of Uncertain Thoughts |

---

## Paper-Model Matrix

Detailed breakdown of which models each paper evaluated on.

| Paper | Models |
|-------|--------|
| **Reasoning in Flux** | LLaMA-2-7B, LLaMA-2-13B, LLaMA-2-70B, Mistral-7B, Mistral-8x7B |
| **UnCert-CoT (code)** | DeepSeek-Coder-6.7B-Base, CodeLlama-13B-python-hf, Qwen2.5-Coder-7B-Base |
| **MUR** | Qwen3-1.7B, Qwen3-4B, Qwen3-8B |
| **DeepConf** | DeepSeek-8B, Qwen3-8B, Qwen3-32B, GPT-OSS-20B, GPT-OSS-120B |
| **Let's Verify Step by Step** | GPT-4 |
| **Tree of Thoughts** | GPT-4, GPT-3.5 |
| **Tree of Uncertain Thoughts** | Llama 2-70B-Chat, Cohere, PaLM 2, Claude 2, GPT-3.5-turbo, GPT-4 |
| **φ-Decoding** | LLaMA3.1-8B-Instruct, Mistral-v0.3-7B-Instruct, Qwen2.5-3B-Instruct, LLaMA3.1-70B-Instruct, R1-Distill-LLaMA-8B |

> **Note**: Some papers (SMART, Entro-duction, AdaDec) were removed from this matrix pending verification of their exact model lists.

---

## Model Selection Rationale

We selected **Qwen2.5-7B**, **Qwen3-8B**, and **Qwen3-32B** for our benchmark because:

1. **Open-source**: Fully reproducible experiments
2. **Strong reasoning**: Competitive performance on math tasks
3. **Size diversity**: 7B/8B/32B enables scaling analysis
4. **Well-supported**: Good HuggingFace integration, active community
5. **Used in literature**: MUR and DeepConf papers use Qwen3 family

### Configuration

> **Important**: For Qwen3 models, always use **non-thinking mode** (`disable_thinking_mode: true`). This disables the internal reasoning phase and produces direct answers.
>
> According to the [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B):
> - **Non-thinking mode**: Temperature=0.7, TopP=0.8, TopK=20, MinP=0
> - **Thinking mode**: Temperature=0.6, TopP=0.95, TopK=20, MinP=0
>
> **DO NOT use greedy decoding** (temperature=0), as it can lead to performance degradation and endless repetitions.

**vLLM (Recommended for full-trace strategies)**

For local evaluations, use vLLM to avoid OOM errors on long reasoning sequences:

```yaml
# config/model/vllm_qwen3.yaml
model:
  type: "vllm"
  model_path: Qwen/Qwen3-8B  # or Qwen/Qwen2.5-7B, Qwen/Qwen3-32B
  device: cuda
  gpu_memory_utilization: 0.9
  tensor_parallel_size: 1
  enable_prefix_caching: true
  trust_remote_code: true
  max_model_len: 32768
  disable_thinking_mode: true  # Required for Qwen3 models

generation:
  temperature: 0.7  # Standard for non-thinking mode
```

> ⚠️ **Limitation**: vLLM does not support custom stopping criteria callbacks (only exact string matching via `stop` parameter). This means step-by-step strategies like Online Best-of-N cannot use dynamic step detection during generation with vLLM. Use HuggingFace inference for such strategies. See [evaluation README](README.md#vllm-recommended) for details.

**HuggingFace (Required for step-by-step strategies)**

```yaml
# config/model/hf_qwen3.yaml
model:
  model_path: Qwen/Qwen3-8B  # or Qwen/Qwen2.5-7B, Qwen/Qwen3-32B
  device_map: auto
  torch_dtype: float16
```

> **Note**: HuggingFace inference is prone to OOM errors on long reasoning sequences. Use vLLM for production evaluations when possible.

**Advantages over vLLM:**
- Supports custom `StoppingCriteria` callbacks for dynamic step boundary detection
- Required for thinking mode TTS with semantic step detection (`ThinkingStepStoppingCriteria`)
- Required for Online Best-of-N and Beam Search with non-explicit step markers

---

## References

- [Reasoning in Flux (UAG)](https://aclanthology.org/2024.acl-long.xxx/) - LLaMA-2 and Mistral evaluation
- [MUR](https://arxiv.org/abs/2507.14958) - Qwen3 scaling analysis
- [DeepConf](https://arxiv.org/abs/2508.15260) - DeepSeek and Qwen3 evaluation
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) - GPT-4 process reward model
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601) - GPT-4 deliberate reasoning
- [Tree of Uncertain Thoughts](https://openreview.net/pdf?id=ZWyLjimciT) - Multi-model uncertainty-aware planning
- [φ-Decoding](https://arxiv.org/abs/2503.13288) - Multi-model instruction-tuned evaluation
- [UnCert-CoT](https://link.springer.com/chapter/10.1007/978-981-95-0014-7_36) - Code generation with uncertainty
