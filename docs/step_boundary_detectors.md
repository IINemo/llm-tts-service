# Step Boundary Detectors

This document describes approaches for detecting step boundaries in LLM reasoning traces, particularly for native thinking mode outputs (e.g., `<think>` tags in Qwen3, o1-style models).

## Motivation

When analyzing LLM reasoning traces, we need to split continuous thinking content into discrete reasoning steps. The goal is to identify logical boundaries that separate distinct reasoning units - similar to how a strong LLM (e.g., GPT-4.1) would segment the same content.

**Why match strong LLM behavior?**
- Strong LLMs use semantic understanding to find meaningful boundaries
- They split at logical transitions, not arbitrary positions
- Matching their behavior gives us semantically meaningful steps without the cost/latency of running an LLM for every segmentation

## Experimental Setup

To develop and evaluate step boundary detectors, we analyzed thinking traces from:

- **Model**: Qwen3-8B with native thinking mode (`<think>` tags)
- **Dataset**: AIME 2025 (first 5 samples)
- **Source**: `outputs/2025-12-11/aime2025_thinking_vllm_22-50-50/results.json`
- **Baseline**: GPT-4.1-mini as reference for "good" step boundaries

### Process

1. Run Qwen3-8B on AIME 2025 problems with thinking mode enabled
2. Extract thinking content from `<think>` tags
3. Apply different step boundary detectors
4. Compare results with GPT-4.1-mini segmentation
5. Analyze statistical and semantic alignment

## Detector Types

### 1. Sentence-based Detectors (`ThinkingSentenceDetector`)

Split text by structural boundaries (paragraphs, sentences).

| Variant | `split_mode` | Description |
|---------|--------------|-------------|
| `sentence_paragraph` | `"paragraph"` | Split only at paragraph breaks (`\n\n`) |
| `sentence_both` | `"both"` | Split by paragraphs, then by sentences for long paragraphs |

**Parameters:**
- `min_step_chars`: Merge steps shorter than this (default: 50)
- `max_step_chars`: Split steps longer than this (default: 800-1000)

**Pros:** Very fast (<1ms), simple
**Cons:** No semantic understanding, splits may not align with reasoning transitions

### 2. Marker-based Detectors (`ThinkingMarkerDetector`)

Split at linguistic transition markers that indicate reasoning flow.

**Marker Categories:**
- **Sequence**: `first`, `second`, `then`, `next`, `finally`
- **Conclusion**: `so`, `therefore`, `thus`, `hence`, `this means`
- **Thinking**: `let me`, `let's`, `wait`, `hmm`, `okay`, `actually`
- **Verification**: `to verify`, `let's check`, `substituting`
- **Reasoning** (v2): `for example`, `given that`, `similarly`

| Variant | Description |
|---------|-------------|
| `marker_all` | All marker categories + structure (paragraphs, bullets) |
| `marker_semantic` | Semantic markers only (no structure markers) |
| `marker_semantic_v2` | Optimized for GPT-4.1 alignment (recommended) |

**Pros:** Fast (~40ms), semantically meaningful splits
**Cons:** Pattern-based, can't handle context-dependent splits

### 3. Hybrid Detector (`ThinkingHybridDetector`)

Combines marker detection with fallback to sentence-based detection.

**Strategy:**
1. Try marker-based detection first
2. If too few steps (`< min_steps`), fall back to sentence detection
3. Normalize step sizes, limit to `max_steps`

**Pros:** Robust across different content types
**Cons:** May produce very long steps if markers are sparse

### 4. Adaptive Detector (`ThinkingAdaptiveDetector`)

Analyzes content first, then selects the best strategy.

**Selection Logic:**
- Has markers + 2+ paragraphs → use marker detector
- Has list structure → use marker detector
- Short text, no structure → use sentence detector
- Complex cases → use hybrid detector

### 5. LLM-based Detectors (`ThinkingLLMDetector`)

Use a secondary LLM to semantically parse thinking content.

| Variant | Model | Description |
|---------|-------|-------------|
| `llm_gpt4` | GPT-4.1-mini | OpenAI API, 8K chunks, verbatim mode |
| `llm_qwen3` | Qwen3-8B | Local vLLM inference |

**Features:**
- Verbatim mode: Only adds "Step N:" markers, preserves all text
- Chunking: Long content split into 8K char chunks with overlap
- Coverage: ~97-99% of original text preserved

**Pros:** Best semantic understanding
**Cons:** Expensive (~$0.01-0.05/sample), slow (~195 sec/sample)

## Experimental Results

### Statistical Comparison

| Detector | Total Steps | Avg Steps/Trace | Avg Chars/Step | Median Chars |
|----------|-------------|-----------------|----------------|--------------|
| sentence_paragraph | 859 | 171.8 | 225.0 | 139 |
| sentence_both | 1183 | 236.6 | 163.1 | 106 |
| marker_all | 1733 | 346.6 | 111.0 | 88 |
| marker_semantic | 1038 | 207.6 | 186.8 | 155 |
| **marker_semantic_v2** | **1075** | **215.0** | **180.1** | **152** |
| hybrid | 142 | 28.4 | 1365.9 | 1405 |
| adaptive | 1733 | 346.6 | 111.0 | 88 |
| **llm_gpt4 (reference)** | **1084** | **216.8** | **179.1** | **116** |

### Alignment with GPT-4.1

| Detector | Diff from GPT-4.1 | Notes |
|----------|-------------------|-------|
| marker_semantic | 4.2% | Good baseline |
| **marker_semantic_v2** | **0.8%** | Best alignment |
| marker_all | 59.9% | Over-splits |
| sentence_paragraph | 20.8% | Under-splits |

### Boundary Position Alignment

Comparing where detectors place step boundaries:

| Metric | marker_semantic_v2 vs GPT-4.1 |
|--------|-------------------------------|
| Exact matches (within 10 chars) | 13.3% |
| Close matches (within 100 chars) | **85.9%** |
| Content similarity (>50%) | 62.9% |

**Interpretation:** 85.9% of GPT-4.1 boundaries have a marker_semantic_v2 boundary within 100 characters. The detectors find similar logical breakpoints.

## Analysis: Why Certain Markers Work

### GPT-4.1 Step Starting Words (from 1084 steps)

| Word/Phrase | Count | In marker_semantic? |
|-------------|-------|---------------------|
| wait | 128 | Yes |
| so | 114 | Yes |
| let me | 100 | Yes |
| but | 102 | No (causes over-splitting) |
| alternatively | 89 | No (causes over-splitting) |
| for example | 18 | Yes (v2) |
| however | 17 | No (causes over-splitting) |
| similarly | 10 | Yes (v2) |
| given that | 7 | Yes (v2) |

### Key Findings

1. **More markers ≠ better alignment**
   - Adding `but`, `however`, `alternatively` caused over-splitting (11%+ diff)
   - GPT-4.1 is selective about WHEN to split on these words
   - Rule-based detection can't replicate this selectivity

2. **Multi-word phrases are safer**
   - `for example`, `given that`, `similarly` are unambiguous
   - They rarely appear mid-sentence as noise
   - Adding these 3 phrases improved alignment from 4.2% to 0.8%

3. **Sentence-start constraints aren't enough**
   - Pattern `(?<=[.!?\n])\s*\bbut\b` still over-splits
   - Many sentences legitimately start with "but" without being new steps

4. **The 80/20 rule applies**
   - 3 selective markers gave most of the improvement
   - Diminishing returns from adding more

## Recommendations

### For Production Use

**Recommended: `marker_semantic_v2`**
- 0.8% statistical difference from GPT-4.1
- 85.9% boundary alignment
- ~5400x faster than LLM-based detection
- Zero API cost

```python
from llm_tts.step_boundary_detectors import ThinkingMarkerDetector

detector = ThinkingMarkerDetector(
    use_sequence=True,
    use_conclusion=True,
    use_thinking=True,
    use_verification=True,
    use_structure=False,  # No paragraph splitting
    use_reasoning=True,   # Includes: for example, given that, similarly
    use_sentence_start=False,  # Avoid over-splitting on but/however
    use_correction=False,
    min_step_chars=100,
    max_step_chars=600,
)
steps = detector.detect_steps(thinking_content)
```

### Use Case Guide

| Use Case | Recommended Detector | Reason |
|----------|---------------------|--------|
| Production (cost-sensitive) | `marker_semantic_v2` | Best quality/cost ratio |
| Real-time streaming | `sentence_paragraph` | Fastest (<1ms) |
| High-quality analysis | `llm_gpt4` | Best semantic understanding |
| Coarse chunking | `hybrid` | Few, large steps |

## Files

- **Detectors**: `llm_tts/step_boundary_detectors/`
- **Analysis script**: `scripts/analyze_thinking_steps.py`
- **Results**: `outputs/2025-12-11/aime2025_thinking_vllm_22-50-50/by_detector/`
- **Detailed analysis log**: `outputs/2025-12-11/aime2025_thinking_vllm_22-50-50/marker_semantic_v2_analysis.log`

## Future Work

1. **Human annotation baseline**: Create ground-truth step boundaries for validation
2. **Domain-specific markers**: Different reasoning styles may need different markers
3. **Adaptive marker selection**: Learn which markers work best for each content type
4. **Lighter LLM baseline**: Test smaller models (GPT-4o-mini, Claude Haiku) as cheaper references
