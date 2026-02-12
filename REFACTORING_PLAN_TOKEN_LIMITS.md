# Refactoring Plan: Token Limit Variables

## Current State Analysis

### The Variables

| Variable | Where Defined | Default | What It Actually Controls |
|----------|---------------|---------|---------------------------|
| `max_new_tokens` | `__init__` param | 4096 | **Generation limit** for thinking phase AND answer phase in thinking mode |
| `max_answer_tokens` | `__init__` param | 512 | **Generation limit** for answer phase in NON-thinking mode only |
| `max_step_tokens` | Detector attribute | 300 | **Step detection** (splitting into steps) + **EOS detection threshold** |
| `max_tokens_override` | Method param | None | Override for a specific generation call |
| `effective_max_tokens` | Computed locally | - | Actual limit used: `max_tokens_override or max_step_tokens` |
| `answer_max_tokens` | Computed locally | - | `max_new_tokens` (thinking mode) or `max_answer_tokens` (non-thinking) |

### The Callers

**Thinking Phase Generation:**
```python
# strategy_offline_best_of_n.py line 397-402
batch_results = self.step_generator.generate_step_candidates_batch(
    ...
    max_tokens_override=self.step_generator.max_new_tokens,  # 32K for thinking
)
```

**Answer Phase Generation:**
```python
# vllm.py line 1192-1200
answer_max_tokens = self.max_new_tokens if self.thinking_mode else self.max_answer_tokens
results = self._generate_step_candidates_impl(
    ...
    max_tokens_override=answer_max_tokens,
)
```

### The Problems

1. **`max_step_tokens` (300) is NOT a generation limit** - it's used for:
   - Step boundary detection (finding where to split thinking into steps)
   - EOS detection threshold (`stopped_at_eos = ... len(token_ids) < self.max_step_tokens`)
   - But it was **accidentally used for truncation** (the bug we just fixed)

2. **`max_new_tokens` is used for BOTH phases in thinking mode** - confusing because:
   - Name suggests it's for "new tokens" generally
   - But it's actually the limit for BOTH thinking AND answer in thinking mode

3. **`max_answer_tokens` only used in non-thinking mode** - but:
   - Name sounds like it should be the answer limit for all modes
   - Actually only used when `thinking_mode=False`

4. **`effective_max_tokens` is a workaround** - it exists because:
   - We need to override the default on a per-call basis
   - The default (`max_step_tokens`) was wrong for answer phase

## Root Cause

The fundamental issue is that **we have 3 different concepts mixed together**:

1. **Generation Limit** - How many tokens can the model generate?
2. **Step Detection Threshold** - How do we split long thinking into steps?
3. **EOS Detection Threshold** - How many tokens before we consider it "hit max tokens"?

These should be separate variables with clear names!

## Proposed Refactoring

### Phase 1: Rename and Clarify (Low Risk)

```python
class VLLMStepGenerator:
    def __init__(self, ...):
        # Generation limits (how many tokens can be generated)
        self.thinking_generation_limit = max_new_tokens      # 32768 for thinking phase
        self.answer_generation_limit = max_answer_tokens     # 512 for non-thinking answer

        # Step detection (for splitting thinking into logical steps)
        self.step_detection_threshold = 300  # tokens per step boundary

        # Context limit (total budget)
        self.context_limit = max_context_budget  # 32768
```

**Changes:**
- `max_new_tokens` → `thinking_generation_limit`
- `max_answer_tokens` → `answer_generation_limit`
- `max_step_tokens` → `step_detection_threshold`
- `max_context_budget` → `context_limit`

### Phase 2: Simplify Method Signature

Replace `max_tokens_override` with a cleaner pattern:

```python
def _generate_step_candidates_impl(
    self,
    ...,
    generation_limit: Optional[int] = None,  # Clear name!
):
    """
    Args:
        generation_limit: Max tokens to generate. If None, uses:
            - thinking_generation_limit for thinking phase
            - answer_generation_limit for non-thinking mode
    """
    if generation_limit is None:
        generation_limit = self.thinking_generation_limit

    # Use generation_limit directly, no more "effective_*" variable
```

### Phase 3: Separate EOS Detection from Step Detection

Currently `max_step_tokens` (300) is used for BOTH:
- Step boundary detection
- EOS detection (`len(token_ids) < self.max_step_tokens`)

These should be separate:

```python
# Step detection - for splitting thinking into logical parts
self.step_detection_threshold = 300

# EOS detection - for determining if model stopped naturally
self.eos_detection_threshold = 300  # Can be same initially
```

### Phase 4: Add a LimitsConfig Dataclass (Optional, Higher Risk)

```python
@dataclass
class GenerationLimits:
    """Token limits for generation phases."""
    thinking: int = 32768      # Max tokens for thinking phase
    answer: int = 512         # Max tokens for answer phase (non-thinking mode)
    step_boundary: int = 300  # Tokens per step for detection
    context_total: int = 32768 # Total context budget

    def get_limit(self, phase: str, thinking_mode: bool) -> int:
        """Get generation limit for a specific phase."""
        if phase == "thinking":
            return self.thinking
        elif phase == "answer":
            return self.thinking if thinking_mode else self.answer
        raise ValueError(f"Unknown phase: {phase}")
```

## Recommended Implementation Order

1. **NOW: Fix the bug** (already done) - Use `effective_max_tokens` for truncation check

2. **NEXT PR: Rename variables for clarity** (Phase 1)
   - Add new names as aliases
   - Deprecate old names with warnings
   - Low risk, high clarity improvement

3. **LATER: Simplify method signatures** (Phase 2)
   - Replace `max_tokens_override` with `generation_limit`
   - Remove `effective_max_tokens` variable

4. **OPTIONAL: Add LimitsConfig** (Phase 4)
   - Only if we want to pass limits around as a group
   - Higher risk, may not be necessary

## Quick Fix (Immediate)

The bug fix already applied:

```python
# Line 686-688: Use effective_max_tokens instead of max_step_tokens
actual_hit_max_tokens = stop_reason == "length" or (
    stop_reason is None and len(token_ids) >= effective_max_tokens
)
```

This ensures:
- Thinking phase: truncation at `thinking_generation_limit` (32768)
- Answer phase: truncation at `thinking_generation_limit` (32768) for thinking mode
- Non-thinking: truncation at `answer_generation_limit` (512)

## Summary

| Before | After (Proposed) | Purpose |
|--------|------------------|---------|
| `max_new_tokens` | `thinking_generation_limit` | Max tokens for thinking generation |
| `max_answer_tokens` | `answer_generation_limit` | Max tokens for answer (non-thinking only) |
| `max_step_tokens` | `step_detection_threshold` | Step boundary detection |
| `max_tokens_override` | `generation_limit` | Override for specific call |
| `effective_max_tokens` | (remove) | No longer needed |
| `max_context_budget` | `context_limit` | Total context budget |
