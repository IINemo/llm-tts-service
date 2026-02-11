# Analysis: Token Limit Variables in vLLM Generator

## Current Variables (Confusing!)

### 1. Instance Variables (set in `__init__`)

| Variable | Source | Default | Purpose |
|----------|--------|---------|---------|
| `max_new_tokens` | Config `generation.max_new_tokens` | 4096 | Max tokens for thinking phase in thinking mode |
| `max_answer_tokens` | Config `generation.max_answer_tokens` | 512 | Max tokens for answer phase in NON-thinking mode |
| `max_step_tokens` | Detector attribute | 300 | Used for step boundary detection and truncation |
| `min_step_tokens` | Detector attribute | 0 | Minimum tokens per step |
| `max_context_budget` | Config | 32768 | Total context limit for prompt + generation |

### 2. Local Variables (computed per call)

| Variable | Computed In | Value | Purpose |
|----------|-------------|-------|---------|
| `effective_max_tokens` | `_generate_step_candidates_impl` | `max_tokens_override or max_step_tokens` | Actual limit for generation |
| `max_tokens_override` | Parameter passed to methods | Varies | Override default limit |
| `answer_max_tokens` | `generate_answer_candidates_batch` | `max_new_tokens` (thinking) or `max_answer_tokens` (non-thinking) | Limit for answer phase |

## Problems

### 1. **Naming is inconsistent and confusing**
- `max_new_tokens` vs `max_step_tokens` vs `max_tokens_override` vs `effective_max_tokens` vs `answer_max_tokens`
- Hard to understand what each does

### 2. **max_step_tokens has TWO different purposes**
- Step boundary detection (splitting thinking into steps)
- Truncation limit (cutting off generation)
- These should be separate!

### 3. **max_answer_tokens is only used for non-thinking mode**
- In thinking mode, answer phase uses `max_new_tokens` (32K)
- In non-thinking mode, answer phase uses `max_answer_tokens` (512)
- This is confusing!

### 4. **effective_max_tokens is a workaround**
- Created to fix the bug where `max_step_tokens` was used for truncation
- But the logic is still convoluted

## Flow Analysis

### Thinking Mode

**Phase 1: Generate Thinking**
1. Call `generate_step_candidates_batch`
2. `_generate_step_candidates_impl` with `max_tokens_override=None`
3. `effective_max_tokens = max_step_tokens (300)`
4. Generate until stop token (`</think/topics>`)
5. If hit 300 tokens → truncate at sentence boundary

**Phase 2: Generate Answer**
1. Call `generate_answer_candidates_batch`
2. `answer_max_tokens = max_new_tokens (32768)`
3. `_generate_step_candidates_impl` with `max_tokens_override=32768`
4. `effective_max_tokens = 32768`
5. Generate until stop token (`<end of response>`)
6. If hit 32768 tokens → truncate at sentence boundary

### Non-Thinking Mode

**Single Phase**
1. Call `generate_step_candidates_batch`
2. `_generate_step_candidates_impl` with `max_tokens_override=None`
3. `effective_max_tokens = max_step_tokens (300)`
4. Generate until stop token or hit 300 tokens
5. Answer is extracted from the single generation

## Proposed Refactoring

### Option A: Clearer Variable Names

```python
# Instance variables
self.thinking_max_tokens = max_new_tokens  # For thinking phase (32K)
self.answer_max_tokens = max_answer_tokens  # For answer in non-thinking mode (512)
self.step_boundary_tokens = 300  # For step detection only (not truncation!)
self.context_limit = max_context_budget  # Total context limit

# Remove max_step_tokens entirely or rename to step_boundary_tokens
```

### Option B: Mode-Specific Configs

```python
# Thinking mode limits
self.limits = {
    "thinking": max_new_tokens,      # 32K
    "answer": max_new_tokens,        # 32K (same as thinking in thinking mode)
}

# Non-thinking mode limits
self.limits = {
    "thinking": None,                # Not used
    "answer": max_answer_tokens,     # 512
}
```

### Option C: Phase-Aware Generation

Instead of `max_tokens_override`, pass the phase explicitly:

```python
def _generate_step_candidates_impl(
    self,
    ...,
    phase: Literal["thinking", "answer"] = "thinking",
):
    if phase == "thinking":
        max_tokens = self.thinking_max_tokens
    else:
        max_tokens = self.answer_max_tokens if not self.thinking_mode else self.thinking_max_tokens
```

## Immediate Fix (Already Applied)

Changed truncation check to use `effective_max_tokens` instead of `max_step_tokens`:

```python
# Before (bug):
actual_hit_max_tokens = ... len(token_ids) >= self.max_step_tokens

# After (fixed):
actual_hit_max_tokens = ... len(token_ids) >= effective_max_tokens
```

## Recommended Next Steps

1. **Rename variables for clarity**:
   - `max_step_tokens` → `step_boundary_tokens` (make clear it's for detection, not truncation)
   - `max_tokens_override` → `generation_limit` (clearer purpose)
   - `effective_max_tokens` → can stay, or use `generation_limit` directly

2. **Separate concerns**:
   - Step boundary detection limit
   - Generation truncation limit
   - These can be different!

3. **Add documentation**:
   - Document what each limit does
   - Document the flow for thinking vs non-thinking mode

4. **Consider a LimitsConfig class**:
   ```python
   @dataclass
   class GenerationLimits:
       thinking_phase: int  # Max tokens for thinking
       answer_phase: int    # Max tokens for answer
       step_boundary: int   # For step detection
       context_total: int   # Total context budget
   ```
