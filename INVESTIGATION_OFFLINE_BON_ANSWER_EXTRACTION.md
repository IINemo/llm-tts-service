# Investigation: Offline BoN Answer Extraction Bug

## Problem Summary

**Symptom**: GPQA Diamond offline_bon with entropy scorer has only ~14% accuracy (27/198 correct) vs baseline's ~57% accuracy.

**Root Cause**: The `generated_answer` field is empty for 158/198 samples because the answer (`\boxed{B}`) is not being extracted properly from the trajectory.

## Key Findings

### 1. Data Structure

For each sample in `results.json`:
- `generated_trajectory` (from `full_text`): The complete trajectory including thinking + answer phase
- `generated_answer`: Should contain the extracted answer (e.g., "B")
- `answer_step`: The answer phase content (stored separately)
- `steps`: List of reasoning steps

### 2. The Discrepancy

Comparing sample 1:
- `generated_trajectory` ends with: `...answer is:<end of response>` (22,935 chars)
- `answer_step` ends with: `...answer is:\n\n$$\n\boxed{B}\n$$` (2,807 chars)

The `\boxed{B}` is present in `answer_step` but NOT in `generated_trajectory`.

### 3. Code Flow Analysis

**Answer generation happens in 2 phases:**

#### Phase 1: Generate Thinking
- `generate_step_candidates_batch` generates thinking until `</think/topics>` stop token
- Creates `StepCandidate` with `text` and `raw_text`
- For thinking mode: `text = raw_text` (line 672 in vllm.py)

#### Phase 2: Generate Answer
- `generate_answer_candidates_batch` generates answer until `<end of response>` stop token
- vLLM stops at `<end of response>` and EXCLUDES it from output
- Then appends `<end of response>` pattern if not in `c.text` (lines 1197-1201)

**Strategy result construction:**
```python
# Line 328 in strategy_offline_best_of_n.py
traj_data["full_text"] = convert_trajectory_to_string(trajectory)

# Line 335-336
answer_text = answer_step.raw_text or answer_step.text
traj_data["answer_step"] = answer_text
```

### 4. The Bug

The `convert_trajectory_to_string` function (line 55 in base.py):
```python
return "".join([step.text for step in trajectory])
```

It uses `step.text`, but `answer_step.text` has been modified (pattern appended) while `answer_step.raw_text` has the original content.

**Critical Discovery:**
- Position 2788 of both strings shows the divergence
- `answer_step` (raw_text): `...answer is:\n\n$$\n\boxed{B}\n$$`
- `trajectory` (text): `...answer is:<end of response>`

Both are 2805 chars but:
- `text` has `<end of response>` (17 chars) where `raw_text` has `\n\n$$\n\boxed{B}\n$$` (17 chars)

**This means the code is TRUNCATING the answer and appending `<end of response>` instead.**

### 5. Current Fixes Applied

1. **`llm_tts/generators/base.py`**: Changed `convert_trajectory_to_string` to use `raw_text`:
```python
return "".join([step.raw_text or step.text for step in trajectory])
```

2. **`llm_tts/strategies/strategy_offline_best_of_n.py`**: Extract answer from `answer_step` first:
```python
answer_text = best_result.get("answer_step") or best_result.get("full_text", "")
extracted = extract_answer(answer_text)
```

3. **`scripts/run_llm_judge_local.py`**: Handle parentheses in Grade format.

## ROOT CAUSE FOUND

The truncation at line 685-689 was using `self.max_step_tokens` (300) even for answer generation:

```python
actual_hit_max_tokens = stop_reason == "length" or (
    stop_reason is None and len(token_ids) >= self.max_step_tokens  # BUG: Uses 300
)
if actual_hit_max_tokens and not thinking_complete:
    text = self._truncate_at_sentence_boundary(text)
```

For answer phase:
- `thinking_complete = False` (answer text doesn't contain "```")
- `len(token_ids) = 702` (answer is long)
- `self.max_step_tokens = 300`
- So truncation happens!

The answer got truncated at sentence boundary, losing `\boxed{B}` at the end.

## THE FIX

Changed line 687 to use `effective_max_tokens` instead of `self.max_step_tokens`:

```python
actual_hit_max_tokens = stop_reason == "length" or (
    stop_reason is None and len(token_ids) >= effective_max_tokens  # FIXED: Uses override
)
```

Now for answer phase, `effective_max_tokens = 32768`, so no truncation happens.

## Files Modified

- `llm_tts/strategies/strategy_offline_best_of_n.py`
- `llm_tts/generators/base.py`
- `scripts/run_llm_judge_local.py`

## Commit

```
1cb2b32 fix: extract answer from answer_step in offline_bon for thinking mode
```
