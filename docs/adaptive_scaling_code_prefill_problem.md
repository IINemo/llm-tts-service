# Adaptive Scaling for Code: The Prefill Problem with Chat APIs

## Summary

Adaptive scaling (MUR) fails for code generation when using OpenAI-compatible chat APIs (e.g., GPT-4o-mini via OpenRouter) because these APIs **do not support assistant prefilling**. The model treats the accumulated trajectory as a complete prior response and starts a fresh answer at every step, producing degenerate looping output.

## Background

Adaptive scaling generates text step-by-step:
1. Generate N candidates for the next step
2. Score candidates using entropy/perplexity signal
3. Select the best candidate
4. Append it to the trajectory
5. Repeat until completion

This requires the model to **continue** from the accumulated trajectory — i.e., given the partial output so far, generate the next chunk. This works with:
- **vLLM / local models** — trajectory is injected directly into the KV cache as a prompt prefix
- **Anthropic Claude API** — has explicit `prefix` support for assistant messages
- **Legacy Completion APIs** — raw text continuation by design

## The Problem

OpenAI's Chat Completions API (and OpenRouter proxying to OpenAI) does **not** support assistant prefilling. When the code appends the trajectory as an assistant message:

```python
# From llm_tts/generators/api.py (_prepare_request)
request_with_trajectory.append({
    "role": "assistant",
    "content": convert_trajectory_to_string(trajectory),
    "prefix": True,  # Silently ignored by OpenAI/OpenRouter
})
```

The `"prefix": True` field is **not a standard OpenAI/OpenRouter field** and is silently ignored. The assistant message is treated as a **complete prior turn** in the conversation, not as a partial response to continue from.

## What Happens in Practice

### Step 1 (initial generation)

The model receives:
```
User: "You are given the following Python function signature and docstring.
       Generate a self-contained Python script...
       def flip_case(string: str) -> str: ..."
```

The model generates until hitting the first `\n` boundary (after min_step_tokens):
```
"Certainly! Below is a self-contained Python script...

```python
def flip_case(string: str) -> str:
    """ For a given string, flip lowercase characters to uppercase...
    >>>"
```

### Step 2 (continuation attempt)

The code sends:
```
User:      "Generate a self-contained Python script..."
Assistant: "Certainly! Below is... def flip_case(...):\n    >>>"   ← COMPLETE prior turn
[Model generates NEW assistant response]
```

The model sees its "previous response" as complete and starts a **new, better** response:
```
"Here is the complete self-contained Python script that implements the `flip_case` function:

```python
def flip_case(string: str) -> str:
    """ For a given string, flip lowercase characters to uppercase...
    >>>"
```

### Steps 3-250 (degenerate loop)

Each step appends another restart to the trajectory. By step 8, the trajectory looks like:

```
Certainly! Below is a self-contained Python script...
def flip_case(string: str) -> str:
    """ For a given string...
    >>>Here is the complete self-contained Python script...
def flip_case(string: str) -> str:
    """ For a given string...
    >>>Here is the complete self-contained Python script...
def flip_case(string: str) -> str:
    """ For a given string...
    >>>Here is the complete self-contained Python script...
    >>>
```

The model sees an increasingly long, nonsensical trajectory of repeated function stubs and keeps restarting.

### Final result

- All samples hit `max_steps=250`
- Each step is ~50-80 tokens of the same restarted function signature
- Validity scores stay uniformly high (0.85-0.95) because each individual step looks like valid code
- The concatenated trajectory is garbage
- **Accuracy: 5/32 (15.6%)** — only samples where the first step happened to contain a complete correct solution

## Attempted Fix: Increasing min_step_tokens

Changed `min_step_tokens` from 10 to 200 so each step covers a larger code block.

**Result: worse (0% accuracy)**

- ~75% of samples now complete at step 0 — the model generates a full response (200+ tokens), hits EOS naturally, and the entire output is one single step. No scaling happens, so it degrades to single-shot baseline.
- ~25% of samples still loop with the same restart pattern, just with bigger chunks per step.
- The single-shot GPT-4o-mini responses happen to be incorrect, giving 0% accuracy.

## Root Cause

The adaptive scaling strategy assumes the ability to do **text continuation** — append partial output and have the model continue from it. This is a property of:
- Completion APIs (text in → text out)
- Models with explicit prefill support (Claude, vLLM)

Chat Completion APIs fundamentally work as **turn-based conversation**. An assistant message is a complete turn, and the model generates a new turn in response. There is no mechanism to say "continue this partial response."

## Possible Solutions

### 1. Use offline Best-of-N for code (recommended)
Generate N complete solutions in one shot, score them, pick the best. Already implemented and working. No prefill needed.

### 2. Use a model/API that supports prefill
- Local models via vLLM (trajectory injected into KV cache)
- Anthropic Claude API (native `prefix` support)

### 3. Prompt-based continuation (experimental)
Modify `_prepare_request()` to detect non-prefill APIs and use a different message structure:
```
User:      "Write function..."
Assistant: "def flip_case(string: str) -> str:\n    ..."
User:      "Continue writing from exactly where you left off. Do not restart or repeat any code."
```
This would require changes to the generator and is not guaranteed to work reliably.

## Experiment Details

| Config | `min_step_tokens=10` | `min_step_tokens=200` |
|--------|---------------------|----------------------|
| Accuracy | 5/32 (15.6%) | 0/32+ (0%) |
| Avg steps | 250 (all max) | 1 (75%) or 250 (25%) |
| Avg tokens/sample | 17,443 | TBD |
| Degenerate restarts | Yes, every step | Yes, for non-EOS samples |

**Config**: `adaptive_scaling_openrouter_gpt4o_mini_human_eval_plus_entropy.yaml`
**Model**: `openai/gpt-4o-mini` via OpenRouter
**Strategy**: Adaptive scaling with newline boundaries (`\n`, `\n\n`)
**Date**: 2026-02-15
