# Inference Pipeline Architecture

This document describes the two main inference paradigms: **Offline** (full trace generation) and **Online** (step-by-step generation).

## Strategy Types Overview

| Type | Strategies | Generation | Components |
|------|------------|------------|------------|
| **Offline** | Self-Consistency, DeepConf, CoT | Full traces | model + chain scorer |
| **Online** | Best-of-N, Phi Decoding, Adaptive Scaling, Beam Search | Step-by-step | step_generator + detector + step_scorer |

---

## Offline Strategies (Full Trace Generation)

Offline strategies generate complete reasoning traces and then select/aggregate answers.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            STRATEGY                                      │
│  (StrategySelfConsistency, StrategyDeepConf, StrategyChainOfThought)    │
│                                                                          │
│  1. Generate N complete traces (parallel or batched)                     │
│  2. Extract answers from each trace                                      │
│  3. Filter traces (optional, e.g., by confidence)                        │
│  4. Select final answer via voting/aggregation                           │
└─────────────────────────────────────────────────────────────────────────┘
          │                                      │
          ▼                                      ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│        MODEL            │          │     CHAIN SCORER        │
│ (vLLM, HuggingFace,     │          │ (ChainMajorityVoting,   │
│  BlackboxModel)         │          │  ConfidenceFiltering)   │
│                         │          │                         │
│ Generates full traces   │          │ Scores/filters complete │
│ in single call per trace│          │ traces for aggregation  │
└─────────────────────────┘          └─────────────────────────┘
```

### Self-Consistency Flow

```python
strategy = StrategySelfConsistency(
    model=model,
    num_paths=16,
    temperature=0.7,
    scorer=ChainMajorityVotingScorer(),
)

# 1. Generate N complete traces
traces = []
for i in range(num_paths):
    trace = model.generate(prompt, max_tokens=max_tokens)
    traces.append(trace)

# 2. Extract answers from each trace
answers = [extract_answer(trace) for trace in traces]

# 3. Majority voting
answer_counts = Counter(answers)
selected_answer = answer_counts.most_common(1)[0][0]
```

### DeepConf Flow

```python
strategy = StrategyDeepConf(
    model=model,
    mode="offline",
    budget=16,
    filter_method="top10",
    window_size=10,
)

# 1. Generate N traces WITH logprobs
traces = []
for i in range(budget):
    trace, logprobs = model.generate(prompt, output_scores=True)
    confidence = compute_min_window_confidence(logprobs, window_size)
    traces.append({"text": trace, "confidence": confidence})

# 2. Filter by confidence (top-K or threshold)
filtered = sorted(traces, key=lambda t: t["confidence"], reverse=True)[:k]

# 3. Weighted majority voting
for trace in filtered:
    answer = extract_answer(trace["text"])
    answer_weights[answer] += trace["confidence"]

selected_answer = max(answer_weights, key=answer_weights.get)
```

### Component Responsibilities (Offline)

| Component | Class | Responsibility |
|-----------|-------|----------------|
| **Strategy** | `StrategySelfConsistency`, `StrategyDeepConf` | Orchestrates generation and voting |
| **Model** | `VLLMModel`, `WhiteboxModel`, `BlackboxModel` | Generates complete traces |
| **Chain Scorer** | `ChainMajorityVotingScorer` | Extracts answers, performs voting |
| **Answer Extractor** | `extract_answer()` | Parses answer from trace text |

---

## Online Strategies (Step-by-Step Generation)

Online strategies generate reasoning step-by-step, scoring and selecting at each step.

### Sequence Diagram

```
STRATEGY                 STEP_GENERATOR              DETECTOR                 SCORER
   │                          │                          │                       │
   │  generate_candidates()   │                          │                       │
   │─────────────────────────>│                          │                       │
   │                          │                          │                       │
   │                          │  ┌──────────────────────────────────────────┐   │
   │                          │  │ FOR each candidate (1..N):               │   │
   │                          │  │                                          │   │
   │                          │  │   model.generate() with StoppingCriteria │   │
   │                          │  │         │                                │   │
   │                          │  │         │  is_step_complete(text)?       │   │
   │                          │  │         │────────────────────────────>│  │   │
   │                          │  │         │                             │  │   │
   │                          │  │         │  True/False + answer_found  │  │   │
   │                          │  │         │<────────────────────────────│  │   │
   │                          │  │         │                                │   │
   │                          │  │   if True: stop generation              │   │
   │                          │  │   return StepCandidate                  │   │
   │                          │  └──────────────────────────────────────────┘   │
   │                          │                          │                       │
   │  List[StepCandidate]     │                          │                       │
   │<─────────────────────────│                          │                       │
   │                          │                          │                       │
   │  score_candidates(candidates)                       │                       │
   │────────────────────────────────────────────────────────────────────────────>│
   │                          │                          │                       │
   │                          │                          │       List[float]     │
   │<────────────────────────────────────────────────────────────────────────────│
   │                          │                          │                       │
   │  select best candidate   │                          │                       │
   │  add to trajectory       │                          │                       │
   │                          │                          │                       │
   │  if is_trajectory_complete: break                   │                       │
   │  else: repeat for next step                         │                       │
   │                          │                          │                       │
```

### Component Relationships

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STRATEGY                                        │
│  (StrategyOnlineBestOfN, PhiDecoding, AdaptiveScalingBestOfN)               │
│                                                                              │
│  - Holds step_generator and scorer                                           │
│  - Runs main loop: generate → score → select → repeat                        │
│  - Checks is_trajectory_complete to know when answer is found                │
└─────────────────────────────────────────────────────────────────────────────┘
                    │ owns                              │ owns
                    ▼                                   ▼
┌───────────────────────────────────┐    ┌────────────────────────────────────┐
│         STEP GENERATOR            │    │           STEP SCORER              │
│ (StepCandidateGeneratorThrough    │    │ (StepScorerUncertainty,            │
│  Huggingface)                     │    │  StepScorerPRM)                    │
│                                   │    │                                    │
│ - Holds model + detector          │    │ - Scores candidates independently  │
│ - Creates StoppingCriteria        │    │ - Returns List[float] scores       │
│ - Returns List[StepCandidate]     │    │                                    │
└───────────────────────────────────┘    └────────────────────────────────────┘
                    │ owns
                    ▼
┌───────────────────────────────────┐
│      STEP BOUNDARY DETECTOR       │
│ (StructuredStepDetector,          │
│  ThinkingMarkerDetector)          │
│                                   │
│ - Defines step_patterns           │
│ - Defines answer_patterns         │
│ - is_step_complete(text) → bool   │
│ - is_trajectory_complete → bool   │
└───────────────────────────────────┘
                    │ used by
                    ▼
┌───────────────────────────────────┐
│       STOPPING CRITERIA           │
│ (BatchStepStoppingCriteria,       │
│  ThinkingStepStoppingCriteria)    │
│                                   │
│ - HuggingFace callback            │
│ - Called after each token         │
│ - Calls detector.is_step_complete │
│ - Returns True to stop generation │
└───────────────────────────────────┘
```

### Online Best-of-N Flow

```python
strategy = StrategyOnlineBestOfN(
    step_generator=step_generator,  # Has detector inside
    scorer=scorer,
    max_steps=10,
    candidates_per_step=4,
)

trajectory = []

# For each step in trajectory:
for step_num in range(max_steps):

    # 1. Step generator creates candidates
    #    Internally uses detector via StoppingCriteria to know when to stop
    candidates = step_generator(
        request,
        trajectory=trajectory,
        candidates_per_step=4,
    )

    # 2. Scorer evaluates candidates
    scores = scorer.score_candidates(request, candidates)

    # 3. Strategy selects best candidate
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    selected = candidates[best_idx]

    # 4. Add to trajectory
    trajectory.append(selected)

    # 5. Check completion (detector.answer_patterns matched)
    if selected.is_trajectory_complete:
        break
```

### Component Responsibilities (Online)

| Component | Class | Responsibility |
|-----------|-------|----------------|
| **Strategy** | `StrategyOnlineBestOfN`, `PhiDecoding`, etc. | Orchestrates generation loop, selects best candidates |
| **Step Generator** | `StepCandidateGeneratorThroughHuggingface` | Generates candidate steps using LLM |
| **Step Boundary Detector** | `StructuredStepDetector`, `ThinkingMarkerDetector` | Detects step boundaries via patterns |
| **Step Scorer** | `StepScorerUncertainty`, `StepScorerPRM` | Scores candidates for selection |
| **Stopping Criteria** | `BatchStepStoppingCriteria`, `ThinkingStepStoppingCriteria` | HuggingFace callback that uses detector |

### Step Generation Flow (Inside Step Generator)

```
┌──────────────────────────────────────────────────────────────────┐
│                     STEP GENERATOR                                │
│                                                                   │
│  1. Build prompt from request + trajectory                        │
│  2. Create StoppingCriteria with detector                         │
│  3. Call model.generate() with stopping_criteria                  │
│  4. StoppingCriteria checks each token:                           │
│     - Decode generated text                                       │
│     - detector.is_step_complete(text) → stop if True             │
│  5. Return StepCandidate with:                                    │
│     - text: generated step content                                │
│     - is_trajectory_complete: detector.is_trajectory_complete()  │
└──────────────────────────────────────────────────────────────────┘
```

### Answer Detection Flow

When `is_trajectory_complete=True` (answer pattern detected):

```python
if selected_candidate.is_trajectory_complete:
    # Check if answer content actually exists after pattern
    # (stopping criteria may have stopped AT "<Answer>:" without content)
    if not self._has_answer_content(selected_candidate):
        # Remove incomplete step, generate proper answer
        trajectory.pop()
        final_answer = self._generate_final_answer(request, trajectory)
        trajectory.append(final_answer)
    break
```

`_has_answer_content()` uses `step_generator.detector.answer_patterns` to check if there's actual content after the answer marker.

---

## When to Use Which

| Use Case | Strategy Type | Recommended |
|----------|---------------|-------------|
| High throughput, simple voting | Offline | Self-Consistency with vLLM |
| Confidence-based filtering | Offline | DeepConf |
| Fine-grained control per step | Online | Best-of-N, Phi Decoding |
| Adaptive compute allocation | Online | Adaptive Scaling Best-of-N |
| Thinking mode with semantic steps | Online | Best-of-N with ThinkingMarkerDetector |

---

## File References

### Offline Strategies
- Self-Consistency: `llm_tts/strategies/strategy_self_consistency.py`
- DeepConf: `llm_tts/strategies/deepconf/strategy.py`
- Chain of Thought: `llm_tts/strategies/strategy_chain_of_thought.py`

### Online Strategies
- Strategy base: `llm_tts/strategies/strategy_base.py`
- Online Best-of-N: `llm_tts/strategies/strategy_online_best_of_n.py`
- Phi Decoding: `llm_tts/strategies/phi.py`
- Adaptive Scaling: `llm_tts/strategies/adaptive_scaling_best_of_n.py`
- Beam Search: `llm_tts/strategies/strategy_beam_search.py`

### Shared Components
- Step generators: `llm_tts/generators/`
- Step boundary detectors: `llm_tts/step_boundary_detectors/`
- Scorers: `llm_tts/scorers/`
