# Inference Pipeline Architecture

This document describes how **Strategy**, **Step Generator**, **Step Boundary Detector**, and **Step Scorer** work together for step-by-step inference.

## Component Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            STRATEGY                                      │
│  (StrategyOnlineBestOfN, PhiDecoding, AdaptiveScalingBestOfN, etc.)     │
│                                                                          │
│  Orchestrates the generation loop:                                       │
│  1. Call step_generator to get candidates                                │
│  2. Call scorer to rank candidates                                       │
│  3. Select best candidate, add to trajectory                             │
│  4. Repeat until complete or max_steps                                   │
└─────────────────────────────────────────────────────────────────────────┘
          │                                      │
          ▼                                      ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│    STEP GENERATOR       │          │      STEP SCORER        │
│ (StepCandidateGenerator │          │ (StepScorerUncertainty, │
│  ThroughHuggingface)    │          │  StepScorerPRM, etc.)   │
│                         │          │                         │
│ Generates N candidate   │          │ Scores candidates to    │
│ steps using the model   │          │ select the best one     │
└─────────────────────────┘          └─────────────────────────┘
          │
          ▼
┌─────────────────────────┐
│ STEP BOUNDARY DETECTOR  │
│ (StructuredStepDetector,│
│  ThinkingMarkerDetector)│
│                         │
│ Detects step boundaries │
│ during generation via   │
│ StoppingCriteria        │
└─────────────────────────┘
```

## Data Flow

```python
# 1. Strategy initializes components
strategy = StrategyOnlineBestOfN(
    step_generator=step_generator,  # Has detector inside
    scorer=scorer,
    max_steps=10,
    candidates_per_step=4,
)

# 2. For each step in trajectory:
for step_num in range(max_steps):

    # 3. Step generator creates candidates
    #    Internally uses detector via StoppingCriteria to know when to stop
    candidates = step_generator(
        request,
        trajectory=trajectory,
        candidates_per_step=4,
    )

    # 4. Scorer evaluates candidates
    scores = scorer.score_candidates(request, candidates)

    # 5. Strategy selects best candidate
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    selected = candidates[best_idx]

    # 6. Add to trajectory
    trajectory.append(selected)

    # 7. Check completion (detector.answer_patterns matched)
    if selected.is_trajectory_complete:
        break
```

## Component Responsibilities

| Component | Class | Responsibility |
|-----------|-------|----------------|
| **Strategy** | `StrategyOnlineBestOfN`, `PhiDecoding`, etc. | Orchestrates generation loop, selects best candidates |
| **Step Generator** | `StepCandidateGeneratorThroughHuggingface` | Generates candidate steps using LLM |
| **Step Boundary Detector** | `StructuredStepDetector`, `ThinkingMarkerDetector` | Detects step boundaries via patterns |
| **Step Scorer** | `StepScorerUncertainty`, `StepScorerPRM` | Scores candidates for selection |
| **Stopping Criteria** | `BatchStepStoppingCriteria`, `ThinkingStepStoppingCriteria` | HuggingFace callback that uses detector |

## Step Generation Flow (Inside Step Generator)

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

## Answer Detection Flow

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

## File References

- Strategy base: `llm_tts/strategies/strategy_base.py`
- Online Best-of-N: `llm_tts/strategies/strategy_online_best_of_n.py`
- Phi Decoding: `llm_tts/strategies/phi.py`
- Adaptive Scaling: `llm_tts/strategies/adaptive_scaling_best_of_n.py`
- Step generators: `llm_tts/generators/`
- Step boundary detectors: `llm_tts/step_boundary_detectors/`
