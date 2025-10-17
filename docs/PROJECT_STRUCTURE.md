# Project Structure

This document provides a comprehensive overview of the LLM Test-Time Scaling Service codebase organization.

---

## Directory Tree

```
llm-tts-service/
├── config/                           # Hydra configuration files
│   ├── experiments/                  # Complete experiment configs
│   │   ├── deepconf/                # DeepConf experiments (offline/online)
│   │   ├── chain_of_thought/        # Chain-of-thought experiments
│   │   └── self_consistency/        # Self-consistency experiments
│   ├── dataset/                      # Dataset configs (gsm8k, math, etc.)
│   ├── model/                        # Model configs (openai, openrouter, hf, etc.)
│   ├── strategy/                     # Strategy-specific parameters
│   ├── scorer/                       # Scorer configs (PRM, uncertainty, etc.)
│   ├── generation/                   # Generation parameters
│   ├── evaluation/                   # Evaluation configs (llm_judge, alignscore)
│   └── system/                       # System settings (device, seed, etc.)
│
├── llm_tts/                          # Main library package
│   ├── strategies/                   # TTS strategy implementations
│   │   ├── deepconf/                # DeepConf strategy (offline/online modes)
│   │   │   ├── strategy.py          # Main strategy implementation
│   │   │   └── utils.py             # Confidence computation utilities
│   │   ├── strategy_base.py         # Abstract base class for all strategies
│   │   ├── strategy_online_best_of_n.py
│   │   ├── strategy_self_consistency.py
│   │   └── strategy_chain_of_thought.py
│   │
│   ├── models/                       # Model wrappers
│   │   ├── blackboxmodel_with_streaming.py  # OpenAI-compatible model with streaming
│   │   └── base.py                  # Base model interface
│   │
│   ├── scorers/                      # Step scoring implementations
│   │   ├── step_scorer_base.py      # Base scorer interface
│   │   ├── step_scorer_prm.py       # Process Reward Model scorer
│   │   ├── step_scorer_uncertainty.py # Uncertainty-based scorer
│   │   └── majority_voting.py       # Majority voting scorer
│   │
│   ├── evaluation/                   # Evaluation methods
│   │   ├── llm_as_a_judge.py       # LLM-based correctness verification
│   │   ├── exact_match.py          # Direct answer comparison
│   │   └── alignscore.py           # Semantic similarity evaluation
│   │
│   ├── datasets/                     # Dataset utilities
│   │   └── gsm8k.py                # GSM8K dataset loading and processing
│   │
│   ├── early_stopping.py            # Early stopping conditions for streaming
│   ├── step_boundary_detector.py    # Detects step/answer boundaries
│   ├── step_candidate_generator_base.py
│   ├── step_candidate_generator_through_api.py
│   └── step_candidate_generator_through_huggingface.py
│
├── scripts/
│   └── run_tts_eval.py              # Main evaluation script
│
├── tests/                            # Test suite
│   ├── strategy_registry.py         # Strategy registry and validation
│   ├── deepconf/                    # DeepConf strategy tests
│   │   ├── test_deepconf_accurate.py
│   │   ├── test_online_mode.py
│   │   └── test_deepconf_math.py
│   ├── online_best_of_n/            # Best-of-N strategy tests
│   └── run_tts_eval/                # Integration tests
│
├── docs/                             # Documentation
│   ├── deepconf/                    # DeepConf strategy guide
│   │   └── DeepConf.md
│   ├── datasets/                    # Dataset documentation
│   │   └── GSM8K/
│   ├── STRATEGY_REGISTRATION.md     # Strategy registry guide
│   └── PROJECT_STRUCTURE.md         # This file
│
├── lm-polygraph/                     # Submodule: uncertainty estimation library
│
├── Makefile                          # Development commands (format, lint, test)
├── pyproject.toml                    # Package configuration and dependencies
├── setup.py                          # Package setup
├── setup.sh                          # Installation script
└── .github/workflows/                # CI/CD pipelines
    └── test.yml
```

---

## Core Components

### 1. Strategies (`llm_tts/strategies/`)

TTS strategy implementations that control how test-time computation is used.

#### Strategy Base (`strategy_base.py`)
Abstract base class that all strategies must inherit from:
```python
class StrategyBase:
    def generate_trajectory(self, prompt) -> Dict:
        """
        Generate reasoning trajectory for a prompt.

        Returns:
            dict with keys: trajectory, steps, validity_scores, completed, metadata
        """
        raise NotImplementedError
```

#### DeepConf (`deepconf/`)
Confidence-based test-time scaling strategy.

**Files:**
- `strategy.py` - Main implementation with offline/online modes
- `utils.py` - Confidence computation using lm-polygraph methods

**Modes:**
- **Offline**: Generate N traces → compute confidence → filter → majority vote
- **Online**: Warmup phase → adaptive generation with early stopping

**Key Features:**
- Uses lm-polygraph for uncertainty estimation
- Supports multiple filtering methods (top5, top10, threshold)
- Answer extraction for math problems (`\boxed{answer}`)

#### Online Best-of-N (`strategy_online_best_of_n.py`)
Step-by-step generation with process reward model scoring.

**Algorithm:**
1. Generate K candidates for current step
2. Score each candidate with PRM
3. Select best candidate
4. Repeat until answer reached

#### Self-Consistency (`strategy_self_consistency.py`)
Majority voting across multiple complete reasoning chains.

**Algorithm:**
1. Generate N independent solutions
2. Extract final answers
3. Return most frequent answer

#### Chain-of-Thought (`strategy_chain_of_thought.py`)
Single-pass step-by-step reasoning without test-time scaling.

---

### 2. Models (`llm_tts/models/`)

Model wrappers that provide unified interface to different LLM APIs.

#### BlackboxModelWithStreaming (`blackboxmodel_with_streaming.py`)
Main model wrapper extending lm-polygraph's `BlackboxModel`.

**Features:**
- Streaming generation support
- OpenAI API compatibility
- Custom base_url for OpenRouter
- Early stopping integration
- Logprobs support for confidence computation

**Usage:**
```python
model = BlackboxModelWithStreaming(
    openai_api_key=api_key,
    model_path="openai/gpt-4o-mini",
    supports_logprobs=True,
    base_url="https://openrouter.ai/api/v1",  # For OpenRouter
    early_stopping=BoundaryEarlyStopping(detector)
)
```

#### Base Model (`base.py`)
Abstract interface that all model wrappers must implement.

---

### 3. Scorers (`llm_tts/scorers/`)

Scoring functions for evaluating generation steps/candidates.

#### Step Scorer Base (`step_scorer_base.py`)
Abstract base for all step scorers:
```python
class StepScorerBase:
    def score_steps(self, trajectory, candidates) -> List[float]:
        """Score candidate next steps."""
        raise NotImplementedError
```

#### Process Reward Model (`step_scorer_prm.py`)
Uses trained PRM to score reasoning steps.

#### Uncertainty Scorer (`step_scorer_uncertainty.py`)
Scores based on model uncertainty (entropy, etc.).

#### Majority Voting (`majority_voting.py`)
Scores based on frequency across multiple samples.

---

### 4. Evaluation (`llm_tts/evaluation/`)

Methods for evaluating solution correctness.

#### LLM as a Judge (`llm_as_a_judge.py`)
Uses LLM to verify if solution matches gold answer.

**Prompt Template:**
```
Problem: {problem}
Student Solution: {solution}
Gold Answer: {gold_answer}

Is the student solution correct?
<Grade>: Correct/Incorrect
```

#### Exact Match (`exact_match.py`)
Direct string comparison of extracted answers.

#### AlignScore (`alignscore.py`)
Semantic similarity between solution and gold answer.

---

### 5. Configuration (`config/`)

Hierarchical Hydra configuration system.

#### Structure:
```
config/
├── experiments/          # Complete configs (compose from components)
├── dataset/             # Dataset-specific settings
├── model/               # Model providers and parameters
├── strategy/            # Strategy hyperparameters
├── scorer/              # Scorer configurations
├── generation/          # Generation parameters (temp, top_p, etc.)
├── evaluation/          # Evaluation method configs
└── system/              # System settings (device, seed)
```

#### Example Experiment Config:
```yaml
# experiments/deepconf/run_gsm8k_deepconf_offline.yaml

defaults:
  - /dataset/gsm8k
  - /model/openrouter
  - /strategy/deepconf
  - /generation/default
  - /evaluation/llm_judge
  - /system/default

# Override specific parameters
strategy:
  mode: offline
  budget: 8
  filter_method: top5

model:
  model_path: "openai/gpt-3.5-turbo"
```

See `config/README.md` for detailed configuration guide.

---

### 6. Early Stopping (`llm_tts/early_stopping.py`)

Pluggable conditions for stopping streaming generation.

#### Available Conditions:

**BoundaryEarlyStopping**
- Stops when step/answer boundary detected
- Used by Best-of-N strategies

**ConfidenceEarlyStopping**
- Stops when confidence drops below threshold
- Used by DeepConf online mode

**CompositeEarlyStopping**
- Combines multiple conditions
- Stops when any condition triggers

**NoEarlyStopping**
- Never stops early (generates until max_tokens)

---

### 7. Datasets (`llm_tts/datasets/`)

Dataset loading and processing utilities.

#### GSM8K (`gsm8k.py`)
Grade School Math 8K dataset utilities:
- Load from HuggingFace
- Extract answers from `\boxed{}` format
- Validation helpers

---

### 8. Tests (`tests/`)

Comprehensive test suite with strategy registry.

#### Structure:
```
tests/
├── strategy_registry.py          # Central registry
├── deepconf/                     # Strategy-specific tests
│   ├── test_deepconf_accurate.py
│   ├── test_online_mode.py
│   └── test_deepconf_math.py
├── online_best_of_n/
│   └── test_online_best_of_n.py
└── run_tts_eval/
    └── test_run_tts_eval.py
```

See [Strategy Registration Guide](STRATEGY_REGISTRATION.md) for testing requirements.

---

### 9. Scripts (`scripts/`)

Main evaluation and utility scripts.

#### run_tts_eval.py
Main evaluation script with two-phase pipeline:

**Phase 1: Generation**
- Load dataset and model
- Run strategy on each sample
- Save trajectories incrementally
- Resume support for long runs

**Phase 2: Evaluation**
- Load generated trajectories
- Run evaluators (LLM Judge, Exact Match, etc.)
- Compute metrics
- Save results

**Usage:**
```bash
python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name experiments/deepconf/run_gsm8k_deepconf_offline \
  dataset.subset=10
```

---

## Development Workflow

### 1. Adding a New Strategy

```bash
# 1. Implement strategy
touch llm_tts/strategies/strategy_my_new.py

# 2. Create tests
mkdir tests/my_new
touch tests/my_new/test_my_new.py

# 3. Register in strategy registry
# Edit tests/strategy_registry.py

# 4. Validate
python tests/strategy_registry.py --validate

# 5. Run tests
pytest tests/my_new/ -v
```

### 2. Adding a New Dataset

```bash
# 1. Implement dataset loader
touch llm_tts/datasets/my_dataset.py

# 2. Create config
touch config/dataset/my_dataset.yaml

# 3. Add documentation
mkdir docs/datasets/MY_DATASET
touch docs/datasets/MY_DATASET/README.md
```

### 3. Adding a New Scorer

```bash
# 1. Implement scorer
touch llm_tts/scorers/step_scorer_my_scorer.py

# 2. Create config
touch config/scorer/my_scorer.yaml

# 3. Update run_tts_eval.py to support new scorer
```

---

## Key Design Patterns

### 1. Strategy Pattern
All strategies inherit from `StrategyBase` and implement `generate_trajectory()`.

### 2. Dependency Injection
Models, scorers, and evaluators are injected into strategies via constructor.

### 3. Configuration Composition
Hydra configs compose from smaller config files using `defaults:`.

### 4. Plugin Architecture
Early stopping conditions are pluggable - strategies accept any `EarlyStopping` implementation.

### 5. Registry Pattern
Strategy registry tracks all strategies and enforces test requirements.

---

## External Dependencies

### lm-polygraph
Uncertainty estimation library installed as git submodule.

**Used for:**
- Confidence computation (entropy, perplexity)
- Uncertainty methods (`Categorical`, `MaximumTokenProbability`)
- Base model interfaces

**Installation:**
```bash
./setup.sh  # Installs lm-polygraph dev branch
```

---

## References

- **Configuration Guide**: `config/README.md`
- **Strategy Registration**: `docs/STRATEGY_REGISTRATION.md`
- **Testing Guide**: `tests/README.md`
- **DeepConf Documentation**: `docs/deepconf/DeepConf.md`
- **GSM8K Dataset**: `docs/datasets/GSM8K/README.md`
