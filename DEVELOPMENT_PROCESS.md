# Development Process

This document tracks the development history and major changes to the LLM Test-Time Scaling (TTS) service.

## Recent Changes

### 2024-10-16: Model Architecture Cleanup

**Summary**: Removed redundant model adapter files and consolidated to unified `BlackboxModelWithStreaming` approach.

**Changes Made**:
1. **Removed redundant files** (~700 lines of code):
   - `llm_tts/models/openrouter.py` - OpenRouter adapter (redundant)
   - `llm_tts/models/together_ai.py` - Together AI adapter (redundant)
   - `llm_tts/models/factory.py` - Factory pattern (unused)

2. **Updated imports**:
   - `llm_tts/models/__init__.py` - Now exports only `BaseModel` and `BlackboxModelWithStreaming`
   - `tests/deepconf/test_deepconf_accurate.py` - Updated to use `BlackboxModelWithStreaming` directly
   - `tests/deepconf/test_deepconf_math.py` - Updated to use `BlackboxModelWithStreaming` directly
   - `docs/deepconf/DeepConf.md` - Updated documentation examples

3. **Simplified test imports**:
   - Removed complex `importlib.util` workarounds
   - Now using standard imports from `llm_tts.strategies`
   - Fixed strategy class name references (`DeepConfStrategy` → `StrategyDeepConf`)

**Rationale**:
- `BlackboxModelWithStreaming` already handles all providers via the `base_url` parameter:
  - OpenRouter: `base_url="https://openrouter.ai/api/v1"`
  - OpenAI: `base_url=None` (default)
  - Together AI: Compatible with OpenAI API, can use custom `base_url`
- Unified approach reduces code duplication and maintenance burden
- All strategies (DeepConf, Best-of-N, Self-Consistency, CoT) work with the single model class

**Testing**:
- ✅ All imports validated successfully
- ✅ DeepConf strategy imports working
- ✅ All strategy imports (Base, DeepConf, OnlineBestOfN, ChainOfThought, SelfConsistency) working
- ✅ Main evaluation script imports validated
- ✅ No linting errors

**Migration Guide**:
```python
# OLD (removed):
from llm_tts.models import create_model
model = create_model(
    provider="openrouter",
    model_name="openai/gpt-4o-mini",
    api_key=api_key,
    top_logprobs=20
)

# NEW (current):
from llm_tts.models import BlackboxModelWithStreaming
model = BlackboxModelWithStreaming(
    openai_api_key=api_key,
    model_path="openai/gpt-4o-mini",
    supports_logprobs=True,
    base_url="https://openrouter.ai/api/v1"
)
```

---

## Architecture Overview

### Core Components

**Models** (`llm_tts/models/`):
- `base.py` - Abstract model interface
- `blackboxmodel_with_streaming.py` - Unified model adapter with streaming + logprobs support

**Strategies** (`llm_tts/strategies/`):
- `strategy_base.py` - Abstract strategy interface
- `strategy_deepconf.py` - Confidence-based test-time scaling (offline/online modes)
- `strategy_online_best_of_n.py` - Step-by-step generation with PRM scoring
- `strategy_self_consistency.py` - Majority voting across reasoning paths
- `strategy_chain_of_thought.py` - Single-pass step-by-step reasoning

**Scorers** (`llm_tts/scorers/`):
- `step_scorer_base.py` - Abstract scorer interface
- `step_scorer_prm.py` - Process Reward Model scoring
- `majority_voting.py` - Frequency-based scoring
- `step_scorer_uncertainty.py` - Uncertainty-based scoring

**Evaluators**:
- `evaluator_gold_standard.py` - DeepSeek-based answer verification

### Model Usage Pattern

All strategies use `BlackboxModelWithStreaming` which provides:
- ✅ Streaming generation with `stream=True`
- ✅ Token log probabilities with `logprobs=True`
- ✅ Confidence scoring via token probabilities
- ✅ Early stopping support (via callbacks)
- ✅ Multi-provider support (OpenRouter, OpenAI, Together AI)

**Provider Configuration**:
```python
# OpenRouter
model = BlackboxModelWithStreaming(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    model_path="openai/gpt-4o-mini",
    supports_logprobs=True,
    base_url="https://openrouter.ai/api/v1"
)

# OpenAI (default)
model = BlackboxModelWithStreaming(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_path="gpt-4o-mini",
    supports_logprobs=True,
    base_url=None  # or omit this parameter
)
```

---

## Development Workflow

### Setup
```bash
# Initial setup
./setup.sh

# Install dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
make hooks
```

### Code Quality
```bash
# Auto-fix formatting
make fix

# Linting
make lint

# Run tests
make test
```

### Running Experiments
```bash
# DeepConf offline mode
python scripts/run_tts_eval.py \
  --config-name experiments/deepconf/run_gsm8k_deepconf_offline \
  dataset.subset=10

# DeepConf online mode
python scripts/run_tts_eval.py \
  --config-name experiments/deepconf/run_gsm8k_deepconf_online \
  strategy.warmup_traces=3 \
  strategy.total_budget=10
```

---

## Configuration System

Uses Hydra for hierarchical configuration:

```
config/
├── experiments/     # Complete experiment configs
├── dataset/        # Dataset configs (gsm8k, math, proofnet)
├── model/          # Model configs (openrouter, openai, etc.)
├── strategy/       # Strategy-specific parameters
├── generation/     # Generation parameters
└── system/        # System settings (device, seed)
```

**Override pattern**:
```bash
python scripts/run_tts_eval.py \
  --config-name experiments/deepconf/run_gsm8k_deepconf_offline \
  model.model_name="openai/gpt-4o" \
  dataset.subset=100 \
  strategy.budget=16
```

---

## Testing Strategy

### Unit Tests
- `tests/deepconf/` - DeepConf strategy tests
- `tests/online_best_of_n/` - Best-of-N strategy tests
- `tests/run_tts_eval/` - Evaluation script tests

### Integration Tests
```bash
# DeepConf accuracy tests
python tests/deepconf/test_deepconf_accurate.py

# DeepConf math problems
python tests/deepconf/test_deepconf_math.py --budget 10 --verbose
```

### API Keys Required
- `OPENROUTER_API_KEY` - For model inference via OpenRouter
- `DEEPSEEK_API_KEY` - For answer verification (optional, evaluation only)

---

## Code Style Guidelines

1. **Imports**: Use absolute imports from `llm_tts.*`
2. **Type hints**: Use type hints for function signatures
3. **Docstrings**: Use Google-style docstrings
4. **Formatting**: Black + isort (enforced via pre-commit hooks)
5. **Linting**: flake8 (max line length: 100)

---

## Git Workflow

### Pre-commit Hooks
Automatically run on `git commit`:
- `black` - Code formatting
- `isort` - Import sorting
- `flake8` - Linting

### Commit Guidelines
```bash
# Auto-fix before committing
make fix

# Check for issues
make lint

# Commit (hooks run automatically)
git commit -m "feat: add new feature"
```

---

## Future Improvements

### Planned Features
- [ ] Add support for local HuggingFace models
- [ ] Implement more test-time scaling strategies
- [ ] Add comprehensive benchmarking suite
- [ ] Improve online mode early stopping heuristics

### Technical Debt
- [ ] Add more comprehensive unit tests
- [ ] Improve error handling in streaming mode
- [ ] Add logging configuration options
- [ ] Document all configuration parameters

---

## References

- **DeepConf Paper**: [arxiv.org/abs/2508.15260](https://arxiv.org/abs/2508.15260)
- **lm-polygraph**: Used for BlackboxModel base class
- **Hydra**: Configuration management
- **OpenRouter**: Multi-model API access
