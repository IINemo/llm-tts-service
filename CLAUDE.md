# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an LLM Test-Time Scaling (TTS) service that implements various strategies for improving LLM reasoning through test-time computation. The main strategies include:
- **DeepConf**: Confidence-based test-time scaling with trace filtering
- **Best-of-N**: Generate N candidates, select best by scorer
- **Self-Consistency**: Majority voting across multiple reasoning paths
- **Chain-of-Thought**: Single-pass step-by-step reasoning

## Setup and Installation

```bash
# Initial setup (installs package + lm-polygraph dev branch)
./setup.sh

# Update lm-polygraph later
./setup.sh --update

# Install dev dependencies and pre-commit hooks
pip install -e ".[dev]"
make hooks
```

## Development Workflow

```bash
# Auto-fix formatting issues before committing
make fix

# Check for linting errors
make lint

# Commit (hooks run automatically)
git commit -m "message"
```

Pre-commit hooks will block commits that fail checks (black, isort, flake8).

## Running Evaluations

The main evaluation script is `scripts/run_tts_eval.py` which uses Hydra for configuration.

**Basic usage:**
```bash
python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name experiments/run_gsm8k_deepconf
```

**Common overrides:**
```bash
# Run on subset of data
python scripts/run_tts_eval.py \
  --config-name experiments/run_gsm8k_deepconf \
  dataset.subset=10 \
  strategy.budget=4

# Change model
python scripts/run_tts_eval.py \
  --config-name experiments/run_gsm8k_deepconf \
  model.model_name="openai/gpt-4o-mini"

# Environment variables needed
export OPENROUTER_API_KEY="your-key"
export DEEPSEEK_API_KEY="your-key"  # For evaluation
```

**Makefile shortcuts:**
```bash
make eval-gsm8k          # Full GSM8K evaluation
make eval-gsm8k-subset   # Quick test (3 samples)
```

## Architecture Overview

### Core Components

**1. Models** (`llm_tts/models/`)
- `base.py`: Abstract model interface
- `blackboxmodel_with_streaming.py`: Wrapper around lm-polygraph's BlackboxModel with streaming support and custom base_url for OpenRouter
- Models support both streaming (for online strategies) and non-streaming with logprobs (for DeepConf offline)

**2. Strategies** (`llm_tts/strategies/`)
All strategies inherit from `StrategyBase` and implement `generate_trajectory(prompt) -> Dict`:
- `strategy_deepconf.py`: Confidence-based filtering with offline/online modes
- `strategy_online_best_of_n.py`: Step-by-step generation with PRM scoring
- `strategy_self_consistency.py`: Majority voting across complete chains
- `strategy_chain_of_thought.py`: Single reasoning chain

**3. Scorers** (`llm_tts/scorers/`)
Used by Best-of-N strategies to evaluate step candidates:
- `direct_prm_scorer.py`: Process Reward Model scoring
- `majority_voting.py`: Frequency-based scoring for self-consistency
- All inherit from `StepScorerBase`

**4. Evaluators** (`llm_tts/`)
- `evaluator_gold_standard_deepseek.py`: DeepSeek-based answer verification for correctness checking

### Configuration System (Hydra)

Configurations are in `config/` with hierarchical structure:
```
config/
├── experiments/        # Complete experiment configs
│   └── run_gsm8k_deepconf.yaml
├── dataset/           # Dataset configs (gsm8k, etc.)
├── model/            # Model configs (openrouter, openai, etc.)
├── strategy/         # Strategy-specific params
├── generation/       # Generation parameters
└── system/          # System settings (device, seed)
```

Experiment configs use `defaults:` to compose from other configs, then override specific values.

## Key Implementation Details

### DeepConf Strategy

**Two modes:**
- **Offline**: Generate N traces → compute confidence → filter → majority vote
- **Online**: Warmup phase → adaptive generation with early stopping (streaming TODO)

**Critical design:**
- Requires models with `supports_logprobs=True`
- Uses `BlackboxModelWithStreaming` which supports both streaming and non-streaming
- For offline mode: calls parent's `generate_texts()` with `output_scores=True` to get logprobs
- Confidence computation uses entropy over top-k logprobs with sliding window averaging
- Answer extraction expects `\boxed{answer}` format

**Model setup for DeepConf:**
```python
# BlackboxModelWithStreaming supports base_url for OpenRouter
model = BlackboxModelWithStreaming(
    openai_api_key=api_key,
    model_path="openai/gpt-4o-mini",
    supports_logprobs=True,
    base_url="https://openrouter.ai/api/v1"  # For OpenRouter
)
# The model.openai_api is set to the custom client
# Parent's generate_texts() uses this for logprobs
```

### Model Architecture Pattern

`BlackboxModelWithStreaming` inherits from lm-polygraph's `BlackboxModel`:
- Parent has non-streaming `generate_texts()` with logprobs support
- Child overrides `generate_texts()` for streaming mode
- Both use `self.openai_api` client attribute
- Setting `base_url` in `__init__` creates custom OpenAI client for OpenRouter

### Evaluation Pipeline

`scripts/run_tts_eval.py` has two phases:
1. **Generation** (`generate_trajectories`): Run strategy on dataset, save results
2. **Evaluation** (`evaluate_results`): Check correctness using DeepSeek evaluator

Results are saved incrementally with resume support.

## Important Patterns

### Adding a New Strategy

1. Create `llm_tts/strategies/strategy_yourname.py` inheriting from `StrategyBase`
2. Implement `generate_trajectory(prompt) -> Dict` with keys: `trajectory`, `steps`, `validity_scores`, `completed`, `metadata`
3. Add to `llm_tts/strategies/__init__.py`
4. Add case in `scripts/run_tts_eval.py:create_tts_strategy()`
5. Create config in `config/strategy/yourname.yaml`
6. Create experiment config in `config/experiments/`

### Configuration Override Pattern

Hydra allows runtime overrides:
```bash
python script.py \
  config.param=value \
  nested.config.param=value \
  +new.param=value  # Add new parameter
```

### Prompt Format Handling

Strategies accept both string prompts and message lists:
```python
# String format
prompt = "Question: ..."

# Message format (converted internally)
prompt = [
    {"role": "system", "content": "You are..."},
    {"role": "user", "content": "Question: ..."}
]
```

DeepConf converts message lists to strings by extracting user content.

## API Keys and Environment

Required environment variables:
- `OPENROUTER_API_KEY`: For model inference via OpenRouter
- `DEEPSEEK_API_KEY`: For answer verification (optional, for evaluation only)

Models supporting logprobs via OpenRouter:
- `openai/gpt-4o-mini` (recommended - fast and cheap)
- `openai/gpt-4o`
- `openai/gpt-3.5-turbo`

## Testing

```bash
# Run all tests
make test

# Linting
make lint

# Format code
make format
```

## Common Issues

**1. Import errors with lm-polygraph**
- Ensure lm-polygraph dev branch is installed via `./setup.sh`
- It's installed in editable mode from `lm-polygraph/` subdirectory

**2. DeepConf logprobs not working**
- Verify model supports logprobs (OpenAI models via OpenRouter)
- Check `supports_logprobs=True` in model config
- Ensure `top_logprobs` parameter is set (max 20)

**3. Hydra config not found**
- Use `--config-path ../config` (relative to script location)
- Config name should NOT include `.yaml` extension

**4. Model type mismatch for DeepConf**
- DeepConf requires `BlackboxModel` or `BlackboxModelWithStreaming`
- Set `model.type: openai_api` and `strategy.type: deepconf` in config
- Rule of thumb not to use default parameters in functions. Try reduce the number of default parameters as much as possible
- Alwats use relevant agents for all tasks (among them, which I created with u)
- Do not commit .agents and agents/prompts
- do not commit any files related to claude (it is common repository)