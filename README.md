# LLM Test-Time Scaling Service

## Install

```bash
./setup.sh
```

This installs:
- Package dependencies (via pip install -e .)
- lm-polygraph dev branch

Update lm-polygraph later:
```bash
./setup.sh --update
```

## Configuration

**API Keys:**
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys
# OPENROUTER_API_KEY=your-key-here
# DEEPSEEK_API_KEY=your-key-here
```

The script will automatically load API keys from `.env` file.

## Development

Install dev dependencies and pre-commit hooks:
```bash
pip install -e ".[dev]"
make hooks
```

**Daily workflow:**
```bash
# Make your changes...

# Before committing - auto-fix all issues
make fix     # Runs pre-commit on all files (black, isort, etc.)

# Check for remaining issues
make lint    # Run flake8

# Commit (hooks will run automatically)
git commit -m "your message"
```

**Manual formatting:**
```bash
make format  # Just run black + isort (without other hooks)
make lint    # Check with flake8
```

Pre-commit hooks run automatically on `git commit` and will block commits that fail checks.

## Structure
* config -- hydra configuration files.
* llm_tts -- the library with test time scaling strategies.
* scripts/run_tts_eval.py -- the script for running evaluation of test time scaling methods.

# TODO:
1. Add new scorers
2. Add tree of thought


# Running Experiments

See strategy-specific documentation:
- [DeepConf Strategy](docs/deepconf/DeepConf.md) - Confidence-based test-time scaling

## Examples

**Note:** API keys are loaded from `.env` file - no need to specify them in the command.

### Online Best-of-N Strategy

Reasoner with Qwen-3 (local):
```bash
python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name run_tts_eval \
  dataset=small_gsm8k \
  dataset.subset=1 \
  model=hf_qwen3
```

Reasoner with ChatGPT:
```bash
python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name run_tts_eval \
  dataset=small_gsm8k \
  dataset.subset=1 \
  model=openai \
  model.model_path="gpt-4o-mini"
```

With uncertainty scorer:
```bash
python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name run_tts_eval \
  dataset=small_gsm8k \
  dataset.subset=1 \
  model=hf_qwen3 \
  scorer=uncertainty
```

### DeepConf Strategy

**Offline mode** (generate N traces, filter by confidence, majority vote):
```bash
python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name experiments/deepconf/run_gsm8k_deepconf_offline \
  model.model_path="openai/gpt-3.5-turbo" \
  dataset.subset=10
```

**Online mode** (adaptive generation with confidence-based early stopping):
```bash
python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name experiments/deepconf/run_gsm8k_deepconf_online \
  model.model_path="openai/gpt-3.5-turbo" \
  dataset.subset=10
```
