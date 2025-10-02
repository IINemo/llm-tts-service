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


# Example:
OPENAI_API_KEY=<key> PYTHONPATH=./ python ./scripts/run_tts_eval.py --config-path=../config/ --config-name=run_tts_eval.yaml dataset=small_gsm8k dataset.subset=1
