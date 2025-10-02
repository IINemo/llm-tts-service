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

## Structure
* config -- hydra configuration files.
* llm_tts -- the library with test time scaling strategies.
* scripts/run_tts_eval.py -- the script for running evaluation of test time scaling methods.

# TODO:
1. Add new scorers
2. Add tree of thought
