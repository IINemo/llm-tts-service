# Install
1. Install dev branch of lm_polygraph
2. Install the library

# Structure
* config -- hydra configuration files.
* llm_tts -- the library with test time scaling strategies.
* scripts/run_tts_eval.py -- the script for running evaluation of test time scaling methods.

# TODO:
1. Add new scorers
2. Add tree of thought


# Example:
OPENAI_API_KEY=<key> PYTHONPATH=./ python ./scripts/run_tts_eval.py --config-path=../config/ --config-name=run_tts_eval.yaml dataset=small_gsm8k dataset.subset=1