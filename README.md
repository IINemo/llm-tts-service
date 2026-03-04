<img width="130" height="130" alt="ThinkBooster" src="assets/logo.png" />

# ThinkBooster

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://thinkbooster.s3.us-east-1.amazonaws.com/thinkbooster.pdf)

## What is ThinkBooster?

ThinkBooster is an open-source framework for **test-time compute scaling** of large language models. It implements multiple search strategies — beam search, best-of-N, self-consistency, DeepConf, and more — scored by process reward models (PRMs) and uncertainty estimators. The framework includes an evaluation pipeline for math reasoning benchmarks, an OpenAI-compatible REST API, and an interactive visual debugger for inspecting strategy behavior step by step.

---

## Key Features

- **Multiple TTS strategies** — beam search, online/offline best-of-N, self-consistency, DeepConf, adaptive scaling, chain-of-thought, extended thinking, and tree-of-thoughts
- **Process Reward Model (PRM) and uncertainty-based scoring** — step-level scoring with configurable aggregation (min, mean, max, product) and sliding window
- **OpenAI-compatible REST API** — drop-in replacement that works with any OpenAI SDK; select strategy via request parameters
- **Visual debugger** — interactive web UI for comparing strategies, inspecting step-by-step reasoning traces and confidence signals
- **Evaluation pipeline** — GSM8K, MATH-500, AIME, OlympiadBench, GaoKao with crash-resistant resume support

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/IINemo/thinkbooster.git
cd thinkbooster

# Create conda environment
conda create -n thinkbooster python=3.11 -y
conda activate thinkbooster

# Install dependencies
./setup.sh
pip install latex2sympy2 --no-deps   # math evaluation (separate due to antlr4 conflict)
pip install ".[vllm]"                # vLLM for fast local inference

# Configure API keys
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### Run an Experiment

```bash
# Beam search on GSM8K (3 samples for quick verification)
python scripts/run_tts_eval.py \
  --config-name experiments/beam_search/gsm8k/window_all/mean/beam_search_vllm_qwen25_math_7b_instruct_gsm8k_prm \
  dataset.subset=3
```

Results are saved to `outputs/` with full config snapshots for reproducibility. Add `--resume` to continue interrupted runs.

### REST API

```bash
pip install -e ".[service]"
python service_app/main.py   # starts on http://localhost:8001
```

Use with any OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="your-key")

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Reason step by step, put answer in \\boxed{}."},
        {"role": "user", "content": "What is 15 * 7?"}
    ],
    extra_body={
        "tts_strategy": "self_consistency",
        "num_paths": 5
    }
)
print(response.choices[0].message.content)
```

See [Service API Guide](docs/service/SERVICE_API_GUIDE.md) for the full reference.

---

## Visual Debugger

The interactive debugger lets you compare multiple TTS strategies side by side on the same problem. Inspect per-step decisions (escalate, stop, prune, select), view confidence and uncertainty signals, and drill into sampled candidates and tree expansions.

After starting the REST API service, open:

```
http://localhost:8001/debugger
```

See [service_app/README.md](service_app/README.md) for details on cached examples and custom input modes.

---

## Supported Strategies

| Strategy | Type | Scorer | Description |
|---|---|---|---|
| `beam_search` | Online | PRM / uncertainty | Beam search over reasoning steps with step-level pruning |
| `online_bon` | Online | PRM / uncertainty | Step-level candidate selection with early stopping |
| `offline_bon` | Offline | PRM / uncertainty | Generate N full trajectories, pick best by aggregate score |
| `self_consistency` | Offline | Majority vote | Majority voting over multiple reasoning paths |
| `deepconf` | Online/Offline | Confidence | Confidence-based adaptive generation and filtering |
| `chain_of_thought` | Baseline | — | Standard chain-of-thought prompting |
| `extended_thinking` | Baseline | — | Extended thinking with native model capabilities |
| `baseline` | Baseline | — | Single-pass generation (no scaling) |

---

## Project Structure

```
thinkbooster/
├── llm_tts/              # Core library
│   ├── strategies/       # TTS strategy implementations
│   ├── models/           # Model wrappers (vLLM, HuggingFace, API)
│   ├── scorers/          # Step scoring (PRM, uncertainty, voting)
│   ├── evaluation/       # Correctness evaluation (exact match, LLM judge)
│   └── datasets/         # Dataset loaders and utilities
├── config/               # Hydra configuration system
├── scripts/              # Evaluation scripts (run_tts_eval.py)
├── service_app/          # REST API service + visual debugger
├── tests/                # Test suite with strategy registry
├── docs/                 # Documentation
└── lm-polygraph/         # Submodule: uncertainty estimation
```

See [Project Structure](docs/PROJECT_STRUCTURE.md) for a detailed architecture overview.

---

## Documentation

- [Project Structure](docs/PROJECT_STRUCTURE.md) — architecture and component descriptions
- [Evaluation Protocol](docs/evaluation/README.md) — datasets, metrics (accuracy, tokens, FLOPs), and reporting
- [Strategy Registration](docs/STRATEGY_REGISTRATION.md) — how to add new strategies with tests
- [Service API Guide](docs/service/SERVICE_API_GUIDE.md) — REST API reference and configuration
- [DeepConf Guide](docs/deepconf/DeepConf.md) — confidence-based test-time scaling
- [Robustness Guide](docs/ROBUSTNESS.md) — incremental saving, resume, and reproducibility

---

## Contributing

We welcome contributions! Whether it's a new strategy, scorer, dataset, or bug fix — see the [Contributing Guide](docs/CONTRIBUTING.md) for setup instructions, development workflow, and coding standards.

---

## Citation

If you use ThinkBooster in your research, please cite:

```bibtex
@inproceedings{thinkbooster2026,
  title     = {ThinkBooster: An Open-Source Framework for Test-Time Compute Scaling of LLMs},
  author    = {Smirnov, Vladislav and Karantonis, Stylianos and Shelmanov, Artem},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics: System Demonstrations},
  year      = {2026},
  url       = {https://thinkbooster.s3.us-east-1.amazonaws.com/thinkbooster.pdf}
}
```

---

## Troubleshooting

<details>
<summary>vLLM engine fails to start</summary>

**Corrupted torch compile cache:** If you see `RuntimeError: Engine core initialization failed`:

```bash
rm -rf ~/.cache/vllm/torch_compile_cache/
```

**Missing C compiler:** If Triton can't find `gcc`:

```bash
conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y
ln -s $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc $CONDA_PREFIX/bin/gcc
ln -s $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ $CONDA_PREFIX/bin/g++
```

</details>

<details>
<summary>ANTLR version mismatch warnings</summary>

```
ANTLR runtime and generated code versions disagree: 4.9.3!=4.7.2
```

This is expected — Hydra uses ANTLR 4.9.3, latex2sympy2 was built with 4.7.2. Both work correctly.

</details>

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
