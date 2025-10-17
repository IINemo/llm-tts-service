# Configuration Directory Structure

Hydra-based configuration for LLM Test-Time Scaling evaluation framework.

## Directory Structure

```
config/
├── experiments/          # Full experiment configs (entry points)
│   ├── run_tts_eval.yaml              # Default evaluation config
│   ├── deepconf/                      # DeepConf experiments
│   │   ├── run_gsm8k_deepconf.yaml    # GSM8K + DeepConf
│   │   └── deepconf_api_test.yaml     # DeepConf API test
│   ├── chain_of_thought/              # Chain-of-thought experiments
│   │   ├── chain_of_thought_test.yaml
│   │   └── chain_of_thought_api_test.yaml
│   └── self_consistency/              # Self-consistency experiments
│       ├── self_consistency_test.yaml
│       └── self_consistency_api_test.yaml
│
├── dataset/              # Dataset configurations
│   ├── default.yaml      # Default dataset config
│   ├── gsm8k.yaml        # GSM8K dataset
│   └── small_gsm8k.yaml  # Small GSM8K subset
│
├── model/                # Model configurations
│   ├── default.yaml      # Default model config
│   ├── openrouter.yaml   # OpenRouter API models
│   ├── openai.yaml       # OpenAI API models
│   └── together_ai.yaml  # Together AI models
│
├── strategy/             # Strategy configurations
│   ├── deepconf.yaml            # DeepConf strategy
│   ├── chain_of_thought.yaml   # Chain-of-thought
│   └── self_consistency.yaml   # Self-consistency
│
├── generation/           # Generation settings
│   └── default.yaml
│
├── output/               # Output settings
│   └── default.yaml
│
└── system/               # System settings
    └── default.yaml
```

## Usage

### Run an experiment config:

```bash
python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name experiments/deepconf/run_gsm8k_deepconf
```

### Override specific values:

```bash
python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name experiments/deepconf/run_gsm8k_deepconf \
  dataset.subset=100 \
  strategy.budget=16 \
  model.model_name="openai/gpt-4o"
```

## Experiment Configs

Experiment configs in `experiments/` combine modular components:

```yaml
# experiments/run_gsm8k_deepconf.yaml
defaults:
  - dataset: gsm8k          # From dataset/gsm8k.yaml
  - model: openrouter       # From model/openrouter.yaml
  - generation: default     # From generation/default.yaml
  - output: default         # From output/default.yaml
  - system: default         # From system/default.yaml
  - _self_

# Override specific values
model:
  model_name: "openai/gpt-4o-mini"

strategy:
  type: deepconf
  budget: 8
```

## Component Configs

### Dataset (`dataset/`)

Configure which dataset to use:
- `dataset_path`: HuggingFace dataset path
- `dataset_split`: train/test
- `subset`: Number of samples (null = all)

### Model (`model/`)

Configure model settings:
- `provider`: openrouter, together_ai, or local
- `model_name`: Model identifier
- `top_logprobs`: For DeepConf (API models only)

### Strategy (`strategy/`)

Configure reasoning strategy:
- `type`: deepconf, direct_online_best_of_n_reason_eval_separate, etc.
- Strategy-specific parameters

### Generation (`generation/`)

Configure text generation:
- `max_new_tokens`: Max tokens per generation
- `temperature`: Sampling temperature
- `top_p`, `top_k`: Nucleus/top-k sampling

## Creating New Experiments

1. **Choose the right subfolder** based on strategy:
   - `experiments/deepconf/` - DeepConf experiments
   - `experiments/chain_of_thought/` - Chain-of-thought experiments
   - `experiments/self_consistency/` - Self-consistency experiments

2. **Copy existing config:**
   ```bash
   cp config/experiments/deepconf/run_gsm8k_deepconf.yaml \
      config/experiments/deepconf/my_experiment.yaml
   ```

3. **Modify as needed:**
   ```yaml
   defaults:
     - /dataset/my_dataset
     - /model/openrouter
     - /generation/default
     - _self_

   model:
     model_name: "openai/gpt-4o"

   strategy:
     type: deepconf
     budget: 32
   ```

4. **Run:**
   ```bash
   python scripts/run_tts_eval.py \
     --config-path ../config \
     --config-name experiments/deepconf/my_experiment
   ```

## Best Practices

1. **Organize by strategy** - Keep experiments in method-specific subfolders
2. **Use modular components** - Reuse dataset/model/strategy configs
3. **Name descriptively** - `run_<dataset>_<method>.yaml` or `<method>_<variant>.yaml`
4. **Document overrides** - Comment why values are overridden
5. **Absolute paths in defaults** - Use `/dataset/...` not `dataset/...` for clarity
