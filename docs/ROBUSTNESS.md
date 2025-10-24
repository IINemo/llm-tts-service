# Robustness Features

This guide covers the robustness and reliability features built into the LLM TTS evaluation system.

## Table of Contents
- [Incremental Saving](#incremental-saving)
- [Resume Interrupted Evaluations](#resume-interrupted-evaluations)
- [Experiment Reproducibility](#experiment-reproducibility)
- [Best Practices](#best-practices)

---

## Incremental Saving

**All evaluations save results after each sample** - no more lost work from crashes or interruptions!

### What's Saved

- ✅ **Generation phase**: Results saved after each sample (not every 10)
- ✅ **Evaluation phase**: Results saved after each sample is evaluated
- ✅ **At most 1 sample lost** if process is interrupted

### Output Structure

Every experiment run creates a timestamped directory with complete metadata:

```
outputs/YYYY-MM-DD/HH-MM-SS/
├── results.json           # Incrementally updated results
├── run_tts_eval.log       # Execution logs
└── .hydra/                # Full experiment configuration
    ├── config.yaml        # Resolved config with all parameters
    ├── overrides.yaml     # Command-line overrides used
    └── hydra.yaml         # Hydra settings
```

### Implementation Details

**Generation Phase** (`scripts/run_tts_eval.py:422-424`):
```python
# Save after each sample (enables resuming with minimal data loss)
save_results_json(results, save_path_file)
log.info(f"Saved result for sample {i} to {save_path_file}")
```

**Evaluation Phase** (`scripts/run_tts_eval.py:450-533`):
```python
# Evaluate samples one at a time and save after each
for i, result in enumerate(results):
    # ... evaluate single sample
    eval_result = evaluator_fn(
        [result["question"]],
        [result["generated_answer"]],
        [result["gold_answer"]],
    )
    # ... process result
    results[i].setdefault("eval", {})[eval_name] = eval_data

    # Save after EACH sample evaluation
    save_results_json(results, save_path_file)
```

---

## Resume Interrupted Evaluations

If your evaluation is interrupted (crash, network error, manual stop), you can resume from where it left off.

### Quick Start

**Resume from latest run:**
```bash
python scripts/run_tts_eval.py \
  --config-name experiments/deepconf/run_gsm8k_deepconf_offline \
  --resume
```

**Resume from specific directory:**
```bash
python scripts/run_tts_eval.py \
  --config-name experiments/deepconf/run_gsm8k_deepconf_offline \
  --resume-from outputs/2025-10-18/23-50-46
```

### What Gets Resumed

- ✅ **Skips already processed samples** during generation
- ✅ **Skips already evaluated samples** during evaluation
- ✅ **Uses same output directory** (no duplicate work)
- ✅ **Preserves original configuration**

### Example Workflow

```bash
# Start evaluation
python scripts/run_tts_eval.py \
  --config-name experiments/deepconf/run_gsm8k_deepconf_offline \
  dataset.subset=100

# Process interrupted after 60 samples? No problem!
# Just add --resume and it continues from sample 61
python scripts/run_tts_eval.py \
  --config-name experiments/deepconf/run_gsm8k_deepconf_offline \
  --resume
```

### How It Works

**Resume Logic** (`scripts/utils/results.py:65-94`):
```python
def load_results_json(json_path: Path):
    """Load existing results from a JSON file for resuming evaluation."""
    if not json_path.exists():
        return [], set()

    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Extract processed indices
    processed_indices = {r["index"] for r in results if "index" in r}

    log.info(f"Loaded {len(results)} existing results from {json_path}")
    log.info(f"Resuming from {len(processed_indices)} processed samples")

    return results, processed_indices
```

**Finding Latest Output** (`scripts/utils/results.py:97-118`):
```python
def find_latest_output_dir():
    """Find the most recent output directory."""
    outputs_root = Path("outputs")
    if not outputs_root.exists():
        return None

    # Find all timestamped directories (YYYY-MM-DD/HH-MM-SS)
    all_dirs = []
    for date_dir in outputs_root.iterdir():
        if date_dir.is_dir() and date_dir.name.count("-") == 2:
            for time_dir in date_dir.iterdir():
                if time_dir.is_dir() and time_dir.name.count("-") == 2:
                    all_dirs.append(time_dir)

    # Sort by modification time, return most recent
    latest_dir = max(all_dirs, key=lambda p: p.stat().st_mtime)
    return latest_dir
```

---

## Experiment Reproducibility

Every experiment automatically saves complete configuration snapshots, enabling exact reproduction of results.

### What's Saved Automatically

1. **Full Configuration** (`.hydra/config.yaml`):
   - All parameters with resolved values
   - Defaults from config hierarchy
   - Command-line overrides applied

2. **Command-Line Overrides** (`.hydra/overrides.yaml`):
   - Exact parameters you specified on command line
   - Useful for re-running with same overrides

3. **Execution Logs** (`run_tts_eval.log`):
   - Complete execution trace
   - Debug information
   - Error messages if any

4. **Results** (`results.json`):
   - Generated trajectories with metadata
   - Evaluation results from all evaluators
   - Per-sample confidence scores and traces

### Reproducing an Experiment

**Step 1: Check what parameters were used**
```bash
# View the full configuration
cat outputs/2025-10-18/23-50-46/.hydra/config.yaml

# View just the overrides
cat outputs/2025-10-18/23-50-46/.hydra/overrides.yaml
# Output:
# - dataset.subset=20
# - strategy.budget=4
```

**Step 2: Re-run with exact same config**
```bash
python scripts/run_tts_eval.py \
  --config-name experiments/deepconf/run_gsm8k_deepconf_offline \
  dataset.subset=20 \
  strategy.budget=4
```

### Configuration Hierarchy

Hydra uses a hierarchical configuration system:

```
config/
├── experiments/
│   └── run_gsm8k_deepconf.yaml      # Experiment config (top level)
├── dataset/
│   └── gsm8k.yaml                    # Dataset defaults
├── model/
│   └── openrouter.yaml               # Model defaults
├── strategy/
│   └── deepconf.yaml                 # Strategy defaults
└── generation/
    └── default.yaml                   # Generation defaults
```

The final configuration in `.hydra/config.yaml` is the result of merging all these files with your command-line overrides.

---

## Best Practices

### For Long-Running Evaluations

1. **Use screen or tmux** to prevent SSH disconnections:
   ```bash
   screen -S eval
   python scripts/run_tts_eval.py ...
   # Detach: Ctrl+A, D
   # Reattach: screen -r eval
   ```

2. **Monitor progress** from another terminal:
   ```bash
   tail -f outputs/YYYY-MM-DD/HH-MM-SS/run_tts_eval.log
   ```

3. **Keep --resume handy** for quick recovery:
   ```bash
   # If evaluation stops, just re-run with --resume
   python scripts/run_tts_eval.py \
     --config-name your_experiment \
     --resume
   ```

### For Reproducible Research

1. **Commit configs before running**:
   ```bash
   git add config/
   git commit -m "Add experiment config"
   git push
   ```

2. **Save git commit hash** with results (optional enhancement):
   ```bash
   git rev-parse HEAD > outputs/YYYY-MM-DD/HH-MM-SS/git_commit.txt
   ```

3. **Document experiment** in commit message:
   ```bash
   git commit -m "Run DeepConf with temp=0.6, top_p=0.95

   Config: experiments/deepconf/run_gsm8k_deepconf_offline
   Dataset: GSM8K (subset=100)
   Budget: 4 traces per sample
   Results: outputs/2025-10-18/23-50-46/"
   ```

### For Team Collaboration

1. **Use descriptive output names** (optional enhancement):
   ```yaml
   # In config file
   hydra:
     run:
       dir: outputs/${experiment_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}
   ```

2. **Share experiment configs** in version control:
   ```bash
   # All configs tracked in git
   git log config/experiments/
   ```

3. **Document results** in experiment logs:
   ```bash
   # Create summary after evaluation
   echo "Accuracy: 85%" >> outputs/YYYY-MM-DD/HH-MM-SS/SUMMARY.txt
   ```

---

## Troubleshooting

### Q: Resume doesn't skip already-evaluated samples

**A:** Check that the output directory has the `results.json` file:
```bash
ls outputs/YYYY-MM-DD/HH-MM-SS/
# Should show: results.json, run_tts_eval.log, .hydra/
```

If `results.json` is missing, the evaluation was interrupted before any saves. Start fresh.

### Q: How do I know if incremental saving is working?

**A:** Watch the logs during evaluation:
```bash
tail -f outputs/YYYY-MM-DD/HH-MM-SS/run_tts_eval.log | grep "Saved result"
```

You should see:
```
[INFO] - Saved result for sample 0 to .../results.json
[INFO] - Saved result for sample 1 to .../results.json
...
```

### Q: Can I resume from a different machine?

**A:** Yes! Just copy the entire output directory:
```bash
# On original machine
tar -czf eval_2025-10-18_23-50-46.tar.gz outputs/2025-10-18/23-50-46/

# On new machine
tar -xzf eval_2025-10-18_23-50-46.tar.gz
python scripts/run_tts_eval.py \
  --resume-from outputs/2025-10-18/23-50-46
```

### Q: What if I want to re-evaluate existing results?

**A:** Use `--resume-from` without the generated trajectories:
```bash
# Remove evaluation results but keep generations
cd outputs/2025-10-18/23-50-46
jq 'map(del(.eval))' results.json > results_no_eval.json
mv results_no_eval.json results.json

# Now re-run evaluation
python scripts/run_tts_eval.py \
  --config-name your_experiment \
  --resume-from outputs/2025-10-18/23-50-46
```

---

## Related Documentation

- **[Project Structure](PROJECT_STRUCTURE.md)** - Architecture overview
- **[DeepConf Guide](deepconf/DeepConf.md)** - DeepConf strategy details
- **[Configuration Guide](../config/README.md)** - Hydra config system
