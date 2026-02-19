#!/bin/bash
#
# Local experiment submission using Task Spooler (tsp)
# Drop-in replacement for scripts/slurm/submit.sh on local machines.
#
# Prerequisites:
#   apt update && apt install task-spooler
#
# Usage: ./submit.sh --strategy <strategy> --dataset <dataset> [OPTIONS]
#

set -e

# Default values
STRATEGY=""
DATASET=""
MODEL="qwen25_7b"
SCORERS=""
SEED=42
SEEDS=1          # number of seeds (42, 43, 44, ...)
GPU=""           # specific GPU id(s), e.g. "0" or "0,1"
TIMEOUT=""       # in seconds, e.g. 14400 = 4h
DRY_RUN="no"
LABEL=""         # custom tsp label

# Detect project directory (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Help message
show_help() {
    cat << 'EOF'
Usage: ./submit.sh --strategy <strategy> --dataset <dataset> [OPTIONS]

Local alternative to SLURM using Task Spooler (tsp).

Required:
  --strategy <name>     Strategy: baseline, self_consistency, offline_bon, online_bon, beam_search, adaptive_scaling
  --dataset <name>      Dataset: aime2024, aime2025, amc23, math500, olympiadbench, gaokao2023en, minerva_math, gpqa_diamond

Optional:
  --model <name>         Model: qwen25_7b (default), qwen3_8b_thinking, qwen3_8b, qwen25_math_7b, qwen25_math_15b
  --scorers <list>       Scorers: all, prm, entropy, perplexity, sequence_prob
  --seed <n>             Starting seed (default: 42)
  --seeds <n>            Number of seeds to run (default: 1; queues seed, seed+1, ...)
  --gpu <id>             GPU id(s) to use (default: auto via tsp; use "0,1" for multi-gpu)
  --timeout <seconds>    Timeout in seconds (default: auto based on strategy)
  --label <name>         Custom label for tsp job
  --dry-run              Show commands without submitting
  -h, --help             Show this help

Useful tsp commands:
  tsp                    Show job queue
  tsp -c <id>            Show stdout of job <id>
  tsp -i <id>            Show info about job <id>
  tsp -r <id>            Remove queued job <id>
  tsp -k <id>            Kill running job <id>
  tsp -l <id>            Show label of job <id>
  tsp -S <n>             Set max simultaneous jobs (default: 1)

Examples:
  # Baseline on AMC23
  ./submit.sh --strategy baseline --dataset amc23 --model qwen25_math_7b

  # Offline BoN with all scorers (queued sequentially)
  ./submit.sh --strategy offline_bon --dataset math500 --scorers all

  # Beam search with PRM on 2 GPUs
  ./submit.sh --strategy beam_search --dataset olympiadbench --scorers prm --gpu 0,1

  # Multiple seeds (queues 4 jobs with seeds 42,43,44,45)
  ./submit.sh --strategy baseline --dataset aime2024 --model qwen3_8b_thinking --seeds 4

  # Dry run
  ./submit.sh --strategy online_bon --dataset gaokao2023en --scorers entropy --dry-run
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy)   STRATEGY="$2"; shift 2 ;;
        --dataset)    DATASET="$2"; shift 2 ;;
        --model)      MODEL="$2"; shift 2 ;;
        --scorers)    SCORERS="$2"; shift 2 ;;
        --seed)       SEED="$2"; shift 2 ;;
        --seeds)      SEEDS="$2"; shift 2 ;;
        --gpu)        GPU="$2"; shift 2 ;;
        --timeout)    TIMEOUT="$2"; shift 2 ;;
        --label)      LABEL="$2"; shift 2 ;;
        --dry-run)    DRY_RUN="yes"; shift ;;
        -h|--help)    show_help ;;
        *)            echo "Unknown option: $1"; show_help ;;
    esac
done

# Validate required arguments
if [[ -z "$STRATEGY" || -z "$DATASET" ]]; then
    echo "Error: --strategy and --dataset are required"
    echo ""
    show_help
fi

# Check tsp is installed (skip for dry-run)
if [[ "$DRY_RUN" != "yes" ]] && ! command -v tsp &>/dev/null; then
    echo "Error: task-spooler (tsp) is not installed."
    echo "Install: apt update && apt install task-spooler"
    exit 1
fi

# --- Config mappings (same as SLURM version) ---

declare -A STRATEGY_CONFIGS
STRATEGY_CONFIGS[baseline]="baseline"
STRATEGY_CONFIGS[self_consistency]="self_consistency"
STRATEGY_CONFIGS[offline_bon]="offline_best_of_n"
STRATEGY_CONFIGS[online_bon]="online_best_of_n"
STRATEGY_CONFIGS[beam_search]="beam_search"
STRATEGY_CONFIGS[adaptive_scaling]="adaptive_scaling"

declare -A DATASET_CONFIGS
DATASET_CONFIGS[aime2024]="aime2024"
DATASET_CONFIGS[aime2025]="aime2025"
DATASET_CONFIGS[amc23]="amc23"
DATASET_CONFIGS[math500]="math500"
DATASET_CONFIGS[olympiadbench]="olympiadbench"
DATASET_CONFIGS[gaokao2023en]="gaokao2023en"
DATASET_CONFIGS[minerva_math]="minerva_math"
DATASET_CONFIGS[gpqa_diamond]="gpqa_diamond"

declare -A SCORER_CONFIGS
SCORER_CONFIGS[entropy]="entropy"
SCORER_CONFIGS[perplexity]="perplexity"
SCORER_CONFIGS[sequence_prob]="sequence_prob"
SCORER_CONFIGS[prm]="prm"

declare -A MODEL_CONFIGS
MODEL_CONFIGS[qwen25_7b]="vllm_nothink_qwen25_7b"
MODEL_CONFIGS[qwen3_8b_thinking]="vllm_thinking_qwen3_8b"
MODEL_CONFIGS[qwen3_8b]="vllm_qwen3_8b"
MODEL_CONFIGS[qwen25_math_7b]="vllm_qwen25_math_7b_instruct"
MODEL_CONFIGS[qwen25_math_15b]="vllm_qwen25_math_15b_instruct"

# --- Helper functions ---

get_config_name() {
    local strategy=$1 dataset=$2 scorer=$3 model=$4
    local strategy_key=${STRATEGY_CONFIGS[$strategy]}
    local dataset_key=${DATASET_CONFIGS[$dataset]}
    local model_key=${MODEL_CONFIGS[$model]}

    if [[ -z "$strategy_key" ]]; then echo "Error: Unknown strategy: $strategy" >&2; exit 1; fi
    if [[ -z "$dataset_key" ]]; then echo "Error: Unknown dataset: $dataset" >&2; exit 1; fi
    if [[ -z "$model_key" ]]; then echo "Error: Unknown model: $model" >&2; exit 1; fi

    if [[ "$strategy" == "baseline" ]]; then
        echo "experiments/${strategy_key}/${dataset_key}/baseline_${model_key}_${dataset_key}"
    elif [[ "$strategy" == "self_consistency" ]]; then
        echo "experiments/${strategy_key}/${dataset_key}/self_consistency_${model_key}_${dataset_key}"
    else
        local scorer_key=${SCORER_CONFIGS[$scorer]}
        if [[ -z "$scorer_key" ]]; then echo "Error: Unknown scorer: $scorer" >&2; exit 1; fi
        echo "experiments/${strategy_key}/${dataset_key}/${strategy}_${model_key}_${dataset_key}_${scorer_key}"
    fi
}

detect_gpus() {
    # Returns comma-separated list of GPU ids (e.g. "0,1,2")
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ',' | sed 's/,$//'
}

AVAILABLE_GPUS=""
NUM_GPUS=1

get_num_gpus_for_job() {
    local scorer=$1
    if [[ -n "$GPU" ]]; then
        # Count commas + 1
        echo $(( $(echo "$GPU" | tr -cd ',' | wc -c) + 1 ))
    elif [[ "$scorer" == "prm" ]]; then
        echo "2"
    else
        echo "1"
    fi
}

get_timeout() {
    if [[ -n "$TIMEOUT" ]]; then
        echo "$TIMEOUT"
    elif [[ "$STRATEGY" == "baseline" || "$STRATEGY" == "self_consistency" ]]; then
        echo "14400"   # 4h
    elif [[ "$STRATEGY" == "adaptive_scaling" ]]; then
        echo "259200"  # 72h
    else
        echo "86400"   # 24h
    fi
}

get_job_label() {
    local strategy=$1 dataset=$2 scorer=$3
    if [[ -n "$LABEL" ]]; then
        echo "$LABEL"
    elif [[ -n "$scorer" && "$scorer" != "none" ]]; then
        echo "${strategy}_${dataset}_${scorer}"
    else
        echo "${strategy}_${dataset}"
    fi
}

# --- Submit function ---

submit_tsp_job() {
    local config_name=$1
    local num_gpus=$2
    local timeout=$3
    local label=$4
    local seed=$5

    if [[ "$DRY_RUN" == "yes" ]]; then
        echo "  Config:  $config_name"
        echo "  Seed:    $seed"
        echo "  GPUs:    $num_gpus"
        echo "  Timeout: ${timeout}s"
        echo "  Label:   $label"
        echo "  Command: python scripts/run_tts_eval.py --config-path=../config --config-name=${config_name} system.seed=${seed}"
        echo ""
    else
        local wrapper="/tmp/tsp_${label}_${seed}.sh"
        # TS_VISIBLE_DEVICES is set by tsp at runtime with the assigned GPU ids
        cat > "$wrapper" << WRAPPER_EOF
#!/bin/bash
eval "\$(grep '^export ' ~/.bashrc 2>/dev/null)"
export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-\$TS_VISIBLE_DEVICES}
cd "${PROJECT_DIR}" || exit 1

echo "============================================"
echo "Label: ${label}"
echo "Seed: ${seed}"
echo "Host: \$(hostname)"
echo "GPUs: \$CUDA_VISIBLE_DEVICES"
echo "Start: \$(date)"
echo "============================================"

python scripts/run_tts_eval.py --config-path=../config --config-name=${config_name} system.seed=${seed}

echo "============================================"
echo "End: \$(date)"
echo "============================================"
WRAPPER_EOF
        chmod +x "$wrapper"
        local tsp_args="-L $label -G $num_gpus"
        if [[ -n "$GPU" ]]; then
            tsp_args="-L $label -g $GPU"
        fi
        local job_id
        job_id=$(tsp $tsp_args timeout ${timeout} bash ${wrapper})
        echo "Queued job $job_id: $label (seed=$seed, gpus=$num_gpus, timeout=${timeout}s)"
    fi
}

# --- Main ---

# Detect GPUs and configure tsp
AVAILABLE_GPUS=$(detect_gpus)
if [[ -z "$AVAILABLE_GPUS" ]]; then
    AVAILABLE_GPUS="0"
    NUM_GPUS=1
else
    IFS=',' read -ra _gpu_arr <<< "$AVAILABLE_GPUS"
    NUM_GPUS=${#_gpu_arr[@]}
fi
# Tell tsp which GPUs are available and allow parallel jobs
export TS_VISIBLE_DEVICES="$AVAILABLE_GPUS"
if [[ "$DRY_RUN" != "yes" ]]; then
    tsp -S "$NUM_GPUS" 2>/dev/null
fi

echo "============================================"
echo "Local Experiment Submission (Task Spooler)"
echo "============================================"
echo "Strategy: $STRATEGY"
echo "Dataset:  $DATASET"
echo "Model:    $MODEL"
echo "GPUs:     $AVAILABLE_GPUS ($NUM_GPUS available)"
if [[ "$SEEDS" -gt 1 ]]; then
    echo "Seeds:    $SEED..$((SEED + SEEDS - 1)) ($SEEDS runs)"
else
    echo "Seed:     $SEED"
fi
if [[ -n "$SCORERS" ]]; then
    echo "Scorers:  $SCORERS"
fi
echo "============================================"
echo ""

# Determine scorers
if [[ "$STRATEGY" == "baseline" || "$STRATEGY" == "self_consistency" ]]; then
    scorer_list=("none")
elif [[ "$SCORERS" == "all" ]]; then
    scorer_list=("entropy" "perplexity" "sequence_prob" "prm")
elif [[ -n "$SCORERS" ]]; then
    IFS=',' read -ra scorer_list <<< "$SCORERS"
else
    echo "Error: --scorers required for $STRATEGY strategy"
    echo "Options: all, prm, entropy, perplexity, sequence_prob"
    exit 1
fi

if [[ "$DRY_RUN" == "yes" ]]; then
    echo "DRY RUN — would queue:"
    echo ""
fi

# Submit jobs (scorer × seed)
for scorer in "${scorer_list[@]}"; do
    if [[ "$scorer" == "none" ]]; then
        config_name=$(get_config_name "$STRATEGY" "$DATASET" "" "$MODEL")
    else
        config_name=$(get_config_name "$STRATEGY" "$DATASET" "$scorer" "$MODEL")
    fi

    timeout=$(get_timeout)
    base_label=$(get_job_label "$STRATEGY" "$DATASET" "$scorer")

    for ((s=0; s<SEEDS; s++)); do
        seed_val=$((SEED + s))
        num_gpus=$(get_num_gpus_for_job "$scorer")
        if [[ "$SEEDS" -gt 1 ]]; then
            label="${base_label}_s${seed_val}"
        else
            label="$base_label"
        fi
        submit_tsp_job "$config_name" "$num_gpus" "$timeout" "$label" "$seed_val"
    done
done

if [[ "$DRY_RUN" != "yes" ]]; then
    echo ""
    echo "Parallel slots: $NUM_GPUS (one per GPU). Run 'tsp' to view queue status."
fi
