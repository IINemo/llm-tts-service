#!/bin/bash
#
# Unified SLURM experiment submission script
# Usage: ./submit.sh --strategy <strategy> --dataset <dataset> [OPTIONS]
#

set -e

# Default values
STRATEGY=""
DATASET=""
MODEL="qwen25_7b"  # default model
SCORERS=""
SEEDS=1
SEED=42
MODE="parallel"  # parallel or sequential
DRY_RUN="no"
GPUS=""
TIME_LIMIT=""

# Help message
show_help() {
    cat << EOF
Usage: $0 --strategy <strategy> --dataset <dataset> [OPTIONS]

Required:
  --strategy <name>     Strategy: baseline, self_consistency, offline_bon, online_bon, beam_search, adaptive_scaling
  --dataset <name>      Dataset: aime2024, aime2025, math500, olympiadbench, gaokao2023en, minerva_math, gpqa_diamond

Optional:
  --model <name>         Model: qwen25_7b (default), qwen3_8b_thinking, qwen3_8b, qwen25_math_7b, qwen25_math_15b
  --scorers <list>       Scorers: all, prm, entropy, perplexity, sequence_prob (default: none for baseline)
  --seeds <n>            Number of seeds to run (default: 1)
  --seed <n>             Single seed to use (default: 42)
  --mode <mode>          Execution mode: parallel (default) or sequential
                        - parallel: array jobs run simultaneously on different GPUs
                        - sequential: for loop runs on same GPU, one after another
  --sequential           Alias for --mode sequential
  --gpus <n>             Number of GPUs (default: auto)
  --time <hh:mm:ss>      Time limit (default: auto)
  --dry-run              Show commands without submitting
  -h, --help             Show this help

Examples:
  # Single baseline run (default seed=42)
  $0 --strategy baseline --dataset aime2024

  # Baseline with 4 seeds (parallel array jobs)
  $0 --strategy baseline --dataset aime2024 --seeds 4

  # Offline BoN with all scorers (sequential, saves job slots)
  $0 --strategy offline_bon --dataset math500 --scorers all --mode sequential

  # Offline BoN with all scorers (parallel array jobs, faster)
  $0 --strategy offline_bon --dataset math500 --scorers all --mode parallel

  # Online BoN with specific scorer
  $0 --strategy online_bon --dataset olympiadbench --scorers prm

  # Adaptive scaling with all scorers (sequential to bypass job limits)
  $0 --strategy adaptive_scaling --dataset gaokao2023en --scorers all --sequential
EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --scorers)
            SCORERS="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            SEEDS=1
            shift 2
            ;;
        --no-array)
            MODE="sequential"
            shift
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --sequential)
            MODE="sequential"
            shift
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="yes"
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Validate required arguments
if [[ -z "$STRATEGY" || -z "$DATASET" ]]; then
    echo "Error: --strategy and --dataset are required"
    echo ""
    show_help
fi

# Project directory
PROJECT_DIR="/home/artem.shelmanov/vlad/llm-tts-service"
cd "$PROJECT_DIR"

# Config mappings
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
MODEL_CONFIGS[qwen3_8b_thinking]="vllm_qwen3_8b_thinking"
MODEL_CONFIGS[qwen3_8b]="vllm_qwen3_8b"
MODEL_CONFIGS[qwen25_math_7b]="vllm_qwen25_math_7b_instruct"
MODEL_CONFIGS[qwen25_math_15b]="vllm_qwen25_math_15b_instruct"

# Function to get config path
get_config_name() {
    local strategy=$1
    local dataset=$2
    local scorer=$3
    local model=$4

    local strategy_key=${STRATEGY_CONFIGS[$strategy]}
    local dataset_key=${DATASET_CONFIGS[$dataset]}
    local model_key=${MODEL_CONFIGS[$model]}

    if [[ -z "$strategy_key" || -z "$dataset_key" ]]; then
        echo "Error: Unknown strategy or dataset"
        exit 1
    fi

    if [[ -z "$model_key" ]]; then
        echo "Error: Unknown model: $model"
        exit 1
    fi

    # Build config name
    if [[ "$strategy" == "baseline" ]]; then
        # Baseline doesn't use scorers
        echo "experiments/${strategy_key}/${dataset_key}/baseline_${model_key}_${dataset_key}"
    elif [[ "$strategy" == "self_consistency" ]]; then
        echo "experiments/${strategy_key}/${dataset_key}/self_consistency_${model_key}_${dataset_key}"
    else
        local scorer_key=${SCORER_CONFIGS[$scorer]}
        if [[ -z "$scorer_key" ]]; then
            echo "Error: Unknown scorer: $scorer"
            exit 1
        fi
        echo "experiments/${strategy_key}/${dataset_key}/${strategy}_${model_key}_${dataset_key}_${scorer_key}"
    fi
}

# Function to determine if PRM is needed
is_prm() {
    [[ "$1" == "prm" ]]
}

# Function to get GPU count
get_gpu_count() {
    local scorer=$1
    if [[ -n "$GPUS" ]]; then
        echo "$GPUS"
    elif [[ "$STRATEGY" == "baseline" || "$STRATEGY" == "self_consistency" ]]; then
        echo "1"
    elif is_prm "$scorer"; then
        echo "2"
    else
        echo "1"
    fi
}

# Function to get time limit
get_time_limit() {
    if [[ -n "$TIME_LIMIT" ]]; then
        echo "$TIME_LIMIT"
    elif [[ "$STRATEGY" == "baseline" || "$STRATEGY" == "self_consistency" ]]; then
        echo "04:00:00"
    elif [[ "$STRATEGY" == "offline_bon" || "$STRATEGY" == "online_bon" || "$STRATEGY" == "beam_search" ]]; then
        echo "24:00:00"
    elif [[ "$STRATEGY" == "adaptive_scaling" ]]; then
        echo "72:00:00"
    else
        echo "24:00:00"
    fi
}

# Function to generate job name
get_job_name() {
    local strategy=$1
    local dataset=$2
    local scorer=$3

    local short_strat
    case $strategy in
        baseline) short_strat="bl" ;;
        self_consistency) short_strat="sc" ;;
        offline_bon) short_strat="obon" ;;
        online_bon) short_strat="bon" ;;
        beam_search) short_strat="beam" ;;
        adaptive_scaling) short_strat="adapt" ;;
    esac

    local short_dataset
    case $dataset in
        aime2024) short_dataset="aime24" ;;
        aime2025) short_dataset="aime25" ;;
        olympiadbench) short_dataset="olymp" ;;
        gaokao2023en) short_dataset="gaokao" ;;
        minerva_math) short_dataset="minerva" ;;
        gpqa_diamond) short_dataset="gpqa" ;;
        *) short_dataset="$dataset" ;;
    esac

    local short_scorer
    if [[ -n "$scorer" ]]; then
        case $scorer in
            entropy) short_scorer="ent" ;;
            perplexity) short_scorer="ppl" ;;
            sequence_prob) short_scorer="sp" ;;
            prm) short_scorer="prm" ;;
        esac
        echo "${short_strat}_${short_scorer}_${short_dataset}"
    else
        echo "${short_strat}_${short_dataset}"
    fi
}

# Function to submit a single job
submit_job() {
    local config_name=$1
    local seed=$2
    local job_name=$3
    local gpus=$4
    local time_limit=$5
    local use_array=$6
    local array_size=$7

    local output_file="logs/${job_name}_${seed}_%j.out"
    local error_file="logs/${job_name}_${seed}_%j.err"

    # For array jobs, template the seed and update output files
    if [[ "$use_array" == "yes" && "$array_size" -gt 1 ]]; then
        output_file="logs/${job_name}_%a_%j.out"
        error_file="logs/${job_name}_%a_%j.err"
    fi

    # Build sbatch command
    local sbatch_cmd="#!/bin/bash
#SBATCH -J ${job_name}
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${gpus}
#SBATCH -p long
#SBATCH -t ${time_limit}
#SBATCH -o ${output_file}
#SBATCH -e ${error_file}
#SBATCH --cpus-per-task=16
"

    # Add array directive if needed
    if [[ "$use_array" == "yes" && "$array_size" -gt 1 ]]; then
        sbatch_cmd="${sbatch_cmd}#SBATCH --array=0-$((array_size-1))
#SBATCH --cpus-per-task=16

# Generate seeds array and select by task ID
SEEDS=({$seed})
SEED=\${SEEDS[\$SLURM_ARRAY_TASK_ID]}
"
    fi

    sbatch_cmd="${sbatch_cmd}
PROJECT_DIR=\"$PROJECT_DIR\"
cd \"\$PROJECT_DIR\"
mkdir -p logs

module load rocm 2>/dev/null || true

source ~/.bashrc
conda activate lm-polygraph-env

echo \"============================================\"
echo \"SLURM Job ID: \$SLURM_JOB_ID\"
echo \"Seed: \${SEED}\"
echo \"Running on node: \$(hostname)\"
echo \"GPUs allocated: \$CUDA_VISIBLE_DEVICES\"
echo \"Start time: \$(date)\"
echo \"============================================\"

nvidia-smi || true

# Setup qwen-eval environment
QWEN_EVAL_PYTHON=\"\$HOME/miniconda3/envs/qwen-eval/bin/python\"
if [ ! -f \"\$QWEN_EVAL_PYTHON\" ]; then
    echo \"Setting up qwen-eval environment...\"
    conda create -n qwen-eval python=3.11 -y
    conda run -n qwen-eval pip install sympy==1.12 antlr4-python3-runtime==4.11.1 regex
else
    echo \"qwen-eval environment found\"
fi

SEED=${seed}
python scripts/run_tts_eval.py \\
    --config-path=../config \\
    --config-name=${config_name} \\
    system.seed=\${SEED}
"

    # Write to temp file and submit
    local temp_script="/tmp/job_${job_name}_${seed}.sh"
    echo "$sbatch_cmd" > "$temp_script"

    if [[ "$DRY_RUN" == "yes" ]]; then
        echo "Would submit:"
        echo "  Config: $config_name"
        echo "  Seed: $seed"
        echo "  GPUs: $gpus"
        echo "  Time: $time_limit"
        echo "  Script: $temp_script"
        echo ""
        cat "$temp_script"
        rm -f "$temp_script"
    else
        local job_id=$(sbatch "$temp_script" | grep -oP '\d+')
        echo "Submitted job $job_id: $job_name (seed=$seed)"
        rm -f "$temp_script"
    fi
}

# Function to submit sequential wrapper job (multiple scorers in one job)
submit_sequential_job() {
    local num_configs=$1
    shift
    local configs=("${@:1:$num_configs}")
    shift $num_configs
    local strategy=$1
    local dataset=$2
    local gpus=$3
    local time_limit=$4
    local seed=$5

    local job_name="${strategy}_seq"
    local output_file="logs/${job_name}_%j.out"
    local error_file="logs/${job_name}_%j.err"

    # Calculate number of experiments
    local num_exps=$num_configs
    # Adjust time for sequential runs
    local adjusted_time
    local hours=$(echo "$time_limit" | cut -d: -f1)
    hours=$((hours * num_exps))
    adjusted_time="${hours}:00:00"

    # Build sbatch command with for loop
    local sbatch_cmd="#!/bin/bash
#SBATCH -J ${job_name}
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${gpus}
#SBATCH -p long
#SBATCH -t ${adjusted_time}
#SBATCH -o ${output_file}
#SBATCH -e ${error_file}
#SBATCH --cpus-per-task=16

set -e

PROJECT_DIR=\"$PROJECT_DIR\"
cd \"\$PROJECT_DIR\"
mkdir -p logs

module load rocm 2>/dev/null || true

source ~/.bashrc
conda activate lm-polygraph-env

echo \"============================================\"
echo \"SLURM Job ID: \$SLURM_JOB_ID\"
echo \"Mode: Sequential (${num_exps} experiments)\"
echo \"Seed: ${seed}\"
echo \"Running on node: \$(hostname)\"
echo \"GPUs allocated: \$CUDA_VISIBLE_DEVICES\"
echo \"Start time: \$(date)\"
echo \"============================================\"

nvidia-smi || true

# Setup qwen-eval environment
QWEN_EVAL_PYTHON=\"\$HOME/miniconda3/envs/qwen-eval/bin/python\"
if [ ! -f \"\$QWEN_EVAL_PYTHON\" ]; then
    echo \"Setting up qwen-eval environment...\"
    conda create -n qwen-eval python=3.11 -y
    conda run -n qwen-eval pip install sympy==1.12 antlr4-python3-runtime==4.11.1 regex
else
    echo \"qwen-eval environment found\"
fi

# Array of config names
declare -a CONFIG_ARRAY=(${configs[*]})

# Run experiments sequentially
for i in \"\${!CONFIG_ARRAY[@]}\"; do
    echo \"\"
    echo \"============================================\"
    echo \"Starting experiment: \$i\"
    echo \"Time: \$(date)\"
    echo \"============================================\"

    python scripts/run_tts_eval.py \\
        --config-path=../config \\
        --config-name=\$i \\
        system.seed=${seed}

    echo \"\"
    echo \"============================================\"
    echo \"Finished experiment: \$i\"
    echo \"Time: \$(date)\"
    echo \"============================================\"
done

echo \"\"
echo \"============================================\"
echo \"All ${num_exps} experiments completed!\"
echo \"End time: \$(date)\"
echo \"============================================\"
"

    # Write to temp file and submit
    local temp_script="/tmp/job_${job_name}_${seed}.sh"
    echo "$sbatch_cmd" > "$temp_script"

    if [[ "$DRY_RUN" == "yes" ]]; then
        echo "Would submit:"
        echo "  Mode: sequential (${num_exps} experiments)"
        echo "  Configs: ${configs[*]}"
        echo "  Seed: $seed"
        echo "  GPUs: $gpus"
        echo "  Time: $time_limit (each)"
        echo "  Total time: $adjusted_time"
        echo "  Script: $temp_script"
        echo ""
        head -30 "$temp_script"
        echo "..."
        rm -f "$temp_script"
    else
        local job_id=$(sbatch "$temp_script" | grep -oP '\d+')
        echo "Submitted job $job_id: $job_name (${num_exps} experiments, sequential)"
        rm -f "$temp_script"
    fi
}

# Main execution
echo "============================================"
echo "Experiment Configuration"
echo "============================================"
echo "Strategy: $STRATEGY"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Mode: $MODE"
if [[ -n "$SCORERS" ]]; then
    echo "Scorers: $SCORERS"
fi
if [[ "$SEEDS" -gt 1 ]]; then
    echo "Seeds: $SEEDS"
else
    echo "Seed: $SEED"
fi
echo "============================================"
echo ""

# Determine which scorers to run
if [[ "$STRATEGY" == "baseline" || "$STRATEGY" == "self_consistency" ]]; then
    # These don't use scorers
    scorer_list=("none")
elif [[ "$SCORERS" == "all" ]]; then
    scorer_list=("entropy" "perplexity" "sequence_prob" "prm")
elif [[ -n "$SCORERS" ]]; then
    IFS=',' read -ra scorer_list <<< "$SCORERS"
    scorer_list=("${scorer_list[@]}")
else
    echo "Error: --scorers required for $STRATEGY strategy"
    echo "Options: all, prm, entropy, perplexity, sequence_prob"
    exit 1
fi

# Build configs and job names for all scorers
declare -a config_names
declare -a job_names

for scorer in "${scorer_list[@]}"; do
    if [[ "$scorer" == "none" ]]; then
        config_name=$(get_config_name "$STRATEGY" "$DATASET" "" "$MODEL")
        job_name=$(get_job_name "$STRATEGY" "$DATASET" "")
    else
        config_name=$(get_config_name "$STRATEGY" "$DATASET" "$scorer" "$MODEL")
        job_name=$(get_job_name "$STRATEGY" "$DATASET" "$scorer")
    fi
    config_names+=("$config_name")
    job_names+=("$job_name")
done

# Get common time limit
time_limit=$(get_time_limit)

# For sequential mode, get max GPU count across all scorers
max_gpus=1
for scorer in "${scorer_list[@]}"; do
    scorer_gpus=$(get_gpu_count "$scorer")
    if [[ "$scorer_gpus" -gt "$max_gpus" ]]; then
        max_gpus=$scorer_gpus
    fi
done

# Submit based on mode
if [[ "$MODE" == "sequential" && "${#scorer_list[@]}" -gt 1 && "$SCORERS" == "all" ]]; then
    # Sequential mode with all scorers: single wrapper job
    submit_sequential_job ${#config_names[@]} "${config_names[@]}" "$STRATEGY" "$DATASET" "$max_gpus" "$time_limit" "$SEED"
elif [[ "$MODE" == "sequential" && "${#scorer_list[@]}" -gt 1 ]]; then
    # Sequential mode with specific scorers: single wrapper job
    submit_sequential_job ${#config_names[@]} "${config_names[@]}" "$STRATEGY" "$DATASET" "$max_gpus" "$time_limit" "$SEED"
else
    # Parallel mode: submit separate jobs for each scorer
    for i in "${!config_names[@]}"; do
        config_name="${config_names[$i]}"
        job_name="${job_names[$i]}"
        scorer="${scorer_list[$i]}"

        # Get GPU count for this scorer
        scorer_gpus=$(get_gpu_count "$scorer")

        # Handle seeds
        if [[ "$SEEDS" -gt 1 ]]; then
            if [[ "$MODE" == "parallel" ]]; then
                # Array jobs (parallel) - generate seed sequence like "42,43,44,45"
                seed_seq=$(seq -s, $SEED $(($SEED+$SEEDS-1)))
                submit_job "$config_name" "$seed_seq" "$job_name" "$scorer_gpus" "$time_limit" "yes" "$SEEDS"
            else
                # Sequential seeds (separate jobs)
                for ((s=0; s<SEEDS; s++)); do
                    seed_val=$(($SEED+s))
                    submit_job "$config_name" "$seed_val" "$job_name" "$scorer_gpus" "$time_limit" "no" "1"
                done
            fi
        else
            # Single seed
            submit_job "$config_name" "$SEED" "$job_name" "$scorer_gpus" "$time_limit" "no" "1"
        fi
    done
fi

echo ""
echo "============================================"
echo "Done!"
echo "============================================"
