#!/bin/bash
#
# Submit GPQA Diamond experiment in sequential parts
# This script splits the dataset into chunks and runs them sequentially in a single SLURM job
#
# Usage: ./submit_gpqa_diamond.sh <strategy> <scorer> <seed> [options]
#
# Examples:
#   ./submit_gpqa_diamond.sh offline_bon entropy 42
#   ./submit_gpqa_diamond.sh offline_bon perplexity 42 --parts 6
#   ./submit_gpqa_diamond.sh beam_search sequence_prob 43 --parts 4
#

set -e

# Default values
STRATEGY=""
SCORER=""
SEED=""
MODEL="qwen3_8b_thinking"
PARTS=4  # Number of parts to split GPQA Diamond (198 samples / 4 = ~50 per part)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        offline_bon|online_bon|beam_search|baseline|self_consistency|adaptive_scaling)
            STRATEGY="$1"
            shift
            ;;
        entropy|perplexity|sequence_prob|prm)
            SCORER="$1"
            shift
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --parts)
            PARTS="$2"
            shift 2
            ;;
        -h|--help)
            cat << EOF
Usage: $0 <strategy> <scorer> <seed> [OPTIONS]

Arguments:
  strategy      offline_bon, online_bon, beam_search, baseline, self_consistency, adaptive_scaling
  scorer        entropy, perplexity, sequence_prob, prm
  seed          Random seed (e.g., 42, 43, 44)

Options:
  --model       Model to use (default: qwen3_8b_thinking)
  --parts       Number of parts to split dataset into (default: 4, ~50 samples each)
  -h, --help    Show this help

Examples:
  $0 offline_bon entropy 42
  $0 offline_bon perplexity 42 --parts 6
  $0 beam_search sequence_prob 43 --parts 4
EOF
            exit 0
            ;;
        *)
            if [[ -z "$SEED" ]]; then
                SEED="$1"
            else
                echo "Unknown option: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$STRATEGY" || -z "$SCORER" || -z "$SEED" ]]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 <strategy> <scorer> <seed>"
    echo "Run '$0 --help' for more information"
    exit 1
fi

# GPQA Diamond has 198 samples
TOTAL_SAMPLES=198
SAMPLES_PER_PART=$((TOTAL_SAMPLES / PARTS))

echo "============================================"
echo "GPQA Diamond Sequential Experiment"
echo "============================================"
echo "Strategy: $STRATEGY"
echo "Scorer: $SCORER"
echo "Model: $MODEL"
echo "Seed: $SEED"
echo "Parts: $PARTS"
echo "Samples per part: $SAMPLES_PER_PART"
echo "============================================"
echo ""

# Build config name
get_config_name() {
    local strategy=$1
    local scorer=$2

    case $strategy in
        offline_bon) strategy_key="offline_best_of_n" ;;
        online_bon) strategy_key="online_best_of_n" ;;
        beam_search) strategy_key="beam_search" ;;
        baseline) strategy_key="baseline" ;;
        self_consistency) strategy_key="self_consistency" ;;
        adaptive_scaling) strategy_key="adaptive_scaling" ;;
    esac

    case $scorer in
        entropy) scorer_key="entropy" ;;
        perplexity) scorer_key="perplexity" ;;
        sequence_prob) scorer_key="sequence_prob" ;;
        prm) scorer_key="prm" ;;
    esac

    echo "experiments/${strategy_key}/gpqa_diamond/${strategy}_vllm_thinking_qwen3_8b_gpqa_diamond_${scorer_key}"
}

CONFIG_BASE=$(get_config_name "$STRATEGY" "$SCORER")
JOB_NAME="gpqa_${STRATEGY:0:4}_${SCORER:0:3}"

# Calculate time per part (8 hours per part to be safe)
TIME_PER_PART="08:00:00"
# Total time = parts * time_per_part
TOTAL_HOURS=$((PARTS * 8))
TOTAL_TIME="${TOTAL_HOURS}:00:00"

# Get GPUs
GPUS=1
if [[ "$MODEL" == "qwen3_8b_thinking" ]]; then
    GPUS=1
fi

# Get currently used nodes
USED_NODES=$(squeue -h -o "%N" -u "$USER" 2>/dev/null | sort -u | tr '\n' ',' | sed 's/,$//')

# Always exclude gpu-24 (broken GPU)
if [[ -n "$USED_NODES" ]]; then
    EXCLUDE_LINE="#SBATCH --exclude=${USED_NODES},gpu-24"
else
    EXCLUDE_LINE="#SBATCH --exclude=gpu-24"
fi

# Create SLURM script
SBATCH_SCRIPT="#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gpus-per-node=${GPUS}
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1984MB
#SBATCH --time=${TOTAL_TIME}
#SBATCH --output=slurm/gpqa_diamond_%j.log
#SBATCH --error=slurm/gpqa_diamond_%j.log
${EXCLUDE_LINE}

set -e

echo \"============================================\"
echo \"GPQA Diamond Sequential Experiment\"
echo \"============================================\"
echo \"SLURM Job ID: \$SLURM_JOB_ID\"
echo \"Strategy: ${STRATEGY}\"
echo \"Scorer: ${SCORER}\"
echo \"Seed: ${SEED}\"
echo \"Parts: ${PARTS}\"
echo \"Running on node: \$(hostname)\"
echo \"GPUs allocated: \$CUDA_VISIBLE_DEVICES\"
echo \"Start time: \$(date)\"
echo \"============================================\"
echo \"\"

# Move to project directory
cd /home/artem.shelmanov/vlad/llm-tts-service || exit 1

nvidia-smi || true

# Run each part sequentially
TOTAL_SAMPLES=${TOTAL_SAMPLES}
NUM_PARTS=${PARTS}
SAMPLES_PER_PART=\$((TOTAL_SAMPLES / NUM_PARTS))
REMAINDER=\$((TOTAL_SAMPLES % NUM_PARTS))

for PART in \$(seq 0 \$((NUM_PARTS - 1))); do
    # Calculate subset size for this part (add remainder to first part)
    if [ \"\$PART\" -eq 0 ]; then
        THIS_SUBSET=\$((SAMPLES_PER_PART + REMAINDER))
    else
        THIS_SUBSET=\$SAMPLES_PER_PART
    fi

    OFFSET=\$((PART * SAMPLES_PER_PART))
    if [ \"\$PART\" -gt 0 ]; then
        OFFSET=\$((OFFSET + REMAINDER))
    fi

    echo \"\"
    echo \"============================================\"
    echo \"Starting Part \$PART (offset=\$OFFSET, max_samples=\${THIS_SUBSET})\"
    echo \"Time: \$(date)\"
    echo \"============================================\"

    python scripts/run_tts_eval.py \\
        --config-path=/home/artem.shelmanov/vlad/llm-tts-service/config \\
        --config-name=${CONFIG_BASE} \\
        system.seed=${SEED} \\
        dataset.offset=\$OFFSET \\
        dataset.subset=\${THIS_SUBSET}

    echo \"\"
    echo \"============================================\"
    echo \"Finished Part \$PART\"
    echo \"Time: \$(date)\"
    echo \"============================================\"
done

echo \"\"
echo \"============================================\"
echo \"All ${PARTS} parts completed!\"
echo \"End time: \$(date)\"
echo \"============================================\"
"

# Write to temp file
TEMP_SCRIPT="/tmp/gpqa_diamond_${JOB_NAME}_${SEED}.sh"
echo "$SBATCH_SCRIPT" > "$TEMP_SCRIPT"

# Submit job
JOB_ID=$(sbatch "$TEMP_SCRIPT" | grep -oP '\d+' || true)
rm -f "$TEMP_SCRIPT"

if [[ -n "$JOB_ID" ]]; then
    echo "Submitted job $JOB_ID: $JOB_NAME ($PARTS parts, seed=$SEED)"
    echo "Log file: slurm/gpqa_diamond_${JOB_ID}.log"
else
    echo "Failed to submit job"
    exit 1
fi
