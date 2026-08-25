#!/bin/zsh

set -euo pipefail

PROJECT_DIR=${0:A:h}
CONDA_ENV=${KIMODO_CONDA_ENV:-kimodo-macos}
DEVICE=${1:-mps}
PROMPT=${2:-"A person walks forward, turns to the right, and stops."}
DURATION=${3:-3.0}
DIFFUSION_STEPS=${4:-100}
OUTPUT_NAME=${5:-"${DEVICE}_motion"}
SUMMARY_PROMPT=${6:-"$PROMPT"}
SEED=${7:-7}
MODEL="Kimodo-SOMA-RP-v1"
FPS=30.0
NUM_SAMPLES=1
SAMPLE_INDEX=0
NUM_TRANSITION_FRAMES=5
GENERATOR_REPOSITORY="https://github.com/atticus-lv/kimodo"
TEXT_ENCODER_BASE=${TEXT_ENCODER_BASE:-"meta-llama/Meta-Llama-3-8B-Instruct"}
TEXT_ENCODER_ADAPTER_MNTP="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
TEXT_ENCODER_ADAPTER_SUPERVISED="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"

if [[ "$DEVICE" != "mps" && "$DEVICE" != "cpu" ]]; then
    print -u2 "DEVICE must be 'mps' or 'cpu': $DEVICE"
    exit 2
fi
if ! command -v conda >/dev/null 2>&1; then
    print -u2 "Conda is required. Run setup_macos.sh first."
    exit 2
fi
if ! conda run -n "$CONDA_ENV" python --version >/dev/null 2>&1; then
    print -u2 "Conda environment does not exist: $CONDA_ENV"
    print -u2 "Run setup_macos.sh first."
    exit 2
fi

export HF_HUB_CACHE="$PROJECT_DIR/hf-cache/hub"
if [[ -z ${HF_HUB_OFFLINE+x} ]]; then
    if [[ -e "$PROJECT_DIR/checkpoints/$MODEL/config.yaml" && -d "$HF_HUB_CACHE" ]]; then
        export HF_HUB_OFFLINE=1
    else
        export HF_HUB_OFFLINE=0
    fi
fi
export CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"
export TEXT_ENCODER_MODE=local
export TEXT_ENCODER_DEVICE="$DEVICE"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

mkdir -p "$PROJECT_DIR/outputs"
OUTPUT_PATH="$PROJECT_DIR/outputs/${OUTPUT_NAME}.npz"
if [[ -e "$OUTPUT_PATH" ]]; then
    print -u2 "Output already exists: $OUTPUT_PATH"
    exit 3
fi
GENERATOR_REVISION=$(git -C "$PROJECT_DIR/kimodo" rev-parse HEAD)
METADATA_FLAGS=()
if [[ "$HF_HUB_OFFLINE" == "1" ]]; then
    METADATA_FLAGS+=(--hf-hub-offline)
fi
if [[ "$PYTORCH_ENABLE_MPS_FALLBACK" == "1" ]]; then
    METADATA_FLAGS+=(--mps-fallback)
fi

cd "$PROJECT_DIR"
/usr/bin/time -p conda run --no-capture-output -n "$CONDA_ENV" kimodo_gen \
    "$PROMPT" \
    --model "$MODEL" \
    --duration "$DURATION" \
    --diffusion_steps "$DIFFUSION_STEPS" \
    --num_samples "$NUM_SAMPLES" \
    --num_transition_frames "$NUM_TRANSITION_FRAMES" \
    --device "$DEVICE" \
    --output "$PROJECT_DIR/outputs/$OUTPUT_NAME" \
    --no-postprocess \
    --seed "$SEED"

conda run --no-capture-output -n "$CONDA_ENV" python \
    "$PROJECT_DIR/embed_motion_metadata.py" \
    "$OUTPUT_PATH" \
    --prompt-full "$PROMPT" \
    --prompt-summary "$SUMMARY_PROMPT" \
    --model "$MODEL" \
    --device "$DEVICE" \
    --output-name "$OUTPUT_NAME" \
    --requested-duration-sec "$DURATION" \
    --fps "$FPS" \
    --diffusion-steps "$DIFFUSION_STEPS" \
    --num-samples "$NUM_SAMPLES" \
    --sample-index "$SAMPLE_INDEX" \
    --num-transition-frames "$NUM_TRANSITION_FRAMES" \
    --seed "$SEED" \
    --cfg-type "model_default" \
    --constraints "none" \
    --generator-repository "$GENERATOR_REPOSITORY" \
    --generator-revision "$GENERATOR_REVISION" \
    --text-encoder-base "$TEXT_ENCODER_BASE" \
    --text-encoder-adapter "$TEXT_ENCODER_ADAPTER_MNTP" \
    --text-encoder-adapter "$TEXT_ENCODER_ADAPTER_SUPERVISED" \
    "${METADATA_FLAGS[@]}"

conda run --no-capture-output -n "$CONDA_ENV" python \
    "$PROJECT_DIR/validate_motion.py" \
    "$OUTPUT_PATH"
