#!/bin/zsh

set -euo pipefail

PROJECT_DIR=${0:A:h}
CONDA_ENV=${KIMODO_CONDA_ENV:-kimodo-macos}

if ! command -v conda >/dev/null 2>&1; then
    print -u2 "Conda is required. Install Miniforge, then run this script again."
    exit 2
fi

git -C "$PROJECT_DIR" submodule update --init --recursive

if ! conda run -n "$CONDA_ENV" python --version >/dev/null 2>&1; then
    conda env create --file "$PROJECT_DIR/environment.yml" --name "$CONDA_ENV"
else
    conda env update \
        --file "$PROJECT_DIR/environment.yml" \
        --name "$CONDA_ENV" \
        --prune
fi

conda run -n "$CONDA_ENV" python -m pip install --upgrade pip
conda run -n "$CONDA_ENV" python -m pip install "torch==2.13.0"

SKIP_MOTION_CORRECTION_IN_SETUP=1 conda run -n "$CONDA_ENV" \
    python -m pip install \
    "$PROJECT_DIR/kimodo"

conda run -n "$CONDA_ENV" python -c \
    'import torch; from kimodo.model import load_model; print("torch:[%s] mps:[%s]"%(torch.__version__,torch.backends.mps.is_available()))'

print "Kimodo macOS environment is ready."
print "Authenticate once with: conda run -n $CONDA_ENV hf auth login"
print "Then generate with:     HF_HUB_OFFLINE=0 $PROJECT_DIR/run_motion.sh mps"
