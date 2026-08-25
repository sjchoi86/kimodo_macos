#!/bin/zsh

set -euo pipefail

PROJECT_DIR=${0:A:h}
RIM_ROOT=${1:-"$PROJECT_DIR/../rim_v6"}
TARGET_DIR="$RIM_ROOT/notebooks/15_motion_retargeting/motion_data"

if [[ ! -d "$RIM_ROOT" ]]; then
    print -u2 "RIM v6 repository does not exist: $RIM_ROOT"
    exit 2
fi

mkdir -p "$TARGET_DIR"

source_paths=("$PROJECT_DIR"/outputs/*.npz(N))
if (( ${#source_paths} == 0 )); then
    print -u2 "No Kimodo motion archives found in: $PROJECT_DIR/outputs"
    exit 3
fi

for source_path in "${source_paths[@]}"; do
    destination_path="$TARGET_DIR/kimodo_soma_rp_v1_${source_path:t}"
    if [[ -e "$destination_path" ]]; then
        if cmp -s "$source_path" "$destination_path"; then
            print "Current: $destination_path"
            continue
        fi
        print -u2 "Destination already exists with different contents: $destination_path"
        exit 4
    fi
    cp "$source_path" "$destination_path"
    print "Copied:  $destination_path"
done
