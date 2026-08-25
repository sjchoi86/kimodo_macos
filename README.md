# Kimodo on Apple Silicon macOS

Run Kimodo SOMA77 text-to-motion generation locally on an Apple Silicon Mac,
save each motion and its complete prompt metadata in one portable NPZ file,
and load the result directly in NumPy or RIM v6.

This repository uses **Conda only**. It does not use uv, venv, or Poetry.

## What this repository provides

- A pinned Kimodo fork with Apple Metal/MPS support.
- A repeatable Conda installation script.
- One command for MPS or CPU text-to-motion generation.
- Atomic, pickle-free metadata embedding in the generated NPZ.
- Strict validation of SOMA77 arrays and generation metadata.
- Three checked-in, five-second example motions.
- A safe copy command and a Python loading example for RIM v6.

The optional native motion-correction extension is not installed by default.
Generation uses Kimodo's `--no-postprocess` option. The text encoder and core
diffusion motion generator work on both MPS and CPU.

## Requirements

- An Apple Silicon Mac.
- macOS with Command Line Tools and Git. Run `xcode-select --install` if Git is
  not already available.
- Conda. Miniforge is recommended.
- 32 GB or more of unified memory is recommended for local text encoding.
- At least 25 GB of free disk space for the environment, text encoder, and
  Kimodo checkpoint.
- A Hugging Face account with a read token. Accept the model terms for
  [Meta Llama 3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
  before the first download.

## Quick start for a new user

If Conda is already installed, this is the complete flow from clone to the
first generated motion. The Hugging Face login command is interactive and asks
you to paste a read token.

```bash
git clone --recurse-submodules https://github.com/sjchoi86/kimodo_macos.git
cd kimodo_macos

./setup_macos.sh

conda run --no-capture-output -n kimodo-macos hf auth login
conda run --no-capture-output -n kimodo-macos hf auth whoami

HF_HUB_OFFLINE=0 ./run_motion.sh \
    mps \
    "A person walks forward for four steps and stops upright with both arms relaxed" \
    5.0 \
    100 \
    first_walk_5s \
    "Walk forward and stop" \
    7

conda run -n kimodo-macos python validate_motion.py \
    outputs/first_walk_5s.npz
```

The first generation downloads the text encoder and Kimodo checkpoint and can
take substantially longer than later runs. The final motion is saved at:

```text
outputs/first_walk_5s.npz
```

Continue with the detailed instructions below if Conda is not installed or if
you want to understand each step.

## Detailed installation from a fresh Mac

If Conda is already available, skip the first block.

### 1. Install Miniforge with Homebrew

```bash
brew install --cask miniforge
/opt/homebrew/Caskroom/miniforge/base/bin/conda init zsh
exec zsh
```

Confirm that Conda is available:

```bash
conda --version
uname -m
```

`uname -m` should print `arm64` on a supported Apple Silicon Mac.

### 2. Clone and install Kimodo

Copy and paste the complete block:

```bash
git clone --recurse-submodules https://github.com/sjchoi86/kimodo_macos.git
cd kimodo_macos
./setup_macos.sh
```

The setup script:

1. initializes the pinned Kimodo submodule if necessary;
2. creates or updates the `kimodo-macos` Conda environment from
   `environment.yml`;
3. installs PyTorch 2.13.0 and the pinned Kimodo package; and
4. prints the installed PyTorch version and MPS availability.

A successful final line looks like this:

```text
torch:[2.13.0] mps:[True]
```

`setup_macos.sh` installs Python packages only. It does not download the large
text encoder or Kimodo checkpoint; those are downloaded during the first
generation.

### 3. Accept the model terms and create a token

Before the first download:

1. sign in to [Hugging Face](https://huggingface.co/);
2. open the
   [Meta Llama 3 8B Instruct model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
   and accept its terms if requested; and
3. create a read token in the Hugging Face access-token settings.

Do not place the token in a shell script, Markdown file, generated NPZ, or Git
configuration.

### 4. Authenticate with Hugging Face once

```bash
conda run --no-capture-output -n kimodo-macos hf auth login
conda run --no-capture-output -n kimodo-macos hf auth whoami
```

Paste a Hugging Face access token when prompted. The token is stored by the
Hugging Face client in the user's standard Hugging Face configuration, not in
this repository or in a generated motion file. `run_motion.sh` keeps only the
downloaded model cache under `hf-cache/hub`, so the standard login remains
available while large model files stay local to this repository.

### 5. Verify the installed environment

```bash
conda run -n kimodo-macos python -c \
    'import torch; import kimodo; print(torch.__version__); print(torch.backends.mps.is_available())'

conda run -n kimodo-macos python validate_motion.py \
    outputs/side_steps_arms_open_5s.npz
```

The first command should print PyTorch `2.13.0` and `True`. The second command
validates a small example motion already included in the repository and does
not download a model.

## Generate a motion

The first run must be online because it downloads the checkpoint and text
encoder into `hf-cache/hub`. Copy and paste this example from the repository
root:

```bash
HF_HUB_OFFLINE=0 ./run_motion.sh \
    mps \
    "A person starts upright, takes two small steps to the right while raising both arms outward to shoulder height, returns to the center, and finishes upright with arms relaxed" \
    5.0 \
    100 \
    my_side_steps_5s \
    "Side steps with arm raises" \
    7
```

The positional arguments are:

```text
backend full_prompt duration_seconds diffusion_steps output_name summary_prompt seed
```

| Argument | Meaning | Example |
| --- | --- | --- |
| `backend` | `mps` or `cpu` | `mps` |
| `full_prompt` | Exact text sent to Kimodo | `"A person walks forward..."` |
| `duration_seconds` | Requested motion length | `5.0` |
| `diffusion_steps` | Number of denoising steps | `100` |
| `output_name` | Output filename without `.npz` | `my_motion_5s` |
| `summary_prompt` | Short human-readable label | `"Walk and stop"` |
| `seed` | Generation seed | `7` |

To use CPU, change only the first argument:

```bash
HF_HUB_OFFLINE=0 ./run_motion.sh \
    cpu \
    "A person marches in place with natural opposite arm swings and finishes standing still" \
    5.0 \
    100 \
    my_march_5s \
    "March in place" \
    7
```

After all models have been downloaded, offline generation is available:

```bash
HF_HUB_OFFLINE=1 ./run_motion.sh \
    mps \
    "A person walks forward for four steps and stops upright" \
    5.0 \
    100 \
    walk_and_stop_5s \
    "Walk and stop" \
    7
```

The script refuses to overwrite an existing output. Choose a new
`output_name` if the destination already exists.

## Where files are stored

Generated motion:

```text
outputs/<output_name>.npz
```

For the first example above, the exact path is:

```text
outputs/my_side_steps_5s.npz
```

Downloaded model data is kept separately:

```text
hf-cache/       Hugging Face text-encoder and model cache
checkpoints/    Optional local Kimodo checkpoint directory or links
outputs/        Generated and example motion NPZ files
```

`hf-cache/` and `checkpoints/` are excluded from Git. New files under
`outputs/` are also ignored by default so large experiments are not committed
accidentally. Three small, validated example archives are deliberately tracked:

- `outputs/side_steps_arms_open_5s.npz`
- `outputs/march_arm_swing_5s.npz`
- `outputs/forward_back_arm_reach_5s.npz`

## What is stored in each NPZ

`run_motion.sh` first asks Kimodo to write its numeric motion arrays. It then
runs `embed_motion_metadata.py`, which atomically rewrites the same NPZ with a
complete schema-v1 generation record. No sidecar JSON file is required.

Important motion arrays include:

| Key | Meaning | SOMA77 shape |
| --- | --- | --- |
| `posed_joints` | Global joint positions | `(T,77,3)` |
| `global_rot_mats` | Global joint rotations | `(T,77,3,3)` |
| `local_rot_mats` | Parent-relative joint rotations | `(T,77,3,3)` |
| `foot_contacts` | Six SOMA foot contact channels | `(T,6)` |
| `root_positions` | Root trajectory | `(T,3)` |
| `smooth_root_pos` | Smoothed root representation | model-dependent |
| `global_root_heading` | Root heading representation | model-dependent |
| `fps` | Scalar frames per second | scalar |

The same archive also contains:

- `text_prompt_full` and `text_prompt_summary`;
- model name, device, requested and actual duration, and frame count;
- diffusion steps, seed, sample index, transition frames, and CFG mode;
- postprocessing, constraint, offline, and MPS-fallback flags;
- generation UTC timestamp and exact generator repository revision;
- Python, PyTorch, platform, and text-encoder identifiers; and
- `metadata_schema_version`.

All strings use NumPy Unicode dtypes. No Python objects or dictionaries are
stored, so every file can be opened safely with `allow_pickle=False`.

## Validate a generated motion

`run_motion.sh` validates each output automatically. You can repeat validation
at any time:

```bash
conda run -n kimodo-macos python validate_motion.py \
    outputs/my_side_steps_5s.npz
```

The validator checks required keys, object-free dtypes, SOMA77 shapes, finite
numeric values, non-empty prompts, positive FPS, and frame/FPS/duration
consistency.

Validate every local motion with this copy-paste command:

```bash
for motion_path in outputs/*.npz; do
    conda run -n kimodo-macos python validate_motion.py "$motion_path"
done
```

## Load a motion with NumPy

This example needs only NumPy and does not depend on RIM v6:

```python
from pathlib import Path

import numpy as np

motion_path = Path("outputs/side_steps_arms_open_5s.npz")

with np.load(motion_path,allow_pickle=False) as archive:
    positions = np.asarray(archive["posed_joints"],dtype=np.float64)
    rotations = np.asarray(archive["global_rot_mats"],dtype=np.float64)
    contacts = np.asarray(archive["foot_contacts"],dtype=bool)
    fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
    full_prompt = str(
        np.asarray(archive["text_prompt_full"]).reshape(-1)[0]
    )
    summary_prompt = str(
        np.asarray(archive["text_prompt_summary"]).reshape(-1)[0]
    )

frame_count = positions.shape[0]
duration_sec = frame_count/fps

print("positions:",positions.shape)
print("rotations:",rotations.shape)
print("contacts:",contacts.shape)
print("duration:",duration_sec)
print("summary prompt:",summary_prompt)
print("full prompt:",full_prompt)
```

Run the example directly from the terminal:

```bash
conda run --no-capture-output -n kimodo-macos python - <<'PY'
from pathlib import Path
import numpy as np

path = Path("outputs/side_steps_arms_open_5s.npz")
with np.load(path,allow_pickle=False) as archive:
    positions = np.asarray(archive["posed_joints"])
    rotations = np.asarray(archive["global_rot_mats"])
    contacts = np.asarray(archive["foot_contacts"])
    fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
    prompt = str(np.asarray(archive["text_prompt_full"]).reshape(-1)[0])

print("positions:",positions.shape)
print("rotations:",rotations.shape)
print("contacts:",contacts.shape)
print("fps:",fps)
print("full prompt:",prompt)
PY
```

## Use a motion from RIM v6

There are two supported workflows.

### Option A: load the original file directly

Assume these two repositories are siblings:

```text
workspace/
|-- kimodo_macos/
`-- rim_v6/
```

Run this Python code from the `rim_v6` repository root with its supported
environment:

```python
from pathlib import Path

from rim_v6.kinematics import rpy2r
from rim_v6.motion import Soma77Motion,Soma77Skeleton

motion_path = Path(
    "../kimodo_macos/outputs/side_steps_arms_open_5s.npz"
)
motion = Soma77Motion.from_kimodo_npz(
    motion_path,
    skeleton=Soma77Skeleton(),
    R_source_to_target=rpy2r([90.0,0.0,0.0],unit="deg"),
    fps_default=30.0,
)

print(motion.frame_count)
print(motion.fps)
print(motion.duration_sec)
print(motion.prompt_summary)
print(motion.prompt_full)
```

`R_source_to_target` explicitly converts the Kimodo source frame to the
MuJoCo Z-up frame used by the RIM v6 motion-retargeting notebooks.

### Option B: copy motions into the RIM v6 notebook data folder

From the `kimodo_macos` repository root:

```bash
./copy_to_rim_v6.sh ../rim_v6
```

The destination is:

```text
rim_v6/notebooks/15_motion_retargeting/motion_data/
```

The script adds a `kimodo_soma_rp_v1_` prefix. For example:

```text
outputs/my_side_steps_5s.npz
```

becomes:

```text
rim_v6/notebooks/15_motion_retargeting/motion_data/
kimodo_soma_rp_v1_my_side_steps_5s.npz
```

The copy is conservative:

- a missing destination is copied;
- an existing byte-identical destination is reported as current; and
- an existing destination with different contents stops the script without
  overwriting anything.

Add the copied motion to a notebook's `MOTION_SPECS`:

```python
MOTION_SPECS = {
    "my_side_steps_5s":{
        "display_name":"Side steps with arm raises",
        "path":"motion_data/kimodo_soma_rp_v1_my_side_steps_5s.npz",
        "fps":30.0,
    },
}
```

The strict RIM v6 loader requires prompt-rich schema-v1 metadata by default.
Every file created by this repository's `run_motion.sh` satisfies that
contract.

## Reproducibility notes

- MPS and CPU are not bitwise deterministic relative to each other, even with
  the same prompt and seed. Record the backend as part of an experiment.
- The exact full prompt is stored in the NPZ; the summary prompt is only a
  concise display label.
- The generator revision, backend, seed, sampling configuration, and runtime
  versions are stored with the motion.
- An existing output is never overwritten.
- Detailed measurements are available in [RESULTS.md](RESULTS.md).

## Troubleshooting

### `conda: command not found`

Install Miniforge and initialize zsh:

```bash
brew install --cask miniforge
/opt/homebrew/Caskroom/miniforge/base/bin/conda init zsh
exec zsh
```

### The Kimodo submodule is empty

```bash
git submodule update --init --recursive
```

`setup_macos.sh` also runs this command automatically.

### Hugging Face returns `401`, `403`, or a gated-model error

1. Open the Meta Llama model page and accept its terms.
2. Create a Hugging Face read token.
3. Log in again:

```bash
conda run --no-capture-output -n kimodo-macos hf auth login
```

### Offline mode reports a missing model

Run once with downloads enabled:

```bash
HF_HUB_OFFLINE=0 ./run_motion.sh mps
```

### The output already exists

Choose a different `output_name`. The repository intentionally never
overwrites a motion archive.

### MPS is unavailable

Check the environment directly:

```bash
conda run -n kimodo-macos python -c \
    'import torch; print(torch.__version__); print(torch.backends.mps.is_available())'
```

## Validated baseline

- Apple Silicon M4 Max, 16 CPU cores, 128 GB unified memory.
- macOS 26.5.
- Conda Python 3.10.
- PyTorch 2.13.0 with MPS available.
- Kimodo MPS fork commit
  `598fee96ca39bff9403db652d756d9046f089fc3`.
- Kimodo checkpoint `Kimodo-SOMA-RP-v1`.

For a five-second, 150-frame, 100-step motion, measured total runtime was
14.48 seconds on MPS and 33.41 seconds on CPU. See [RESULTS.md](RESULTS.md) for
the complete measurements and limitations.

## License and upstream projects

The `kimodo/` directory is a pinned Git submodule of an Apache-2.0 Kimodo MPS
fork. Model checkpoints, the text encoder, and datasets have separate license
terms. Review those terms before redistributing outputs or using them
commercially.

- [Official Kimodo repository](https://github.com/nv-tlabs/kimodo)
- [Pinned MPS fork commit](https://github.com/atticus-lv/kimodo/commit/598fee96ca39bff9403db652d756d9046f089fc3)
- [Official Kimodo installation guide](https://github.com/nv-tlabs/kimodo/blob/main/docs/source/getting_started/installation.md)
- [Official Kimodo NPZ format](https://github.com/nv-tlabs/kimodo/blob/main/docs/source/user_guide/output_formats.md)
