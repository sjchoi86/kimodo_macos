# Kimodo macOS validation results

Validation was performed on August 24, 2026, using an Apple Silicon M4 Max
with 16 CPU cores and 128 GB of unified memory on macOS 26.5. Each measurement
used an independent CLI process, so total runtime includes loading the Llama 3
text encoder and the Kimodo checkpoint.

## MPS and CPU measurements

| Backend | Duration | Diffusion steps | Frames | Total time | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| MPS | 1.0 s | 5 | 30 | 15.53 s | pass |
| CPU | 1.0 s | 5 | 30 | 12.74 s | pass |
| MPS | 3.0 s | 100 | 90 | 19.08 s | pass |
| MPS | 5.0 s | 100 | 150 | 14.48 s | pass |
| CPU | 5.0 s | 100 | 150 | 33.41 s | pass |

For the same five-second prompt and seed, MPS completed the full process about
2.3 times faster than CPU. Considering only the 100-step diffusion stage, MPS
took approximately 5.54 seconds and CPU took approximately 23.47 seconds, so
MPS was about 4.2 times faster. The short five-step smoke test is not a useful
backend benchmark because model-loading time dominates it.

Every measured result opened successfully with `allow_pickle=False` and
contained finite numeric arrays with these required shapes:

```text
posed_joints:    (T,77,3)
global_rot_mats: (T,77,3,3)
foot_contacts:   (T,6)
```

The three retained example files also passed the installed RIM v6
`Soma77Motion.from_kimodo_npz(...)` loader.

## Reproducibility limitations

MPS and CPU are not bitwise deterministic relative to each other, even when
the prompt and seed are identical. In the one-second smoke test, the maximum
absolute joint-position difference was approximately `0.7001`, and the contact
masks also differed. Generation records should therefore include both the
backend and seed.

These tests used `--no-postprocess`. The core Kimodo text encoder and diffusion
generation work on MPS and CPU, but the optional native motion-correction build
and visual motion quality require separate evaluation.

The validated Transformers and PEFT combination prints an
`Already found a peft_config` warning while loading the second LLM2Vec adapter.
The process still completes and produces a valid NPZ. This validation covers
runtime compatibility and the RIM input contract; prompt alignment and adapter
quality require rendered motion comparisons.

## Prompt-rich source set

Earlier performance-test files and initial motion candidates were removed
because their prompts were not embedded in the archives. The retained source
set contains only three self-contained metadata schema-v1 files. Each file was
generated with MPS, a five-second duration, 150 frames, 30 FPS, 100 diffusion
steps, seed 7, and postprocessing disabled.

| Output | Summary prompt | Total time | SHA-256 |
| --- | --- | ---: | --- |
| `side_steps_arms_open_5s.npz` | Side steps with arms opening | 17.61 s | `ef203d00eb7f1468272dbc57d40bb66ef1961d7f97045aeaf1b2415cb62f91f0` |
| `march_arm_swing_5s.npz` | March in place with arm swings | 14.64 s | `b015c13f11d5017148cb2d73548535fc99534fb12469b7586e770bc04dbd866b` |
| `forward_back_arm_reach_5s.npz` | Forward and backward steps with arm reaches | 13.92 s | `3b4f5c273b56f3f256e1984cfbb618268b58972fdc9034666239e7a94f3e2e7f` |

All three archives passed these checks:

- complete required metadata;
- no object-dtype arrays;
- valid SOMA77 array shapes;
- finite positions and rotations;
- consistent frame count, FPS, and duration; and
- non-empty exact full and summary prompts.

Each archive stores the generator revision, model, seed, sampling
configuration, text encoder, device flags, UTC timestamp, Python version,
PyTorch version, and platform information. All fields remain readable from the
same file with `allow_pickle=False`.

## RIM v6 validation

All three files were copied unchanged into the RIM v6 motion data directory and
validated again with the strict `Soma77Motion.from_kimodo_npz(...)` loader.
`side_steps_arms_open_5s` also completed the ordered K1 notebook pipelines
`03` through `05`.

The remaining desktop-only checks are visual rather than numeric: Qt prompt
layout, keyboard input, camera behavior, and skin synchronization should be
confirmed with `Restart Kernel and Run All` in the desktop notebook runtime.
