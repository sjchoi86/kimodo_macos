"""Validate the numeric SOMA77 arrays produced by the local Kimodo CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def validate_motion(path: Path) -> None:
    """Load one archive without pickle and validate the RIM SOMA77 contract."""
    required_keys = {
        "foot_contacts",
        "fps",
        "generation_cfg_type",
        "generation_constraints",
        "generation_created_utc",
        "generation_device",
        "generation_diffusion_steps",
        "generation_duration_sec",
        "generation_frame_count",
        "generation_hf_hub_offline",
        "generation_model",
        "generation_mps_fallback",
        "generation_num_samples",
        "generation_num_transition_frames",
        "generation_output_name",
        "generation_postprocess",
        "generation_prompt_segment_count",
        "generation_requested_duration_sec",
        "generation_sample_index",
        "generation_seed",
        "generator_repository",
        "generator_revision",
        "global_rot_mats",
        "metadata_schema_version",
        "platform_machine",
        "platform_release",
        "platform_system",
        "posed_joints",
        "python_version",
        "text_encoder_adapters",
        "text_encoder_base",
        "text_prompt_full",
        "text_prompt_summary",
        "torch_version",
    }

    with np.load(path, allow_pickle=False) as archive:
        keys = tuple(sorted(archive.files))
        missing_keys = sorted(required_keys - set(keys))
        if missing_keys:
            raise ValueError(f"Missing Kimodo arrays in {path}: {missing_keys}")

        positions = np.asarray(archive["posed_joints"])
        rotations = np.asarray(archive["global_rot_mats"])
        contacts = np.asarray(archive["foot_contacts"])
        object_keys = [
            key for key in keys
            if np.asarray(archive[key]).dtype.hasobject
        ]
        if object_keys:
            raise TypeError(
                f"Kimodo archive contains object arrays: {object_keys}"
            )
        schema_version = int(np.asarray(
            archive["metadata_schema_version"],dtype=np.int64,
        ).reshape(-1)[0])
        prompt_full = str(np.asarray(
            archive["text_prompt_full"],dtype=np.str_,
        ).reshape(-1)[0]).strip()
        prompt_summary = str(np.asarray(
            archive["text_prompt_summary"],dtype=np.str_,
        ).reshape(-1)[0]).strip()
        fps = float(np.asarray(
            archive["fps"],dtype=np.float64,
        ).reshape(-1)[0])
        duration_sec = float(np.asarray(
            archive["generation_duration_sec"],dtype=np.float64,
        ).reshape(-1)[0])

    if positions.ndim != 3 or positions.shape[1:] != (77, 3):
        raise ValueError(
            f"posed_joints must have shape (T,77,3): {positions.shape}"
        )
    frame_count = positions.shape[0]
    if frame_count < 1:
        raise ValueError("Kimodo motion must contain at least one frame.")
    if rotations.shape != (frame_count, 77, 3, 3):
        raise ValueError(
            "global_rot_mats must have shape (T,77,3,3): "
            f"{rotations.shape}"
        )
    if contacts.shape != (frame_count, 6):
        raise ValueError(
            f"foot_contacts must have shape (T,6): {contacts.shape}"
        )
    if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
        raise ValueError("Kimodo motion contains NaN or infinity.")
    if schema_version != 1:
        raise ValueError(
            f"Unsupported metadata schema version: {schema_version}"
        )
    if not prompt_full or not prompt_summary:
        raise ValueError("Kimodo prompts must be non-empty.")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Kimodo fps must be positive: {fps}")
    if not np.isclose(duration_sec,frame_count/fps,atol=1e-9):
        raise ValueError(
            "generation_duration_sec does not match frames/fps: "
            f"{duration_sec} != {frame_count/fps}"
        )

    print(f"path: {path}")
    print(f"keys: {keys}")
    print(f"frames: {frame_count}")
    print(f"posed_joints: {positions.shape} {positions.dtype}")
    print(f"global_rot_mats: {rotations.shape} {rotations.dtype}")
    print(f"foot_contacts: {contacts.shape} {contacts.dtype}")
    print(f"fps: {fps}")
    print(f"summary prompt: {prompt_summary}")
    print(f"full prompt: {prompt_full}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    arguments = parser.parse_args()
    validate_motion(arguments.path)
