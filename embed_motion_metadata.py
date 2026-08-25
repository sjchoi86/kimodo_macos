"""Embed portable, pickle-free generation metadata in one Kimodo NPZ."""

from __future__ import annotations

import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np
import torch


METADATA_SCHEMA_VERSION = 1


def _text(value: str) -> np.ndarray:
    """Return one pickle-free Unicode scalar array."""
    return np.asarray(str(value),dtype=np.str_)


def embed_motion_metadata(arguments: argparse.Namespace) -> None:
    """Rewrite one Kimodo archive atomically with complete generation metadata."""
    path = arguments.path.resolve()
    with np.load(path,allow_pickle=False) as archive:
        arrays = {
            key:np.asarray(archive[key]).copy()
            for key in archive.files
        }
    object_keys = [
        key for key,value in arrays.items()
        if value.dtype.hasobject
    ]
    if object_keys:
        raise TypeError(
            "Kimodo archive contains object arrays: %s"%object_keys
        )

    positions = np.asarray(arrays["posed_joints"])
    frame_count = int(positions.shape[0])
    fps = float(arguments.fps)
    duration_sec = frame_count/fps
    prompt_segments = [
        text.strip() for text in arguments.prompt_full.split(".")
        if text.strip()
    ]
    created_utc = datetime.now(timezone.utc).isoformat().replace("+00:00","Z")

    arrays.update({
        "metadata_schema_version":np.asarray(
            METADATA_SCHEMA_VERSION,dtype=np.int64,
        ),
        "text_prompt_full":_text(arguments.prompt_full),
        "text_prompt_summary":_text(arguments.prompt_summary),
        "generation_model":_text(arguments.model),
        "generation_device":_text(arguments.device),
        "generation_output_name":_text(arguments.output_name),
        "generation_requested_duration_sec":np.asarray(
            arguments.requested_duration_sec,dtype=np.float64,
        ),
        "generation_duration_sec":np.asarray(duration_sec,dtype=np.float64),
        "generation_frame_count":np.asarray(frame_count,dtype=np.int64),
        "generation_prompt_segment_count":np.asarray(
            len(prompt_segments),dtype=np.int64,
        ),
        "fps":np.asarray(fps,dtype=np.float64),
        "generation_diffusion_steps":np.asarray(
            arguments.diffusion_steps,dtype=np.int64,
        ),
        "generation_num_samples":np.asarray(
            arguments.num_samples,dtype=np.int64,
        ),
        "generation_sample_index":np.asarray(
            arguments.sample_index,dtype=np.int64,
        ),
        "generation_num_transition_frames":np.asarray(
            arguments.num_transition_frames,dtype=np.int64,
        ),
        "generation_seed":np.asarray(arguments.seed,dtype=np.int64),
        "generation_postprocess":np.asarray(
            arguments.postprocess,dtype=np.bool_,
        ),
        "generation_cfg_type":_text(arguments.cfg_type),
        "generation_constraints":_text(arguments.constraints),
        "generation_hf_hub_offline":np.asarray(
            arguments.hf_hub_offline,dtype=np.bool_,
        ),
        "generation_mps_fallback":np.asarray(
            arguments.mps_fallback,dtype=np.bool_,
        ),
        "generation_created_utc":_text(created_utc),
        "generator_repository":_text(arguments.generator_repository),
        "generator_revision":_text(arguments.generator_revision),
        "python_version":_text(platform.python_version()),
        "torch_version":_text(torch.__version__),
        "platform_system":_text(platform.system()),
        "platform_release":_text(platform.release()),
        "platform_machine":_text(platform.machine()),
        "text_encoder_base":_text(arguments.text_encoder_base),
        "text_encoder_adapters":np.asarray(
            arguments.text_encoder_adapters,dtype=np.str_,
        ),
    })

    path.parent.mkdir(parents=True,exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
                prefix=path.stem+"_metadata_",
                suffix=".npz",
                dir=path.parent,
                delete=False,
            ) as temp_file:
            temp_path = Path(temp_file.name)
        np.savez_compressed(temp_path,**arrays)
        with np.load(temp_path,allow_pickle=False) as archive:
            if "text_prompt_full" not in archive.files:
                raise RuntimeError("Embedded prompt is missing after rewrite.")
            for key in archive.files:
                if np.asarray(archive[key]).dtype.hasobject:
                    raise TypeError(
                        "Embedded archive contains object array:[%s]"%key
                    )
        os.replace(temp_path,path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    print("Embedded metadata schema:[%d]"%METADATA_SCHEMA_VERSION)
    print("Summary prompt:[%s]"%arguments.prompt_summary)
    print("Full prompt:[%s]"%arguments.prompt_full)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path",type=Path)
    parser.add_argument("--prompt-full",required=True)
    parser.add_argument("--prompt-summary",required=True)
    parser.add_argument("--model",required=True)
    parser.add_argument("--device",required=True)
    parser.add_argument("--output-name",required=True)
    parser.add_argument("--requested-duration-sec",type=float,required=True)
    parser.add_argument("--fps",type=float,required=True)
    parser.add_argument("--diffusion-steps",type=int,required=True)
    parser.add_argument("--num-samples",type=int,required=True)
    parser.add_argument("--sample-index",type=int,required=True)
    parser.add_argument("--num-transition-frames",type=int,required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--postprocess",action="store_true")
    parser.add_argument("--cfg-type",required=True)
    parser.add_argument("--constraints",required=True)
    parser.add_argument("--hf-hub-offline",action="store_true")
    parser.add_argument("--mps-fallback",action="store_true")
    parser.add_argument("--generator-repository",required=True)
    parser.add_argument("--generator-revision",required=True)
    parser.add_argument("--text-encoder-base",required=True)
    parser.add_argument(
        "--text-encoder-adapter",
        dest="text_encoder_adapters",
        action="append",
        required=True,
    )
    embed_motion_metadata(parser.parse_args())
