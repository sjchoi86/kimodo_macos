# Kimodo macOS agent instructions

- Use the Conda environment named `kimodo-macos` for every supported command.
- Treat `environment.yml` and `setup_macos.sh` as the environment source of
  truth. Do not introduce uv, venv, Poetry, or another environment manager.
- Keep model checkpoints and Hugging Face caches out of Git.
- New motion archives belong in `outputs/`; only deliberately selected example
  archives may be unignored and committed.
- Every published motion archive must pass `validate_motion.py` and retain its
  full prompt, summary prompt, generation settings, and runtime provenance in
  the same pickle-free NPZ file.
- Keep the Kimodo source pinned through the `kimodo/` Git submodule. Do not
  rewrite upstream history in this wrapper repository.
- For published README media, treat the source display aspect ratio reported by
  `ffprobe` as authoritative. Do not infer it from the coded width and height;
  encode GIF and MP4 derivatives with square pixels while preserving that DAR.
- Use a new asset path when replacing published media so GitHub cannot serve a
  stale branch-image cache. After pushing, verify both the natural dimensions
  and rendered dimensions on the public README page.
