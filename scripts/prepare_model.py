"""Resolve and cache an immutable Hugging Face model snapshot for reproducible runs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", required=True)
    parser.add_argument("--model", required=True, help="Official model alias, HF repo ID, or local model path.")
    parser.add_argument("--revision", default=None, help="Optional HF revision; resolved to an immutable commit SHA.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--alias-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--print-path", action="store_true")
    args = parser.parse_args()

    upstream = Path(args.upstream_dir).resolve()
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(upstream / "src"))
    from utils import resolve_model_path

    from huggingface_hub import HfApi, snapshot_download

    resolved, was_alias = resolve_model_path(args.model)
    local_candidate = Path(resolved).expanduser()
    model_repo_id = resolved
    if local_candidate.exists():
        snapshot_path = str(local_candidate.resolve())
        revision = args.revision or "local-files"
        source = "local_model_directory"
    else:
        api = HfApi()
        info = api.model_info(model_repo_id, revision=args.revision)
        revision = info.sha
        snapshot_path = snapshot_download(
            repo_id=model_repo_id,
            revision=revision,
            cache_dir=args.cache_dir,
        )
        source = "huggingface_hub_snapshot"

    alias_name = Path(args.model.rstrip("/")).name.replace(" ", "_")
    alias_root = Path(args.alias_dir).resolve()
    alias_root.mkdir(parents=True, exist_ok=True)
    alias_path = alias_root / alias_name
    if alias_path.is_symlink() or alias_path.is_file():
        alias_path.unlink()
    elif alias_path.exists():
        raise FileExistsError(
            f"Refusing to replace non-symlink model alias directory: {alias_path}"
        )
    os.symlink(snapshot_path, alias_path, target_is_directory=True)

    manifest = {
        "model_argument": args.model,
        "model_repo_id_or_local_path": model_repo_id,
        "resolved_revision": revision,
        "snapshot_path": snapshot_path,
        "stable_local_alias": str(alias_path),
        "source": source,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Remote Hugging Face models are downloaded at an immutable commit SHA. "
            "The stable local alias is passed to the pinned upstream runner."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.print_path:
        print(alias_path)
    else:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
