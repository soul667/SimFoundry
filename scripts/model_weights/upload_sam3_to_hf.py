#!/usr/bin/env python3
"""Upload an externally obtained SAM3 checkpoint to a private HF model repo.

This helper runs OUTSIDE the SAM3 Docker image. It does not download the
checkpoint from Meta and it never stores credentials in this repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="HF model repo, e.g. user/simfoundry-sam3-weights")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local official-compatible sam3.pt")
    parser.add_argument(
        "--license",
        dest="license_path",
        type=Path,
        required=True,
        help="Local copy of Meta's SAM LICENSE to redistribute with the weights",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create a public repo. Default is private; use public only if you intentionally want redistribution.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.license_path.is_file():
        raise FileNotFoundError(args.license_path)

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=not args.public,
        exist_ok=True,
    )
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        path_or_fileobj=str(args.checkpoint),
        path_in_repo="sam3.pt",
        commit_message="Add SAM3 checkpoint",
    )
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        path_or_fileobj=str(args.license_path),
        path_in_repo="LICENSE",
        commit_message="Add SAM license",
    )
    print(f"Uploaded sam3.pt and LICENSE to {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
