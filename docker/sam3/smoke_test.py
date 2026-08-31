#!/usr/bin/env python3
"""Weight-free smoke test for the standalone SAM3 image.

The real checkpoint is intentionally not available during image build. This test
checks only the CLI/runtime contract and verifies that missing mounted weights
fail locally instead of triggering a Hugging Face download.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


SCRIPT = Path("/opt/simfoundry/segment_sam3.py")


def main() -> int:
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    subprocess.run([sys.executable, str(SCRIPT), "--help"], check=True, stdout=subprocess.DEVNULL)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        image_path = tmp_path / "frame.png"
        Image.new("RGB", (32, 32), (0, 0, 0)).save(image_path)
        missing = tmp_path / "missing-sam3.pt"
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--image",
                str(image_path),
                "--prompt",
                "cup",
                "--checkpoint",
                str(missing),
                "--device",
                "cpu",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.returncode != 0
        combined = proc.stdout + proc.stderr
        assert "checkpoint not found" in combined.lower()
        assert "huggingface.co" not in combined.lower()

    print("SAM3 external-weight contract smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
