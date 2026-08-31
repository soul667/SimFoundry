#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone SimFoundry-style Gaussian-splat training wrapper."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def _bool_text(value: bool) -> str:
    return "True" if value else "False"


def _latest_config(outputs_dir: Path) -> Path:
    configs = sorted(
        outputs_dir.glob("**/config.yml"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not configs:
        raise FileNotFoundError(f"No nerfstudio config.yml found under {outputs_dir}")
    return configs[0]


def _dataset_has_seed_ply(data_dir: Path) -> bool:
    transforms_path = data_dir / "transforms.json"
    try:
        payload = json.loads(transforms_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    rel = payload.get("ply_file_path")
    return bool(rel and (data_dir / rel).exists())


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str], dry_run: bool) -> None:
    print("+", shlex.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and export a Nerfstudio Gaussian splat with SimFoundry's "
            "metric-pose and optional depth-supervision settings."
        )
    )
    parser.add_argument("--data", type=Path, required=True, help="Nerfstudio dataset directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/output"),
        help="Training root. Nerfstudio writes outputs/ and export/ below it.",
    )
    parser.add_argument("--method", default="splatfacto-big", choices=("splatfacto", "splatfacto-big"))
    parser.add_argument("--iterations", type=int, default=80_000)
    parser.add_argument(
        "--camera-optimizer-mode",
        default="SO3xR3",
        choices=("off", "SO3xR3", "SE3"),
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=None,
        help="Optional frame_*.npy depth directory. Enables the SimFoundry depth-loss patch.",
    )
    parser.add_argument("--depth-loss-mult", type=float, default=0.5)
    parser.add_argument("--depth-loss-min", type=float, default=0.1)
    parser.add_argument(
        "--load-3d-points",
        choices=("auto", "true", "false"),
        default="auto",
        help="Initialize Gaussians from transforms.json ply_file_path.",
    )
    parser.add_argument("--viewer-port", type=int, default=7007)
    parser.add_argument("--keep-viewer", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--export-name", default="splat.ply")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data.resolve()
    output_dir = args.output_dir.resolve()

    if not (data_dir / "transforms.json").is_file():
        raise FileNotFoundError(f"Expected {data_dir / 'transforms.json'}")
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")

    if args.load_3d_points == "auto":
        load_3d_points = _dataset_has_seed_ply(data_dir)
    else:
        load_3d_points = args.load_3d_points == "true"

    env = os.environ.copy()
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["TORCH_COMPILE_DISABLE"] = "1"
    env.setdefault("TORCH_CUDA_ARCH_LIST", "7.0;7.5;8.0;8.6;8.9;9.0;12.0")

    if args.depth_dir is not None:
        depth_dir = args.depth_dir.resolve()
        depths = sorted(depth_dir.glob("frame_*.npy"))
        if not depths:
            raise FileNotFoundError(f"No frame_*.npy depth maps under {depth_dir}")
        env["NERFSTUDIO_DEPTH_LOSS"] = "1"
        env["NERFSTUDIO_DEPTH_LOSS_MULT"] = str(args.depth_loss_mult)
        env["NERFSTUDIO_DEPTH_LOSS_MIN"] = str(args.depth_loss_min)
        env["NERFSTUDIO_DEPTH_DIR"] = str(depth_dir)
    else:
        for key in (
            "NERFSTUDIO_DEPTH_LOSS",
            "NERFSTUDIO_DEPTH_LOSS_MULT",
            "NERFSTUDIO_DEPTH_LOSS_MIN",
            "NERFSTUDIO_DEPTH_DIR",
        ):
            env.pop(key, None)

    output_dir.mkdir(parents=True, exist_ok=True)
    quit_on_done = not args.keep_viewer
    train_cmd = [
        "ns-train",
        args.method,
        "--max-num-iterations",
        str(args.iterations),
        "--data",
        str(data_dir),
        "--vis",
        "viewer",
        "--viewer.websocket-port",
        str(args.viewer_port),
        "--viewer.quit-on-train-completion",
        _bool_text(quit_on_done),
        f"--pipeline.model.camera-optimizer.mode={args.camera_optimizer_mode}",
        "nerfstudio-data",
        "--auto-scale-poses",
        "False",
        "--center-method",
        "none",
        "--orientation-method",
        "none",
        "--scale-factor",
        "1.0",
        "--load-3D-points",
        _bool_text(load_3d_points),
    ]
    _run(train_cmd, cwd=output_dir, env=env, dry_run=args.dry_run)

    manifest: dict[str, object] = {
        "data": str(data_dir),
        "method": args.method,
        "iterations": args.iterations,
        "camera_optimizer_mode": args.camera_optimizer_mode,
        "depth_loss": args.depth_dir is not None,
        "load_3d_points": load_3d_points,
        "train_command": train_cmd,
    }

    if not args.no_export and not args.dry_run:
        load_config = _latest_config(output_dir / "outputs")
        export_dir = output_dir / "export"
        export_dir.mkdir(parents=True, exist_ok=True)

        # torch >=2.6 changed torch.load(weights_only=True) by default. Nerfstudio
        # 1.1.5 predates that change, so use the same compatibility shim as SimFoundry.
        shim = (
            "import torch,sys;"
            "_orig=torch.load;"
            "torch.load=lambda *a,**kw:_orig(*a,**{**{'weights_only':False},**kw});"
            f"sys.argv=['ns-export','gaussian-splat','--load-config',{str(load_config)!r},"
            f"'--output-dir',{str(export_dir)!r}];"
            "from nerfstudio.scripts.exporter import entrypoint;entrypoint()"
        )
        export_cmd = [sys.executable, "-c", shim]
        _run(export_cmd, cwd=output_dir, env=env, dry_run=False)

        default_ply = export_dir / "splat.ply"
        requested_ply = export_dir / args.export_name
        if not default_ply.exists():
            raise FileNotFoundError(f"Nerfstudio export did not create {default_ply}")
        if requested_ply != default_ply:
            if requested_ply.exists():
                requested_ply.unlink()
            default_ply.rename(requested_ply)
        manifest["config"] = str(load_config)
        manifest["export"] = str(requested_ply)

    (output_dir / "gaussian_train_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
