#!/usr/bin/env python3
"""Build-time end-to-end smoke test for direct SimFoundry-scene registration."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
import trimesh


def make_sphere_depth(h: int, w: int, K: np.ndarray, center: np.ndarray, radius: float):
    v, u = np.indices((h, w), dtype=np.float64)
    dx = (u - K[0, 2]) / K[0, 0]
    dy = (v - K[1, 2]) / K[1, 1]
    dirs = np.stack((dx, dy, np.ones_like(dx)), axis=-1)

    a = np.sum(dirs * dirs, axis=-1)
    b = -2.0 * np.sum(dirs * center.reshape(1, 1, 3), axis=-1)
    c = float(np.dot(center, center) - radius * radius)
    disc = b * b - 4.0 * a * c
    valid = disc > 0

    depth = np.zeros((h, w), dtype=np.float32)
    root = np.zeros_like(disc)
    root[valid] = np.sqrt(disc[valid])
    t = np.zeros_like(disc)
    t[valid] = (-b[valid] - root[valid]) / (2.0 * a[valid])
    valid &= t > 0
    depth[valid] = t[valid].astype(np.float32)  # ray z component is 1, so z-depth == t
    return depth, valid


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="simfoundry-reg-smoke-") as tmp:
        root = Path(tmp)
        scene = root / "Data" / "synthetic_scene"
        out = root / "out"
        (scene / "s2_da" / "da" / "exports" / "npz").mkdir(parents=True)
        (scene / "s3_ground").mkdir(parents=True)
        (scene / "s4_frame").mkdir(parents=True)
        (scene / "s5_scene" / "removal_mask").mkdir(parents=True)
        (scene / "s7_mesh" / "textured_mesh" / "hunyuan").mkdir(parents=True)

        h, w = 64, 80
        K = np.array([[85.0, 0.0, (w - 1) / 2], [0.0, 85.0, (h - 1) / 2], [0.0, 0.0, 1.0]])
        depth, mask = make_sphere_depth(h, w, K, np.array([0.0, 0.0, 1.0]), 0.20)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[..., 0] = 90
        rgb[..., 1] = 140
        rgb[..., 2] = 210

        np.savez(
            scene / "s2_da" / "da" / "exports" / "npz" / "results.npz",
            image=rgb[None],
            depth=depth[None],
            intrinsics=K[None],
        )
        (scene / "s3_ground" / "frame_selection.json").write_text(
            json.dumps({"selected_idx": 0}), encoding="utf-8"
        )
        np.save(scene / "s4_frame" / "image_0_cam2world.npy", np.eye(4, dtype=np.float64))

        Image.fromarray(rgb).save(scene / "s5_scene" / "source_padded_resized.png")
        Image.fromarray((mask.astype(np.uint8) * 255)).save(
            scene / "s5_scene" / "removal_mask" / "iter_0.png"
        )

        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        mesh.export(scene / "s7_mesh" / "textured_mesh" / "hunyuan" / "iter_0_mesh.glb")

        cmd = [
            sys.executable,
            "/opt/simfoundry/register_rgbd.py",
            "--simfoundry-scene-dir", str(scene),
            "--object-index", "0",
            "--mesh-backend", "hunyuan",
            "--samples", "2",
            "--mesh-points", "500",
            "--output-dir", str(out),
        ]
        subprocess.run(cmd, check=True)

        result_path = out / "registration.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        assert payload["input"]["mode"] == "simfoundry-scene"
        assert payload["input"]["frame_index"] == 0
        assert payload["input"]["object_index"] == 0
        assert payload["num_points"] > 10
        assert "registration" in payload
        assert "world_pose" in payload["registration"]
        assert (out / "object_point_cloud_camera.ply").is_file()
        assert (out / "object_point_cloud_world.ply").is_file()
        assert (out / "aligned_mesh_camera.glb").is_file()
        assert (out / "aligned_mesh_world.glb").is_file()
        print("Direct SimFoundry scene smoke test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
