#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RGB-D backprojection and SimFoundry-style mesh scale/pose registration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from probreg import cpd, transformation
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh


MIN_POINTS = 10


def load_matrix(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.load(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("K", "matrix", "transform", "T_cam_world", "camera_to_world"):
                if key in payload:
                    payload = payload[key]
                    break
        value = np.asarray(payload, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    if value.shape != expected_shape:
        raise ValueError(f"{path} has shape {value.shape}; expected {expected_shape}")
    return value


def load_depth(path: Path, scale: float) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(path)
    else:
        depth = np.asarray(Image.open(path))
    depth = np.asarray(depth, dtype=np.float32) * scale
    if depth.ndim != 2:
        raise ValueError(f"Depth must be HxW, got {depth.shape}")
    return depth


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(Image.open(path).convert("L"))
    if mask.shape != shape:
        mask = np.asarray(
            Image.fromarray(mask).resize(
                (shape[1], shape[0]),
                resample=Image.Resampling.NEAREST,
            )
        )
    return mask > 127


def backproject(
    depth: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
    *,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    v, u = np.indices((h, w), dtype=np.float64)
    valid = mask & np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    z = depth[valid].astype(np.float64)
    x = (u[valid] - K[0, 2]) * z / K[0, 0]
    y = (v[valid] - K[1, 2]) * z / K[1, 1]
    points = np.stack((x, y, z), axis=1)
    pixels = np.stack(np.nonzero(valid), axis=1)
    return points, pixels


def make_cloud(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def denoise(cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    if len(cloud.points) < 50:
        return cloud
    filtered, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return filtered


def obb_diag(cloud: o3d.geometry.PointCloud) -> float:
    extent = np.asarray(cloud.get_oriented_bounding_box().extent, dtype=np.float64)
    diag = float(np.linalg.norm(extent))
    if not np.isfinite(diag) or diag <= 0:
        raise ValueError("Degenerate point-cloud OBB")
    return diag


def symmetric_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)
    d_ba = tree_a.query(b, k=1, workers=-1)[0]
    d_ab = tree_b.query(a, k=1, workers=-1)[0]
    return float(np.mean(d_ba) + np.mean(d_ab))


def apply_similarity(points: np.ndarray, tf: transformation.RigidTransformation) -> np.ndarray:
    scale = np.asarray(tf.scale, dtype=np.float64)
    return scale * (points @ np.asarray(tf.rot, dtype=np.float64).T) + np.asarray(tf.t, dtype=np.float64)


def fit_mesh_to_partial_cloud(
    source_cloud: o3d.geometry.PointCloud,
    raw_mesh: trimesh.Trimesh,
    *,
    samples: int,
    mesh_points: int,
    seed: int,
    refine_scale: bool,
) -> dict[str, object]:
    target_points_raw, _ = trimesh.sample.sample_surface(raw_mesh, mesh_points)
    target_cloud_raw = make_cloud(target_points_raw)

    source_diag = obb_diag(source_cloud)
    target_diag = obb_diag(target_cloud_raw)
    pre_scale = source_diag / target_diag
    target_points = target_points_raw * pre_scale

    voxel = source_diag / 30.0
    source_ds = source_cloud.voxel_down_sample(voxel)
    target_ds = make_cloud(target_points).voxel_down_sample(voxel)
    source_np = np.asarray(source_ds.points)
    target_np = np.asarray(target_ds.points)
    if len(source_np) < MIN_POINTS or len(target_np) < MIN_POINTS:
        raise ValueError("Too few points after voxel downsampling")

    rng = np.random.default_rng(seed)
    rotations = Rotation.random(samples, random_state=rng).as_matrix()
    best: dict[str, object] | None = None

    scale_modes = (False, True) if refine_scale else (False,)
    for update_scale in scale_modes:
        for init_rot in rotations:
            init_tf = transformation.RigidTransformation(rot=init_rot)
            rotated_source = make_cloud(init_tf.transform(source_np))
            mesh_target = make_cloud(target_np)

            fit_tf, _, _ = cpd.registration_cpd(
                rotated_source,
                mesh_target,
                tf_type_name="rigid",
                use_color=False,
                use_cuda=False,
                update_scale=update_scale,
            )
            source_to_mesh = fit_tf * init_tf
            mesh_to_source = source_to_mesh.inverse()

            residual_scale = float(np.asarray(mesh_to_source.scale).reshape(-1)[0])
            if not np.isfinite(residual_scale) or residual_scale <= 0:
                continue
            if update_scale and not (0.1 <= residual_scale <= 10.0):
                continue

            aligned = apply_similarity(target_np, mesh_to_source)
            score = symmetric_chamfer(source_np, aligned)
            candidate = {
                "score": score,
                "pre_scale": pre_scale,
                "residual_scale": residual_scale,
                "rotation": np.asarray(mesh_to_source.rot, dtype=np.float64),
                "translation": np.asarray(mesh_to_source.t, dtype=np.float64),
                "scale_refined": update_scale,
            }
            if best is None or score < float(best["score"]):
                best = candidate

    if best is None:
        raise RuntimeError("CPD registration produced no valid fit")
    return best


def transform_mesh(
    raw_mesh: trimesh.Trimesh,
    *,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> trimesh.Trimesh:
    mesh = raw_mesh.copy()
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation * scale
    matrix[:3, 3] = translation
    mesh.apply_transform(matrix)
    return mesh


def pose_record(rotation: np.ndarray, translation: np.ndarray, scale: float) -> dict[str, object]:
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    return {
        "translation": translation.tolist(),
        "rotation_matrix": rotation.tolist(),
        "quaternion_xyzw": quat_xyzw.tolist(),
        "scale": float(scale),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backproject RGB-D into a metric partial point cloud and register a generated "
            "mesh using the OBB-scale + multi-start CPD strategy used by SimFoundry Stage 8."
        )
    )
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--mesh", type=Path, default=None, help="Optional generated GLB/OBJ/PLY mesh.")
    parser.add_argument("--camera-to-world", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("/output"))
    parser.add_argument("--depth-scale", type=float, default=1.0, help="Multiply stored depth by this to get metres.")
    parser.add_argument("--min-depth", type=float, default=0.02)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--mesh-points", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-scale-refine", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples <= 0 or args.mesh_points < MIN_POINTS:
        raise ValueError("--samples must be >0 and --mesh-points must be >=10")

    rgb = load_rgb(args.rgb)
    depth = load_depth(args.depth, args.depth_scale)
    if rgb.shape[:2] != depth.shape:
        raise ValueError(f"RGB shape {rgb.shape[:2]} does not match depth {depth.shape}")

    K = load_matrix(args.intrinsics, (3, 3))
    mask = load_mask(args.mask, depth.shape)
    points, pixels = backproject(
        depth,
        K,
        mask,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    if len(points) < MIN_POINTS:
        raise ValueError(f"Only {len(points)} valid RGB-D points; need at least {MIN_POINTS}")

    colors = rgb[pixels[:, 0], pixels[:, 1]].astype(np.float64) / 255.0
    cloud = denoise(make_cloud(points, colors))
    if len(cloud.points) < MIN_POINTS:
        raise ValueError("Too few points remain after denoising")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    camera_cloud_path = out_dir / "object_point_cloud_camera.ply"
    o3d.io.write_point_cloud(str(camera_cloud_path), cloud)

    result: dict[str, object] = {
        "units": "metres",
        "camera_convention": "OpenCV: +x right, +y down, +z forward",
        "point_cloud_camera": str(camera_cloud_path),
        "num_points": len(cloud.points),
    }

    T_cam_world = None
    if args.camera_to_world is not None:
        T_cam_world = load_matrix(args.camera_to_world, (4, 4))
        world_points = np.asarray(cloud.points) @ T_cam_world[:3, :3].T + T_cam_world[:3, 3]
        world_cloud = make_cloud(world_points, np.asarray(cloud.colors))
        world_cloud_path = out_dir / "object_point_cloud_world.ply"
        o3d.io.write_point_cloud(str(world_cloud_path), world_cloud)
        result["point_cloud_world"] = str(world_cloud_path)

    if args.mesh is not None:
        loaded = trimesh.load(args.mesh, force="mesh")
        if not isinstance(loaded, trimesh.Trimesh) or len(loaded.vertices) < 4:
            raise ValueError(f"Could not load a triangle mesh from {args.mesh}")

        fit = fit_mesh_to_partial_cloud(
            cloud,
            loaded,
            samples=args.samples,
            mesh_points=args.mesh_points,
            seed=args.seed,
            refine_scale=not args.no_scale_refine,
        )
        pre_scale = float(fit["pre_scale"])
        residual_scale = float(fit["residual_scale"])
        total_scale = pre_scale * residual_scale
        R_cam = np.asarray(fit["rotation"], dtype=np.float64)
        t_cam = np.asarray(fit["translation"], dtype=np.float64)

        aligned_camera = transform_mesh(
            loaded,
            scale=total_scale,
            rotation=R_cam,
            translation=t_cam,
        )
        aligned_camera_path = out_dir / "aligned_mesh_camera.glb"
        aligned_camera.export(aligned_camera_path)

        registration = {
            "score_symmetric_chamfer": float(fit["score"]),
            "pre_scale_factor": pre_scale,
            "residual_scale": residual_scale,
            "scale_refined": bool(fit["scale_refined"]),
            "camera_pose": pose_record(R_cam, t_cam, total_scale),
            "aligned_mesh_camera": str(aligned_camera_path),
        }

        if T_cam_world is not None:
            R_world = T_cam_world[:3, :3] @ R_cam
            t_world = T_cam_world[:3, :3] @ t_cam + T_cam_world[:3, 3]
            aligned_world = transform_mesh(
                loaded,
                scale=total_scale,
                rotation=R_world,
                translation=t_world,
            )
            aligned_world_path = out_dir / "aligned_mesh_world.glb"
            aligned_world.export(aligned_world_path)
            registration["world_pose"] = pose_record(R_world, t_world, total_scale)
            registration["aligned_mesh_world"] = str(aligned_world_path)

        result["registration"] = registration

    result_path = out_dir / "registration.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
