#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RGB-D backprojection and SimFoundry-style mesh scale/pose registration."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from probreg import cpd, transformation
from scipy.ndimage import binary_erosion
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


def load_mask(path: Path | None, shape: tuple[int, int], *, erode: bool = True) -> np.ndarray:
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
    mask_bool = mask > 127
    if erode:
        mask_bool = binary_erosion(mask_bool, structure=np.ones((3, 3), dtype=bool))
    return mask_bool


def map_simfoundry_mask(
    mask_path: Path,
    source_canvas_path: Path,
    target_shape: tuple[int, int],
    *,
    erode: bool = True,
) -> np.ndarray:
    """Map Stage-5's padded mask back into the canonical DA3 image geometry.

    Stage 8 pads the DA3 RGB to Stage 5's source canvas aspect ratio, resizes the
    Stage-5 mask into that padded canvas, then removes the padding. Reproducing that
    mapping avoids aspect-ratio distortion when the capture is, e.g., 16:9 but Stage 5
    used a 4:3 decomposition canvas.
    """
    mask = np.asarray(Image.open(mask_path).convert("L"))
    canvas = Image.open(source_canvas_path)
    canvas_w, canvas_h = canvas.size
    target_ratio = canvas_w / canvas_h
    h, w = target_shape
    current_ratio = w / h

    if np.isclose(current_ratio, target_ratio, rtol=0, atol=1e-6):
        resized = np.asarray(Image.fromarray(mask).resize((w, h), Image.Resampling.NEAREST))
    elif current_ratio > target_ratio:
        padded_w = w
        padded_h = max(h, int(round(w / target_ratio)))
        mapped = np.asarray(
            Image.fromarray(mask).resize((padded_w, padded_h), Image.Resampling.NEAREST)
        )
        y0 = max(0, (padded_h - h) // 2)
        resized = mapped[y0:y0 + h, :w]
    else:
        padded_h = h
        padded_w = max(w, int(round(h * target_ratio)))
        mapped = np.asarray(
            Image.fromarray(mask).resize((padded_w, padded_h), Image.Resampling.NEAREST)
        )
        x0 = max(0, (padded_w - w) // 2)
        resized = mapped[:h, x0:x0 + w]

    if resized.shape != (h, w):
        resized = np.asarray(
            Image.fromarray(resized).resize((w, h), Image.Resampling.NEAREST)
        )
    mask_bool = resized > 127
    if erode:
        mask_bool = binary_erosion(mask_bool, structure=np.ones((3, 3), dtype=bool))
    return mask_bool


def _selection_index_from_json(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    preferred = {"selected_idx", "selected_index", "img_idx", "image_idx", "frame_idx"}

    def walk(value):
        if isinstance(value, dict):
            for key in preferred:
                if key in value and isinstance(value[key], (int, np.integer)):
                    return int(value[key])
            for nested in value.values():
                found = walk(nested)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for nested in value:
                found = walk(nested)
                if found is not None:
                    return found
        return None

    return walk(payload)


def resolve_simfoundry_frame_index(scene_dir: Path, override: int | None) -> int:
    if override is not None:
        return override

    selected = _selection_index_from_json(scene_dir / "s3_ground" / "frame_selection.json")
    if selected is not None:
        return selected

    candidates = sorted((scene_dir / "s4_frame").glob("image_*_cam2world.npy"))
    parsed: list[int] = []
    for path in candidates:
        match = re.match(r"image_(\d+)_cam2world\.npy$", path.name)
        if match:
            parsed.append(int(match.group(1)))
    if len(parsed) == 1:
        return parsed[0]
    if not parsed:
        raise FileNotFoundError(
            f"Could not resolve canonical frame: no frame_selection.json index and no "
            f"image_*_cam2world.npy under {scene_dir / 's4_frame'}"
        )
    raise ValueError(
        f"Multiple canonical-frame artifacts found {parsed}; pass --frame-index explicitly."
    )


def resolve_simfoundry_mesh(scene_dir: Path, object_index: int, backend: str | None) -> Path:
    mesh_root = scene_dir / "s7_mesh" / "textured_mesh"
    if backend is not None:
        candidate = mesh_root / backend / f"iter_{object_index}_mesh.glb"
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    candidates = sorted(mesh_root.glob(f"*/iter_{object_index}_mesh.glb"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No generated mesh found for object {object_index} under {mesh_root}"
        )
    raise ValueError(
        "Multiple mesh backends found for object "
        f"{object_index}: {[p.parent.name for p in candidates]}; pass --mesh-backend."
    )


def load_simfoundry_scene_inputs(
    scene_dir: Path,
    object_index: int,
    *,
    frame_index: int | None,
    mesh_backend: str | None,
    erode_mask: bool,
) -> dict[str, object]:
    scene_dir = scene_dir.resolve()
    idx = resolve_simfoundry_frame_index(scene_dir, frame_index)
    npz_path = scene_dir / "s2_da" / "da" / "exports" / "npz" / "results.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    with np.load(npz_path) as results:
        for key in ("image", "depth", "intrinsics"):
            if key not in results:
                raise KeyError(f"{npz_path} does not contain {key!r}")
        if not (0 <= idx < len(results["image"])):
            raise IndexError(f"frame index {idx} is outside DA3 result length {len(results['image'])}")
        rgb = np.asarray(results["image"][idx]).copy()
        depth = np.asarray(results["depth"][idx], dtype=np.float32).copy()
        K = np.asarray(results["intrinsics"][idx], dtype=np.float64).copy()

    mask_path = scene_dir / "s5_scene" / "removal_mask" / f"iter_{object_index}.png"
    canvas_path = scene_dir / "s5_scene" / "source_padded_resized.png"
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    if not canvas_path.is_file():
        raise FileNotFoundError(canvas_path)
    mask = map_simfoundry_mask(
        mask_path,
        canvas_path,
        depth.shape,
        erode=erode_mask,
    )

    mesh_path = resolve_simfoundry_mesh(scene_dir, object_index, mesh_backend)
    cam2world_path = scene_dir / "s4_frame" / f"image_{idx}_cam2world.npy"
    if not cam2world_path.is_file():
        raise FileNotFoundError(cam2world_path)
    T_cam_world = load_matrix(cam2world_path, (4, 4))

    return {
        "rgb": rgb,
        "depth": depth,
        "K": K,
        "mask": mask,
        "mesh_path": mesh_path,
        "T_cam_world": T_cam_world,
        "frame_index": idx,
        "object_index": object_index,
        "npz_path": npz_path,
        "mask_path": mask_path,
        "cam2world_path": cam2world_path,
    }


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
    source = parser.add_argument_group("input source")
    source.add_argument(
        "--simfoundry-scene-dir",
        type=Path,
        default=None,
        help="Completed Data/<scene> directory. Auto-resolves DA3 RGB-D/K, Stage-5 mask, Stage-7 mesh, and cam2world.",
    )
    source.add_argument("--object-index", type=int, default=None, help="Stage-5/7 iter index; required with --simfoundry-scene-dir.")
    source.add_argument("--frame-index", type=int, default=None, help="Override canonical frame index in SimFoundry mode.")
    source.add_argument("--mesh-backend", default=None, help="Select s7_mesh/textured_mesh/<backend> in SimFoundry mode.")
    source.add_argument("--rgb", type=Path, default=None)
    source.add_argument("--depth", type=Path, default=None)
    source.add_argument("--intrinsics", type=Path, default=None)
    source.add_argument("--mask", type=Path, default=None)
    source.add_argument("--mesh", type=Path, default=None, help="Generated GLB/OBJ/PLY mesh. Requires --mask in explicit-file mode.")
    source.add_argument("--camera-to-world", type=Path, default=None)

    parser.add_argument("--output-dir", type=Path, default=Path("/output"))
    parser.add_argument("--depth-scale", type=float, default=1.0, help="Explicit-file mode: multiply stored depth by this to get metres.")
    parser.add_argument("--min-depth", type=float, default=0.02)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--mesh-points", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-scale-refine", action="store_true")
    parser.add_argument("--no-mask-erode", action="store_true", help="Disable the Stage-8-style 3x3 object-mask erosion.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples <= 0 or args.mesh_points < MIN_POINTS:
        raise ValueError("--samples must be >0 and --mesh-points must be >=10")

    provenance: dict[str, object]
    mesh_path: Path | None
    T_cam_world: np.ndarray | None

    if args.simfoundry_scene_dir is not None:
        if any(v is not None for v in (args.rgb, args.depth, args.intrinsics, args.mask, args.mesh, args.camera_to_world)):
            raise ValueError("Do not mix --simfoundry-scene-dir with explicit --rgb/--depth/--mask/--mesh inputs.")
        if args.object_index is None:
            raise ValueError("--object-index is required with --simfoundry-scene-dir")
        bundle = load_simfoundry_scene_inputs(
            args.simfoundry_scene_dir,
            args.object_index,
            frame_index=args.frame_index,
            mesh_backend=args.mesh_backend,
            erode_mask=not args.no_mask_erode,
        )
        rgb = np.asarray(bundle["rgb"])
        depth = np.asarray(bundle["depth"], dtype=np.float32)
        K = np.asarray(bundle["K"], dtype=np.float64)
        mask = np.asarray(bundle["mask"], dtype=bool)
        mesh_path = Path(bundle["mesh_path"])
        T_cam_world = np.asarray(bundle["T_cam_world"], dtype=np.float64)
        provenance = {
            "mode": "simfoundry-scene",
            "scene_dir": str(args.simfoundry_scene_dir.resolve()),
            "frame_index": int(bundle["frame_index"]),
            "object_index": int(bundle["object_index"]),
            "da3_npz": str(bundle["npz_path"]),
            "mask": str(bundle["mask_path"]),
            "mesh": str(mesh_path),
            "camera_to_world": str(bundle["cam2world_path"]),
        }
    else:
        missing = [name for name, value in (("--rgb", args.rgb), ("--depth", args.depth), ("--intrinsics", args.intrinsics)) if value is None]
        if missing:
            raise ValueError(f"Explicit-file mode requires {', '.join(missing)}")
        if args.mesh is not None and args.mask is None:
            raise ValueError("Object mesh registration requires --mask. Omit --mesh for a full-frame RGB-D point cloud.")

        rgb = load_rgb(args.rgb)
        depth = load_depth(args.depth, args.depth_scale)
        if rgb.shape[:2] != depth.shape:
            raise ValueError(f"RGB shape {rgb.shape[:2]} does not match depth {depth.shape}")
        K = load_matrix(args.intrinsics, (3, 3))
        mask = load_mask(args.mask, depth.shape, erode=not args.no_mask_erode)
        mesh_path = args.mesh
        T_cam_world = load_matrix(args.camera_to_world, (4, 4)) if args.camera_to_world is not None else None
        provenance = {
            "mode": "explicit-files",
            "rgb": str(args.rgb.resolve()),
            "depth": str(args.depth.resolve()),
            "intrinsics": str(args.intrinsics.resolve()),
            "mask": str(args.mask.resolve()) if args.mask is not None else None,
            "mesh": str(args.mesh.resolve()) if args.mesh is not None else None,
            "camera_to_world": str(args.camera_to_world.resolve()) if args.camera_to_world is not None else None,
        }

    if rgb.shape[:2] != depth.shape:
        raise ValueError(f"RGB shape {rgb.shape[:2]} does not match depth {depth.shape}")
    if K.shape != (3, 3):
        raise ValueError(f"Intrinsics must be 3x3, got {K.shape}")
    if mask.shape != depth.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match depth {depth.shape}")

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
        "input": provenance,
    }

    if T_cam_world is not None:
        world_points = np.asarray(cloud.points) @ T_cam_world[:3, :3].T + T_cam_world[:3, 3]
        world_cloud = make_cloud(world_points, np.asarray(cloud.colors))
        world_cloud_path = out_dir / "object_point_cloud_world.ply"
        o3d.io.write_point_cloud(str(world_cloud_path), world_cloud)
        result["point_cloud_world"] = str(world_cloud_path)

    if mesh_path is not None:
        loaded = trimesh.load(mesh_path, force="mesh")
        if not isinstance(loaded, trimesh.Trimesh) or len(loaded.vertices) < 4:
            raise ValueError(f"Could not load a triangle mesh from {mesh_path}")

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
