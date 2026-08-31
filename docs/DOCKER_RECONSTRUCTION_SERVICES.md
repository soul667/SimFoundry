# Reconstruction Docker services

This directory exposes two narrow services from the SimFoundry reconstruction stack so they can be called from another simulator or orchestration system without installing all of OmniGibson / Isaac Sim.

## 1. Gaussian trainer

Image after merge to `main`: `ghcr.io/soul667/simfoundry-gaussian-trainer:latest`

This image follows the current SimFoundry auto-background training stack:

- CUDA 12.8
- PyTorch 2.7.1 + cu128
- Nerfstudio 1.1.5
- gsplat 1.5.3
- SimFoundry's environment-variable-gated Splatfacto depth-loss patch
- metric-pose settings: no pose auto-scaling, no automatic centering/orientation, `scale_factor=1.0`

Input is an already prepared Nerfstudio dataset directory containing `transforms.json`, images, and optionally a seed PLY referenced by `ply_file_path`. Optional depth supervision is a directory of `frame_*.npy` maps in metres, at training-image resolution, where 0 means invalid.

```bash
docker run --rm --gpus all \
  -p 7007:7007 \
  -v "$PWD/ns_data:/data:ro" \
  -v "$PWD/gs_out:/output" \
  ghcr.io/soul667/simfoundry-gaussian-trainer:latest \
  --data /data \
  --output-dir /output \
  --method splatfacto-big \
  --iterations 80000 \
  --camera-optimizer-mode SO3xR3 \
  --depth-dir /data/depths
```

If `transforms.json` contains a valid `ply_file_path`, the trainer automatically enables Nerfstudio's `--load-3D-points`. Use `--load-3d-points true|false` to override that behavior.

Outputs:

```text
/output/
  outputs/                       # Nerfstudio checkpoints/configs
  export/splat.ply               # exported Gaussian splat
  gaussian_train_manifest.json   # resolved settings + command
```

## 2. RGB-D registration

Image after merge to `main`: `ghcr.io/soul667/simfoundry-rgbd-registration:latest`

This image extracts the geometry core of SimFoundry Stage 8:

1. backproject metric RGB-D with the camera intrinsics;
2. apply an optional object mask;
3. write the object's partial point cloud;
4. initialize generated-mesh scale from the point-cloud / mesh OBB diagonal ratio;
5. run multi-start CPD with both rigid-only and residual-scale fits;
6. export camera-frame and, when `T_cam_world` is supplied, world-frame poses.

The CPD step in SimFoundry itself runs on CPU (`use_cuda=False`), so this image intentionally stays CPU-only and small. SimFoundry's optional FoundationPose refinement is not bundled here; its CUDA extensions and multi-GB scorer/refiner weights remain part of the full SimFoundry environment. This service is intended to produce the deterministic geometric initialization that can be refined later if required.

Required inputs:

- RGB image
- depth image (`.npy`, or an image such as 16-bit PNG)
- 3x3 intrinsics (`.npy` or JSON)

Optional inputs:

- object mask (`>127` is foreground)
- generated object mesh (`.glb`, `.obj`, `.ply`, ...)
- 4x4 camera-to-world transform (`.npy` or JSON)

Depth values must become metres after multiplying by `--depth-scale`. For a millimetre uint16 depth PNG, pass `--depth-scale 0.001`.

```bash
docker run --rm \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/reg_out:/output" \
  ghcr.io/soul667/simfoundry-rgbd-registration:latest \
  --rgb /input/rgb.png \
  --depth /input/depth.npy \
  --intrinsics /input/K.npy \
  --mask /input/object_mask.png \
  --mesh /input/object.glb \
  --camera-to-world /input/T_cam_world.npy \
  --output-dir /output
```

Outputs:

```text
/output/
  object_point_cloud_camera.ply
  object_point_cloud_world.ply       # only if camera-to-world was supplied
  aligned_mesh_camera.glb            # only if mesh was supplied
  aligned_mesh_world.glb             # only if mesh + camera-to-world were supplied
  registration.json
```

`registration.json` stores translation in metres, rotation matrix, quaternion in XYZW order, the OBB pre-scale, residual CPD scale, total scale, and symmetric-Chamfer fit score.

## GitHub Actions / GHCR

`.github/workflows/docker-images.yml` builds both Dockerfiles as a matrix.

- Any branch push touching the Docker build inputs: build both images immediately.
- Same-repository pull requests: the duplicate PR build job is skipped because the branch push already validates it.
- Fork pull requests: build both images without registry push.
- Push to `main`: build and additionally push `latest` plus a full commit-SHA tag to GHCR.
- Manual `workflow_dispatch`: build only.

The workflow uses the repository `GITHUB_TOKEN`; no additional registry secret is required as long as GitHub Actions has package write permission for the repository.
