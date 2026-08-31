# RGB-D registration: direct SimFoundry scene mode

The `rgbd-registration` container does **not** bundle SAM3, Depth Anything 3, Hunyuan3D, TRELLIS, or FoundationPose weights. It is a Stage-8-style geometry service: RGB-D backprojection plus generated-mesh scale and pose registration.

When a full SimFoundry reconstruction has already run, no perception models are needed inside this container. The container can now read the upstream artifacts directly from `Data/<scene>/`.

## Direct scene usage

Mount one completed scene directory and select the Stage-5/7 object iteration:

```bash
docker run --rm \
  -v "$PWD/Data/my_scene:/scene:ro" \
  -v "$PWD/reg_out:/output" \
  ghcr.io/soul667/simfoundry-rgbd-registration:latest \
  --simfoundry-scene-dir /scene \
  --object-index 3 \
  --output-dir /output
```

The adapter resolves:

- `s2_da/da/exports/npz/results.npz` for canonical RGB, metric depth, and intrinsics;
- `s3_ground/frame_selection.json` when available, otherwise the unique `s4_frame/image_*_cam2world.npy`, for the canonical frame index;
- `s5_scene/removal_mask/iter_<N>.png` for the object mask;
- `s7_mesh/textured_mesh/*/iter_<N>_mesh.glb` for the generated mesh;
- `s4_frame/image_<frame>_cam2world.npy` for camera-to-world placement.

If multiple Stage-7 mesh backends exist, select one explicitly:

```bash
--mesh-backend hunyuan
```

If stale Stage-4 outputs leave multiple possible canonical frames, select one explicitly:

```bash
--frame-index 4
```

## Mask geometry

Stage 5 often operates on a padded/resized decomposition canvas whose aspect ratio differs from the DA3 canonical RGB-D image. Direct scene mode reproduces the Stage-8 geometry mapping: it pads the DA3 geometry to the Stage-5 canvas aspect ratio, maps the object mask, removes the padding, and then applies a 3x3 erosion. Use `--no-mask-erode` only for debugging or very small objects.

## Explicit-file mode

The original standardized interface remains available:

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

For object mesh registration, `--mask` is required in explicit-file mode. If `--mesh` is omitted, the service may be used to backproject the full RGB-D frame into a point cloud without a mask.

## What remains outside this image

If the only input is raw RGB or RGB-D and there is no object mask or generated mesh yet, run the relevant upstream SimFoundry stages (or another segmentation / image-to-3D pipeline) first. This keeps model downloads, gated checkpoints, and CUDA-heavy FoundationPose refinement out of the lightweight deterministic registration service.
