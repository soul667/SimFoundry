# SAM3 + external Meshy GLB pipeline

The object mesh generator is not part of the SAM3 container. The intended modular flow is:

```text
RGB image ----> SAM3 Docker ----> object mask ----+ 
                                                   |
RGB-D + K ----------------------------------------+--> rgbd-registration --> metric mesh pose/scale
                                                   |
Meshy / other external service ----> object.glb ---+
```

The existing `rgbd-registration` explicit-file mode accepts external `GLB`, `OBJ`, or `PLY` meshes through `--mesh`; the file does not need to come from SimFoundry Stage 7.

Example:

```bash
docker run --rm \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/sam3_out:/sam3:ro" \
  -v "$PWD/meshy:/meshy:ro" \
  -v "$PWD/reg_out:/output" \
  ghcr.io/soul667/simfoundry-rgbd-registration:latest \
  --rgb /input/frame.png \
  --depth /input/depth.npy \
  --intrinsics /input/K.npy \
  --mask /sam3/object_mask.png \
  --mesh /meshy/object.glb \
  --camera-to-world /input/T_cam_world.npy \
  --output-dir /output
```

The GLB's original arbitrary scale is expected: registration derives an initial scale from the masked RGB-D object's OBB diagonal and then performs CPD pose/scale fitting.

Current limitation: the convenience `--simfoundry-scene-dir` mode auto-resolves the Stage-7 mesh and does not currently accept an external `--mesh` override. Use explicit-file mode for Meshy GLBs. This does not change the underlying registration algorithm.
