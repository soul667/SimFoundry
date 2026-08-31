# SAM3 image contract

- Runtime/source code is baked into the image.
- Model weights are never baked into the image.
- Model weights are never downloaded by the container.
- Default checkpoint path: `/models/sam3.pt`.
- Mount `/models` read-only from the host.
- The image sets `HF_HUB_OFFLINE=1`.

Example:

```bash
docker run --rm --gpus all \
  -v /srv/simfoundry-models:/models:ro \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/output:/output" \
  ghcr.io/soul667/simfoundry-sam3:latest \
  --image /input/frame.png \
  --prompt "cup" \
  --mask-out /output/object_mask.png
```

See `docs/SAM3_DOCKER.md` for the host-side Hugging Face staging workflow.
