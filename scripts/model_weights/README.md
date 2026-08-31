# Model weight staging helpers

These scripts are host-side utilities. They are intentionally not part of any Docker runtime.

For SAM3:

```bash
python -m pip install -U huggingface_hub
hf auth login

python scripts/model_weights/upload_sam3_to_hf.py \
  --repo-id YOUR_HF_USER/simfoundry-sam3-weights \
  --checkpoint /path/to/sam3.pt \
  --license /path/to/SAM3/LICENSE

bash scripts/model_weights/download_sam3_from_hf.sh \
  YOUR_HF_USER/simfoundry-sam3-weights \
  /srv/simfoundry-models
```

The upload helper creates a private model repository by default and uploads both `sam3.pt` and the SAM license. The download helper materializes those files on the Docker host so they can be bind-mounted read-only.
