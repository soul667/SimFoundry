# SAM3 image design decision

The standalone SAM3 image deliberately separates three concerns:

1. GHCR stores the reproducible runtime and pinned SAM3 code.
2. Hugging Face (or any equivalent host-side artifact store) stores the checkpoint separately.
3. Docker receives the checkpoint only through a read-only bind mount.

No credentials are passed into the container. No model download occurs from inside the container. This keeps the GHCR image small, avoids gated-weight coupling during GitHub Actions builds, and lets the checkpoint be versioned independently from the runtime.
