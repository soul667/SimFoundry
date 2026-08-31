FROM python:3.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir \
      "numpy<2" \
      scipy \
      pillow \
      trimesh \
      open3d \
      probreg && \
    python -c "import numpy, open3d, probreg, scipy, trimesh; print('Registration stack OK:', numpy.__version__, open3d.__version__)"

COPY docker/rgbd-registration/register.py /opt/simfoundry/register_rgbd.py
RUN python -m py_compile /opt/simfoundry/register_rgbd.py

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace
ENTRYPOINT ["python", "/opt/simfoundry/register_rgbd.py"]
CMD ["--help"]
