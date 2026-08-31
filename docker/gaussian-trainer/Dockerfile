FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgl1 \
    libglib2.0-0 \
    ninja-build \
    patch \
    python3 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir \
      torch==2.7.1 torchvision==0.22.1 \
      --index-url https://download.pytorch.org/whl/cu128 && \
    python3 -m pip install --no-cache-dir \
      nerfstudio==1.1.5 "hydra-core>=1.3,<1.4" && \
    python3 -m pip install --no-cache-dir --no-deps gsplat==1.5.3

COPY patches/splatfacto_depth_loss.patch /tmp/splatfacto_depth_loss.patch
RUN set -eux; \
    SP_DIR="$(python3 -c 'import nerfstudio, os; print(os.path.dirname(os.path.dirname(nerfstudio.__file__)))')"; \
    patch -p1 -d "${SP_DIR}" --forward < /tmp/splatfacto_depth_loss.patch; \
    rm /tmp/splatfacto_depth_loss.patch

COPY docker/gaussian-trainer/train.py /opt/simfoundry/train_gaussian.py
RUN python3 -m py_compile /opt/simfoundry/train_gaussian.py

ENV PYTHONUNBUFFERED=1 \
    TORCHDYNAMO_DISABLE=1 \
    TORCH_COMPILE_DISABLE=1

WORKDIR /workspace
ENTRYPOINT ["python3", "/opt/simfoundry/train_gaussian.py"]
CMD ["--help"]
