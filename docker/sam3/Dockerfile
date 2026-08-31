# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.11-slim

FROM ${PYTHON_IMAGE} AS sam3-source
ARG SAM3_COMMIT=46957e47805eaa273f4aa7bbbd25a88bca9108ce
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
RUN git init /src/sam3 \
    && git -C /src/sam3 remote add origin https://github.com/facebookresearch/sam3.git \
    && git -C /src/sam3 fetch --depth 1 origin "${SAM3_COMMIT}" \
    && git -C /src/sam3 checkout --detach FETCH_HEAD \
    && rm -rf /src/sam3/.git

FROM ${PYTHON_IMAGE}
ARG TORCH_VERSION=2.7.0
ARG TORCHVISION_VERSION=0.22.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    HF_HUB_CACHE=/models/huggingface/hub

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        --index-url "${TORCH_INDEX_URL}"

COPY --from=sam3-source /src/sam3 /opt/sam3
RUN python -m pip install -e /opt/sam3 \
    && mkdir -p /licenses \
    && cp /opt/sam3/LICENSE /licenses/SAM3-LICENSE.txt

COPY docker/sam3/segment.py /opt/simfoundry/segment_sam3.py
RUN python -m py_compile /opt/simfoundry/segment_sam3.py \
    && python -c "import torch, sam3; from sam3.model_builder import build_sam3_image_model; from sam3.model.sam3_image_processor import Sam3Processor; print('SAM3 runtime import OK:', torch.__version__)"

WORKDIR /workspace
VOLUME ["/models"]
ENTRYPOINT ["python", "/opt/simfoundry/segment_sam3.py"]
CMD ["--help"]
