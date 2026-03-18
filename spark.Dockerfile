# GR00T N1.5 — DGX Spark (aarch64 SBSA, Blackwell iGPU sm_121, CUDA 13)
# Reference: https://github.com/NVIDIA/Isaac-GR00T/issues/474
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# System dependencies + deadsnakes PPA for Python 3.10 (Ubuntu 24.04 ships 3.12)
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common gpg-agent && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3.10-venv \
      build-essential cmake ninja-build make nasm yasm pkg-config git curl wget \
      ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
      libhdf5-serial-dev libtesseract-dev libgtk-3-0 libtbb12 \
      libatlas-base-dev libopenblas-dev \
      libgnutls28-dev libvpx-dev libopus-dev libvorbis-dev \
      libmp3lame-dev libfreetype-dev libass-dev libaom-dev libdav1d-dev \
      libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev \
      libavutil-dev libswresample-dev libswscale-dev \
      pybind11-dev \
      tmux vim less sudo htop ca-certificates zip unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Make python3.10 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Remove system packaging that conflicts with pip bootstrap, then install pip for 3.10
RUN rm -rf /usr/lib/python3/dist-packages/packaging* && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Allow pip to install globally without venv constraint
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# NVPL LAPACK/BLAS — required by the Jetson torch wheel on aarch64
# The CUDA base image already has the NVIDIA apt repo configured
RUN apt-get update && \
    apt-get install -y libnvpl-lapack0 libnvpl-blas0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ── Install PyTorch + core deps from cu130 index ────────────────────────────
# Using the official PyTorch cu130 index for aarch64 wheels
RUN pip install --no-cache-dir \
      torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu130

# ── Install GR00T N1.5 source ──────────────────────────────────────────────
COPY pyproject.toml .
COPY gr00t /workspace/gr00t
COPY scripts /workspace/scripts
COPY getting_started /workspace/getting_started
COPY demo_data /workspace/demo_data
COPY Makefile /workspace/Makefile

# Install gr00t package without deps (to avoid pip overriding torch)
# then install remaining deps manually
RUN pip install --no-cache-dir -e . --no-deps

# ── Remaining N1.5 dependencies (pinned to compatible versions) ─────────────
RUN pip install --no-cache-dir \
      "albumentations==1.4.18" \
      "av==12.3.0" \
      "blessings==1.7" \
      "dm_tree==0.1.8" \
      "einops==0.8.1" \
      "gymnasium==1.0.0" \
      "h5py==3.12.1" \
      "hydra-core==1.3.2" \
      "imageio==2.34.2" \
      "kornia==0.7.4" \
      "matplotlib==3.10.0" \
      "numpy>=1.23.5,<2.0.0" \
      "numpydantic==1.6.7" \
      "omegaconf==2.3.0" \
      "opencv_python_headless==4.11.0.86" \
      "pandas==2.2.3" \
      "pydantic==2.10.6" \
      "PyYAML==6.0.2" \
      "ray==2.40.0" \
      "Requests==2.32.3" \
      "tianshou==0.5.1" \
      "timm==1.0.14" \
      "tqdm==4.67.1" \
      "transformers==4.51.3" \
      "typing_extensions==4.12.2" \
      "pyarrow==14.0.1" \
      "wandb==0.18.0" \
      "fastparquet==2024.11.0" \
      "accelerate>=1.2.1" \
      "peft==0.17.0" \
      "protobuf==4.25.1" \
      "onnx==1.18.0" \
      "tyro" \
      "pytest" \
      "diffusers==0.30.2" \
      "pyzmq" \
      "gpustat"

# ── decord2 (aarch64 drop-in replacement for decord) ───────────────────────
RUN pip install --no-cache-dir decord2

# ── flash-attn (prebuilt community wheel for aarch64 + cu130 + torch2.9) ───
# Source: https://github.com/mjun0812/flash-attention-prebuild-wheels
RUN pip install --no-cache-dir \
      https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu130torch2.10-cp310-cp310-linux_aarch64.whl

# ── PyTorch3D (must compile from source on aarch64) ────────────────────────
# TORCH_CUDA_ARCH_LIST must be set explicitly — sm_121 (Blackwell) is not
# auto-detected by pytorch3d's build system, causing an empty arch list error.
RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="12.1" \
    pip install --no-cache-dir --no-build-isolation \
      "git+https://github.com/facebookresearch/pytorch3d.git"

# ── Environment ─────────────────────────────────────────────────────────────
ENV PYTHONPATH=/workspace:${PYTHONPATH}
ENV CUDA_HOME=/usr/local/cuda-13.0
ENV CUDA_PATH=/usr/local/cuda-13.0
ENV TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
ENV CPATH="/usr/local/cuda-13.0/include:${CPATH}"

# Expose NVIDIA pip package libs (cuBLAS, cuDNN, etc.) to the runtime linker.
# Without this, torch gets CUBLAS_STATUS_NOT_INITIALIZED on the first matmul.
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cu13/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cusparselt/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nvshmem/lib:/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"

# Note: Use --video-backend torchvision_av when running finetuning/inference
# Example: python scripts/gr00t_finetune.py --video-backend torchvision_av

WORKDIR /workspace
