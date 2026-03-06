# Elixir (hexpm) base for mix/elixir. CUDA 12.9 toolkit (nvcc) for XLA_TARGET=cuda12 (matches host driver 576.x / CUDA 12.9).
FROM hexpm/elixir:1.18.0-erlang-27.2.1-debian-bookworm-20251103

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# Build tools + XLA-from-source deps (Bazel 7.7.0, Clang, Python3+NumPy, xxd for embed_gpu_specs_gen). See https://hexdocs.pm/xla/0.10.0/XLA.html
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake curl gnupg git ca-certificates apt-util dos2unix wget xxd \
    python3 python3-pip python3-venv sudo \
    clang-14 \
    unzip zip \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --break-system-packages --no-cache-dir numpy \
    && (wget -q https://github.com/bazelbuild/bazel/releases/download/7.7.0/bazel-7.7.0-installer-linux-x86_64.sh -O /tmp/bazel-installer.sh \
        && chmod +x /tmp/bazel-installer.sh \
        && /tmp/bazel-installer.sh \
        && rm /tmp/bazel-installer.sh) || true

# Install lazygit from GitHub releases (no linuxbrew)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && LAZYGIT_VERSION=$(wget -qO- "https://api.github.com/repos/jesseduffield/lazygit/releases/latest" | grep -oP '"tag_name": "v\K(.*)(?=")') \
    && wget -q "https://github.com/jesseduffield/lazygit/releases/download/v${LAZYGIT_VERSION}/lazygit_${LAZYGIT_VERSION}_Linux_x86_64.tar.gz" -O /tmp/lazygit.tar.gz \
    && tar -C /usr/local/bin -xzf /tmp/lazygit.tar.gz lazygit \
    && rm /tmp/lazygit.tar.gz \
    && chmod +x /usr/local/bin/lazygit

# CUDA 12.9 keyring + toolkit (nvcc) + NCCL + NVSHMEM + cuDNN 9 for EXLA. Use 12.x so container runtime matches host driver (e.g. 576.x supports 12.9; cuda13 needs r580+).
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cuda-toolkit-12-9 \
        cuda-nvrtc-12-9 \
        libnccl2 \
        libnvshmem3-cuda-12 \
        cudnn9-cuda-12-9 \
    && rm -rf /var/lib/apt/lists/*

# XLA extension was linked against NVSHMEM transport .so.3; package provides .so.4 — copy and set soname so loader accepts
RUN apt-get update && apt-get install -y --no-install-recommends patchelf && rm -rf /var/lib/apt/lists/* \
    && NVDIR=/usr/lib/x86_64-linux-gnu/nvshmem/12 \
    && for lib in nvshmem_transport_ibrc nvshmem_transport_ibgda nvshmem_transport_ibdevx nvshmem_transport_libfabric nvshmem_transport_ucx; do \
      if [ -f "$$NVDIR/$$lib.so.4" ]; then \
        cp -a "$$NVDIR/$$lib.so.4" "$$NVDIR/$$lib.so.3" && patchelf --set-soname "$$lib.so.3" "$$NVDIR/$$lib.so.3"; \
      fi; \
    done

# Register NVSHMEM, CUDA, and cuDNN lib dirs in system loader so Beam finds them
RUN echo "/usr/lib/x86_64-linux-gnu/nvshmem/12" > /etc/ld.so.conf.d/nvshmem.conf \
    && echo "/usr/local/cuda-12.9/lib64" >> /etc/ld.so.conf.d/nvshmem.conf \
    && echo "/usr/local/cuda-12.9/targets/x86_64-linux/lib" >> /etc/ld.so.conf.d/nvshmem.conf \
    && echo "/usr/lib/x86_64-linux-gnu" >> /etc/ld.so.conf.d/nvshmem.conf \
    && (CUDNN_LIB="$$(find /usr -name 'libcudnn_engines_precompiled.so.9*' -type f 2>/dev/null | head -1)" \
        && [ -n "$$CUDNN_LIB" ] && echo "$$(dirname $$CUDNN_LIB)" >> /etc/ld.so.conf.d/nvshmem.conf) || true \
    && (NVRTC_LIB="$$(find /usr -name 'libnvrtc-builtins.so.12*' -type f 2>/dev/null | head -1)" \
        && [ -n "$$NVRTC_LIB" ] && echo "$$(dirname $$NVRTC_LIB)" >> /etc/ld.so.conf.d/nvshmem.conf) || true \
    && (dpkg -L cudnn9-cuda-12-9 2>/dev/null | grep -E '/libcudnn.*\.so' | xargs -I{} dirname {} 2>/dev/null | sort -u >> /etc/ld.so.conf.d/nvshmem.conf) || true \
    && ldconfig

# NVIDIA Nsight Systems CLI (GPU profiling). See https://developer.nvidia.com/nsight-systems/get-started
RUN (wget -q "https://developer.download.nvidia.com/devtools/nsight-systems/2026_1/NsightSystems-linux-cli-public-2026.1.1.204-3717666.deb" -O /tmp/nsys.deb \
      && dpkg -i /tmp/nsys.deb || (apt-get update -qq && apt-get install -f -y -qq && dpkg -i /tmp/nsys.deb) \
      && rm -f /tmp/nsys.deb) || echo "Nsight Systems install skipped; install manually if needed for GPU profiling."

ENV PATH="/usr/local/cuda-12.9/bin:/opt/nvidia/nsight-systems/2026.1.1.204/bin:/usr/local/bin:${PATH}"
# CUDA + NVSHMEM + cuDNN (multiarch) so EXLA NIF finds libs even if ldconfig path differs
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.9/targets/x86_64-linux/lib:/usr/local/cuda-12.9/lib64:/usr/lib/x86_64-linux-gnu/nvshmem/12:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
# Always build XLA from source so the binary matches this image's CUDA/cuDNN/NVSHMEM (see https://hexdocs.pm/xla/0.10.0/XLA.html)
ENV XLA_TARGET=cuda12

RUN mix local.hex --force && mix local.rebar --force
