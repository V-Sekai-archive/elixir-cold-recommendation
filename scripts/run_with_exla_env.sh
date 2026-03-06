#!/usr/bin/env bash
# Set LD_LIBRARY_PATH so EXLA NIF finds NVSHMEM and CUDA libs (e.g. nvshmem_transport_ibrc.so.3).
# Run from repo root: ./scripts/run_with_exla_env.sh mix recgpt.serve ...
#
# As root, once per machine (if not using rebuilt devcontainer):
#   sudo bash scripts/setup_exla_libs.sh   # Fedora and Debian/Ubuntu supported
set -e

# Detect Fedora vs Debian for lib paths
if [ -d /usr/lib64/nvshmem/12 ]; then
  NVSHMEM_DIR="/usr/lib64/nvshmem/12"
  CUDA_LIB64="/usr/local/cuda/lib64"
  CUDA_TARGETS_LIB="/usr/local/cuda/targets/x86_64-linux/lib"
  [ -d /usr/local/cuda-12.9 ] && CUDA_LIB64="/usr/local/cuda-12.9/lib64" && CUDA_TARGETS_LIB="/usr/local/cuda-12.9/targets/x86_64-linux/lib"
  CUDNN_FALLBACK="/usr/lib64"
else
  NVSHMEM_DIR="/usr/lib/x86_64-linux-gnu/nvshmem/12"
  CUDA_LIB64="/usr/local/cuda-12.9/lib64"
  CUDA_TARGETS_LIB="/usr/local/cuda-12.9/targets/x86_64-linux/lib"
  CUDNN_FALLBACK="/usr/lib/x86_64-linux-gnu"
fi

NVRTC_LIB="$(find /usr -name 'libnvrtc-builtins.so.12*' -type f 2>/dev/null | head -1)"
[ -n "$NVRTC_LIB" ] && NVRTC_DIR="$(dirname "$NVRTC_LIB")" || NVRTC_DIR=""
CUDNN_LIB="$(find /usr -name 'libcudnn_engines_precompiled.so.9*' -type f 2>/dev/null | head -1)"
CUDNN_DIR="$CUDNN_FALLBACK"
[ -n "$CUDNN_LIB" ] && CUDNN_DIR="$(dirname "$CUDNN_LIB")"

export LD_LIBRARY_PATH="${NVRTC_DIR:+$NVRTC_DIR:}${CUDA_TARGETS_LIB}:${CUDA_LIB64}:${NVSHMEM_DIR}:${CUDNN_DIR}:${CUDNN_FALLBACK}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/usr/local/cuda-12.9/bin:/usr/local/cuda/bin${PATH:+:$PATH}"

if [ -z "$NVRTC_LIB" ]; then
  echo "Warning: libnvrtc-builtins.so.12* not found. On Fedora see CONTRIBUTING.md (CUDA repo)." >&2
  echo "  On Debian: apt-get install -y cuda-nvrtc-12-9" >&2
fi
exec "$@"
