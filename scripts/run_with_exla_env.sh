#!/usr/bin/env bash
# Set LD_LIBRARY_PATH so EXLA NIF finds NVSHMEM and CUDA libs (e.g. nvshmem_transport_ibrc.so.3).
# Run from repo root: ./scripts/run_with_exla_env.sh mix recgpt.serve ...
#
# As root, once per machine (if not using rebuilt devcontainer):
#   apt-get install -y patchelf
#   NVDIR=/usr/lib/x86_64-linux-gnu/nvshmem/12
#   for lib in nvshmem_transport_ibrc nvshmem_transport_ibgda nvshmem_transport_ibdevx nvshmem_transport_libfabric nvshmem_transport_ucx; do
#     [ -f "$NVDIR/$lib.so.4" ] && cp -a "$NVDIR/$lib.so.4" "$NVDIR/$lib.so.3" && patchelf --set-soname "$lib.so.3" "$NVDIR/$lib.so.3"
#   done
#   echo "/usr/lib/x86_64-linux-gnu/nvshmem/12" > /etc/ld.so.conf.d/nvshmem.conf
#   echo "/usr/local/cuda-12.9/lib64" >> /etc/ld.so.conf.d/nvshmem.conf
#   echo "/usr/lib/x86_64-linux-gnu" >> /etc/ld.so.conf.d/nvshmem.conf
#   CUDNN_LIB=$(find /usr -name 'libcudnn_engines_precompiled.so.9*' 2>/dev/null | head -1)
#   [ -n "$CUDNN_LIB" ] && echo "$(dirname "$CUDNN_LIB")" >> /etc/ld.so.conf.d/nvshmem.conf
#   ldconfig
# If libcudnn_engines_precompiled.so.9 is missing, install: apt-get install -y cudnn9-cuda-12-9
# If libnvrtc-builtins.so.12* is missing, install: apt-get install -y cuda-nvrtc-12-9
set -e
# Resolve paths for libs the EXLA NIF needs (so loader finds them when mix runs)
NVRTC_LIB="$(find /usr -name 'libnvrtc-builtins.so.12*' -type f 2>/dev/null | head -1)"
NVRTC_DIR="${NVRTC_DIR:-}"
[ -n "$NVRTC_LIB" ] && NVRTC_DIR="$(dirname "$NVRTC_LIB")"
CUDNN_LIB="$(find /usr -name 'libcudnn_engines_precompiled.so.9*' -type f 2>/dev/null | head -1)"
CUDNN_DIR="/usr/lib/x86_64-linux-gnu"
[ -n "$CUDNN_LIB" ] && CUDNN_DIR="$(dirname "$CUDNN_LIB")"
# NVRTC can be in lib64 or targets/x86_64-linux/lib; include both so libnvrtc-builtins.so.12* is found
CUDA_LIB64="/usr/local/cuda-12.9/lib64"
CUDA_TARGETS_LIB="/usr/local/cuda-12.9/targets/x86_64-linux/lib"
export LD_LIBRARY_PATH="${NVRTC_DIR:+$NVRTC_DIR:}${CUDA_TARGETS_LIB}:${CUDA_LIB64}:/usr/lib/x86_64-linux-gnu/nvshmem/12:${CUDNN_DIR}:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/usr/local/cuda-12.9/bin${PATH:+:$PATH}"
if [ -z "$NVRTC_LIB" ]; then
  echo "Warning: libnvrtc-builtins.so.12* not found. Install with: apt-get install -y cuda-nvrtc-12-9" >&2
  echo "  (Ensure CUDA repo is configured; run ./scripts/setup_exla_libs.sh as root first.)" >&2
fi
exec "$@"
