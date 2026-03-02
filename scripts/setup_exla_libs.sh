#!/usr/bin/env bash
# One-time setup so EXLA NIF finds NVSHMEM/CUDA/cuDNN libs. Run as root inside the container:
#   ./scripts/setup_exla_libs.sh
# Then run: mix recgpt.serve ... or ./scripts/run_with_exla_env.sh mix recgpt.serve ...
set -e
echo "Setting up EXLA library paths..."

# 1. Install patchelf if missing
if ! command -v patchelf >/dev/null 2>&1; then
  echo "Installing patchelf..."
  apt-get update -qq && apt-get install -y -qq patchelf
fi

# 2. NVSHMEM: XLA wants .so.3, package has .so.4 — copy and set soname
NVDIR="/usr/lib/x86_64-linux-gnu/nvshmem/12"
if [ -d "$NVDIR" ]; then
  for lib in nvshmem_transport_ibrc nvshmem_transport_ibgda nvshmem_transport_ibdevx nvshmem_transport_libfabric nvshmem_transport_ucx; do
    if [ -f "$NVDIR/$lib.so.4" ]; then
      cp -a "$NVDIR/$lib.so.4" "$NVDIR/$lib.so.3"
      patchelf --set-soname "$lib.so.3" "$NVDIR/$lib.so.3"
      echo "  Created $NVDIR/$lib.so.3"
    fi
  done
else
  echo "  Warning: $NVDIR not found. Install with: apt-get update && apt-get install -y libnvshmem3-cuda-12"
  echo "  Then run this script again. (Requires CUDA repo: cuda-keyring + apt-get update.)"
fi

# 3. Install NVRTC/cuDNN if libs missing (need cuda repo from cuda-toolkit or keyring)
NVRTC_LIB="$(find /usr -name 'libnvrtc-builtins.so.12*' -type f 2>/dev/null | head -1)"
if [ -z "$NVRTC_LIB" ]; then
  echo "Installing cuda-nvrtc-12-9 (libnvrtc-builtins.so.12)..."
  apt-get update -qq && apt-get install -y -qq cuda-nvrtc-12-9 || true
  NVRTC_LIB="$(find /usr -name 'libnvrtc-builtins.so.12*' -type f 2>/dev/null | head -1)"
fi
CUDNN_LIB="$(find /usr -name 'libcudnn_engines_precompiled.so.9*' -type f 2>/dev/null | head -1)"
if [ -z "$CUDNN_LIB" ]; then
  echo "Installing cudnn9-cuda-12-9..."
  apt-get update -qq && apt-get install -y -qq cudnn9-cuda-12-9 || true
fi

# 4. NVRTC: XLA may expect specific soname; ensure lib dir is registered (no copy needed for 12.x)
NVRTC_DIR="/usr/local/cuda-12.9/targets/x86_64-linux/lib"

# 5. Register lib dirs with system loader
echo "/usr/lib/x86_64-linux-gnu/nvshmem/12" > /etc/ld.so.conf.d/nvshmem.conf
echo "/usr/local/cuda-12.9/lib64" >> /etc/ld.so.conf.d/nvshmem.conf
echo "/usr/local/cuda-12.9/targets/x86_64-linux/lib" >> /etc/ld.so.conf.d/nvshmem.conf
echo "/usr/lib/x86_64-linux-gnu" >> /etc/ld.so.conf.d/nvshmem.conf
CUDNN_LIB="$(find /usr -name 'libcudnn_engines_precompiled.so.9*' -type f 2>/dev/null | head -1)"
[ -n "$CUDNN_LIB" ] && echo "$(dirname "$CUDNN_LIB")" >> /etc/ld.so.conf.d/nvshmem.conf
NVRTC_LIB="$(find /usr -name 'libnvrtc-builtins.so.12*' -type f 2>/dev/null | head -1)"
[ -n "$NVRTC_LIB" ] && echo "$(dirname "$NVRTC_LIB")" >> /etc/ld.so.conf.d/nvshmem.conf
ldconfig
echo "Done. Run: mix recgpt.serve ... or ./scripts/run_with_exla_env.sh mix recgpt.serve ..."
