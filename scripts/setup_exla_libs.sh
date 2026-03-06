#!/usr/bin/env bash
# One-time setup so EXLA NIF finds NVSHMEM/CUDA/cuDNN libs. Takes no arguments.
# Run as root (e.g. sudo bash scripts/setup_exla_libs.sh from repo root).
# Supports Fedora (dnf) and Debian/Ubuntu (apt-get). Then run: ./scripts/run_with_exla_env.sh mix test
set -e
echo "Setting up EXLA library paths..."

# Detect package manager and NVSHMEM path
if command -v dnf >/dev/null 2>&1 && [ -f /etc/fedora-release ] || grep -q '^ID=fedora' /etc/os-release 2>/dev/null; then
  PKG_MGR="dnf"
  NVDIR="/usr/lib64/nvshmem/12"
  LIBDIR_EXTRA="/usr/lib64"
elif command -v apt-get >/dev/null 2>&1; then
  PKG_MGR="apt"
  NVDIR="/usr/lib/x86_64-linux-gnu/nvshmem/12"
  LIBDIR_EXTRA="/usr/lib/x86_64-linux-gnu"
else
  echo "Unsupported OS: need dnf (Fedora) or apt-get (Debian/Ubuntu)." >&2
  exit 1
fi

# 1. Install patchelf if missing (Debian path); Fedora uses symlinks so patchelf optional
if [ "$PKG_MGR" = "apt" ] && ! command -v patchelf >/dev/null 2>&1; then
  echo "Installing patchelf..."
  apt-get update -qq && apt-get install -y -qq patchelf
fi

# 2. NVSHMEM: XLA wants .so.3, package has .so.4 — symlink (Fedora) or copy+patchelf (Debian)
if [ -d "$NVDIR" ]; then
  for lib in nvshmem_transport_ibrc nvshmem_transport_ibgda nvshmem_transport_ibdevx nvshmem_transport_libfabric nvshmem_transport_ucx; do
    if [ -f "$NVDIR/$lib.so.4" ]; then
      if [ "$PKG_MGR" = "dnf" ]; then
        ln -sf "$NVDIR/$lib.so.4" "$NVDIR/$lib.so.3"
        echo "  Linked $NVDIR/$lib.so.3 -> $lib.so.4"
      else
        cp -a "$NVDIR/$lib.so.4" "$NVDIR/$lib.so.3"
        patchelf --set-soname "$lib.so.3" "$NVDIR/$lib.so.3"
        echo "  Created $NVDIR/$lib.so.3"
      fi
    fi
  done
else
  if [ "$PKG_MGR" = "dnf" ]; then
    echo "  Warning: $NVDIR not found. Install with (see CONTRIBUTING.md):"
    echo "    sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo"
    echo "    sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/hpc-sdk/rhel/nvhpc.repo"
    echo "    sudo dnf install libnccl libnccl-devel libcudnn9-cuda-12 libcudnn9-devel-cuda-12 nvshmem"
  else
    echo "  Warning: $NVDIR not found. Install with: apt-get update && apt-get install -y libnvshmem3-cuda-12"
  fi
  echo "  Then run this script again."
fi

# 3. (Debian only) Install NVRTC/cuDNN if libs missing
if [ "$PKG_MGR" = "apt" ]; then
  NVRTC_LIB="$(find /usr -name 'libnvrtc-builtins.so.12*' -type f 2>/dev/null | head -1)"
  if [ -z "$NVRTC_LIB" ]; then
    echo "Installing cuda-nvrtc-12-9 (libnvrtc-builtins.so.12)..."
    apt-get update -qq && apt-get install -y -qq cuda-nvrtc-12-9 || true
  fi
  CUDNN_LIB="$(find /usr -name 'libcudnn_engines_precompiled.so.9*' -type f 2>/dev/null | head -1)"
  if [ -z "$CUDNN_LIB" ]; then
    echo "Installing cudnn9-cuda-12-9..."
    apt-get update -qq && apt-get install -y -qq cudnn9-cuda-12-9 || true
  fi
fi

# 4. Register lib dirs with system loader
: > /etc/ld.so.conf.d/nvshmem.conf
echo "$NVDIR" >> /etc/ld.so.conf.d/nvshmem.conf
echo "$LIBDIR_EXTRA" >> /etc/ld.so.conf.d/nvshmem.conf
# CUDA paths (version may vary)
for cuda_base in /usr/local/cuda-12.9 /usr/local/cuda; do
  [ -d "$cuda_base/lib64" ] && echo "$cuda_base/lib64" >> /etc/ld.so.conf.d/nvshmem.conf
  [ -d "$cuda_base/targets/x86_64-linux/lib" ] && echo "$cuda_base/targets/x86_64-linux/lib" >> /etc/ld.so.conf.d/nvshmem.conf
done
CUDNN_LIB="$(find /usr -name 'libcudnn_engines_precompiled.so.9*' -type f 2>/dev/null | head -1)"
[ -n "$CUDNN_LIB" ] && echo "$(dirname "$CUDNN_LIB")" >> /etc/ld.so.conf.d/nvshmem.conf
NVRTC_LIB="$(find /usr -name 'libnvrtc-builtins.so.12*' -type f 2>/dev/null | head -1)"
[ -n "$NVRTC_LIB" ] && echo "$(dirname "$NVRTC_LIB")" >> /etc/ld.so.conf.d/nvshmem.conf
ldconfig

# 5. (Fedora) Profile script so LD_LIBRARY_PATH is set for non-root sessions (CUDA SONAME quirks)
if [ "$PKG_MGR" = "dnf" ]; then
  CUDA_TARGETS="/usr/local/cuda/targets/x86_64-linux/lib"
  [ -d /usr/local/cuda-12.9 ] && CUDA_TARGETS="/usr/local/cuda-12.9/targets/x86_64-linux/lib"
  echo "export LD_LIBRARY_PATH=$NVDIR:${CUDA_TARGETS}:\$LD_LIBRARY_PATH" > /etc/profile.d/nvshmem.sh
  echo "  Wrote /etc/profile.d/nvshmem.sh; run: source /etc/profile.d/nvshmem.sh (or log in again)"
fi

echo "Done. Run: ./scripts/run_with_exla_env.sh mix test"
