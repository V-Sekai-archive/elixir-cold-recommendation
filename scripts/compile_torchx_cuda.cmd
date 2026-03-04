@echo off
:: Compile torchx with CUDA (cu129). Run from project root.
:: Uses VS Build Tools + CUDA env so CMake finds the CUDA toolset.

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set LIBTORCH_TARGET=cu129
set CUDA_PATH=%CUDA_ROOT%
set CUDAToolkit_ROOT=%CUDA_ROOT%
set PATH=%CUDA_ROOT%\bin;%PATH%

call "%VCVARS%"
if errorlevel 1 (
  echo vcvars64.bat failed
  exit /b 1
)

cd /d "%~dp0.."
mix deps.compile torchx --force
