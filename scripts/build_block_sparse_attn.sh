#!/usr/bin/env bash
# Build and install Block-Sparse Attention (BSA) into the streamer venv.
#
# BSA provides CUDA kernels for sparse attention used by FlashVSR.
# It must be compiled from source because the kernels are architecture-specific.
#
# Prerequisites:
#   - CUDA toolkit with nvcc (>= 11.7) at $CUDA_HOME
#
# Usage:
#   bash scripts/build_block_sparse_attn.sh          # auto-detect GPU arch
#   BLOCK_SPARSE_ATTN_CUDA_ARCHS="120" bash scripts/build_block_sparse_attn.sh  # force arch
#
# Environment variables:
#   BLOCK_SPARSE_ATTN_CUDA_ARCHS  Semicolon-separated CUDA arch list (default: auto-detect)
#   CUDA_HOME                     Path to CUDA toolkit (default: /usr/local/cuda)
#   MAX_JOBS                      Max parallel nvcc compilation jobs (default: auto)
#   BSA_CLONE_DIR                 Where to clone BSA (default: /tmp/Block-Sparse-Attention)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BSA_CLONE_DIR="${BSA_CLONE_DIR:-/tmp/Block-Sparse-Attention}"
BSA_GIT_URL="https://github.com/mit-han-lab/Block-Sparse-Attention.git"

# ---------------------------------------------------------------------------
# Colours for output
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[BSA]${NC} $*"; }
warn()  { echo -e "${YELLOW}[BSA]${NC} $*"; }
error() { echo -e "${RED}[BSA]${NC} $*" >&2; }

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

cd "$PROJECT_DIR"

# Resolve the venv python early so all python invocations use the same interpreter.
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
    error "Venv python not found at $VENV_PYTHON — run 'uv sync' first."
    exit 1
fi

# Check if already installed
if "$VENV_PYTHON" -c "import block_sparse_attn; print(f'v{block_sparse_attn.__version__}')" 2>/dev/null; then
    info "Block-Sparse Attention is already installed."
    info "To force rebuild, uninstall first: uv pip uninstall block-sparse-attn"
    exit 0
fi

# Resolve CUDA_HOME
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    error "nvcc not found at $CUDA_HOME/bin/nvcc"
    error "Set CUDA_HOME to the CUDA toolkit directory containing bin/nvcc."
    exit 1
fi

NVCC_VERSION=$("$CUDA_HOME/bin/nvcc" --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
info "CUDA toolkit: $CUDA_HOME (nvcc $NVCC_VERSION)"

TORCH_CUDA=$("$VENV_PYTHON" -c "import torch; print(torch.version.cuda)")
info "PyTorch CUDA: $TORCH_CUDA"

# Auto-detect GPU compute capability if not overridden
if [ -z "${BLOCK_SPARSE_ATTN_CUDA_ARCHS:-}" ]; then
    GPU_ARCH=$("$VENV_PYTHON" -c "
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    arch = cap[0] * 10 + cap[1]
    if arch >= 120:
        archs = '80;90;100;120'
    elif arch >= 100:
        archs = '80;90;100'
    elif arch >= 90:
        archs = '80;90'
    else:
        archs = '80'
    print(archs)
else:
    print('80')
")
    export BLOCK_SPARSE_ATTN_CUDA_ARCHS="$GPU_ARCH"
    info "Auto-detected GPU arch(s): $BLOCK_SPARSE_ATTN_CUDA_ARCHS"
else
    info "Using user-specified arch(s): $BLOCK_SPARSE_ATTN_CUDA_ARCHS"
fi

# ---------------------------------------------------------------------------
# Clone or update BSA
# ---------------------------------------------------------------------------

if [ -d "$BSA_CLONE_DIR/.git" ]; then
    info "BSA repo already cloned at $BSA_CLONE_DIR"
    cd "$BSA_CLONE_DIR"
    git fetch --depth=1 origin main 2>/dev/null || true
    git reset --hard FETCH_HEAD
else
    info "Cloning Block-Sparse Attention..."
    git clone --depth=1 "$BSA_GIT_URL" "$BSA_CLONE_DIR"
    cd "$BSA_CLONE_DIR"
fi

# Ensure cutlass submodule is initialised
info "Initialising cutlass submodule..."
git submodule update --init csrc/cutlass

# ---------------------------------------------------------------------------
# Install build deps and build
# ---------------------------------------------------------------------------

cd "$PROJECT_DIR"

info "Installing build dependencies..."
uv pip install --python "$VENV_PYTHON" build packaging ninja psutil wheel setuptools

# Put venv bin on PATH so PyTorch's cpp_extension finds ninja
export PATH="$PROJECT_DIR/.venv/bin:$PATH"

# Verify ninja is visible
if "$VENV_PYTHON" -c "from torch.utils.cpp_extension import is_ninja_available; assert is_ninja_available()" 2>/dev/null; then
    NINJA_VERSION=$("$PROJECT_DIR/.venv/bin/ninja" --version 2>/dev/null || echo "?")
    info "Ninja build system: v$NINJA_VERSION (parallel compilation enabled)"
else
    error "Ninja not found — required for parallel compilation."
    error "Run: uv pip install ninja"
    exit 1
fi

# Auto-resolve MAX_JOBS: cores / 8, capped at 8.
# Each nvcc uses --threads 4 internally and several GB of RAM for templates.
if [ -z "${MAX_JOBS:-}" ]; then
    MAX_JOBS=$("$VENV_PYTHON" -c "import os; print(min(8, max(2, os.cpu_count() // 8)))")
fi
export MAX_JOBS

info "Building Block-Sparse Attention wheel (this may take 5-15 minutes)..."
info "  CUDA_HOME=$CUDA_HOME"
info "  BLOCK_SPARSE_ATTN_CUDA_ARCHS=$BLOCK_SPARSE_ATTN_CUDA_ARCHS"
info "  MAX_JOBS=$MAX_JOBS"

cd "$BSA_CLONE_DIR"

# Clean stale build artefacts
rm -rf dist/ build/ *.egg-info

"$VENV_PYTHON" -m build --wheel --no-isolation

BSA_WHEEL=$(ls -t dist/block_sparse_attn-*.whl 2>/dev/null | head -1)

if [ -z "$BSA_WHEEL" ]; then
    error "Wheel build failed — no .whl file found in $BSA_CLONE_DIR/dist/"
    error "Check the full build output above for errors."
    exit 1
fi

info "Built wheel: $BSA_WHEEL"

if [ "${BSA_SKIP_INSTALL:-1}" = "1" ]; then
    info "Skipping install (BSA_SKIP_INSTALL=1)."
    exit 0
fi

cd "$PROJECT_DIR"
info "Installing wheel into venv..."
uv pip install --python "$VENV_PYTHON" "$BSA_CLONE_DIR/$BSA_WHEEL"

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

BSA_VERSION=$("$VENV_PYTHON" -c "import block_sparse_attn; print(block_sparse_attn.__version__)" 2>/dev/null)
if [ -n "$BSA_VERSION" ]; then
    info "✓ Block-Sparse Attention v$BSA_VERSION installed successfully!"
else
    error "Installation verification failed — could not import block_sparse_attn"
    exit 1
fi

