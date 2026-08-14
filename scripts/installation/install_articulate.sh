#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Error handling: exit and print the offending line on failure.
# Resolve the script path up front: $0 is relative and the script cd's around.
SELF_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
error_handler() {
    local exit_code=$?
    local line_no=$1
    echo "Error occurred at line $line_no: $(sed "${line_no}q;d" "$SELF_PATH")"
    echo "Exit code: $exit_code"
    exit $exit_code
}
trap 'error_handler $LINENO' ERR
set -o errexit
set -o pipefail

if ! command -v mamba >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Error: mamba was not found on PATH.

Install Miniforge, or install mamba into your base conda
environment, then rerun this script.
EOF
  exit 127
fi

eval "$(mamba shell hook --shell bash)"

# Get script dir
# repo dir is grandparent directory, by default
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
source "${SCRIPT_DIR}/git_safe.sh"
project_root="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT=false
# An exported PROJECT_ROOT must not silently override --project-root; note and clear.
INHERITED_PROJECT_ROOT="${PROJECT_ROOT:-}"
unset PROJECT_ROOT

# Parse command-line options
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-root) project_root="$2"; shift 2 ;;
        --default) DEFAULT=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ! ${DEFAULT} == true ]]; then
  read -p "Enter project root (default: $project_root): " PROJECT_ROOT
fi
PROJECT_ROOT=${PROJECT_ROOT:-$project_root}
if [[ -n "${INHERITED_PROJECT_ROOT}" && "${INHERITED_PROJECT_ROOT}" != "${PROJECT_ROOT}" ]]; then
  echo "NOTE: ignoring PROJECT_ROOT='${INHERITED_PROJECT_ROOT}' exported in the environment; using '${PROJECT_ROOT}' (pass --project-root to change)."
fi

# ==============================================================================
# ARTICULATE ENVIRONMENT SETUP
# ==============================================================================

echo "=== Articulate Environment Setup ==="

cd $PROJECT_ROOT  # Change to your project root

# Public articulate-anything fork (SimFoundry release). Large assets
# (embeddings) are stored in Git LFS, so git-lfs must be installed (handled above).
export ARTICULATE_ANYTHING_REPO="${ARTICULATE_ANYTHING_REPO:-https://github.com/nadunRanawaka1/articulate-anything-sf.git}"

# ==============================================================================
# SYSTEM DEPENDENCIES (run first)
# ==============================================================================

# Collect what is missing first so machines with everything present never need sudo.
# git-lfs fetches the LFS assets; ffmpeg + X/GL libs are the Blender runtime.
# Do NOT use the Ubuntu 'blender' apt package.
APT_MISSING=()
command -v git-lfs >/dev/null 2>&1 || APT_MISSING+=(git-lfs)
command -v ffmpeg  >/dev/null 2>&1 || APT_MISSING+=(ffmpeg)
for pkg in libxrender1 libxi6 libxxf86vm1 libxfixes3 libxkbcommon0 libsm6 libgl1; do
  dpkg -s "$pkg" >/dev/null 2>&1 || APT_MISSING+=("$pkg")
done

# The Blender install below needs sudo too; decide now so the preflight covers it.
# Override BLENDER_VERSION to pin a different release.
BLENDER_VERSION="${BLENDER_VERSION:-4.2.3}"
BLENDER_SERIES="${BLENDER_VERSION%.*}"                 # e.g. 4.2.3 -> 4.2
BLENDER_INSTALL_DIR="${BLENDER_INSTALL_DIR:-/opt/blender-${BLENDER_VERSION}}"
NEED_BLENDER=true
if blender --version 2>/dev/null | grep -q "Blender ${BLENDER_VERSION}"; then
  NEED_BLENDER=false
fi

if [ "${#APT_MISSING[@]}" -gt 0 ] || [ "${NEED_BLENDER}" = true ]; then
  SUDO_FOR="${APT_MISSING[*]}"
  if [ "${NEED_BLENDER}" = true ]; then
    SUDO_FOR="${SUDO_FOR:+${SUDO_FOR} }blender-${BLENDER_VERSION}"
  fi
  echo "=== Installing system dependencies: ${SUDO_FOR} (this step may prompt for your sudo password) ==="
  # Prime sudo up front; fail with a clear message when it cannot prompt (headless run).
  if ! sudo -v; then
    echo "ERROR: sudo is required to install: ${SUDO_FOR}" >&2
    echo "       Install them yourself, or rerun this script from a terminal where sudo can prompt." >&2
    exit 1
  fi
  if [ "${#APT_MISSING[@]}" -gt 0 ]; then
    sudo apt-get install -y "${APT_MISSING[@]}"
  fi
else
  echo "=== System dependencies already present — skipping apt-get ==="
fi
git lfs install

echo "=== Installing Blender (official blender.org build) ==="
# NEED_BLENDER and the BLENDER_* vars are set above, where the sudo preflight runs.
if [ "${NEED_BLENDER}" = false ]; then
  echo "Blender ${BLENDER_VERSION} already installed, skipping download."
else
  BLENDER_TARBALL="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
  BLENDER_URL="https://download.blender.org/release/Blender${BLENDER_SERIES}/${BLENDER_TARBALL}"
  TMP_BLENDER="$(mktemp -d)"
  wget -O "${TMP_BLENDER}/${BLENDER_TARBALL}" "${BLENDER_URL}"
  sudo rm -rf "${BLENDER_INSTALL_DIR}"
  sudo mkdir -p "${BLENDER_INSTALL_DIR}"
  sudo tar -xf "${TMP_BLENDER}/${BLENDER_TARBALL}" -C "${BLENDER_INSTALL_DIR}" --strip-components=1
  # Symlink into /usr/local/bin (takes precedence over any /usr/bin/blender).
  sudo ln -sf "${BLENDER_INSTALL_DIR}/blender" /usr/local/bin/blender
  rm -rf "${TMP_BLENDER}"
  hash -r
  echo "Blender installed: $(blender --version 2>/dev/null | head -1)"
fi

# ==============================================================================
# REPOSITORIES & CONDA ENVIRONMENTS (long-running, unattended)
# ==============================================================================

if [ ! -d "deps" ]; then
  mkdir deps
fi
cd deps

if [ ! -d "articulate-anything" ]; then
  git clone -b main "${ARTICULATE_ANYTHING_REPO}" articulate-anything
fi

cd articulate-anything
# Track the latest main, but never discard local work: a dirty/diverged checkout is
# left as-is with a warning.
git_safe_sync_branch "." origin main "deps/articulate-anything"

bash installation_hunyuan.sh   # create hunyuan environment
bash installation_partfield.sh # create partfield environment

# ==============================================================================
# LIBIGL (watertight mesh conversion, required in hunyuan env)
# ==============================================================================

echo "=== Installing libigl in articulate-anything-hunyuan ==="
mamba activate articulate-anything-hunyuan
conda install -c conda-forge igl -y
mamba deactivate

# ==============================================================================
# DOWNLOAD MODEL CHECKPOINTS
# ==============================================================================

echo "=== Downloading P3-SAM weights ==="
P3SAM_WEIGHTS_DIR="deps/Hunyuan3D-Part/P3-SAM/weights"
mkdir -p $P3SAM_WEIGHTS_DIR
if [ ! -f "$P3SAM_WEIGHTS_DIR/p3sam.safetensors" ]; then
    wget -O $P3SAM_WEIGHTS_DIR/p3sam.safetensors \
        https://huggingface.co/tencent/Hunyuan3D-Part/resolve/main/p3sam/p3sam.safetensors
else
    echo "P3-SAM weights already present, skipping download."
fi

