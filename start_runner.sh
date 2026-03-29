#!/usr/bin/env bash
set -euo pipefail

# Detect architecture
ARCH=$(uname -m)
if [[ "${ARCH}" == "aarch64" || "${ARCH}" == "arm64" ]]; then
  IMAGE_TAG="arm"
  RUNNER_ARCH="ARM64"
else
  IMAGE_TAG="latest"
  RUNNER_ARCH="X64"
fi

# Pull the latest image before starting the runner
echo "Pulling latest runner image..."
docker pull "gueraf/self_hosted_cuda_runner:${IMAGE_TAG}"

# Ensure jq is available (install if missing)
if ! command -v jq >/dev/null 2>&1; then
  echo "jq not found, attempting installation via apt..."
  if command -v apt >/dev/null 2>&1; then
    sudo apt update -y && sudo apt install -y jq
  else
    echo "apt not available; please install jq manually." >&2
    exit 1
  fi
fi

# Derive a short-lived registration token using the provided PAT (export GITHUB_ACCESS_TOKEN or PAT before calling)
PAT="${GITHUB_ACCESS_TOKEN:-${PAT:-}}"
if [[ -z "${PAT}" ]]; then
  echo "Missing PAT. Please set GITHUB_ACCESS_TOKEN or PAT in the environment." >&2
  exit 1
fi

echo "Requesting GitHub runner registration token..."
REG_TOKEN="$(curl -fsS -X POST \
  -H 'Accept: application/vnd.github+json' \
  -H "Authorization: token ${PAT}" \
  https://api.github.com/repos/gueraf/block_sparse_attn_wheels/actions/runners/registration-token | jq -r '.token')"

if [[ -z "${REG_TOKEN}" || "${REG_TOKEN}" == "null" ]]; then
  echo "Failed to obtain a runner registration token from GitHub." >&2
  exit 1
fi

echo "Obtained registration token: ${REG_TOKEN}"

# Remove existing container if it exists (force)
if docker ps -a --format '{{.Names}}' | grep -qx gh-runner; then
  echo "Removing existing gh-runner container..."
  docker rm -f gh-runner-bsa >/dev/null
fi

echo "Starting runner container..."
DOCKER_HOST_HOSTNAME="$(hostname)"
echo "Docker host hostname: ${DOCKER_HOST_HOSTNAME}"
echo "Runner will be registered as: ${DOCKER_HOST_HOSTNAME}"
docker run -d --restart=always --name gh-runner-bsa \
    --gpus all \
    --shm-size 8G \
    -e REPO_HTTPS_URL=https://github.com/gueraf/block_sparse_attn_wheels \
    -e REPO_TOKEN="${REG_TOKEN}" \
    -e RUNNER_NAME="${DOCKER_HOST_HOSTNAME}" \
    -e RUNNER_LABELS="self-hosted,Linux,${RUNNER_ARCH},gpu,${DOCKER_HOST_HOSTNAME}" \
    -e DOCKER_HOST_HOSTNAME="${DOCKER_HOST_HOSTNAME}" \
    "gueraf/self_hosted_cuda_runner:${IMAGE_TAG}"
