#!/bin/bash -l
# General-purpose PyTorch DDP on LUMI-G.
#
#   ./cluster/lumi_ddp.sh program [arg ...]
#   NNODES=2 TIME=48:00:00 JOB_NAME=train \
#     ./cluster/lumi_ddp.sh src/main_cl.py --data '$IMAGENET_ROOT' --epochs 100
#
# Three phases, same file:
#   1. Login node (no SLURM_JOB_ID): sbatch this script.
#   2. Compute node, outside the container: env, caches, dataset choice, singularity.
#   3. Inside the container (/.singularity.d): venv, stage data, torchrun the program.
#
# One Slurm task per node; torchrun forks one process per GCD (8 on a full node).
# A literal $IMAGENET_ROOT in the program arguments is replaced with the dataset
# path chosen inside the job (LMDB, else /tmp staged parquet, else ImageFolder).

# --- Slurm resources: one full LUMI-G node (8 GCDs, all RAM so /tmp can hold ImageNet) ---
#SBATCH --account=project_465003400
#SBATCH --partition=standard-g
#SBATCH --job-name=lumi-ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/project_465003400/SSLHydra/outputs/slurm/%x-%j.out
#SBATCH --error=/scratch/project_465003400/SSLHydra/outputs/slurm/%x-%j.err

set -euo pipefail

# Hardcoded: sbatch copies this file into the spool, so BASH_SOURCE is not the repo.
CLUSTER_DIR=/project/project_465003400/SSLHydra/cluster
SCRIPT="${CLUSTER_DIR}/lumi_ddp.sh"

usage() {
  echo "Usage: cluster/lumi_ddp.sh program [arg ...]" >&2
  echo "  NNODES=2 TIME=12:00:00 cluster/lumi_ddp.sh src/main_cl.py --data '\$IMAGENET_ROOT'" >&2
  exit 2
}

# Persistent Find DB lives on scratch ($MIOPEN_CACHE). The run itself uses a
# node-local copy because Lustre cannot lock the MIOpen DB. Call this only
# after the program has exited, so workers have flushed new kernels.
save_miopen() {
  [[ -n "${MIOPEN_USER_DB_PATH:-}" && -d "${MIOPEN_USER_DB_PATH}" ]] || return 0
  [[ -n "${MIOPEN_CACHE:-}" ]] || return 0
  mkdir -p "${MIOPEN_CACHE}"
  cp -a "${MIOPEN_USER_DB_PATH}/." "${MIOPEN_CACHE}/" || true
  echo "MIOPEN save ${MIOPEN_USER_DB_PATH} -> ${MIOPEN_CACHE}"
}

# =============================================================================
# 3. Inside the container
# /.singularity.d exists only in the runtime, so a leaked SINGULARITY_CONTAINER
# from --export=ALL cannot skip the host setup below.
# =============================================================================
if [[ -d /.singularity.d || -d /.apptainer.d ]]; then
  # Host phase exported these. Fail here if a bind or env handoff dropped one.
  : "${PROJECT_ROOT:?}"
  : "${VENV_PATH:?}"
  : "${IMAGENET_ROOT:?}"
  : "${MIOPEN_CACHE:?}"
  : "${MIOPEN_USER_DB_PATH:?}"
  [[ $# -ge 1 ]] || usage

  # Overlay venv on the image's PyTorch. Do not activate: some activate scripts
  # pin VIRTUAL_ENV to another cluster.
  export PATH="${VENV_PATH}/bin:${PATH}"
  export VIRTUAL_ENV="${VENV_PATH}"
  unset PYTHONHOME
  export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  # Study jobs read a committed clone, copied onto this node's tmpfs.
  if [[ -n "${STUDY_SRC:-}" ]]; then
    rm -rf /tmp/SSLHydra
    mkdir -p /tmp/SSLHydra/src
    cp -a "${STUDY_SRC}/." /tmp/SSLHydra/src/
    export PROJECT_ROOT=/tmp/SSLHydra
    export PYTHONPATH="/tmp/SSLHydra/src${PYTHONPATH:+:${PYTHONPATH}}"
    echo "CODE PROJECT_ROOT=/tmp/SSLHydra from ${STUDY_SRC}"
  fi
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  # DataLoader file_system shm. Path was chosen on the host; create it on this node.
  mkdir -p "${TMPDIR}"

  # Parquet → /tmp/IN1K-jpeg. Each node stages its own tmpfs (multi-node included).
  if [[ "${STAGE_IN1K:-0}" == "1" ]]; then
    bash "${CLUSTER_DIR}/jobs/stage_in1k_tmp.sh"
  fi

  # torchrun supplies the interpreter. Accept a leading python so both
  # "script.py" and "python script.py" / "python -m pkg" work.
  if [[ "$1" == python || "$1" == python3 || "$1" == */python || "$1" == */python3 ]]; then
    shift
    [[ $# -ge 1 ]] || usage
  fi

  # Pass the program's arguments through. Substitute the dataset path chosen
  # on the host; the login shell must not expand $IMAGENET_ROOT (quote it).
  ARGS=()
  for arg in "$@"; do
    ARGS+=("${arg//\$IMAGENET_ROOT/${IMAGENET_ROOT}}")
  done

  # DDP: one torchrun per node, nproc = GCDs. Single node uses --standalone.
  # Multi-node meets at RDZV_HOST (rank 0), which the host phase exported.
  cd "${PROJECT_ROOT}"
  NPROC="${NPROC:-${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-8}}}"
  NNODES="${NNODES:-${SLURM_NNODES:-1}}"
  LAUNCH=(--standalone --nnodes=1 --nproc_per_node="${NPROC}")
  if [[ "${NNODES}" -gt 1 ]]; then
    : "${RDZV_HOST:?RDZV_HOST required for multi-node}"
    LAUNCH=(
      --nnodes="${NNODES}"
      --nproc_per_node="${NPROC}"
      --rdzv_backend=c10d
      --rdzv_id="${RDZV_ID:-${SLURM_JOB_ID:-lumi-ddp}}"
      --rdzv_endpoint="${RDZV_HOST}:${RDZV_PORT:-29400}"
      --rdzv_conf=timeout=1800
    )
  fi

  # ffcv's libtiff needs jpeg12_* from this libjpeg. ROCm loads the image
  # libjpeg first, and that one does not export the symbol.
  export LD_PRELOAD="/scratch/project_465003400/cache/vitookit/opencv/lib/libjpeg.so.8${LD_PRELOAD:+:${LD_PRELOAD}}"

  echo "DDP_LAUNCH nnodes=${NNODES} nproc=${NPROC} data=${IMAGENET_ROOT} cmd=${ARGS[*]}"
  # with_miopen.sh stages the scratch DB onto /tmp and also saves on its own
  # EXIT. Save again here, after it returns, so this script still refreshes
  # $MIOPEN_CACHE once every rank has exited.
  rc=0
  bash "${CLUSTER_DIR}/jobs/with_miopen.sh" \
    "${VENV_PATH}/bin/python" -m torch.distributed.run "${LAUNCH[@]}" "${ARGS[@]}" || rc=$?
  save_miopen
  exit "${rc}"
fi

# =============================================================================
# 1. Login node: submit. Flags after -- belong to the program, not to sbatch.
# NNODES, TIME, and JOB_NAME override the #SBATCH defaults above.
# =============================================================================
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  [[ $# -ge 1 ]] || usage
  # shellcheck disable=SC1091
  source "${CLUSTER_DIR}/lumi.env"
  # The interpreter symlink points at /usr/bin/python3.12 inside the image, so
  # -x fails on the login node even when the venv is valid.
  if [[ ! -f "${VENV_PATH}/pyvenv.cfg" ]]; then
    echo "Missing venv ${VENV_PATH}. Run: bash ${CLUSTER_DIR}/setup_lumi_env.sh" >&2
    exit 2
  fi
  mkdir -p "${OUTPUT_ROOT}/slurm"
  echo "Submitting ${JOB_NAME:-lumi-ddp} nodes=${NNODES:-1} time=${TIME:-24:00:00} cmd=$*"
  dep=()
  [[ -z "${DEPENDENCY:-}" ]] || dep=(--dependency="${DEPENDENCY}")
  exec sbatch \
    ${dep[@]+"${dep[@]}"} \
    --job-name="${JOB_NAME:-lumi-ddp}" \
    --time="${TIME:-24:00:00}" \
    --nodes="${NNODES:-1}" \
    --account="${SBATCH_ACCOUNT}" \
    --partition="${PARTITION:-${SBATCH_PARTITION}}" \
    --ntasks-per-node=1 \
    --gpus-per-node="${GPUS:-${SBATCH_GPUS_PER_NODE}}" \
    --cpus-per-task="${CPUS:-${SBATCH_CPUS_PER_TASK}}" \
    --mem="${MEM:-0}" \
    --output="${OUTPUT_ROOT}/slurm/%x-%j.out" \
    --error="${OUTPUT_ROOT}/slurm/%x-%j.err" \
    --export=ALL \
    -- "${SCRIPT}" "$@"
fi

# =============================================================================
# 2. Compute node, outside the container
# =============================================================================

# Environment. lumi.env is the source of truth. Unset first so an Isambard or
# Eureka path carried by --export=ALL cannot win the ${VAR:-default} lines.
unset IMAGENET_ROOT IMAGENET_SQSH IMAGENET_LMDB HF_IN1K_CACHE MIOPEN_CACHE
set -a
# shellcheck disable=SC1091
source "${CLUSTER_DIR}/lumi.env"
set +a
# A study can isolate itself from a stale shared MIOpen DB without changing
# lumi.env for every other job.
if [[ -n "${MIOPEN_CACHE_OVERRIDE:-}" ]]; then
  export MIOPEN_CACHE="${MIOPEN_CACHE_OVERRIDE}"
fi

# Scratch dirs for logs and framework caches. $HOME is 20 GB; nothing heavy goes there.
mkdir -p "${OUTPUT_ROOT}/slurm" "${JOB_TMP_BASE}" \
  "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" "${MIOPEN_CACHE}"

# Cache paths.
# TMPDIR is node tmpfs: DataLoader file_system shm dies on Lustre.
export TMPDIR="/tmp/SSLHydra-${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
export TMP="${TMPDIR}" TEMP="${TMPDIR}"
# HF, torch, Triton, matplotlib, pip. MIOPEN_CACHE (persistent copy) is on scratch.
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"
# FFCV's Numba pipeline. Must be set before Python imports numba.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/scratch/project_465003400/cache/numba}"
mkdir -p "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"
# MIOpen locks its Find DB. Lustre has no usable locks, so the live DB is
# per-job /tmp. save_miopen writes it back to $MIOPEN_CACHE after the program.
export MIOPEN_USER_DB_PATH="/tmp/${USER:-u}-miopen-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"

# Rank 0 logging, one thread per rank, RCCL chatter, Slurm's GPU binding.
# If ROCR_VISIBLE_DEVICES is set, CUDA_VISIBLE_DEVICES makes PyTorch see the wrong devices.
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-hydra}"
export WANDB_ENTITY="${WANDB_ENTITY:-erow}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
  unset CUDA_VISIBLE_DEVICES
fi

# DDP rendezvous. Rank 0's hostname is the c10d endpoint. torchrun reads these
# inside the container; they are not Slurm RANK/LOCAL_RANK (those are unset
# because this job is one task per node).
NNODES="${SLURM_NNODES:-1}"
export NNODES
# Default memhooks monitor hangs multi-node jobs inside these containers.
if [[ "${NNODES}" -gt 1 ]]; then
  export FI_MR_CACHE_MONITOR="${FI_MR_CACHE_MONITOR:-userfaultfd}"
fi
RDZV_HOST="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
export RDZV_HOST
export RDZV_PORT="${RDZV_PORT:-29400}"
export RDZV_ID="${SLURM_JOB_ID}"

# Container. The bindings module attaches Slingshot and the work directory.
module purge
module load Local-LAIF lumi-aif-singularity-bindings
SINGULARITY_BIN="$(command -v singularity)"
: "${SINGULARITY_BIN:?singularity not on PATH after module load}"

# Dataset. LMDB is two files on Lustre. Else stage the HF parquet cache onto
# tmpfs inside the container. Do not loop-mount the squashfs: the kernel cache
# serializes readers. A plain train/ tree is the slow fallback.
BIND_ARGS=(--bind "${PROJECT_ROOT},${SCRATCH_ROOT},/scratch/project_465003400,/tmp")
IN1K_LOCAL="${IN1K_LOCAL:-/tmp/IN1K-jpeg}"
STAGE_IN1K=0
if [[ -f "${IMAGENET_LMDB}/train/data.mdb" ]]; then
  IMAGENET_ROOT="${IMAGENET_LMDB}"
  echo "LMDB ${IMAGENET_LMDB}"
elif [[ -d "${HF_IN1K_CACHE}/snapshots" ]]; then
  IMAGENET_ROOT="${IN1K_LOCAL}"
  STAGE_IN1K=1
  echo "STAGE parquet -> ${IN1K_LOCAL}"
elif [[ -d "${IMAGENET_ROOT}/train" ]]; then
  echo "ImageFolder ${IMAGENET_ROOT}"
else
  echo "WARN no LMDB, HF parquet, or ${IMAGENET_ROOT}/train" >&2
fi
export IMAGENET_ROOT IN1K_LOCAL STAGE_IN1K HF_IN1K_CACHE

[[ $# -ge 1 ]] || usage
echo "DDP_JOB ${SLURM_JOB_ID} host=$(hostname) nnodes=${NNODES} rdzv=${RDZV_HOST}:${RDZV_PORT} data=${IMAGENET_ROOT}"
echo "HF_HOME=${HF_HOME} TORCH_HOME=${TORCH_HOME} XDG_CACHE_HOME=${XDG_CACHE_HOME} MIOPEN_CACHE=${MIOPEN_CACHE}"
echo "CMD: $*"

# Re-exec this file inside the image. Phase 3 sees /.singularity.d and runs torchrun.
LAUNCH_SIF=(
  "${SINGULARITY_BIN}" exec
  "${BIND_ARGS[@]}"
  "${SIF}"
  bash "${SCRIPT}" "$@"
)

# Single node: singularity from this task. Multi-node: one srun task per node,
# same command, --cpu-bind=none so torchrun can place the eight ranks.
rc=0
if [[ "${NNODES}" -gt 1 ]]; then
  srun --label --mpi=none --unbuffered --export=ALL \
    --ntasks="${NNODES}" --ntasks-per-node=1 \
    --gpus-per-node="${SLURM_GPUS_PER_NODE:-8}" --cpu-bind=none \
    "${LAUNCH_SIF[@]}" || rc=$?
else
  "${LAUNCH_SIF[@]}" || rc=$?
fi
# Container exit already saved each node's DB. Copy this node's /tmp again so
# a killed container (trap did not run) still updates the scratch cache.
save_miopen
echo "DDP_DONE exit=${rc} $(date -Is)"
exit "${rc}"
