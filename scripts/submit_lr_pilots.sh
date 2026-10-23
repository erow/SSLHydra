#!/bin/bash
# Step 1 of docs/runs.md: ViT-S 10-epoch LR pilots, Hydra loss and an iBOT-loss control.
#   scripts/submit_lr_pilots.sh            # print the sbatch commands only
#   SUBMIT=1 scripts/submit_lr_pilots.sh   # submit one single-node job per run
#   LRS="5e-5 1e-4" LOSSES="hydra" ...     # subset
# Everything except --lr and --loss is fixed so the runs are comparable.
set -euo pipefail

REPO=/project/project_465003400/SSLHydra
# Clone that has --hydra_lr_scale, --hydra_filter_negatives and --hydra_out_dim.
STUDY_SRC="${STUDY_SRC:-/scratch/project_465003400/SSLHydra/work/hydra-sim-k8-d2048/src}"
OUT=/scratch/project_465003400/SSLHydra/outputs/pilot-vits-10ep
DATA=/scratch/project_465003400/datasets/ffcv/IN1K_train_smart500_q90.ffcv
LRS="${LRS:-2.5e-5 5e-5 1e-4 2e-4}"
LOSSES="${LOSSES:-ibot hydra}"
MOMS="${MOMS:-0.9995}"   # momentum_teacher
HLRS="${HLRS:-0.1}"    # hydra_lr_scale
TEMPS="${TEMPS:-0.2}"   # hydra_temperature
WDS="${WDS:-0.04}"   # constant weight decay (start == end)
SEED="${SEED:-0}"
FILTER="${FILTER:-false}"   # the 10-epoch ViT-B ablation baseline used false

for loss in ${LOSSES}; do
  for lr in ${LRS}; do
   for wd in ${WDS}; do
    for temp in ${TEMPS}; do
     for hlr in ${HLRS}; do
      for mom in ${MOMS}; do
    name="${loss}_lr${lr}_wd${wd#0.}_s${SEED}"
    [[ "${temp}" == 0.2 ]] || name="${name}_t${temp#0.}"
    [[ "${hlr}" == 0.1 ]] || name="${name}_h${hlr#0.}"
    [[ "${mom}" == 0.9995 ]] || name="${name}_m${mom#0.}"
    cmd=(python src/main_ibot.py
      --arch vit_small --epochs 10 --warmup_epochs 2 --warmup_teacher_temp_epochs 2
      --lr "${lr}" --weight_decay "${wd}" --weight_decay_end "${wd}" --seed "${SEED}"
      --batch_size_per_gpu 128 --momentum_teacher "${mom}" --saveckp_freq 5
      --loss "${loss}" --hydra_pair sim --hydra_k 8 --hydra_out_dim 2048
      --hydra_temperature "${temp}" --hydra_tau 0.1 --hydra_lr_scale "${hlr}"
      --hydra_filter_negatives "${FILTER}"
      --num_workers 10 --data_path "${DATA}" --output_dir "${OUT}/${name}")
    echo "[${name}] ${cmd[*]}"
    if [[ "${SUBMIT:-0}" == 1 ]]; then
      STUDY_SRC="${STUDY_SRC}" PARTITION=standard-g TIME=04:00:00 \
        JOB_NAME="pilot-${loss:0:1}-${lr}-wd${wd}-t${temp}-h${hlr}-m${mom}" WANDB_TAGS="${WANDB_TAGS:-tune}" WANDB_NAME="pilot-vits-${name}" \
        "${REPO}/cluster/lumi_ddp.sh" "${cmd[@]}"
    fi
      done
     done
    done
   done
  done
done
