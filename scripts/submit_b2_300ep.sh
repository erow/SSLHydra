#!/bin/bash
# B2 (--hydra_pair rand) with iBOT loss + Hydra (--loss both), 8 local crops, scratch, 300 epochs.
# One job is limited to 2 days, so resubmit with DEP=<previous jobid> (afterany); each job resumes
# fp32 resume: STUDY_SRC=<clone with weights_only=False> USE_FP16=false
# from checkpoint.pth in the output dir (--load_from; a missing file starts fresh).
# Two nodes, same global batch 1024: NNODES=2 BATCH_SIZE_PER_GPU=64.
set -euo pipefail
OUT=/scratch/project_465003400/SSLHydra/outputs/ablation-vits-300ep/B2ibot_lc8_s0
cmd=(python src/main_ibot.py
  --arch vit_small --epochs 300 --warmup_epochs 10 --warmup_teacher_temp_epochs 30
  --lr 0.00005 --weight_decay 0.04 --weight_decay_end 0.04 --seed 0
  --batch_size_per_gpu "${BATCH_SIZE_PER_GPU:-128}" --momentum_teacher 0.996 --saveckp_freq 50
  --loss both --hydra_weight 1.0 --hydra_pair rand --hydra_k 8 --hydra_out_dim 2048 --hydra_mlp_dim 4096
  --hydra_temperature 0.2 --hydra_tau 0.1 --hydra_lr_scale 1.0
  --hydra_filter_negatives false --hydra_force_ones false
  ${USE_FP16:+--use_fp16 ${USE_FP16}}
  --local_crops_number 8
  --num_workers 10 --data_path /scratch/project_465003400/datasets/ffcv/IN1K_train_smart500_q90.ffcv
  --load_from checkpoint.pth --output_dir "${OUT}")
STUDY_SRC="${STUDY_SRC:-/scratch/project_465003400/SSLHydra/work/hydra-sim-k8-d2048/src}" \
PARTITION=standard-g TIME="${TLIM:-2-00:00:00}" JOB_NAME=b2-ibot-lc8 \
WANDB_TAGS="${WANDB_TAGS:-abl-B2-ibot-300ep}" WANDB_NAME=abl-vits-B2ibot-lc8-300ep \
DEPENDENCY="${DEP:+afterany:${DEP}}" \
  /project/project_465003400/SSLHydra/cluster/lumi_ddp.sh "${cmd[@]}"
