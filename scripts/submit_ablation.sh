#!/bin/bash
# Stage 1 of docs/ablation.md: ViT-S, 10 epochs, seed 0, B0 = recommended settings.
#   ROWS="A1 B1" scripts/submit_ablation.sh           # print commands only
#   SUBMIT=1 ROWS=A1 WANDB_TAGS=abl-A1 scripts/submit_ablation.sh
# Each row is B0 plus one change. Training jobs are submitted first; one kNN job
# (dev-g, k=10 20 50) is queued behind all of them with afterok.
set -euo pipefail

REPO=/project/project_465003400/SSLHydra
STUDY_SRC="${STUDY_SRC:-/scratch/project_465003400/SSLHydra/work/hydra-sim-k8-d2048/src}"
# STAGE=2 runs 100 epochs (5 warmup, default 30 teacher-temp warmup) into ablation-vits-100ep.
STAGE="${STAGE:-1}"
if [[ "${STAGE}" == 2 ]]; then
  EPOCHS=100; WARMUP=5; TWARM=30; SAVE=20; TLIM="${TLIM:-1-12:00:00}"
  OUT="${OUT:-/scratch/project_465003400/SSLHydra/outputs/ablation-vits-100ep}"
else
  EPOCHS=10; WARMUP=2; TWARM=2; SAVE=5; TLIM="${TLIM:-04:00:00}"
  OUT="${OUT:-/scratch/project_465003400/SSLHydra/outputs/ablation-vits}"
fi
DATA=/scratch/project_465003400/datasets/ffcv/IN1K_train_smart500_q90.ffcv
IBOT_VITS=/scratch/project_465003400/cache/torch/ibot/vits_16_checkpoint.pth
SEED="${SEED:-0}"
ROWS="${ROWS:-A1 A2 A3 B1 B2 B3 B4 B5 C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 D1 D2 D3 D4 D5 D6 D7}"
TAGS="${WANDB_TAGS:-ablation}"
# Extra name=path pairs for the kNN job, e.g. KNN_EXTRA="random=random B0=/path/checkpoint0010.pth".
KNN_EXTRA="${KNN_EXTRA:-}"

declare -A X=(
  [B0]=""
  [A1]="--init_ibot ${IBOT_VITS}"
  [A2]="--loss both --hydra_weight 1.0"
  [A3]="--loss both --hydra_weight 0.3"
  [B1]="--hydra_force_ones true"
  [B2]="--hydra_pair rand"
  [B2i]="--hydra_pair rand --init_ibot ${IBOT_VITS}"
  [B2i-ibot]="--loss ibot --local_crops_number 0 --init_ibot ${IBOT_VITS}"
  [B3]="--hydra_pair self"
  [B4]="--hydra_pair label"
  [B4x]="--hydra_pair label"   # needs own-view exclusion + SupCon average: STUDY_SRC=work/hydra-label-exclude-20261013/src
  [B5]="--hydra_pair rand --hydra_force_ones true"
  [C1]="--hydra_k 1"   [C2]="--hydra_k 4"   [C3]="--hydra_k 16"
  [C4]="--hydra_tau 0.05" [C5]="--hydra_tau 0.2" [C6]="--hydra_tau 0.5"
  [C7]="--hydra_filter_negatives true"
  [C8]="--hydra_out_dim 768" [C9]="--hydra_out_dim 1024" [C10]="--hydra_out_dim 4096"
  [C11]="--hydra_mlp_dim 1024"
  [D1]="--lr 0.000025" [D2]="--lr 0.0001"
  [D3]="--hydra_temperature 0.1" [D4]="--hydra_temperature 0.4"
  [D5]="--momentum_teacher 0.99" [D6]="--momentum_teacher 0.999"
  [D7]="--weight_decay 0.01 --weight_decay_end 0.01"
)

jobs=(); ckpts=()
for id in ${ROWS}; do
  [[ -v "X[$id]" ]] || { echo "unknown row ${id}" >&2; exit 2; }
  # B0 defaults come first; the row's flags come last and override them (argparse keeps the last).
  cmd=(python src/main_ibot.py
    --arch vit_small --epochs ${EPOCHS} --warmup_epochs ${WARMUP} --warmup_teacher_temp_epochs ${TWARM}
    --lr 0.00005 --weight_decay 0.04 --weight_decay_end 0.04 --seed "${SEED}"
    --batch_size_per_gpu 128 --momentum_teacher 0.996 --saveckp_freq ${SAVE}
    --loss hydra --hydra_pair sim --hydra_k 8 --hydra_out_dim 2048 --hydra_mlp_dim 4096
    --hydra_temperature 0.2 --hydra_tau 0.1 --hydra_lr_scale 1.0
    --hydra_filter_negatives false --hydra_force_ones false
    --num_workers 10 --data_path "${DATA}"
    ${X[$id]}
    ${USE_FP16:+--use_fp16 ${USE_FP16}} ${LOAD_FROM:+--load_from ${LOAD_FROM}}
    --output_dir "${OUT}/${id}_s${SEED}")
  echo "[${id}] ${cmd[*]}"
  ckpts+=("${id}=${OUT}/${id}_s${SEED}/checkpoint$(printf %04d ${EPOCHS}).pth")
  if [[ "${SUBMIT:-0}" == 1 ]]; then
    out=$(STUDY_SRC="${STUDY_SRC}" PARTITION=standard-g TIME="${TLIM}" JOB_NAME="abl${STAGE/1/}-${id}" \
      WANDB_TAGS="${TAGS}" WANDB_NAME="abl-vits-${id}_s${SEED}$([[ ${STAGE} == 2 ]] && echo _100ep)" "${REPO}/cluster/lumi_ddp.sh" "${cmd[@]}")
    echo "${out}"
    jobs+=("$(grep -oE '[0-9]+' <<<"${out}" | tail -1)")
  fi
done

knn=(python scripts/knn_eval_ffcv.py --arch vit_small --nb_knn 10 20 50
  --output "${OUT}/knn_${TAGS}.json"
  --checkpoints ${KNN_EXTRA:+${KNN_EXTRA}} "${ckpts[@]}")
echo "[knn] ${knn[*]}"
if [[ "${SUBMIT:-0}" == 1 ]]; then
  dep=$(IFS=:; echo "afterok:${jobs[*]}")
  env -u STUDY_SRC PARTITION=dev-g TIME=02:00:00 JOB_NAME="abl-knn-${TAGS}" DEPENDENCY="${dep}" \
    "${REPO}/cluster/lumi_ddp.sh" "${knn[@]}"
fi
