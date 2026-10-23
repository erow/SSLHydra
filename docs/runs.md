# How to run the code

```bash
STUDY_SRC=/scratch/project_465003400/SSLHydra/work/hydra-sim-k8-d2048/src \
PARTITION=standard-g TIME=2-00:00:00 \
JOB_NAME=hydra-sim \
./cluster/lumi_ddp.sh python src/main_ibot.py \
  --arch vit_base --epochs 100 --warmup_epochs 5 --lr 0.00005 --weight_decay 0.04 --weight_decay_end 0.04 \
  --momentum_teacher 0.996 --loss hydra --hydra_pair sim --hydra_k 8 --hydra_out_dim 2048 --hydra_temperature 0.2 --hydra_tau 0.1 \
  --hydra_lr_scale 1.0 --hydra_filter_negatives false \
  --num_workers 10 \
  --data_path /scratch/project_465003400/datasets/ffcv/IN1K_train_smart500_q90.ffcv \
  --output_dir /scratch/project_465003400/SSLHydra/outputs/hydra-sim
```



# Parameter tuning

Tune with `vit_small`. Use 10-epoch pilots with 2 warmup epochs to reject unstable settings, then train the selected settings for 100 epochs with 5 warmup epochs. Keep the dataset, global batch size, seed, Hydra pair mode, `hydra_k`, `hydra_out_dim`, and `hydra_tau` fixed across comparisons. Give every run its own `--output_dir`. If starting from a checkpoint, use the same checkpoint and initialization flags for every candidate; compare scratch and pretrained starts separately. An iBOT ViT-S/16 pretrained checkpoint is available [here](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint.pth).

The pilots started from `--lr 0.0001 --weight_decay 0.04 --weight_decay_end 0.04 --hydra_temperature 0.2 --momentum_teacher 0.9995 --hydra_lr_scale 0.1`; the tuned values are in [Tuning results](#tuning-results-vit-s-10-epoch-pilots) below. The earlier 10-epoch ViT-B ablation used a higher effective backbone LR and 1 warmup epoch; most runs became NaN or collapsed, so its temperature and Hydra LR comparisons do not establish a winner.


| Parameter                                    | Pilot values                  | What changes                                                                                                                                                                                                                 |
| -------------------------------------------- | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--lr`                                       | `0.000025`, `0.00005`, `0.0001`, `0.0002` | Backbone peak LR at the reference batch size of 256, scaled by `global_batch / 256` (batch 1024 here, so effective peak is 4x). `0.0005` (effective 0.002) is the setting that NaN'd 14 of 16 ViT-B runs and is left out. Keep global batch fixed. |
| `--weight_decay` / `--weight_decay_end` (WD) | `0.01`, `0.04`, `0.1`         | Set both flags to the same candidate value in 10-epoch pilots so the cosine WD schedule stays constant. Tune a rising end value only after finding a stable baseline.                                                        |
| `--hydra_temperature`                        | `0.2`, `0.4`, `0.6`           | InfoNCE temperature; separate from partner-sampling `--hydra_tau`.                                                                                                                                                           |
| `--momentum_teacher`                         | `0.996`, `0.9995`, `0.9999`   | Initial teacher EMA momentum; the scheduler raises it toward 1.                                                                                                                                                              |
| `--hydra_lr_scale`                           | `0.1`, `0.3`, `1.0`           | Multiplies the scheduled LR for Hydra parameters only; `1.0` gives Hydra the same scheduled LR as the backbone. The backbone LR is unchanged.                                                                                |


First compare the three LR values at baseline settings and retain the most stable one. At that LR, compare the three WD values. Then vary temperature, teacher momentum, and Hydra LR scale one parameter at a time around the best stable setting. If no candidate remains stable for 10 epochs, lower LR before interpreting the other sweeps. Rerun the leading settings for 100 epochs, then repeat the finalists with another seed.

Use `log.txt` in each output directory to compare finite loss, `train_cls_cos`, `train_cls_pair_cos`, `train_zs_std`, and `train_act`. Reject NaN runs and runs whose `cls_cos` rises toward 1: a falling loss alone can accompany backbone collapse. Prefer runs with `cls_pair_cos` above the unpaired similarity, nonzero representation spread, and stable gate activity. Compare representation quality with the same downstream evaluation for each finalist.

## Pilot protocol additions

- Fixed for every pilot: global batch 1024 (`--batch_size_per_gpu 128` on 8 GCDs), `--seed 0`, `--hydra_filter_negatives false` (the ViT-B ablation baseline), `--warmup_teacher_temp_epochs 2` (the default 30 exceeds 10 epochs and crashes the loss schedule), `--saveckp_freq 5`.
- Include an `--loss ibot` control at each LR with the same settings. If it also NaNs or collapses, the recipe is the problem, not Hydra.
- `STUDY_SRC` must be a clone that has `--hydra_lr_scale`; `work/hydra-sim` predates it. Launch Step 1 with `scripts/submit_lr_pilots.sh` (`SUBMIT=1`, `WANDB_TAGS=tune`).
- Select among survivors by kNN (see the kNN notes) on `checkpoint0005`/`checkpoint0010`, with an untrained ViT-S as the floor. Differences of 1-2 points are noise at one seed; rerun the top 2-3 with a second seed.
- Reject when: NaN; `cls_cos` above 0.95 at epoch 3 or still rising at epoch 10; `cls_pair_cos` not above `cls_cos`; `zs_std` near 0.
- `utils.clip_gradients` still clips per tensor, not by global norm, and `gnorm=Infinity` has been seen. Treat clipping as unfixed when interpreting NaNs.

## Tuning results (ViT-S, 10-epoch pilots)

Weighted kNN top-1 (%) on ImageNet-1k, teacher [CLS], k=20 / k=200, `checkpoint0010`, seed 0 unless noted. Untrained ViT-S: 1.37 / 1.83. Pilot flags as in the protocol above (batch 1024, 2 warmup epochs, `hydra_k 8`, `hydra_out_dim 2048`, `hydra_tau 0.1`, no negative filtering). Each sweep varies one parameter from a base of LR 5e-5, WD 0.04, temperature 0.2, momentum 0.9995, Hydra LR scale 0.1, except where noted.

### Recommended settings

| Parameter | Value | Evidence | Confidence |
|---|---|---|---|
| `--lr` | `0.00005` | best of 4; `0.0001` within about 0.5; `0.0002` degrades; `0.0005` NaN'd in ViT-B | high |
| `--momentum_teacher` | `0.996` | 13.65 / 14.20 vs 7.74 / 8.68 at 0.9995; 0.9999 gives 3.41 / 4.25 | high on direction, optimum not bracketed |
| `--hydra_lr_scale` | `1.0` | 7.74 / 8.68 vs 6.40 / 7.25 at 0.1; 0.3 is 7.68 / 8.51 | high |
| `--hydra_temperature` | `0.2` (`0.1` is a candidate) | 0.1 gives +0.3 to +0.4 at both Hydra LR scales; 0.4 and 0.6 are worse | low, near seed noise |
| `--weight_decay` / `--weight_decay_end` | `0.04` / `0.04` | 0.01 to 0.1 differ by 0.5 or less | insensitive |
| `--hydra_filter_negatives` | `false` | not tuned; the pilots used false | untested |

### Sweep results

| Sweep | Value | k=20 | k=200 | Notes |
|---|---|---|---|---|
| LR | 2.5e-5 | 5.25 | 6.24 | |
| LR | **5e-5** | **6.40** | **7.25** | |
| LR | 1e-4 | 5.88 | 7.10 | |
| LR | 2e-4 | 0.90 | 1.35 | loss turns up after epoch 2, `cls_cos` back to 0.61 |
| WD (5e-5) | 0.01 / 0.04 / 0.1 | 6.71 / 6.40 / 6.19 | 7.60 / 7.25 / 7.11 | |
| WD (1e-4) | 0.01 / 0.04 / 0.1 | 5.75 / 5.88 / 5.75 | 6.99 / 7.10 / 6.98 | flat |
| Temperature | 0.1 | 6.71 | 7.67 | |
| Temperature | **0.2** | 6.40 | 7.25 | |
| Temperature | 0.4 | 5.62 | 6.71 | gate act 13 of 2048 |
| Temperature | 0.6 | 4.85 | 5.96 | gate act 9 of 2048 |
| Hydra LR scale | 0.1 | 6.40 | 7.25 | |
| Hydra LR scale | 0.3 | 7.68 | 8.51 | |
| Hydra LR scale | **1.0** | **7.74** | **8.68** | |
| Momentum (scale 1.0) | **0.996** | **13.65** | **14.20** | |
| Momentum (scale 1.0) | 0.9995 | 7.74 | 8.68 | |
| Momentum (scale 1.0) | 0.9999 | 3.41 | 4.25 | |
| Temperature (scale 1.0) | 0.1 | 8.10 | 9.02 | |
| Seed repeat (scale 1.0, seed 1) | | 7.52 | 8.43 | seed noise about 0.25 |
| iBOT-loss control (m 0.9995) | LR 5e-5 / 1e-4 / 2e-4 | 6.66 / 7.05 / 6.91 | 7.23 / 7.45 / 7.47 | no `cls_cos` logged |

### Caveats

- Momentum was swept at Hydra LR scale 1.0, so those two settings were tested together. Not retested at momentum 0.996: LR, WD and temperature, which were tuned at 0.9995. A faster teacher may shift the best LR.
- `0.996` is the lowest momentum tried, so the optimum may be lower. A 10-epoch run has about 12k steps, which favours a fast teacher; the best value at 100 epochs may be higher.
- The iBOT control used momentum 0.9995, so the "Hydra beats iBOT" comparison is **not** like-for-like at 0.996. Rerun the control at the recommended momentum before claiming a gain.
- Gate sparsity: active dimensions fall to about 30 of 2048 at temperature 0.2, and to about 100 at temperature 0.1 with scale 1.0. kNN does not penalise this, but it is the shortcut flagged in the ablation report.
- Single seed except one repeat. Treat differences under about 0.5 as noise.
- Tuned on ViT-S. The ViT-B 100-epoch command above uses these values but has not been validated for ViT-B.
