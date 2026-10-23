# Hydra ablation study (ViT-S, recommended settings)

Purpose: say which parts of Hydra produce the gain over a plain iBOT loss, and whether each recommended setting still holds once the others are fixed. The tuning that produced the base is in [runs.md](./runs.md#tuning-results-vit-s-10-epoch-pilots). The experiment design is followed by results and run status in sections 8–12 (last updated 2026-10-21). Every row runs with **one seed (seed 0)**.

## 1. Base configuration (B0)

Every ablation changes **one** thing from B0. B0 is the recommended setting from the tuning sweeps:


| Flag                                    | Value                                         |
| --------------------------------------- | --------------------------------------------- |
| `--arch`                                | `vit_small`                                   |
| `--lr`                                  | `0.00005` (effective peak 2e-4 at batch 1024) |
| `--weight_decay` / `--weight_decay_end` | `0.04` / `0.04`                               |
| `--momentum_teacher`                    | `0.996`                                       |
| `--loss`                                | `hydra`                                       |
| `--hydra_pair`                          | `sim`                                         |
| `--hydra_k`                             | `8`                                           |
| `--hydra_out_dim` / `--hydra_mlp_dim`   | `2048` / `4096`                               |
| `--hydra_temperature`                   | `0.2`                                         |
| `--hydra_tau`                           | `0.1`                                         |
| `--hydra_lr_scale`                      | `1.0`                                         |
| `--hydra_filter_negatives`              | `false`                                       |
| `--hydra_force_ones`                    | `false`                                       |


Fixed for all runs: batch 1024 (`--batch_size_per_gpu 128`, 8 GCDs), FFCV `IN1K_train_smart500_q90`, `--seed 0`, `--saveckp_freq 5`, `--use_fp16 true`, `--clip_grad 3.0`, `--warmup_teacher_temp_epochs` of 2 (10-epoch stage) or the default 30 (100-epoch stage), same teacher table (`train_teacher_cls.npy`).

B0 seed 0 already exists: `outputs/pilot-vits-10ep/hydra_lr5e-5_wd04_s0_h1.0_m996` (kNN k=20 13.65; k=200 14.20 from the earlier eval). The only seed repeat (momentum 0.9995) differed by about 0.25, so single-seed differences under 1.0 point are treated as noise.

## 2. Questions


| ID  | Question                                                                             | Why it matters                                                                                                                                  |
| --- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Q1  | Does Hydra training improve the released iBOT ViT-S weights? | That checkpoint is the target to beat. It is measured with the same kNN protocol, not taken from the paper. A1 starts from it; B0 starts from scratch and is compared separately. |
| Q2  | Does the gate matter?                                                                | An all-ones gate turns the loss into plain InfoNCE. The gate also collapses to about 30 of 2048 active dimensions, which looks like a shortcut. |
| Q3  | Where does the gain come from: the partner choice, or the frozen iBOT-B reference?   | The reference is a pretrained ViT-B (77% kNN). Part of the gain may be distillation of that model rather than the Hydra objective.              |
| Q4  | How sensitive is the result to the Hydra-specific knobs (K, tau, filtering, widths)? | Sets what has to be tuned when moving to another backbone or dataset.                                                                           |
| Q5  | Do the tuned LR, temperature and momentum still hold together?                       | LR, WD and temperature were tuned at momentum 0.9995 and not rechecked at 0.996.                                                                |




## 3. Experiments

IDs are used in run names. Every row is one run at seed 0. Every row is a flag change on the existing code.

### Group A: objective (Q1)


| ID  | Change from B0                       | Reads as                                     |
| --- | ------------------------------------ | -------------------------------------------- |
| B0  | none                                 | base (random init)                           |
| A1  | `--init_ibot <released iBOT ViT-S checkpoint>` | B0 settings, backbone starts from the iBOT weights (`--init_ibot` copies the teacher into student and teacher; epoch stays 0). Beat the unmodified checkpoint's kNN. |
| A2  | `--loss both` (`--hydra_weight 1.0`) | do the two objectives add?                   |
| A3  | `--loss both --hydra_weight 0.3`     | Hydra as a regularizer on iBOT               |




### Group B: gate and partner (Q2, Q3)


| ID            | Change from B0                                      | Reads as                                                                                         |
| ------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| B1            | `--hydra_force_ones true`                           | gate off; iBOT-B reference still picks partners                                                  |
| B2            | `--hydra_pair rand`                                 | partner carries no reference information; gate still reads `t_p`                                 |
| B3            | `--hydra_pair self`                                 | the positive is the same image; gate reads `t_p` alone                                           |
| B4            | `--hydra_pair label`                                | oracle: same-class partner. Uses ImageNet labels, so it is an upper bound and not an SSL result. |
| B5            | `--hydra_pair rand --hydra_force_ones true`         | no reference signal in the pair, no gate: the loss is a bare InfoNCE on random pairs             |


B2 against B0 isolates reference-guided partner selection. B1 against B0 isolates the gate. B5 is the no-information floor for the pair loss.

### Group C: Hydra-specific knobs (Q4)


| ID            | Change from B0                      |
| ------------- | ----------------------------------- |
| C1 / C2 / C3  | `--hydra_k` 1 / 4 / 16              |
| C4 / C5 / C6  | `--hydra_tau` 0.05 / 0.2 / 0.5      |
| C7            | `--hydra_filter_negatives true`     |
| C8 / C9 / C10 | `--hydra_out_dim` 768 / 1024 / 4096 |
| C11           | `--hydra_mlp_dim` 1024              |


`hydra_tau` only affects `sim`. C4 and C6 move partner similarity toward and away from uniform; B2 is the tau to infinity limit.

### Group D: recheck of tuned settings at the new base (Q5)


| ID      | Change from B0                                 |
| ------- | ---------------------------------------------- |
| D1 / D2 | `--lr` 0.000025 / 0.0001                       |
| D3 / D4 | `--hydra_temperature` 0.1 / 0.4                |
| D5 / D6 | `--momentum_teacher` 0.99 / 0.999              |
| D7      | `--weight_decay` and `--weight_decay_end` 0.01 |


D5 matters most: 0.996 was the lowest value tried and the best, so the optimum is not bracketed.

## 4. Protocol

**Stage 1: 10 epochs, ViT-S, 2 warmup epochs.** All rows above, seed 0 only. No repeats are scheduled; rows within 1.0 point (k=20) of B0 are reported as not distinguishable from B0.

**Stage 2: 100 epochs, ViT-S, 5 warmup epochs,** `--warmup_teacher_temp_epochs 10`**.** B0, the three non-A rows with the largest stage-1 effect (positive or negative), 1 seed each. Rerun D5 here regardless of its stage-1 result, because a 10-epoch run favours a fast teacher.

**Stage 3 (only if stage 2 supports Hydra over iBOT):** the same two arms on ViT-B, 100 epochs, 1 seed. Not scheduled until stage 2 reports.

### Measurements

- Primary: weighted kNN top-1 at k=10, 20 and 50 on `checkpoint0005` and `checkpoint0010` (`scripts/knn_eval_ffcv.py --arch vit_small --nb_knn 10 20 50`). Report top-5 at k=20 as well. Include an untrained ViT-S (floor) and the released iBOT ViT-S checkpoint (target, link in runs.md) in every eval job.
- Diagnostics from `log.txt` at the last epoch: `train_loss`, `train_cls_cos`, `train_cls_pair_cos`, `train_zs_std`, `train_act`, `train_gate_lo`. Hydra and iBOT losses are on different scales, so compare loss only within a row group.
- A run is **rejected** if it has NaN, `cls_cos` above 0.95 at epoch 3 or still rising at epoch 10, or `zs_std` near 0 (see runs.md). Rejected runs are reported, not dropped.



### Decision rules

- One seed per row, so no standard deviation. A difference is called only if it exceeds 1.0 point at k=20 (the one repeat measured about 0.25, so this is roughly 4x).
- **Q1:** Hydra improves iBOT if A1 exceeds the unmodified released checkpoint by more than 1.0 point at k=20 (same eval job, same protocol). Scratch rows (everything else) are ranked against B0 and are not compared with the released model: they train 10 or 100 epochs against its 300.
- **Q3:** if B2 is within 1.0 point of B0, the partner choice adds nothing and the gain comes from elsewhere. If B1 is within 1.0 point of B0, the gate adds nothing.
- Do not adopt a changed setting from Group C or D unless it passes the 1.0-point rule in stage 2. Stage 1 alone only reorders candidates.



## 5. Cost

One 10-epoch ViT-S pilot took about 2 hours on one LUMI-G node (8 GCDs).


| Stage                                               | Runs | Node-hours |
| --------------------------------------------------- | ---- | ---------- |
| 1 (seed 0, all rows; B0 reused)                     | 26   | about 52   |
| 2 (100 epochs, assuming about 20 h per run, 5 runs) | 5    | about 100  |


Stage 1 is about 52 node-hours. Cut first: A3, B3 and B5, which are the least tied to the main questions.

## 6. Launching

`scripts/submit_ablation.sh` holds every row as B0 plus its flag changes. `ROWS="A1 B1" scripts/submit_ablation.sh` prints the commands; `SUBMIT=1` submits one standard-g job per row and queues a single dev-g kNN job (k=10 20 50, with an untrained ViT-S and the released iBOT ViT-S) behind them with `afterok`. Use `WANDB_TAGS=abl-<row>` per submission. Stage 2 (100 epochs) is not scripted yet.

The released iBOT ViT-S checkpoint is at `/scratch/project_465003400/cache/torch/ibot/vits_16_checkpoint.pth`. Earlier ViT-B iBOT-init + Hydra runs lowered kNN by about 6.5 points (see memory `init-control-results`), so A1 may come out negative; that is a result, not a failure.

Output layout: `/scratch/project_465003400/SSLHydra/outputs/ablation-vits/<ID>_s0`, with one kNN JSON per batch next to it.

## 7. Known confounds

- The reference descriptors come from a pretrained ViT-B, so every Hydra arm that reads them has access to that model's features. Group B tells you how much that matters. A1 starts from iBOT weights, so it also carries whatever that model learned without the reference; compare it only with the unmodified checkpoint.
- One seed per row: a single run cannot separate a 1-point effect from noise, and stage 2 is also single-seed. Treat results as a ranking, not a confirmation.
- The 10-epoch regime favours a fast EMA teacher and short horizons. Stage 2 exists to test that.
- Gate sparsity (about 30 of 2048 active) is a known degenerate solution. kNN does not penalize it, so report `train_act` next to every kNN number.
- Per-tensor gradient clipping, not global norm, is still in the code, and `gnorm=Infinity` has been seen. A NaN in one arm is not necessarily the arm's fault.
- Everything is tuned on ViT-S and ImageNet-1k.


## 8. Stage 1 results (ViT-S, 10 epochs, seed 0)

Weighted kNN top-1 (%) on ImageNet-1k, teacher [CLS], `checkpoint0010`. Outputs: `outputs/ablation-vits/knn_abl-A1.json`, `knn_abl-A.json`, `knn_abl-B.json`. Groups C and D were not run (C was cancelled before starting, D was not submitted). B0 is the earlier run `pilot-vits-10ep/hydra_lr5e-5_wd04_s0_h1.0_m996`, re-evaluated at k=10/20/50.

| Run | Change from B0 | k=10 | k=20 | k=50 | Δ k=20 vs B0 | `cls_cos` / `zs_std` / `act` |
|---|---|---|---|---|---|---|
| untrained ViT-S | | 1.20 | 1.37 | 1.53 | | |
| released iBOT ViT-S | target | 75.05 | 74.94 | 74.31 | | |
| A1 | `--init_ibot` released ViT-S + Hydra | 69.84 | 69.44 | 68.38 | -5.5 vs released | 0.13 / 0.15 / 45 |
| B0 | base (scratch) | 12.37 | 13.65 | 14.39 | | 0.18 / 0.39 / 32 |
| A2 | `--loss both` | 14.23 | 15.09 | 15.67 | +1.4 | 0.21 / 0.23 / 39 |
| A3 | `--loss both --hydra_weight 0.3` | 13.14 | 14.05 | 14.48 | +0.4 | 0.21 / 0.23 / 40 |
| B1 | `--hydra_force_ones true` | 20.26 | 21.63 | 22.03 | +8.0 | 0.27 / 0.86 / 2048 |
| B2 | `--hydra_pair rand` | 23.54 | 24.58 | 25.14 | +10.9 | 0.24 / 0.26 / 34 |
| B3 | `--hydra_pair self` | 18.60 | 19.71 | 20.30 | +6.1 | 0.23 / 0.69 / 92 |
| B4 | `--hydra_pair label` | 1.27 | 1.50 | 1.85 | collapsed | 0.95 / 0.002 / 59 |
| B5 | `--hydra_pair rand --hydra_force_ones true` | 1.39 | 1.70 | 2.01 | collapsed | 0.92 / 0.0 / 2048 |

### Reading against the decision rules

- **Q1: not met.** A1 is 5.5 points below the unmodified released checkpoint at k=20 (5.9 at k=50). Hydra training from the iBOT weights lowers kNN, as the earlier ViT-B run did (about -6.5). A1 has the lowest Hydra loss (3.22), so the objective is satisfied while kNN gets worse.
- **Gate and partner choice hurt at 10 epochs.** Turning the gate off (B1) gives +8.0 and random partners (B2) give +10.9, so B0 is not the best arm. The ViT-B reference does not help partner selection here: random partners beat `sim` partners.
- **The gate matters only with uninformative pairs.** B2 (random + gate) is the top row, while B5 (random, no gate) collapses.
- **A2** passes the 1.0-point rule at k=20 (+1.4) and is marginal at k=10 and k=50. A3 is within noise.
- **B4 and B5 are collapses** (`cls_cos` 0.92 to 0.95, `zs_std` near 0), not measurements of the design. B4 has partner entropy 0.36 and a loss of 6.67, which looks like a bug or shortcut and needs checking before it is read as "oracle partners fail".

### Caveats

- One seed and 10 epochs. Only the large gaps (A1, B1 to B5) are reliable.
- `train_gnorm` is NaN or Infinity in many epochs of A2, A3 and B1 to B4 (the per-tensor clipping issue). Training stayed finite, but its effect on these runs was not measured.
- B0's earlier k=200 value (14.20) is not comparable to this k=10/20/50 table.

### Open items

- Stage 2 (100 epochs) for B0, B2 and B1, plus A2.
- B2 with `--init_ibot`, to see whether a random-partner objective damages the iBOT weights less than `sim` does (A1).
- Debug B4 (`label` pairing).
- Groups C and D remain unrun.

## 9. Stage 2 results (ViT-S, 100 epochs, seed 0)

5 warmup epochs, `--warmup_teacher_temp_epochs 30`, `checkpoint0100`. Output: `outputs/ablation-vits-100ep/knn_abl-stage2.json`. B1 was cancelled at epoch 6 because the loss was not dropping (6.92 to 6.77), so it has no 100-epoch result. The released iBOT ViT-S is 75.05 / 74.94 / 74.31 at k=10/20/50.

| Run | Change from B0 | k=10 | k=20 | k=50 | Δ k=20 vs B0 | `cls_cos` / `zs_std` / `act` | 10-ep k=20 |
|---|---|---|---|---|---|---|---|
| B0 | base | 56.78 | 57.13 | 56.22 | | 0.26 / 0.13 / 32 | 13.65 |
| B2 | `--hydra_pair rand` | 61.35 | 61.30 | 60.47 | +4.2 | 0.24 / 0.17 / 33 | 24.58 |
| D5 | `--momentum_teacher 0.99` | 58.07 | 58.17 | 57.52 | +1.0 | 0.25 / 0.13 / 31 | not run |
| B3 | `--hydra_pair self` | 40.58 | 41.35 | 41.06 | -15.8 | 0.41 / 0.51 / 90 | 19.71 |

### Reading

- **Ordering by 10-epoch kNN mostly holds, except B3.** B2 stays the best row (+4.2 over B0). B3 was +6.1 at 10 epochs and is -15.8 at 100 epochs, so a short run overstated it.
- **No row reaches the released iBOT ViT-S.** The best is B2 at 61.3 k=20, 13.6 points below 74.94. These runs train 100 epochs against the released model's 300, so the gap is not a like-for-like comparison.
- **Random partners beat reference-guided partners at 100 epochs too.** The gain from the ViT-B partner table is negative here (B0 < B2), which supports the stage-1 reading.
- **D5 (momentum 0.99) is +1.0 at k=20,** at the edge of the 1.0-point rule and a single seed. It also tracks B0 closely in diagnostics, so it is a mild effect, not a clear improvement. 0.99 was the lowest momentum tried, so the optimum is still unbracketed.
- **B3 drifts to a partly collapsed representation.** `cls_cos` rises from 0.29 to 0.41 and the gate keeps 90 active dimensions, against 32 for B0. Its loss is the lowest (2.96), which is a different objective, not better features.
- **Not run:** B1 (gate off) at 100 epochs, B4/B4x (collapsed at 10 epochs), and Groups C and D except D5.

### Caveat

One seed per row. B2 (+4.2) clears the 1.0-point rule comfortably. D5 does not clear it with confidence.

## 10. B2 from iBOT init (100 ep) and B2 + iBOT loss with local crops (300 ep)

Both original fp16 runs stopped on `Loss is nan, stopping training`. B2i later finished after an fp32 resume; the 300-epoch run remains unfinished (see section 12).

| Run | Setting | Died at | Last good checkpoint | State of the run before the NaN |
|---|---|---|---|---|
| B2i (job 22854621) | `--hydra_pair rand --init_ibot` released ViT-S, 100 ep | epoch 75 (13.8 h) | `checkpoint0060.pth` | Hydra loss 3.14, `cls_cos` 0.25, `zs_std` 0.12, `act` 38; `gnorm` was `inf` in epochs 69 to 71 |
| B2 + iBOT, 8 local crops (job 22854681) | `--loss both --hydra_pair rand --local_crops_number 8`, scratch, 300 ep | epoch 37 (12.3 h) | `checkpoint.pth` (epoch 37; no periodic checkpoint until 50) | total loss 11.76 (iBOT cls 2.58, patch 4.92, Hydra 4.26), `cls_cos` 0.19, `act` 31; `gnorm` was `inf` or `nan` in alternate epochs from epoch 31 |

- **Cause:** the run was healthy up to the NaN. The `gnorm` pattern is the known per-tensor clipping problem (`utils.clip_gradients`), and the NaN is the same issue reaching the loss. Nothing was changed in the objective.
- **Resume bug:** the follow-up jobs 22854701 and 22854702 died in `utils.restart_from_checkpoint` with `UnpicklingError: Weights only load failed`, because `torch.load` in this torch version defaults to `weights_only=True`. Resuming from `checkpoint.pth` is therefore broken until `weights_only=False` is passed.
- **Evaluation:** interim kNN numbers from before the fp32 resumes are in section 11. Final B2i kNN and the latest 300-epoch run status are in section 12.

## 11. Interim kNN before the fp32 resumes

Weighted kNN top-1 (%), `outputs/ablation-vits-100ep/knn_interim.json`. B0 and B2 are the stage-2 runs evaluated at epoch 60 for a same-epoch comparison. The released iBOT ViT-S is 75.05 / 74.94 / 74.31.

| Run | Epoch | k=10 | k=20 | k=50 |
|---|---|---|---|---|
| B2i (random partner, init from released iBOT) | 60 / 100 | 73.29 | 73.02 | 72.09 |
| B2 (random partner, scratch) | 60 / 100 | 59.64 | 59.71 | 58.76 |
| B0 (base, scratch) | 60 / 100 | 55.15 | 55.49 | 54.70 |
| B2 + iBOT loss, 8 local crops (scratch) | 37 / 300 | 40.27 | 40.90 | 40.32 |

- **B2i is 1.9 points below the released checkpoint at k=20 after 60 epochs,** against 5.5 points below for A1 (`sim` partner, 10 epochs). It had not exceeded the release at this checkpoint; the original fp16 attempt stopped at epoch 75. The completed fp32 resume is reported in section 12.
- **At epoch 60 the scratch ordering matches stage 2:** B2 is 4.2 points above B0.
- **The 300-epoch run was evaluated at epoch 37 of 300,** so its 40.9 is not comparable to the 100-epoch rows. Training progress is reported in section 12; the epoch-130 kNN result is in section 13.

## 12. fp32 resume

Both NaN runs were resumed from `checkpoint.pth` with `--use_fp16 false` and the `weights_only=False` fix in `utils.restart_from_checkpoint`, using the clone `work/hydra-fp32-resume-20261019/src`.

- **Failure:** the first two resume attempts per run died in the first forward with `RuntimeError: miopenStatusUnknownError`. The cause was an MIOpen kernel build failure for the fp32 patch-embedding convolution (`Code object build failed. Source: MIOpenConvDirGenFwd.cl`, `local memory (81920) exceeds limit (65536)`), reached through `PatchEmbed.forward` in `models/vision_transformer.py`. The fp16 path had never hit it.
- **Fix:** `PatchEmbed.forward` now computes the same projection as a reshape plus `F.linear` on the conv weights, so no MIOpen convolution is used. The checkpoint format is unchanged. This is in the repo and in the clone.
- **Check:** the resumed runs continue at the loss level they stopped at (B2i epoch 75: 3.14; 300-epoch run epoch 37: 11.4 against 11.8 before).
- **Cost:** fp32 is about 2x slower per epoch than the fp16 runs.


### Status as of 2026-10-21

| Run | fp32 resume job | Status | Latest saved progress | Last completed epoch: loss / `cls_cos` / `zs_std` / `act` / `gnorm` |
|---|---|---|---|---|
| B2i (random partner, released iBOT init) | 22952127 | **Completed**, 2026-10-19; resume took 7 h 13 min | 100 / 100, `ablation-vits-100ep/B2i_s0/checkpoint0100.pth` | 3.057 / 0.252 / 0.122 / 38.7 / 0.285 |
| B2 + iBOT loss, 8 local crops (scratch) | 22952126 | **TIMEOUT**, 2026-10-21 at 09:22 cluster time; 48-hour limit | 130 / 300, `ablation-vits-300ep/B2ibot_lc8_s0/checkpoint.pth` | 10.789 / 0.152 / 0.262 / 31.6 / 3.100 |

The 300-epoch run completed zero-based epoch 129 and timed out during epoch 130 (last logged batch 1080/1251). Its last completed epoch had iBOT cls loss 1.644, patch loss 5.346 and Hydra loss 3.799. Slurm reports a time limit termination; the last logged losses and gradient norm were finite. Periodic checkpoints at 50 and 100 epochs are also available. The run was subsequently resumed on two nodes as job 23020079 (see below) to finish the remaining 170 epochs. The saved epoch-130 checkpoint was subsequently evaluated (section 13).

### B2i final kNN (100 epochs)

Weighted kNN on teacher [CLS], `checkpoint0100.pth`, from `outputs/ablation-vits-100ep/knn_abl-B2i-fp32.json` (evaluation job 22952128, completed 2026-10-20; 1,281,167 training images and 50,000 validation images).

| Run | k=10 top-1 | k=20 top-1 | k=50 top-1 | k=20 top-5 |
|---|---|---|---|---|
| B2i, 100 epochs (fp16 then fp32) | 74.04 | 73.92 | 73.05 | 90.21 |
| Released iBOT ViT-S (earlier baseline evaluation) | 75.05 | 74.94 | 74.31 | — |

B2i improves from 73.02 at epoch 60 to 73.92 at epoch 100 at k=20 (+0.90 points), but remains 1.02 points below the released checkpoint (74.94). It therefore does not meet Q1's requirement of improving the release by more than 1.0 point. The release was evaluated in an earlier job, and B2i combines fp16 training with an fp32 resume, so this is not a controlled comparison of precision modes. These are single-seed results.


## 13. B2 + iBOT loss with local crops: epoch-130 kNN vs B2i

Evaluated 2026-10-21 in job 23019795, using `ablation-vits-300ep/B2ibot_lc8_s0/checkpoint.pth` saved after 130 completed epochs. Results: `outputs/ablation-vits-300ep/knn_B2ibot_lc8_ep130.json`. The protocol matches B2i's evaluation: ViT-S teacher [CLS], L2-normalized features, 224-pixel center crops, weighted kNN at temperature 0.07, 1,281,167 ImageNet training images and 50,000 validation images.

| Run | Completed epochs | Initialization / objective | k=10 top-1 | k=20 top-1 | k=50 top-1 | k=20 top-5 |
|---|---|---|---|---|---|---|
| B2 + iBOT, 8 local crops | 130 / 300 | scratch / iBOT + Hydra, random partners | 59.96 | 60.22 | 59.49 | 81.43 |
| B2i | 100 / 100 | released iBOT / Hydra, random partners | 74.04 | 73.92 | 73.05 | 90.21 |
| Δ (B2 + iBOT minus B2i), points | | | -14.08 | -13.70 | -13.57 | -8.78 |

- **B2i remains ahead:** the scratch combined-loss run is 13.70 points lower at k=20 (60.218 vs 73.922). This compares the current checkpoints; initialization, objective, local crops and training duration differ, so it does not isolate the effect of adding iBOT loss.
- **The combined-loss run improves with training:** k=20 rises from 40.90 at epoch 37 to 60.22 at epoch 130 (+19.32 points), with 170 scheduled epochs still remaining. Its epoch-130 score is 1.08 points below scratch B2 at epoch 100 (61.30), also a comparison across different objectives and training schedules.
- **The released checkpoint remains ahead:** 60.22 is 14.72 points below its 74.94 at k=20. These are single-seed results; the unfinished run's score is interim.


## 14. Two-node B2 + iBOT resume (2026-10-21)

Job **23020079** is running on two LUMI-G nodes (`nid005332`, `nid005333`), with 16 GCDs and a 48-hour limit. It restored the student, teacher, optimizer, iBOT loss and Hydra state from `ablation-vits-300ep/B2ibot_lc8_s0/checkpoint.pth` and resumed at zero-based epoch 130 toward the original 300-epoch target.

The per-GPU batch is reduced from 128 to 64, preserving global batch 1024 and the learning-rate schedule. Training continues in fp32 with the same fixed clone, random partners, combined iBOT + Hydra objective and eight local crops. `scripts/submit_b2_300ep.sh` now accepts `BATCH_SIZE_PER_GPU`.

Reproduce the submission with:

```bash
NNODES=2 BATCH_SIZE_PER_GPU=64 USE_FP16=false \
STUDY_SRC=/scratch/project_465003400/SSLHydra/work/hydra-fp32-resume-20261019/src \
bash scripts/submit_b2_300ep.sh
```

Startup verification: all 16 workers initialized, checkpoint state loaded, and epoch 130 reached batch 30/1251 with finite loss (running mean 10.755) and gradient norm (3.086). After initial kernel tuning, recent steps took about 0.88 s versus about 1.45 s on one node; these early timings are preliminary. Logs: `outputs/slurm/b2-ibot-lc8-23020079.out` and `.err`. Checkpoints continue in the existing run directory.


## 15. iBOT-init, iBOT-loss control (2026-10-22)

Run **`abl-vits-B2i-ibot_s0_100ep`**, tag **`abl-b2i`**, is running as job **23059781** on one LUMI-G node (8 GCDs). It starts a new 100-epoch schedule from the released `/scratch/project_465003400/cache/torch/ibot/vits_16_checkpoint.pth`, with `--loss ibot` and no Hydra objective. This tests whether continued iBOT training also degrades the released weights, against B2i's Hydra result (73.922 at k=20 vs the release's 74.94).

The control matches B2i's seed 0, global batch 1024 (128 per GPU), LR 0.00005, weight decay 0.04, momentum 0.996, five LR warmup epochs, 30 teacher-temperature warmup epochs and zero local crops. It uses fp32 throughout with the fixed `hydra-fp32-resume-20261019/src` clone. The teacher weights loaded without missing keys; training started at epoch 0 with fresh optimizer and loss state. At batch 20/1251 the running mean loss was finite (14.181; cls 4.262, patch 9.919).

Output: `outputs/ablation-vits-100ep/B2i-ibot_s0`. Reproduce with `STAGE=2 ROWS=B2i-ibot SUBMIT=1 USE_FP16=false WANDB_TAGS=abl-b2i STUDY_SRC=/scratch/project_465003400/SSLHydra/work/hydra-fp32-resume-20261019/src bash scripts/submit_ablation.sh`.

kNN job **23059782** is queued with `afterok:23059781` to evaluate the final teacher alongside B2i's `checkpoint0100.pth` and the released iBOT checkpoint in the same job (k=10/20/50). Results will be written to `outputs/ablation-vits-100ep/knn_abl-b2i.json`. W&B: https://wandb.ai/erow/SSLHydra/runs/g4b56y2h.
