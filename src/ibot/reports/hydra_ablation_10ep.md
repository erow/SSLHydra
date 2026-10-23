# Hydra-Sim 十轮消融实验报告

日期：2026-10-09  
分支：`study/hydra-sim-k8-d2048`  
正式作业：Slurm arrays `22661042`、`22661043`

## 摘要

本实验研究在 ImageNet-1K 上从随机初始化 ViT-B/16 backbone 训练 Hydra-Sim 时，预训练 Hydra head、teacher EMA、输出维度、采样次数、temperature、teacher similarity tau、Hydra 学习率、false-negative filtering 和 gate 本身对训练稳定性及 representation collapse 的影响。

主要结果如下。

1. 16 组正式实验中，只有 pretrained-Hydra baseline 和 `momentum_teacher=0.996` 两组完成 10 epochs；其余 14 组均因 loss 变为 NaN 提前退出。
2. 两个完成的实验都发生严重 representation collapse。Baseline 在 epoch 3 达到 `cls_cos=0.9995`，之后保持约 1.0；`momentum_teacher=0.996` 在训练末期同样达到 `cls_cos=0.9995`。
3. Baseline loss 从 6.941 缓慢降至 6.820，但该改善发生在 backbone collapse 之后，不能视为有效学习。全局 batch 为 1024 时，随机均匀分类的交叉熵为 `log(1024)=6.931`。
4. 冻结 iBOT backbone 时，Hydra head 可以稳定优化，loss 从 4.267 降至最低 3.129，且 `cls_cos≈0.091`。但 gate 最终极度稀疏：`act≈44/2048`，约 95.6% 的 gate 小于 0.05。
5. 将该 Hydra head 迁移到随机 backbone 后，没有阻止 collapse。预训练 head 提高了数值存活时间，但没有带来有效 representation learning。
6. All-ones gate 对照同样出现 NaN，且 loss 一直约为 6.93。这说明 gate 不是唯一问题；scratch backbone 的优化动力学是更上游的主要问题。
7. `momentum_teacher=0.9995` 不足以解决问题。十轮实验采用 1 epoch warmup，使 backbone 有效峰值 LR 在一个 epoch 内升至 0.002；这是本轮消融中最重要的共同不稳定因素。

## 研究问题

本轮实验试图回答以下问题。

- 冻结 iBOT backbone 训练得到的 Hydra head 或 gate 能否迁移到 scratch backbone？
- 对 global batch 1024，较慢的 teacher EMA（`momentum_teacher=0.9995`）能否阻止 collapse？
- `hydra_out_dim`、`hydra_k`、`hydra_temperature` 和 `hydra_tau` 是否显著改变稳定性？
- Hydra 独立学习率是否需要低于 backbone？
- false-negative filtering 或关闭 gate 是否能改善训练？

## 实现与监控

相关实现提交如下。

- `35d72e8`：false-negative filtering、W&B gate metrics、逐 epoch checkpoint。
- `8127e5b`：每 10 steps 记录 W&B；补齐 Hydra 模式的输出契约。
- `adcb104`：iBOT 初始化和 frozen-backbone/Hydra-only 模式。
- `b681bd5`：只迁移 Hydra state、Hydra LR multiplier、`cls_pair_cos`。
- `f36960d`：teacher momentum 默认值 0.9995、16 组十轮消融、all-ones gate 对照。
- `fbe02a6`：十轮实验的 teacher-temperature warmup 修正为 1 epoch。

关键监控指标：

- `cls_cos = mean(u @ u.T) = ||mean(u)||²`：全 batch 表征集中度；越接近 1，collapse 越严重。
- `cls_pair_cos`：同一图像两个 global crop 的 cosine similarity。
- `zs_std`：student Hydra latent 的跨样本标准差。
- `zt_std`：teacher projector 后、key BN 后的标准差；BN 会掩盖上游 collapse，不能单独用于判断。
- `act = gate.sum(1).mean()`：平均激活维数。
- `gate_lo`、`gate_hi`：小于 0.05 和大于 0.95 的 gate 比例。
- `ref_h - ref_ent`：模型相对于 teacher similarity graph 的交叉熵差，可理解为参考分布到模型分布的 KL 项。

## 数据、模型和共同训练配置

| 项目 | 配置 |
|---|---|
| 数据 | ImageNet-1K FFCV，1,281,167 个训练样本 |
| Backbone | ViT-B/16，从随机初始化开始 |
| 硬件 | 单节点 8 × MI250 GCD |
| Batch | 128/GPU，全局 batch 1024 |
| 优化器 | AdamW，FP16 GradScaler |
| Epochs | 10 |
| LR warmup | 1 epoch |
| Backbone peak LR | 0.002（`0.0005 × 1024 / 256`） |
| Weight decay | cosine，0.04 → 0.4 |
| Gradient clipping | 3.0；当前实现为逐参数 tensor clipping，不是 global norm clipping |
| Hydra pair | teacher-table similarity sampling |
| Baseline head | frozen-iBOT 实验的 `checkpoint0009.pth` |
| Baseline Hydra LR | backbone LR 的 0.1 倍 |
| Baseline filtering | false |
| Checkpoint | 每个 epoch 保存 |

Teacher EMA 从配置给定的初始 momentum cosine 增长到 1.0。除 momentum 消融外，正式实验统一使用 `momentum_teacher=0.9995`。

## 第一阶段：冻结 iBOT backbone 训练 Hydra head

第一阶段从公开 iBOT ViT-B/16 teacher checkpoint 初始化 student 和 teacher，冻结两个 backbone，只训练 Hydra。配置为 `out_dim=2048`、`k=8`、temperature 0.2、tau 0.1，并启用 false-negative filtering。2 小时作业完成 9 个 epoch。

| Epoch | Loss | cls_cos | act | gate_mean | gate_lo |
|---:|---:|---:|---:|---:|---:|
| 0 | 4.267 | 0.0911 | 130.4 | 0.0637 | 0.8089 |
| 1 | 3.478 | 0.0909 | 36.0 | 0.0176 | 0.9448 |
| 2 | 3.274 | 0.0911 | 40.0 | 0.0195 | 0.9476 |
| 3 | 3.182 | 0.0908 | 42.2 | 0.0206 | 0.9507 |
| 4 | 3.145 | 0.0912 | 43.4 | 0.0212 | 0.9531 |
| 5 | **3.129** | 0.0910 | 44.0 | 0.0215 | 0.9548 |
| 6 | 3.135 | 0.0914 | 44.0 | 0.0215 | 0.9557 |
| 7 | 3.139 | 0.0912 | 44.0 | 0.0215 | 0.9561 |
| 8 | 3.148 | 0.0913 | 44.0 | 0.0215 | 0.9561 |

该阶段数值稳定且 backbone 没有 collapse。Hydra loss 显著下降，但 gate 很快收缩到只激活约 44/2048 个维度。由于 gated cosine 对 gate 的共同缩放近似不敏感，这种极稀疏解可能是 shortcut，并不自动代表 gate 学到了可迁移的语义子空间。

## 正式消融矩阵

配置 0 为统一 baseline：pretrained Hydra、momentum 0.9995、out_dim 2048、k=8、temperature 0.2、tau 0.1、Hydra LR 0.1×、no filtering。其余配置采用单因素变化。

| ID | Run | 变化项 | 完整 epochs | Epoch loss | 最后有效 step / 结果 |
|---:|---|---|---:|---|---|
| 0 | `base_pre_m9995_d2048_k8_t20_tau10_hlr10` | baseline | 10 | 6.941 → 6.820 | 完成；epoch 3 已 collapse |
| 1 | `random_head` | 不加载 pretrained Hydra | 0 | — | epoch 0 step 1130 后 NaN |
| 2 | `momentum_996` | momentum 0.996 | 10 | 6.913 → 6.851 → 6.839 → 6.880 | 完成；下降后反弹并 collapse |
| 3 | `momentum_999` | momentum 0.999 | 1 | 6.859 | epoch 1 step 220 后 NaN；此前已 collapse |
| 4 | `outdim_768_random` | out_dim 768，random head | 1 | 6.722 | epoch 1 step 160 后 NaN |
| 5 | `outdim_4096_random` | out_dim 4096，random head | 0 | — | epoch 0 step 1060 后 NaN |
| 6 | `k_1` | k=1 | 1 | 6.871 | epoch 1 step 460 后 NaN；此前已 collapse |
| 7 | `k_4` | k=4 | 1 | 6.858 | epoch 1 step 190 后 NaN；此前已 collapse |
| 8 | `temperature_010` | temperature 0.1 | 1 | 7.008 | epoch 1 step 80 后 NaN；此前已 collapse |
| 9 | `temperature_040` | temperature 0.4 | 1 | 6.853 | epoch 1 step 740 后 NaN |
| 10 | `tau_005` | tau 0.05 | 1 | 6.820 | epoch 1 step 530 后 NaN；此前已 collapse |
| 11 | `tau_020` | tau 0.2 | 0 | — | epoch 0 step 1030 后 NaN；`cls_cos≈0.9956` |
| 12 | `hydra_lr_003` | Hydra LR 0.03× | 1 | 6.923 | epoch 1 step 20 后 NaN |
| 13 | `hydra_lr_030` | Hydra LR 0.3× | 2 | 6.806 → 6.764 | epoch 2 step 340 后 NaN；epoch 1 `cls_cos=0.999` |
| 14 | `filter_negatives` | filtering=true | 1 | 6.761 | epoch 1 step 30 后 NaN；epoch 1 `cls_cos≈0.9985` |
| 15 | `force_ones_gate` | gate 恒为 1 | 1 | 6.929 | epoch 1 step 230 后 NaN；loss 近随机水平 |

表中的 step 是 NaN 前最后一个每 10 steps 打印的有效记录，因此实际 NaN 发生在该 step 后的最多 10 个 iteration 内。

## 完成实验的逐 epoch 结果

### Pretrained Hydra，momentum 0.9995

| Epoch | Loss | cls_cos | cls_pair_cos | zs_std | zt_std | act |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.9408 | 0.5248 | 0.6495 | 0.01770 | 0.9817 | 83.9 |
| 1 | 6.8747 | 0.7115 | 0.7149 | 0.00283 | 0.9156 | 68.7 |
| 2 | 6.8594 | 0.9693 | 0.9697 | 0.00164 | 0.6107 | 67.1 |
| 3 | 6.8254 | 0.9995 | 0.9995 | 0.00070 | 0.3722 | 75.6 |
| 4 | 6.8281 | 0.9997 | 0.9997 | 0.00055 | 0.3096 | 70.4 |
| 5 | 6.8276 | 1.0000 | 0.9998 | 0.00051 | 0.2642 | 70.4 |
| 6 | 6.8232 | 1.0000 | 0.9998 | 0.00055 | 0.2301 | 71.7 |
| 7 | 6.8196 | 1.0000 | 0.9999 | 0.00099 | 0.2074 | 72.1 |
| 8 | **6.8193** | 1.0000 | 0.9999 | 0.01877 | 0.1946 | 71.8 |
| 9 | 6.8200 | 1.0000 | 0.9999 | 0.04659 | 0.1915 | 71.9 |

Loss 的小幅下降与表示质量无关：epoch 3 后不同图像的 teacher CLS 已几乎完全相同。`cls_pair_cos` 同时接近 1 不能解释为增强不变性，因为全 batch `cls_cos` 也已接近 1。

### Pretrained Hydra，momentum 0.996

| Epoch | Loss | cls_cos | cls_pair_cos |
|---:|---:|---:|---:|
| 0 | 6.9130 | 0.8688 | 0.9034 |
| 1 | 6.8509 | 0.9594 | 0.9651 |
| 2 | **6.8387** | 0.9836 | 0.9855 |
| 3 | 6.8491 | 0.9719 | 0.9774 |
| 4 | 6.8612 | 0.9963 | 0.9968 |
| 5 | 6.8796 | 0.9971 | 0.9971 |
| 6 | 6.9211 | 0.8684 | 0.8694 |
| 7 | 6.9009 | 0.9676 | 0.9673 |
| 8 | 6.8829 | 0.9995 | 0.9995 |
| 9 | 6.8800 | 0.9995 | 0.9996 |

较快 EMA 没有改善最终结果，并出现更明显的震荡。较慢的 0.9995 让轨迹更平滑，但同样收敛到 collapse。

## 分因素分析

### Pretrained Hydra head

Pretrained head baseline 是唯一与 momentum 0.996 一起跑满十轮的配置；random-head d2048 在 epoch 0 结束前 NaN。因此，pretraining 提供了数值稳定性。然而 baseline 仍在三轮内完全 collapse，说明它没有提供可迁移的表征约束。

当前 `--init_hydra` 加载整个 Hydra state，而不是只加载 gate，包括 projector、EMA projector、predictor、key BN buffers 和 gate。因此本实验不能把该稳定性归因于 gate。需要 gate-only 和 projector-only 对照才能解耦。

### Teacher momentum

`0.9995` 相比 `0.996` 使 loss/CLS 轨迹更平滑，但不能阻止 collapse。`0.999` 在 epoch 1 早期 NaN。结果不支持“仅提高 momentum 即可解决问题”。

### Output dimension

out_dim 768 完成一轮后 NaN；4096 在第一轮结束前 NaN。由于只有 2048 维存在 pretrained head，这两个实验同时改变了 out_dim 和是否预训练，不能作为纯粹的维度因果比较。

### Hydra k

k=1 和 k=4 均在第二轮 NaN；k=8 baseline 能跑满。更多 partner draws 可能通过梯度平均改善数值稳定性，但 k=8 仍然 collapse，因此它不是表征层面的解决方案。

### Temperature

temperature 0.1 在 epoch 1 step 80 后 NaN；0.4 持续到 step 740。更高 temperature 看起来能延缓数值失败，但不能阻止表示集中。0.4 是值得在降低 backbone LR 后保留的候选。

### Teacher-similarity tau

tau 0.05 在 epoch 1 中段 NaN；tau 0.2 在 epoch 0 后段 NaN。两者均没有显示稳定收益。不同 tau 会改变 reference entropy，因此不能仅用 raw loss 横向比较。

### Hydra learning rate

Hydra LR 0.03× 在 epoch 1 刚开始即 NaN，而 0.3× 持续到 epoch 2。NaN 时间与 Hydra LR 不呈单调关系，说明共同的 backbone LR/teacher 动力学比 Hydra LR 更关键。

### False-negative filtering

Filtering 稳定移除约 7.4% negatives，但实验在 epoch 1 开始后很快 NaN，且此时已经 collapse。没有证据表明 filtering 是主要控制变量。

### All-ones gate

All-ones gate 将 gated contrast 退化为普通 cosine InfoNCE。其 loss 保持约 6.929，接近随机水平，并在 epoch 1 NaN。关闭 gate 没有修复训练，证明 gate 不是唯一根因；但该对照也没有学到有效实例判别。

## 失败模式分析

### 1. 一轮 warmup 对 scratch ViT 过于激进

全局 batch 1024 将命令行 `lr=0.0005` 线性放大为 peak LR 0.002。所有实验在一个 epoch 内达到该峰值。多数失败发生在 epoch 0 后段或 epoch 1，时间上与 LR 达峰高度一致。降低 Hydra LR 不会降低 backbone LR，因此 `hydra_lr_003` 仍快速失败。

### 2. Collapse 先于 NaN

多个 run 在 loss 仍为有限值时已经达到 `cls_cos>0.99`，之后才出现 NaN。因此 NaN 是已经失稳的优化轨迹的终点，而不是 collapse 的最初原因。

### 3. Gate 解存在尺度不可辨识性和饱和

Gated cosine 对 gate 的共同缩放近似不变，却没有激活率、熵或稀疏度目标。冻结 backbone 时，gate 自动收缩到约 44 个有效维度。迁移后 gate 仍高度稀疏，容易造成条件数恶化和 sigmoid 饱和。

### 4. Key BN 会掩盖 teacher collapse

`zt_std` 位于 teacher projector 后的 key-only BN 之后。即使 raw teacher CLS 已 collapse，BN 仍可维持看似正常的标准差。因此必须以 raw `cls_cos` 为主要 collapse 判断依据。

### 5. 当前 clipping 不是 global gradient clipping

`utils.clip_gradients` 对每个 parameter tensor 独立限制 norm，而不是对整个 student/Hydra 做 global-norm clipping。日志中多次出现 `gnorm=Infinity`；逐 tensor clipping 无法可靠处理非有限梯度。

### 6. 十轮内 weight decay 调度过快

原本面向长训练的 0.04 → 0.4 cosine weight decay 被压缩到 10 epochs。它不是 epoch 0/1 NaN 的唯一原因，但可能加剧后期参数收缩和 representation degeneration。

## 结论

本轮没有找到“Hydra 有效工作”的配置。预训练 Hydra head 能让训练在数值上持续十轮，但 backbone 仍在三轮内完全 collapse。`momentum_teacher=0.9995` 是合理的较慢 EMA 设置，却不能单独解决问题。k、temperature、tau、out_dim、Hydra LR、negative filtering 和关闭 gate 均未改变这一总体结论。

预训练 gate 的准确结论是：它被正确加载并产生高度稀疏、非平凡的 mask，但目前没有证据证明它对 scratch backbone 有效。现有迁移同时加载了整个 Hydra，不能把任何效果单独归因于 gate。

## 下一轮建议

下一轮应先建立一个不会 collapse、不会 NaN 的优化基线，再继续大规模超参数消融。

1. 将 backbone peak LR 从 0.002 至少降低 10 倍。对于当前线性缩放规则，可从 `--lr 5e-5`（有效 peak 2e-4）开始。
2. 对十轮 pilot 使用至少 2–5 epochs warmup，或暂时固定低 LR；1 epoch warmup 不适合当前 scratch ViT。
3. 固定或显著降低 weight decay，例如 0.04，不在十轮内升到 0.4。
4. 使用真正的 global-norm clipping，并记录 backbone、projector、predictor、gate 的独立梯度 norm、GradScaler scale 和 skipped steps。
5. 保留 `momentum_teacher=0.9995`，但不要把它视为主要修复。
6. 加入一个稳定的 ungated instance/self-distillation objective，防止 Hydra 条件 gate 为 backbone 提供 shortcut。
7. 给 gate 增加明确约束，例如 activation target、entropy regularization 或下限 `eps + (1-eps) sigmoid(x)`。
8. 在稳定基线上做严格的五组迁移对照：全随机、gate-only、projector/predictor-only、完整 Hydra、all-ones gate。
9. 下一轮优先保留 `k=8` 和 temperature 0.4 作为较稳定候选；暂不扩大 out_dim。

建议成功标准：10 epochs 内无 NaN；`cls_cos` 不持续上升并保持显著低于 1；`cls_pair_cos` 高于非配对相似度；`zs_std` 不接近 0；loss 明显低于随机基线且 `ref_h-ref_ent` 同时下降。

## 复现与原始数据

- 消融配置入口：`run_hydra_ablation.py`
- 正式输出：`/scratch/project_465003400/SSLHydra/outputs/hydra-ablation-10ep/`
- Array A 日志：`/scratch/project_465003400/SSLHydra/outputs/slurm/hydra-ab10-a-22661042_*.{out,err}`
- Array B 日志：`/scratch/project_465003400/SSLHydra/outputs/slurm/hydra-ab10-b-22661043_*.{out,err}`
- Frozen-backbone Hydra 输出：`/scratch/project_465003400/SSLHydra/outputs/hydra-k8-fneg-t02-ibot-frozen/`
- Frozen-backbone checkpoint：`checkpoint0001.pth` 至 `checkpoint0009.pth`

### 执行异常说明

最初的 array `22660932` 不属于正式实验数据：它一方面超过站点单 array 的 8-task limit，另一方面因默认 `warmup_teacher_temp_epochs=30` 大于总 epochs=10，导致未使用的 iBOTLoss 在初始化 schedule 时出现 negative-dimension 错误。修复后重新提交为两个 8-task arrays，即本报告使用的 `22661042` 和 `22661043`。
