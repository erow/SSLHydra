# Hydra: Self-Supervised Contrastive Learning in Pair-Conditioned Subspaces

Hydra extends the pair-conditioned feature filtering idea of [SimLAP](https://arxiv.org/html/2410.18200v2) to self-supervised learning. Rather than requiring every positive pair to agree across the entire representation space, Hydra learns a gate that selects the coordinates in which a particular pair should be compared. A frozen reference encoder provides the information used to select partners and construct gates, while a momentum teacher supplies the contrastive keys.

The formulation below describes the **current implementation**: two global views, an independent EMA key projector, key-only BN, and no additional instance-discrimination loss. Some details in `docs/hydra.md`, particularly the single-view description and denominator masking, refer to earlier versions.

## 1. Setup

Let \(\mathcal D=\{x_i\}_{i=1}^{M}\) be an unlabeled image dataset. For each image in a minibatch, two independently augmented views are generated:

\[
v_i^{(1)}=\mathcal A_1(x_i),
\qquad
v_i^{(2)}=\mathcal A_2(x_i).
\]

Hydra distinguishes two sources of teacher information.

**Frozen reference encoder.** A pretrained encoder provides a fixed descriptor \(t_i\in\mathbb R^{C}\) for each image. These descriptors are precomputed and standardized coordinate-wise over the dataset:

\[
\widetilde t_{i,c}
=
\frac{t_{i,c}-\mu_c}{\max(\sigma_c,\epsilon)}.
\]

The reference descriptors guide partner sampling and gate construction. They receive no gradients and are not updated by EMA.

**Momentum teacher.** An online encoder \(f_\theta\), projector \(p_\phi\), and predictor \(h_\psi\) produce the query embeddings:

\[
q_i^{(a)}
=
h_\psi\!\left(p_\phi(f_\theta(v_i^{(a)}))\right)
\in\mathbb R^{D}.
\]

Let \(E\) be the backbone embedding width, \(H\) the MLP hidden width, and \(D\) the Hydra output width. These widths are independent: the online and teacher projectors map \(E\to H\to D\), the predictor maps \(D\to H\to D\), key BN has \(D\) channels, and the gate maps the reference descriptor through \(H\) hidden units to \(D\) coordinates. `--hydra_out_dim` controls \(D\); zero preserves the old behavior \(D=E\). `--hydra_mlp_dim` controls \(H\).

For ViT-B, \(E=768\). Binding \(D\) to \(E\) is therefore an architectural choice rather than a requirement of the contrastive objective.

The teacher encoder \(f_{\bar\theta}\) and its independent projector \(p_{\bar\phi}\) produce the keys:

\[
k_i^{(a)}
=
\operatorname{sg}\!\left[
\operatorname{BN}_{\mathrm{key}}
\left(p_{\bar\phi}(f_{\bar\theta}(v_i^{(a)}))\right)
\right],
\]

where \(\operatorname{sg}\) denotes stop-gradient.

The online projector and predictor terminate in Layer Normalization. The teacher projector mirrors the online projector, including its LN, and is followed by an additional non-affine BN applied **only to keys**. During training, key BN uses batch statistics; in distributed training it becomes SyncBatchNorm. The implementation normalizes both global views together before splitting them into their respective key sets.

Teacher parameters follow the online parameters through EMA:

\[
\bar\theta\leftarrow m\bar\theta+(1-m)\theta,
\qquad
\bar\phi\leftarrow m\bar\phi+(1-m)\phi.
\]

Thus, the teacher is frozen with respect to backpropagation, but evolves during training. Key BN maintains its own running statistics rather than obtaining them through EMA.

## 2. Contrastive Learning in Subspaces

### 2.1 Teacher-guided partner sampling

Let \(\mathcal B\) denote the global minibatch containing \(N\) distinct images across all workers. Normalize the reference descriptors:

\[
u_i=\frac{\widetilde t_i}{\|\widetilde t_i\|_2}.
\]

For anchor \(i\), define a distribution over the other images:

\[
\pi_i(j)
=
\begin{cases}
\displaystyle
\frac{\exp(u_i^\top u_j/\tau_{\mathrm p})}
{\sum_{n\ne i}\exp(u_i^\top u_n/\tau_{\mathrm p})},
&j\ne i,\\[8pt]
0,&j=i.
\end{cases}
\]

A positive partner is sampled as

\[
j_i\sim\pi_i.
\]

The partner temperature \(\tau_{\mathrm p}\) controls the concentration of the sampling distribution. Smaller values favor highly similar reference descriptors; larger values approach uniform sampling.

Importantly, similarity controls **which pair is selected**. It does not directly prescribe the student’s similarity score.

### 2.2 Pair-conditioned similarity

For the sampled pair \((i,j)\), let \(g_{ij}\in(0,1)^D\) be its gate. Define

\[
T_{ij}(z)
=
\frac{g_{ij}\odot z}
{\max(\|g_{ij}\odot z\|_2,\epsilon)}.
\]

The cross-view similarity between query \(i\) and candidate key \(n\) is

\[
s_{in\mid j}^{a\rightarrow b}
=
T_{ij}(q_i^{(a)})^\top
T_{ij}(k_n^{(b)}),
\qquad a\ne b.
\]

**The same gate is applied to the query and every candidate key in its denominator.** Each candidate is therefore evaluated in the comparison space selected for the positive pair.

Ignoring numerical stabilization, this is equivalent to

\[
s_{in\mid j}
=
\frac{
q_i^\top M_{ij}k_n
}{
\sqrt{q_i^\top M_{ij}q_i}
\sqrt{k_n^\top M_{ij}k_n}
},
\qquad
M_{ij}=\operatorname{diag}(g_{ij}^{\,2}).
\]

Hydra therefore learns a pair-dependent diagonal similarity metric.

### 2.3 Subspace contrastive objective

For a sampled partner \(j\ne i\),

\[
\ell_{i,j}^{a\rightarrow b}
=
-\log
\frac{
\exp(s_{ij\mid j}^{a\rightarrow b}/\tau_{\mathrm c})
}{
\sum_{n\ne i}
\exp(s_{in\mid j}^{a\rightarrow b}/\tau_{\mathrm c})
}.
\]

Here, \(\tau_{\mathrm c}\) is the contrastive temperature, distinct from \(\tau_{\mathrm p}\).

The current similarity-based configuration excludes the anchor’s own cross-view key \(k_i^{(b)}\) from the denominator. The sampled partner remains in the denominator as the sole positive; all other included keys are negatives.

With \(K\) independent partner draws per anchor and direction,

\[
\mathcal L_{\mathrm{Hydra}}
=
\frac{1}{2NK}
\sum_{(a,b)\in\{(1,2),(2,1)\}}
\sum_{i=1}^{N}
\sum_{r=1}^{K}
\ell_{i,j_i^{(a,b,r)}}^{a\rightarrow b}.
\]

This is the complete Hydra-only objective. **No additional instance-discrimination term is included.**

## 3. Gate Design

### 3.1 Pair representation

The gate input is the mean of the two standardized reference descriptors:

\[
r_{ij}
=
\frac{\widetilde t_i+\widetilde t_j}{2}.
\]

This construction is symmetric:

\[
r_{ij}=r_{ji}.
\]

Consequently, the gate architecture does not encode an ordering between the two members of a pair, although the contrastive loss itself remains directional.

### 3.2 Gate network

The implemented gate is

\[
g_{ij}
=
\sigma\!\left(
W_2\,
\operatorname{ReLU}
\left(
\operatorname{BN}_{g}(W_1r_{ij}+b_1)
\right)
+b_2
\right).
\]

The gate has a hidden BN layer; this is separate from key BN and does not normalize query embeddings.

Each gate coordinate determines the relative contribution of a latent dimension:

- A value near zero suppresses that coordinate.
- A larger value gives that coordinate more influence in the pair-conditioned similarity.

Because sigmoid gates are continuous, the operation is more precisely **soft coordinate reweighting** than an exact lower-dimensional projection. An approximate coordinate subspace emerges when many gate values become small.

### 3.3 Capacity in the pair-specific-coordinate extreme

Suppose every sampled unordered pair must receive one coordinate that no other pair uses. For \(P\) simultaneously represented pairs this construction requires \(D\ge P\). With global batch 256, shared directional partners, and \(K=8\), one step contains at most \(P=256\times8=2048\) sampled anchor-partner pairs, ignoring duplicate and reciprocal draws. This motivates \(D=2048\) as a useful stress test. Assigning a separate coordinate to every possible ImageNet pair would instead require combinatorial width and is not a feasible design.

Hydra does not require one-hot gates: a \(D\)-coordinate sigmoid gate can use distributed and overlapping masks, so \(D\ge P\) is an extreme upper-bound construction rather than a general lower bound. Hidden width still matters. When \(D>H\), the final linear layers produce \(D\) coordinates through an \(H\)-dimensional bottleneck, which can prevent the additional coordinates from behaving independently.

### 3.4 Gradient flow

The reference descriptors, partner draws, and teacher keys are detached. Nevertheless, the gate remains trainable:

\[
\nabla_{\omega}\mathcal L_{\mathrm{Hydra}}\ne0,
\qquad
\nabla_{\bar\theta}\mathcal L_{\mathrm{Hydra}}
=
\nabla_{\bar\phi}\mathcal L_{\mathrm{Hydra}}
=0,
\]

where \(\omega\) denotes gate parameters.

Stop-gradient on a key prevents its producer from receiving gradients; it does **not** prevent gradients through the multiplication of that key by the trainable gate.

No explicit sparsity or binary-gate penalty is used.

## 4. Discussion

### Relation to SimLAP

SimLAP motivates learning shared features of potentially disparate pairs through pair-conditioned feature filtering. Hydra adopts this principle while replacing class-based conditioning with frozen image descriptors and introducing a momentum target branch.

These changes make Hydra a self-supervised extension rather than a direct reproduction. In particular, SimLAP reports that projector normalization materially affects convergence, favoring LN over BN in its evaluated architecture. Hydra’s key-only BN is a separate design choice requiring empirical validation. [SimLAP, §5.2](https://arxiv.org/html/2410.18200v2#S5.SS2)

### Why condition the comparison space?

A pair may share only some visual properties. Enforcing agreement across all coordinates can also suppress properties that distinguish its members. Pair-conditioned weighting allows the objective to emphasize selected coordinates while reducing pressure on others.

This is an architectural motivation, not a guarantee that the learned coordinates correspond to interpretable factors.

### Why separate the key projector?

A shared trainable projector lets the loss change both queries and keys immediately. An independent EMA projector removes that direct optimization path and provides targets that evolve more gradually.

Key BN centers and rescales the target coordinates. It does not guarantee information preservation: identical inputs remain identical after normalization, and BN cannot restore variation that the encoder has already discarded.

### What does \(K\) control?

Increasing \(K\) gives a more accurate Monte Carlo estimate of the expected pair-conditioned loss. It increases the number of gate evaluations and contrastive comparisons, while reusing the same backbone embeddings.

The resulting gates need not be distinct or orthogonal. Thus, \(K\) counts sampled pair-conditioned comparisons, not necessarily \(K\) different learned subspaces.

### Limitations and evaluation

The method depends on the quality and dataset alignment of the frozen descriptors. Sampling can select weakly related partners, and other included keys may share relevant semantics despite being treated as negatives. Pair averaging also discards information that cannot be recovered from the mean.

Moreover, because the gate depends on the sampled partner, the prediction distribution changes with that partner. The entropy of \(\pi_i\) is therefore not generally a lower bound on this gated objective.

Evaluation should measure both representation utility and collapse indicators: downstream accuracy, normalized feature covariance, effective rank, and positive–negative similarity separation. A low training loss or a key standard deviation near one is insufficient by itself.

## 5. Algorithm

**Algorithm 1: Hydra training with an EMA teacher and key-only BN**

**Inputs:** unlabeled dataset \(\mathcal D\); frozen descriptors \(\{\widetilde t_i\}\); temperatures \(\tau_{\mathrm p},\tau_{\mathrm c}\); number of draws \(K\); momentum schedule \(m_t\).

**Initialize:** online encoder, projector, predictor, and gate. Copy the online encoder and projector parameters into their teacher counterparts. Disable gradients for teacher parameters.

```text
repeat for each training iteration:

    Sample images and their dataset indices.
    Generate two augmented views of every image.
    Fetch the corresponding frozen reference descriptors.

    Query branch:
        q = predictor(online_projector(online_encoder(views)))
        # Projector and predictor use LN.

    Teacher branch, without gradients:
        k = teacher_projector(teacher_encoder(views))
        k = key_BN(k)
        # Normalize the two views together, then split them.

    Gather detached keys and reference descriptors across workers.
    Construct reference-based partner distributions.

    loss = 0

    for each direction (view 1 → view 2, view 2 → view 1):
        repeat K times:

            Sample one non-self partner for each anchor.
            Form the mean reference descriptor of each pair.
            Compute the pair-conditioned sigmoid gate.

            Apply that gate to the query and every candidate key.
            L2-normalize the gated vectors.
            Compute cosine logits divided by contrastive temperature.

            Exclude each anchor's own cross-view key.
            Add cross-entropy with the sampled partner as target.

    Average over anchors, directions, and partner draws.

    Update online encoder, projector, predictor, and gate
    using gradients of the averaged loss.

    Update teacher encoder and teacher projector by EMA.
```

## 6. Implementation Details

`src/ibot/main_ibot.py` trains this loss on the first global crop. `--loss ibot` uses only the iBOT objective, `--loss hydra` uses only this loss, and `--loss both` adds them. Student latent `z_s` and momentum-teacher latent `z_t` are the `[CLS]` head outputs. The gate is trained; `t_p` is not.

Hydra builds a contrastive loss (对比损失) on sample pairs. A frozen iBOT teacher picks the partner. The pair's teacher vectors set a gate (门控). The gate projects the student and teacher latents (潜变量) into one subspace (子空间), and InfoNCE treats the partner as the only positive. Repeat this K times and average.

Symbols for one local batch of size `B`. Latent width is `D`. Teacher width is `8192`.

| symbol | shape | meaning |
| --- | --- | --- |
| `idx` | `[B]` | FFCV sample index of each image |
| `t_p` | `[B, 8192]` | teacher vector of each image |
| `t_idx` | `[B]` | partner index inside this batch |
| `gate` | `[B, D]` | subspace weights for each pair |
| `z_s`, `z_t` | `[B, D]` | student latent, teacher latent |

`z_s` and `z_t` are the contrastive embeddings of the same `B` images (student network and momentum teacher, or the two views). They are L2-normalized only after the gate multiply.

## `t_p`

Teacher vectors live at

`/scratch/project_465003400/SSLHydra/outputs/imagenet_teacher_cls_ffcv`

| file | shape | dtype |
| --- | --- | --- |
| `train_teacher_cls.npy` | `[1281167, 8192]` | float16 |
| `train_labels.npy` | `[1281167]` | int64 |
| `val_teacher_cls.npy` | `[50000, 8192]` | float16 |
| `val_labels.npy` | `[50000]` | int64 |

Row `i` is the iBOT teacher `[CLS]` head output for FFCV sample `i` of `IN1K_train_smart500_q90.ffcv` (val: the matching val beton). `meta.json` records `out_dim: 8192` and checkpoint `vitb_16_rand_mask`. These are raw head outputs, not a softmax.

`IBotFFCVLoader` yields `idx`, the FFCV row id for that batch. It is the same id `scripts/imagenet_embed.py` writes back when it merges shards.

```text
table = np.load(train_teacher_cls.npy, mmap_mode="r")   # [1281167, 8192]
t_p = table[idx].float()                                # [B, 8192]
t_p = t_p.detach()
```

No gradient enters `t_p`. Keep the memmap on CPU and copy only the `B` rows.

## `t_idx`

`t_idx[b]` is the partner of sample `b` inside the gathered batch. Three strategies. `sim` and `rand` refuse a self-pair. `self` is the same sample.

### 1. Teacher similarity

Cosine similarity of the teacher vectors, then a temperature softmax. `tau` is this sampling temperature, not the InfoNCE temperature.

```text
u = normalize(t_p, dim=-1)                 # [B, 8192]
m = u @ u.T                                # [B, B]
m.fill_diagonal_(-inf)
p = softmax(m / tau, dim=-1)               # [B, B]
t_idx = multinomial(p, num_samples=1)      # [B], one draw per row
```

`p[b]` is a distribution over the other `B - 1` rows. Draw one partner per row. Draws are independent across the K steps below.

### 2. Random

Uniform over the other indices in the batch.

```text
t_idx = randint(0, B - 1)                  # [B]
t_idx = t_idx + (t_idx >= arange(B))       # shift so t_idx[b] != b
```

### 3. Self

The positive is the teacher of the same sample. Student latent `z_s[b]` is paired with teacher latent `z_t[b]`.

```text
t_idx = arange(B)                          # after the gather, this rank's rows
```

The gate then reads `t_p[b]` alone, because the two teacher vectors are the same.

## `gate = g((t_p + t_p[t_idx]) / 2)`

The gate reads the pair, not a class id. Average the two teacher vectors, then run the gate network `g`.

```text
pair = (t_p + t_p[t_idx]) / 2              # [B, 8192]
gate = g(pair)                             # [B, D], each value in (0, 1)
```

`g` is an MLP ending in `sigmoid`, same role as `BasicGate` in `src/ibot/filter.py`. Input width is `8192`, output width is `D`. There is no label embedding: `t_p` already is the teacher vector. `fuse=True` in `BasicGate` averages two embeddings and then applies the MLP; Hydra averages `t_p` first and does the same.

`gate[b, d]` near 0 drops latent coordinate `d` for pair `b`. `gate[b, d]` near 1 keeps it. One gate is one subspace.

## `loss_hydra`

Project both latents with that gate, then InfoNCE. The positive is the partner. Every sample in the key set stays in the denominator (分母).

Per query `b`, the same `gate[b]` multiplies the student vector and every teacher key:

```text
q = normalize(z_s * gate, dim=-1)                         # [B, D]
k = normalize(z_t[None, :, :] * gate[:, None, :], dim=-1) # [B, B, D]
logits = einsum("bd,bnd->bn", q, k) / temperature         # [B, B]
loss = cross_entropy(logits, t_idx)
```

`logits[b, t_idx[b]]` is the positive. `logits[b, :]` is the denominator: all `B` teacher keys, including the partner. This is `Filter.gated_contrast`: `fused_gated_contrast(z_s, z_t, gate)` returns those cosines; divide by `temperature` and take cross-entropy against `t_idx`.

`temperature` is the InfoNCE temperature. It is not `tau`.

Do not drop the partner, and do not add extra positives. If `z_s` and `z_t` are the same tensor and the strategy is not `self`, set `logits[b, b] = -inf` before the cross-entropy so the anchor is not its own key. `self` keeps that column: it is the positive. If the two tensors come from two networks, leave the diagonal in.

Across ranks, `dist.all_gather` both `z_t` and `t_p`. Both are teacher-side tensors, so the gather has no gradient. Sampling uses the gathered `t_p`, and `t_idx` is already an index into that same tensor. The InfoNCE label is `t_idx`. Do not add `rank * B` again. The denominator is the global batch.

## K subspaces

One step is one partner draw, one gate, one InfoNCE, one subspace. Repeat K times with the same `t_p`, `z_s`, and `z_t`. Each step draws a new `t_idx` and therefore a new gate.

```text
loss_hydra = 0
for _ in range(K):
    t_idx = sample_partner(t_p)                  # strategy 1 or 2
    gate = g((t_p + t_p[t_idx]) / 2)
    logits = fused_gated_contrast(z_s, z_t, gate) / temperature
    if z_s is z_t:
        logits.fill_diagonal_(-inf)
    loss_hydra += cross_entropy(logits, t_idx)
loss_hydra = loss_hydra / K
```

K steps are K subspaces. The scalar `loss_hydra` is their mean. `g` is shared across the K steps; only `t_idx` and `gate` change.

## Do not repeat

The 8×128 run stayed at `log(1024) ≈ 6.931` for 15 epochs. That number is a uniform guess over the global batch: the positive column is not winning.

- Gather `z_t` and `t_p` with `dist.all_gather`, not `gather_with_grad`. `gather_with_grad` was only applied to `z_t`. Partners were drawn from the local `t_p`, then the label was `t_idx + rank * B`. After the partner index became global, that extra offset pointed at the wrong key. One gather, one index, used both as the partner row and as the cross-entropy label.
- On a gathered key matrix of shape `[B, world * B]`, this rank's own key is column `local + rank * B`, not the diagonal. `fill_diagonal_(-inf)` masks the wrong column on every rank except rank 0. And only do that when `z_s is z_t`. Training passes the student and the momentum teacher, two tensors. Leave that column in. It is a different vector and a valid negative.
- A smoke test that draws a new partner every step cannot show that the loss is valid. Job 22522567 moved from 3.46 to 2.55 (delta 0.92) and failed a "drop by more than 1" check, because the target moved. Freeze the draw (`smoke_hydra.py` reseeds each step). Job 22522680 then went from 3.44 to 0.02 on one batch of 32.
