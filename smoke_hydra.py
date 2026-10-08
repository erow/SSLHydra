"""Overfit Hydra on one frozen batch. The loss is valid if it falls."""

import math
import json
import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from fused_gated_contrast import fused_gated_contrast
from loader import IBotFFCVLoader
from main_ibot import (
    Hydra, TeacherTable, _all_gather, _cls_token, _grad_norm, _merge_param_groups,
    get_args_parser, _sample_rows,
)


def _one_batch(loader):
    images, labels, masks, idx = next(iter(loader))
    # Hold a private copy. FFCV reuses the batch storage on the next read.
    return (
        [im.clone() for im in images],
        labels.clone(),
        [m.clone() for m in masks],
        idx.clone(),
    )


def _build(args):
    if args.data_path.endswith('.ffcv'):
        import numba  # noqa: F401
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    pred_size = args.patch_size * 8 if 'swin' in args.arch else args.patch_size
    loader = IBotFFCVLoader(
        args.data_path,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        global_crops_scale=args.global_crops_scale,
        local_crops_scale=args.local_crops_scale,
        global_crops_number=args.global_crops_number,
        local_crops_number=args.local_crops_number,
        patch_size=pred_size,
        distributed=utils.get_world_size() > 1,
        seed=args.seed,
        pred_ratio=args.pred_ratio,
        pred_ratio_var=args.pred_ratio_var,
        pred_aspect_ratio=(0.3, 1 / 0.3),
        pred_shape=args.pred_shape,
        pred_start_epoch=args.pred_start_epoch,
    )
    args.arch = args.arch.replace('deit', 'vit')
    student = models.__dict__[args.arch](
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path,
        return_all_tokens=True,
        masked_im_modeling=args.use_masked_im_modeling,
    )
    teacher = models.__dict__[args.arch](
        patch_size=args.patch_size,
        return_all_tokens=True,
    )
    embed_dim = student.embed_dim
    student = utils.MultiCropWrapper(student, nn.Identity())
    teacher = utils.MultiCropWrapper(teacher, nn.Identity())
    student, teacher = student.cuda(), teacher.cuda()
    student = nn.parallel.DistributedDataParallel(
        student, device_ids=[args.gpu], find_unused_parameters=True)
    teacher.load_state_dict(student.module.state_dict(), strict=False)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    if args.hydra_pair == 'label':
        table = None
        teacher_dim = 0
    else:
        table = TeacherTable(args.teacher_cls)
        teacher_dim = table.dim
    loss_class = DiagnosticHydra if getattr(args, "diagnose", False) else Hydra
    hydra = loss_class(
        teacher_dim, embed_dim, args.hydra_mlp_dim, args.hydra_temperature,
        args.hydra_tau, args.hydra_k, args.hydra_pair,
        out_dim=args.hydra_out_dim or embed_dim,
    ).cuda()
    if utils.has_batchnorms(hydra):
        hydra = nn.SyncBatchNorm.convert_sync_batchnorm(hydra)
    hydra = nn.parallel.DistributedDataParallel(hydra, device_ids=[args.gpu])
    groups = _merge_param_groups(
        utils.get_params_groups(student), utils.get_params_groups(hydra))
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.)
    return loader, student, teacher, hydra, table, opt


def _cls(wrapper, images, masks):
    out = wrapper(images) if masks is None else wrapper(images, mask=masks)
    tokens = out[0] if isinstance(out, tuple) else out
    return _cls_token(tokens)


def _view_cls(wrapper, images, masks, view, b):
    """Backbone [CLS] of one global crop."""
    return _cls(wrapper, images, masks)[view * b:(view + 1) * b]


def _loss(student, teacher, hydra, table, batch, args, freeze_student=False):
    images, labels, masks, idx = batch
    images = [im.cuda(non_blocking=True) for im in images]
    masks = [msk.cuda(non_blocking=True) for msk in masks]
    crops = images[:args.global_crops_number]
    crop_masks = masks[:args.global_crops_number]
    if freeze_student:
        with torch.no_grad():
            cls_s = _cls(student, crops, crop_masks)
    else:
        cls_s = _cls(student, crops, crop_masks)
    cls_t = _cls(teacher, crops, None)
    if args.hydra_pair == 'label':
        guide = labels.cuda(non_blocking=True)
    else:
        guide = table.take(idx).cuda(non_blocking=True)
    return hydra(cls_s, cls_t, guide)


@torch.no_grad()
def _ema(student, teacher, hydra, momentum):
    online = dict(student.module.named_parameters())
    for name, target in teacher.named_parameters():
        if name in online:
            target.mul_(momentum).add_(online[name], alpha=1 - momentum)
    hydra.module.update_teacher(momentum)


def _gate_line(hydra):
    stats = hydra.module.gate_stats
    return (
        f"gate_mean={stats['mean']:.4f} gate_std={stats['std']:.4f} "
        f"gate_lo={stats['lo']:.4f} gate_hi={stats['hi']:.4f}"
    )


def _run_epoch(args, loader, student, teacher, hydra, table, opt):
    """One pass over the training set. A new partner is drawn every step."""
    loader.set_epoch(0)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_fp16)
    student.train()
    n = len(loader)
    first = last = None
    gate_first = gate_last = None
    for step, batch in enumerate(loader):
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(args.use_fp16):
            loss = _loss(student, teacher, hydra, table, batch, args)
        value = float(loss.detach())
        if not math.isfinite(value):
            print(f'hydra_epoch step={step} nonfinite loss={value}', flush=True)
            sys.exit(1)
        scaler.scale(loss).backward()
        if args.clip_grad:
            scaler.unscale_(opt)
            utils.clip_gradients(student, args.clip_grad)
            utils.clip_gradients(hydra, args.clip_grad)
        scaler.step(opt)
        scaler.update()
        _ema(student, teacher, hydra, args.momentum_teacher)
        last = value
        gate_last = dict(hydra.module.gate_stats)
        if first is None:
            first = last
            gate_first = dict(gate_last)
        if step % 100 == 0 or step + 1 == n:
            print(
                f'hydra_epoch step={step}/{n} loss={last:.4f} {_gate_line(hydra)}',
                flush=True,
            )
    line = (
        f'HYDRA_EPOCH steps={n} loss_first={first:.4f} loss_last={last:.4f} '
        f'delta={first - last:.4f} '
        f"gate_mean_first={gate_first['mean']:.4f} gate_mean_last={gate_last['mean']:.4f} "
        f"gate_lo_last={gate_last['lo']:.4f} gate_hi_last={gate_last['hi']:.4f}"
    )
    print(line, flush=True)
    if utils.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'result.txt'), 'w') as f:
            f.write(line + '\n')


def _weights(student, teacher, hydra):
    return tuple(
        {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
        for module in (student, teacher, hydra)
    )


def _restore(student, teacher, hydra, opt, saved):
    for module, state in zip((student, teacher, hydra), saved):
        module.load_state_dict(state)
    opt.state.clear()


def _snapshot(student, teacher, hydra, table, batch, args):
    """How sharp the sim draw is, and whether that column leads in z."""
    images, labels, masks, idx = batch
    images = [im.cuda(non_blocking=True) for im in images]
    masks = [msk.cuda(non_blocking=True) for msk in masks]
    labels = labels.cuda()
    b = labels.shape[0]
    h = hydra.module
    with torch.no_grad():
        crops = images[:args.global_crops_number]
        h_s = _view_cls(student, crops, masks[:args.global_crops_number], 0, b)
        h_t = _view_cls(teacher, crops, None, 1, b)
        t_p = table.take(idx).cuda()
        z_s = h.predictor(h.projector(h_s))
        with torch.no_grad():
            z_t = h.key_bn(h.teacher_projector(h_t))
        keys = _all_gather(z_t)
        t_all = _all_gather(t_p)
        y = _all_gather(labels.view(-1, 1)).view(-1)
        rows = torch.arange(b, device=z_s.device) + utils.get_rank() * b
        sim = F.normalize(t_all[rows], dim=-1) @ F.normalize(t_all, dim=-1).T
        sim[torch.arange(b, device=sim.device), rows] = float('-inf')
        prob = torch.softmax(sim / h.tau, dim=-1)
        mode = prob.argmax(dim=-1)
        ent = float((-(prob * prob.clamp_min(1e-12).log()).sum(-1)).mean())
        same = y.unsqueeze(0).eq(y[rows].unsqueeze(1))
        pair = (t_all[rows] + t_all[mode]) * 0.5
        gate = h.gate(pair).sigmoid()
        logits = fused_gated_contrast(z_s, keys, gate) / h.temperature
        local = torch.arange(b, device=logits.device)
        pos = logits[local, mode]
        neg = (logits.sum(-1) - pos) / (logits.shape[1] - 1)
    n = logits.shape[1]
    return (
        f'n={n} logn={math.log(n):.4f} top1={float(prob.max(-1).values.mean()):.4f} '
        f'ent={ent:.3f}/{math.log(n - 1):.3f} '
        f'same={float(y[mode].eq(y[rows]).float().mean()):.4f} '
        f'same_mass={float((prob * same).sum(-1).mean()):.4f} '
        f'logit_pos={float(pos.mean()):.4f} logit_neg={float(neg.mean()):.4f} '
        f'logit_std={float(logits.float().std()):.4f} '
        f'col_acc={float(logits.argmax(-1).eq(mode).float().mean()):.4f} '
        f'gate_mean={float(gate.float().mean()):.4f}'
    )


def _fit_batch(args, student, teacher, hydra, table, opt, batch, reseed, tag):
    """Fit one batch. reseed=True keeps the partner; False draws a new one."""
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_fp16)
    student.eval()
    hydra.train()
    first = last = None
    for step in range(args.steps):
        if reseed:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(args.use_fp16):
            loss = _loss(student, teacher, hydra, table, batch, args)
        value = float(loss.detach())
        if not math.isfinite(value):
            print(f'{tag} step={step} nonfinite loss={value}', flush=True)
            sys.exit(1)
        scaler.scale(loss).backward()
        if args.clip_grad:
            scaler.unscale_(opt)
            utils.clip_gradients(student, args.clip_grad)
            utils.clip_gradients(hydra, args.clip_grad)
        scaler.step(opt)
        scaler.update()
        _ema(student, teacher, hydra, args.momentum_teacher)
        last = value
        if first is None:
            first = last
        if step % 50 == 0 or step + 1 == args.steps:
            print(
                f'{tag} step={step} loss={last:.4f} {_gate_line(hydra)}',
                flush=True,
            )
    dropped = first - last
    line = f'{tag} first={first:.4f} last={last:.4f} delta={dropped:.4f}'
    print(line, flush=True)
    return line


def _monitor_line(hydra, gnorm):
    m = hydra.module.monitor
    return (
        f"ref_h={m['ref_h']:.4f} ref_ent={m['ref_ent']:.4f} "
        f"zs_std={m['zs_std']:.4f} zt_std={m['zt_std']:.4f} gnorm={gnorm:.4f}"
        + (f" hit={m['hit']:.3f}" if 'hit' in m else '')
    )


def _stream(args, loader, student, teacher, hydra, table, opt, steps, tag, warmup_steps=0):
    """New images every step, as in training. Returns the summary line.

    The first ``warmup_steps`` update only Hydra. The backbone stays eval and
    out of the graph, then the remaining steps train both.
    """
    loader.set_epoch(0)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_fp16)
    student.train()
    hydra.train()
    first = last = at_warmup = None
    for step, batch in enumerate(loader):
        if step >= steps:
            break
        hydra_only = step < warmup_steps
        if hydra_only and step == 0:
            student.eval()
            student.requires_grad_(False)
        elif warmup_steps and step == warmup_steps:
            student.train()
            student.requires_grad_(True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(args.use_fp16):
            loss = _loss(
                student, teacher, hydra, table, batch, args, freeze_student=hydra_only)
        value = float(loss.detach())
        if not math.isfinite(value):
            print(f'{tag} step={step} nonfinite loss={value}', flush=True)
            sys.exit(1)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = _grad_norm(student, hydra)
        if args.clip_grad:
            utils.clip_gradients(student, args.clip_grad)
            utils.clip_gradients(hydra, args.clip_grad)
        scaler.step(opt)
        scaler.update()
        _ema(student, teacher, hydra, args.momentum_teacher)
        last = value
        if first is None:
            first = last
        if hydra_only and step + 1 == warmup_steps:
            at_warmup = value
        phase = 'warmup' if hydra_only else 'full'
        boundary = warmup_steps and step + 1 == warmup_steps
        if step % 20 == 0 or step + 1 == steps or boundary:
            print(
                f'{tag} step={step} phase={phase} loss={last:.4f} '
                f'{_monitor_line(hydra, gnorm)}',
                flush=True,
            )
    warm = f' at_warmup={at_warmup:.4f}' if at_warmup is not None else ''
    line = (
        f'{tag} warmup={warmup_steps} first={first:.4f}{warm} '
        f'last={last:.4f} delta={first - last:.4f} '
        f'{_monitor_line(hydra, gnorm)}'
    )
    print(line, flush=True)
    return line


def _load_ibot(student, teacher):
    """Start from the iBOT teacher that wrote the reference table."""
    dest = os.path.join(
        os.environ.get('TORCH_HOME', '/scratch/project_465003400/cache/torch'),
        'ibot', 'vitb_16_rand_mask_checkpoint.pth')
    if not os.path.isfile(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        url = (
            'https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/'
            'archive/2022/ibot/vitb_16_rand_mask/checkpoint.pth')
        torch.hub.download_url_to_file(url, dest)
    try:
        ckpt = torch.load(dest, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(dest, map_location='cpu')
    state = {k.replace('module.', ''): v for k, v in ckpt['teacher'].items()}
    inc_s = student.module.load_state_dict(state, strict=False)
    inc_t = teacher.load_state_dict(state, strict=False)
    if inc_t.missing_keys:
        raise RuntimeError('teacher checkpoint missing ' + ', '.join(inc_t.missing_keys[:8]))
    print(
        f'loaded ibot teacher from {dest} '
        f'student_missing={len(inc_s.missing_keys)} teacher_missing=0',
        flush=True,
    )


def _run_why(args, loader, student, teacher, hydra, table, opt):
    """H1: constant gate. H2: iBOT initialization. Same stream length for each."""
    saved = _weights(student, teacher, hydra)
    lines = []
    runs = (
        ('WHY_SCRATCH', False, False),
        ('WHY_ONES', True, False),
        ('WHY_IBOT', False, True),
    )
    for tag, force_ones, use_ibot in runs:
        _restore(student, teacher, hydra, opt, saved)
        if use_ibot:
            _load_ibot(student, teacher)
            opt.state.clear()
        hydra.module.force_ones = force_ones
        lines.append(_stream(
            args, loader, student, teacher, hydra, table, opt, args.why_steps, tag))
    if utils.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'result.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')


def _run_warmup(args, loader, student, teacher, hydra, table, opt):
    """Scratch init, learned gate. Warmup fits Hydra, then both train."""
    hydra.module.force_ones = False
    total = args.warmup_steps + args.steps
    print(
        f'WARMUP_SCRATCH init=scratch gate=learnable '
        f'warmup_steps={args.warmup_steps} full_steps={args.steps}',
        flush=True,
    )
    line = _stream(
        args, loader, student, teacher, hydra, table, opt, total,
        'WARMUP_SCRATCH', warmup_steps=args.warmup_steps,
    )
    if utils.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'result.txt'), 'w') as f:
            f.write(line + '\n')


class DiagnosticHydra(Hydra):
    """Smoke-only negative exclusion. Production Hydra is unchanged."""
    exclude_similar = False
    share_partners = False

    def forward(self, h_s, h_t, guide):
        self._shared_partners = []
        self.direction_partners = []
        loss = super().forward(h_s, h_t, guide)
        self.monitor['direction_partner_agreement'] = float(
            (self.direction_partners[0] == self.direction_partners[1]).float().mean())
        return loss

    def _direction(self, z_s, z_t, guide):
        keys = _all_gather(z_t.detach())
        refs = _all_gather(guide.detach())
        b = len(z_s)
        rows = torch.arange(b, device=z_s.device) + utils.get_rank() * b
        local = torch.arange(b, device=z_s.device)
        losses, monitors, partners = [], [], []
        for draw in range(self.k):
            if self.share_partners and len(self._shared_partners) > draw:
                partner = self._shared_partners[draw]
            else:
                partner = _sample_rows(refs, rows, 'sim', self.tau)
                if self.share_partners:
                    self._shared_partners.append(partner)
            partners.append(partner)
            gate = self.gate((refs[rows].float() + refs[partner].float()) * .5).sigmoid()
            logits = fused_gated_contrast(z_s.float(), keys.float(), gate) / self.temperature
            reference = F.normalize(refs[rows].float(), dim=-1) @ F.normalize(refs.float(), dim=-1).T
            excluded = torch.zeros_like(logits, dtype=torch.bool)
            excluded[local, rows] = True
            better = reference > reference[local, partner, None]
            better[local, rows] = False
            if self.exclude_similar:
                excluded |= better
            excluded[local, partner] = False
            logits = logits.masked_fill(excluded, -torch.inf)
            loss = F.cross_entropy(logits, partner)
            with torch.no_grad():
                counts = (~excluded).sum(1).float()
                pos = logits[local, partner]
                self.monitor = {
                    'loss': float(loss), 'candidates': float(counts.mean()),
                    'uniform_loss': float(counts.log().mean()),
                    'gain_over_uniform': float(counts.log().mean() - loss),
                    'top1': float((logits.argmax(1) == partner).float().mean()),
                    'positive_rank': float(1 + (logits > pos[:, None]).sum(1).float().mean()),
                    'reference_better_negative_fraction': float(better.any(1).float().mean()),
                    'zs_std': float(z_s.float().std(0, unbiased=False).mean()),
                    'zt_std': float(keys.float().std(0, unbiased=False).mean()),
                }
                self.probe_monitor = dict(self.monitor)
                self.probe_state = (z_s.detach(), keys.detach(), gate.detach(), excluded, partner)
            losses.append(loss)
            monitors.append(dict(self.monitor))
        self.direction_partners.append(torch.stack(partners))
        self.monitor = {name: sum(m[name] for m in monitors) / self.k for name in monitors[0]}
        return sum(losses) / self.k



def _mean_metrics(metrics):
    names = sorted(metrics)
    values = torch.tensor([metrics[k] for k in names], device='cuda', dtype=torch.float64)
    torch.distributed.all_reduce(values)
    values /= utils.get_world_size()
    return dict(zip(names, values.cpu().tolist()))


@torch.no_grad()
def _diagnostic_probe(args, student, teacher, hydra, table, batch, pair_seed):
    # Preserve running BN statistics and RNG: measurement must not change training.
    modules = (student, teacher, hydra)
    buffers = [(b, b.clone()) for m in modules for b in m.buffers()]
    modes = [m.training for m in modules]
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    student.eval(); teacher.eval(); hydra.train()
    probe_k, probe_shared = hydra.module.k, hydra.module.share_partners
    hydra.module.k, hydra.module.share_partners = 1, False
    torch.manual_seed(pair_seed)
    _loss(student, teacher, hydra, table, batch, args)
    result = dict(hydra.module.probe_monitor)
    q, keys, gate, excluded, target = hydra.module.probe_state
    global_q = _all_gather(q)
    offset = utils.get_rank() * len(q)
    shuffled = global_q.roll(1, 0)[offset:offset + len(q)]
    constant = global_q.mean(0, keepdim=True).expand_as(q)
    for name, changed in [('shuffled_query_loss', shuffled), ('constant_query_loss', constant)]:
        logits = fused_gated_contrast(changed.float(), keys.float(), gate.float()) / hydra.module.temperature
        result[name] = float(F.cross_entropy(logits.masked_fill(excluded, -torch.inf), target))
    result = _mean_metrics(result)
    # All workers already have global tensors. Compute the spectrum only on rank 0.
    if utils.is_main_process():
        for name, features in [('query', global_q), ('key', keys)]:
            z = F.normalize(features.float(), dim=-1)
            result[name + '_mean_cosine'] = float(z.mean(0).square().sum())
            spectrum = torch.linalg.svdvals(z - z.mean(0)).square()
            mass = spectrum / spectrum.sum().clamp_min(1e-30)
            result[name + '_effective_rank'] = float((-(mass * mass.clamp_min(1e-30).log()).sum()).exp())
        result['shuffle_delta'] = result['shuffled_query_loss'] - result['loss']
        result['constant_delta'] = result['constant_query_loss'] - result['loss']
    for buffer, saved in buffers:
        buffer.copy_(saved)
    for module, mode in zip(modules, modes):
        module.train(mode)
    hydra.module.k, hydra.module.share_partners = probe_k, probe_shared
    torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state(gpu_rng)
    return result


def _run_diagnose(args, loader, student, teacher, hydra, table, opt):
    if args.use_fp16 or args.hydra_pair != 'sim' or args.hydra_k != 1:
        raise ValueError('diagnose requires FP32, sim pairing and initial K=1')
    if args.load_from:
        checkpoint = torch.load(args.load_from, map_location='cpu', weights_only=False)
        for name, model in [('student', student), ('teacher', teacher), ('hydra', hydra)]:
            model.load_state_dict(checkpoint[name], strict=True)
        print('DIAG checkpoint_epoch=' + str(checkpoint['epoch']), flush=True)
    saved = _weights(student, teacher, hydra)
    pretrained_saved = None
    if args.stream_study or args.stream_best:
        _load_ibot(student, teacher)
        pretrained_saved = _weights(student, teacher, hydra)
        _restore(student, teacher, hydra, opt, saved)
    utils.fix_random_seeds(args.seed)
    loader.set_epoch(0)
    probe_batch = _one_batch(loader)
    runs = [
        ('fixed_mask', True, True, False),
        ('fixed_nomask', True, False, False),
        ('stream_mask', False, True, False),
        ('stream_nomask', False, False, False),
        ('stream_exclude_mask', False, True, True),
        ('stream_exclude_nomask', False, False, True),
    ]
    if args.partner_control:
        runs = [
            ('fixed_mask', True, True, False),
            ('fixed_redraw_mask', True, True, False),
            ('fixed_nomask', True, False, False),
            ('fixed_redraw_nomask', True, False, False),
        ]
    study_configs = {}
    if args.stream_study:
        runs = []
        for pretrained in (False, True):
            for shared in (False, True):
                for k in (1, 8):
                    tag = '{}_{}_k{}'.format('ibot' if pretrained else 'scratch',
                                             'shared' if shared else 'independent', k)
                    study_configs[tag] = (pretrained, shared, k)
                    runs.append((tag, False, True, False))
    if args.stream_best:
        tag = 'ibot_shared_k8_long'
        study_configs[tag] = (True, True, 8)
        runs = [(tag, False, True, False)]
    output = os.path.join(args.output_dir, 'diagnostics.jsonl')
    if utils.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    def record(data):
        if utils.is_main_process():
            print('DIAG ' + json.dumps(data), flush=True)
            with open(output, 'a') as f:
                f.write(json.dumps(data) + '\n')
    record({'stage': 'config', 'steps': args.steps, 'global_batch': args.batch_size_per_gpu * utils.get_world_size(),
            'lr': args.lr, 'seed': args.seed, 'checkpoint': args.load_from,
            'hydra_out_dim': hydra.module.out_dim, 'hydra_mlp_dim': args.hydra_mlp_dim,
            'precision': 'fp32', 'instance_loss': False, 'partner_control': args.partner_control, 'stream_study': args.stream_study})
    for tag, fixed, masking, exclude in runs:
        fixed_partner = fixed and not tag.startswith('fixed_redraw')
        pretrained, shared, k = study_configs.get(tag, (False, False, 1))
        _restore(student, teacher, hydra, opt, pretrained_saved if pretrained else saved)
        hydra.module.share_partners, hydra.module.k = shared, k
        record({'tag': tag, 'stage': 'condition', 'pretrained_backbone': pretrained,
                'shared_partners': shared, 'K': k, 'masking': masking})
        utils.fix_random_seeds(args.seed)
        loader.set_epoch(0)
        student.module.backbone.masked_im_modeling = masking
        hydra.module.exclude_similar = exclude
        student.train(not fixed); teacher.eval(); hydra.train()
        pair_seed = args.seed if fixed else args.seed + 987
        initial = _diagnostic_probe(args, student, teacher, hydra, table, probe_batch, pair_seed)
        record(dict(tag=tag, stage='before', **initial))
        iterator = None if fixed else iter(loader)
        history = []
        previous_partner = None
        previous_gradient = None
        for step in range(args.steps):
            batch = probe_batch if fixed else next(iterator)
            if fixed_partner:
                # No stochastic depth in eval mode; reseeding fixes both directional partners.
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed(args.seed)
            opt.zero_grad(set_to_none=True)
            loss = _loss(student, teacher, hydra, table, batch, args)
            if not torch.isfinite(loss):
                raise RuntimeError(f'{tag}: nonfinite loss at {step}')
            loss.backward()
            norm = _grad_norm(student, hydra)
            if not math.isfinite(norm):
                raise RuntimeError(f'{tag}: nonfinite gradient at {step}')
            gradient = torch.cat([p.grad.detach().flatten()[::max(1, p.numel() // 128)]
                                  for model in (student, hydra) for p in model.parameters()
                                  if p.grad is not None])
            grad_cos = 0.0 if previous_gradient is None else float(F.cosine_similarity(
                gradient[None], previous_gradient[None]).item())
            previous_gradient = gradient.clone()
            if args.clip_grad:
                utils.clip_gradients(student, args.clip_grad)
                utils.clip_gradients(hydra, args.clip_grad)
            opt.step()
            _ema(student, teacher, hydra, args.momentum_teacher)
            partner = hydra.module.probe_state[-1]
            changed = 0.0 if previous_partner is None else float((partner != previous_partner).float().mean())
            previous_partner = partner.clone()
            metrics = _mean_metrics(dict(hydra.module.monitor, gnorm=norm, partner_change_fraction=changed, consecutive_gradient_cosine=grad_cos))
            history.append(metrics)
            if step % 25 == 0 or step + 1 == args.steps:
                record(dict(tag=tag, stage='train', step=step, **metrics))
        if iterator is not None:
            iterator.close()
        final = _diagnostic_probe(args, student, teacher, hydra, table, probe_batch, pair_seed)
        window = history[-min(50, len(history)):]
        final['last50_gradient_cosine'] = sum(x['consecutive_gradient_cosine'] for x in window) / len(window)
        mean_norm = sum(x['gnorm'] for x in window) / len(window)
        final['last50_gnorm_mean'] = mean_norm
        final['last50_gnorm_cv'] = (sum((x['gnorm'] - mean_norm)**2 for x in window) / len(window))**.5 / max(mean_norm, 1e-12)
        record(dict(tag=tag, stage='after', **final,
                    train_first=history[0]['loss'], train_last=history[-1]['loss'],
                    train_last20=sum(x['loss'] for x in history[-20:]) / min(20, len(history))))
    torch.distributed.barrier()


def main():
    parser = get_args_parser()
    parser.add_argument('--steps', default=1000, type=int)
    parser.add_argument('--diagnose', action='store_true', help='Controlled collapse/false-negative ablations')
    parser.add_argument('--stream_study', action='store_true',
                        help='With --diagnose, factorial pretrained/shared-partner/K=1,8 stream experiments')
    parser.add_argument('--stream_best', action='store_true',
                        help='With --diagnose, run pretrained/shared-partner/K=8 stream validation')
    parser.add_argument('--partner_control', action='store_true',
                        help='With --diagnose, compare fixed vs redrawn partners on identical frozen images')
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--warmup_steps', default=100, type=int)
    parser.add_argument('--one_epoch', action='store_true')
    parser.add_argument('--why', action='store_true')
    parser.add_argument('--why_steps', default=60, type=int)
    args = parser.parse_args()
    loader, student, teacher, hydra, table, opt = _build(args)
    if args.diagnose:
        _run_diagnose(args, loader, student, teacher, hydra, table, opt)
        return
    if args.hydra_pair == 'label' and not (args.warmup or args.one_epoch or args.why):
        hydra.module.force_ones = False
        print('LABEL_SCRATCH init=scratch gate=learnable pair=label', flush=True)
        line = _stream(
            args, loader, student, teacher, hydra, table, opt, args.steps, 'LABEL_SCRATCH')
        if utils.is_main_process():
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, 'result.txt'), 'w') as f:
                f.write(line + '\n')
        return
    if args.warmup:
        _run_warmup(args, loader, student, teacher, hydra, table, opt)
        return
    if args.one_epoch:
        _run_epoch(args, loader, student, teacher, hydra, table, opt)
        return
    if args.why:
        _run_why(args, loader, student, teacher, hydra, table, opt)
        return
    batch = _one_batch(loader)
    student.eval()
    snap = 'HYDRA_SNAP ' + _snapshot(student, teacher, hydra, table, batch, args)
    print(snap, flush=True)
    saved = _weights(student, teacher, hydra)
    # A redrawn partner is what training does. A frozen partner is the control.
    redrawn = _fit_batch(
        args, student, teacher, hydra, table, opt, batch, False, 'HYDRA_REDRAW')
    _restore(student, teacher, hydra, opt, saved)
    frozen = _fit_batch(
        args, student, teacher, hydra, table, opt, batch, True, 'HYDRA_FROZEN')
    lines = [snap, redrawn, frozen]
    if utils.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'result.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
