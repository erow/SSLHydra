"""Overfit Hydra on one frozen batch. The loss is valid if it falls."""

import math
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
    get_args_parser,
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
    table = TeacherTable(args.teacher_cls)
    hydra = Hydra(
        table.dim, embed_dim, args.hydra_mlp_dim, args.hydra_temperature,
        args.hydra_tau, args.hydra_k, args.hydra_pair,
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


def _loss(student, teacher, hydra, table, batch, args):
    images, labels, masks, idx = batch
    images = [im.cuda(non_blocking=True) for im in images]
    masks = [msk.cuda(non_blocking=True) for msk in masks]
    crops = images[:args.global_crops_number]
    crop_masks = masks[:args.global_crops_number]
    cls_s = _cls(student, crops, crop_masks)
    cls_t = _cls(teacher, crops, None)
    t_p = table.take(idx).cuda(non_blocking=True)
    return hydra(cls_s, cls_t, t_p)


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
        z_t = h.projector(h_t)
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
    )


def _stream(args, loader, student, teacher, hydra, table, opt, steps, tag):
    """New images every step, as in training. Returns the summary line."""
    loader.set_epoch(0)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_fp16)
    student.train()
    hydra.train()
    first = last = None
    for step, batch in enumerate(loader):
        if step >= steps:
            break
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(args.use_fp16):
            loss = _loss(student, teacher, hydra, table, batch, args)
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
        last = value
        if first is None:
            first = last
        if step % 20 == 0 or step + 1 == steps:
            print(
                f'{tag} step={step} loss={last:.4f} {_monitor_line(hydra, gnorm)}',
                flush=True,
            )
    line = (
        f'{tag} first={first:.4f} last={last:.4f} delta={first - last:.4f} '
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


def main():
    parser = get_args_parser()
    parser.add_argument('--steps', default=1000, type=int)
    parser.add_argument('--one_epoch', action='store_true')
    parser.add_argument('--why', action='store_true')
    parser.add_argument('--why_steps', default=60, type=int)
    args = parser.parse_args()
    loader, student, teacher, hydra, table, opt = _build(args)
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
