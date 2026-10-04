"""Overfit Hydra on one frozen batch. The loss is valid if it falls."""

import math
import os
import sys

import torch
import torch.nn as nn

import models
import utils
from loader import IBotFFCVLoader
from main_ibot import Hydra, TeacherTable, _merge_param_groups, get_args_parser
from models.head import iBOTHead


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
    student = utils.MultiCropWrapper(student, iBOTHead(
        embed_dim, args.out_dim, patch_out_dim=args.patch_out_dim,
        norm=args.norm_in_head, act=args.act_in_head,
        norm_last_layer=args.norm_last_layer, shared_head=args.shared_head,
    ))
    teacher = utils.MultiCropWrapper(teacher, iBOTHead(
        embed_dim, args.out_dim, patch_out_dim=args.patch_out_dim,
        norm=args.norm_in_head, act=args.act_in_head,
        shared_head=args.shared_head_teacher,
    ))
    student, teacher = student.cuda(), teacher.cuda()
    student = nn.parallel.DistributedDataParallel(
        student, device_ids=[args.gpu], find_unused_parameters=True)
    teacher.load_state_dict(student.module.state_dict(), strict=False)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    table = TeacherTable(args.teacher_cls)
    hydra = Hydra(
        table.dim, args.out_dim, args.hydra_mlp_dim, args.hydra_temperature,
        args.hydra_tau, args.hydra_k, args.hydra_pair,
    ).cuda()
    hydra = nn.parallel.DistributedDataParallel(hydra, device_ids=[args.gpu])
    groups = _merge_param_groups(
        utils.get_params_groups(student), utils.get_params_groups(hydra))
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.)
    return loader, student, teacher, hydra, table, opt


def _loss(student, teacher, hydra, table, batch, args):
    images, labels, masks, idx = batch
    images = [im.cuda(non_blocking=True) for im in images]
    masks = [msk.cuda(non_blocking=True) for msk in masks]
    b = labels.shape[0]
    with torch.no_grad():
        teacher_output = teacher(images[:args.global_crops_number])
    student_output = student(
        images[:args.global_crops_number], mask=masks[:args.global_crops_number])
    t_p = table.take(idx).cuda(non_blocking=True)
    return hydra(student_output[0][:b], teacher_output[0][:b], t_p)


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


def main():
    parser = get_args_parser()
    parser.add_argument('--steps', default=1000, type=int)
    parser.add_argument('--one_epoch', action='store_true')
    args = parser.parse_args()
    loader, student, teacher, hydra, table, opt = _build(args)
    if args.one_epoch:
        _run_epoch(args, loader, student, teacher, hydra, table, opt)
        return
    batch = _one_batch(loader)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_fp16)
    student.eval()
    first = None
    last = None
    for step in range(args.steps):
        # Same partner every step. Otherwise the target moves and the loss cannot fall to zero.
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(args.use_fp16):
            loss = _loss(student, teacher, hydra, table, batch, args)
        value = float(loss.detach())
        if not math.isfinite(value):
            print(f'hydra_overfit step={step} nonfinite loss={value}', flush=True)
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
            gnorm = 0.
            ngrad = 0
            for p in list(student.parameters()) + list(hydra.parameters()):
                if p.grad is not None:
                    gnorm += float(p.grad.detach().float().norm())
                    ngrad += 1
            print(
                f'hydra_overfit step={step} loss={last:.4f} {_gate_line(hydra)} '
                f'grad_norm={gnorm:.4f} ngrad={ngrad}',
                flush=True,
            )
    # Uniform InfoNCE is log(N). A real positive has to beat that by a wide margin.
    dropped = first - last
    ok = dropped > 1.0
    line = f'HYDRA_OVERFIT first={first:.4f} last={last:.4f} delta={dropped:.4f} ok={int(ok)}'
    print(line, flush=True)
    out = os.path.join(args.output_dir, 'result.txt')
    if utils.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        with open(out, 'w') as f:
            f.write(line + '\n')
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
