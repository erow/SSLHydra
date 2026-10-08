# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import os
import shutil
import sys
import datetime
import time
import math
import json
import numpy as np
import utils
import models
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from models.head import iBOTHead
from loader import IBotFFCVLoader, ImageFolderMask
from evaluation.unsupervised.unsup_cls import eval_pred
from fused_gated_contrast import fused_gated_contrast

def get_args_parser():
    parser = argparse.ArgumentParser('iBOT', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'deit_tiny', 'deit_small',
                 'swin_tiny','swin_small', 'swin_base', 'swin_large'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--window_size', default=7, type=int, help="""Size of window - default 7.
        This config is only valid for Swin Transofmer and is ignoired for vanilla ViT architectures.""")
    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
        output for [CLS] token.""")
    parser.add_argument('--patch_out_dim', default=8192, type=int, help="""Dimensionality of
        output for patch tokens.""")
    parser.add_argument('--shared_head', default=False, type=utils.bool_flag, help="""Wether to share 
        the same head for [CLS] token output and patch tokens output. When set to false, patch_out_dim
        is ignored and enforced to be same with out_dim. (Default: False)""")
    parser.add_argument('--shared_head_teacher', default=True, type=utils.bool_flag, help="""See above.
        Only works for teacher model. (Defeault: True)""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--norm_in_head', default=None,
        help="Whether to use batch normalizations in projection head (Default: None)")
    parser.add_argument('--act_in_head', default='gelu',
        help="Whether to use batch normalizations in projection head (Default: gelu)")
    parser.add_argument('--use_masked_im_modeling', default=True, type=utils.bool_flag,
        help="Whether to use masked image modeling (mim) in backbone (Default: True)")
    parser.add_argument('--pred_ratio', default=0.3, type=float, nargs='+', help="""Ratio of partial prediction.
        If a list of ratio is specified, one of them will be randomly choosed for each patch.""")
    parser.add_argument('--pred_ratio_var', default=0, type=float, nargs='+', help="""Variance of partial prediction
        ratio. Length should be indentical to the length of pred_ratio. 0 for disabling. """)
    parser.add_argument('--pred_shape', default='block', type=str, help="""Shape of partial prediction.""")
    parser.add_argument('--pred_start_epoch', default=0, type=int, help="""Start epoch to perform masked
        image prediction. We typically set this to 50 for swin transformer. (Default: 0)""")
    parser.add_argument('--lambda1', default=1.0, type=float, help="""loss weight for dino
        loss over [CLS] tokens (Default: 1.0)""")
    parser.add_argument('--lambda2', default=1.0, type=float, help="""loss weight for beit 
        loss over masked patch tokens (Default: 1.0)""")
        
    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_patch_temp', default=0.04, type=float, help="""See 
        `--warmup_teacher_temp`""")
    parser.add_argument('--teacher_patch_temp', default=0.07, type=float, help=""""See 
        `--teacher_temp`""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--max_steps', default=0, type=int,
        help='Stop each epoch after this many steps. 0 runs the full epoch.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--load_from', default=None, help="""Path to load checkpoints to resume training.""")
    parser.add_argument('--drop_path', type=float, default=0.1, help="""Drop path rate for student network.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_number', type=int, default=2, help="""Number of global
        views to generate. Default is to use two global crops. """)
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.14, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=0, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path',
        default='/scratch/project_465003400/datasets/ffcv/IN1K_train_smart500_q90.ffcv', type=str,
        help='ImageNet train folder, or an FFCV .ffcv file.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--no-wandb', action='store_true', help='Skip Weights & Biases. Rank 0 logs otherwise.')
    parser.add_argument('--saveckp_freq', default=40, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    parser.add_argument('--loss', default='both', choices=['ibot', 'hydra', 'both'],
        help='Train the iBOT loss, the Hydra loss, or the sum of both.')
    # Hydra: gated InfoNCE on teacher-chosen partners. See docs/hydra.md.
    parser.add_argument('--teacher_cls', default=(
        '/scratch/project_465003400/SSLHydra/outputs/imagenet_teacher_cls_ffcv/train_teacher_cls.npy'),
        type=str, help='Memmap of frozen iBOT teacher [CLS] vectors, row = FFCV index.')
    parser.add_argument('--hydra_tau', default=0.1, type=float,
        help='Softmax temperature for teacher-similarity partner sampling.')
    parser.add_argument('--hydra_temperature', default=0.1, type=float,
        help='InfoNCE temperature. Distinct from --hydra_tau.')
    parser.add_argument('--hydra_k', default=1, type=int,
        help='Number of partner draws / subspaces averaged into loss_hydra.')
    parser.add_argument('--hydra_filter_negatives', default=True, type=utils.bool_flag,
        help='Remove denominator keys whose teacher similarity to the anchor exceeds the sampled pair.')
    parser.add_argument('--hydra_pair', default='sim', choices=['sim', 'rand', 'self', 'label'],
        help='Partner index: teacher similarity, uniform, the same sample, or another image of the same class.')
    parser.add_argument('--hydra_mlp_dim', default=4096, type=int,
        help='Hidden width of the gate MLP.')
    parser.add_argument('--hydra_out_dim', default=2048, type=int,
        help='Projector, predictor, key, and gate width. 0 uses the backbone embed_dim.')
    parser.add_argument('--hydra_weight', default=1.0, type=float,
        help='Multiplier on loss_hydra when --loss is hydra or both.')
    parser.add_argument('--init_ibot', default='', type=str,
        help='iBOT checkpoint. Copy its teacher weights into the student and teacher. Epoch stays 0.')
    return parser


def _init_from_ibot(student, teacher, path):
    """Copy an iBOT teacher into both networks. Do not restore epoch or optimizer."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location='cpu')
    raw = ckpt['teacher'] if isinstance(ckpt, dict) and 'teacher' in ckpt else ckpt
    state = {k.replace('module.', ''): v for k, v in raw.items()}
    inc_s = student.module.load_state_dict(state, strict=False)
    inc_t = teacher.load_state_dict(state, strict=False)
    if inc_t.missing_keys:
        raise RuntimeError('teacher checkpoint missing ' + ', '.join(inc_t.missing_keys[:8]))
    print(
        f'loaded ibot teacher from {path} '
        f'student_missing={len(inc_s.missing_keys)} teacher_missing=0',
        flush=True,
    )

def train_ibot(args):
    # Numba records builtins.print at import. init_distributed_mode replaces
    # print afterwards, and a later FFCV import then crashes inside Numba.
    if args.data_path.endswith('.ffcv'):
        import numba  # noqa: F401
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    pred_size = args.patch_size * 8 if 'swin' in args.arch else args.patch_size
    mask_kwargs = dict(
        pred_ratio=args.pred_ratio,
        pred_ratio_var=args.pred_ratio_var,
        pred_aspect_ratio=(0.3, 1/0.3),
        pred_shape=args.pred_shape,
        pred_start_epoch=args.pred_start_epoch,
    )
    if args.data_path.endswith('.ffcv'):
        data_loader = IBotFFCVLoader(
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
            **mask_kwargs,
        )
        print(f"Data loaded: there are {data_loader.num_samples} images.")
    else:
        transform = DataAugmentationiBOT(
            args.global_crops_scale,
            args.local_crops_scale,
            args.global_crops_number,
            args.local_crops_number,
        )
        dataset = ImageFolderMask(
            args.data_path,
            transform=transform,
            patch_size=pred_size,
            **mask_kwargs)
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is of hierechical features (i.e. swin_tiny, swin_small, swin_base)
    if args.arch in models.__dict__.keys() and 'swin' in args.arch:
        student = models.__dict__[args.arch](
            window_size=args.window_size,
            return_all_tokens=True, 
            masked_im_modeling=args.use_masked_im_modeling,
        )
        teacher = models.__dict__[args.arch](
            window_size=args.window_size,
            drop_path_rate=0.0,
            return_all_tokens=True,
        )
        embed_dim = student.num_features
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base, vit_large)
    elif args.arch in models.__dict__.keys():
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
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # Hydra reads the backbone [CLS]. The iBOT head is only for the iBOT loss.
    def _ibot_head(shared):
        return iBOTHead(
            embed_dim, args.out_dim, patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head, act=args.act_in_head,
            norm_last_layer=args.norm_last_layer, shared_head=shared,
        )
    if args.loss == 'hydra':
        student_head, teacher_head = nn.Identity(), nn.Identity()
    else:
        student_head = _ibot_head(args.shared_head)
        teacher_head = _ibot_head(args.shared_head_teacher)
    student = utils.MultiCropWrapper(student, student_head)
    teacher = utils.MultiCropWrapper(teacher, teacher_head)
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], broadcast_buffers=False) if \
            'swin' in args.arch else nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    # Hydra uses the [CLS] head only, so the patch-token parameters get no gradient.
    student = nn.parallel.DistributedDataParallel(
        student, device_ids=[args.gpu], broadcast_buffers='swin' not in args.arch,
        find_unused_parameters=(args.loss == 'hydra'))
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    same_dim = args.shared_head or args.shared_head_teacher
    ibot_loss = iBOTLoss(
        args.out_dim,
        args.out_dim if same_dim else args.patch_out_dim,
        args.global_crops_number,
        args.local_crops_number,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_patch_temp,
        args.teacher_patch_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        mim_start_epoch=args.pred_start_epoch,
    ).cuda()

    if utils.is_main_process():
        if not args.no_wandb:
            import subprocess
            import wandb
            os.environ.setdefault("WANDB_START_METHOD", "thread")
            # Job cwd is /tmp/SSLHydra; the clone (and .git) lives in src/.
            git_root = str(Path(__file__).resolve().parent)
            settings = {"git_root": git_root}
            try:
                settings["git_commit"] = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=git_root, text=True,
                ).strip()
            except (OSError, subprocess.SubprocessError):
                pass
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "SSLHydra"),
                entity=os.environ.get("WANDB_ENTITY") or None,
                name=os.environ.get("WANDB_NAME") or os.environ.get("SLURM_JOB_NAME") or Path(args.output_dir).name,
                dir=args.output_dir,
                config=vars(args),
                settings=settings,
            )
            wandb.define_metric('epoch')
            wandb.define_metric('train/*', step_metric='epoch')
            wandb.define_metric('global_step')
            wandb.define_metric('train_step/*', step_metric='global_step')
        
    # ============ Hydra gate + frozen teacher table ... ============
    use_hydra = args.loss in ('hydra', 'both')
    if use_hydra:
        if args.hydra_pair == 'label':
            teacher_table = None
            teacher_dim = 0
        else:
            teacher_table = TeacherTable(args.teacher_cls)
            teacher_dim = teacher_table.dim
        hydra = Hydra(
            teacher_dim,
            embed_dim,
            args.hydra_mlp_dim,
            args.hydra_temperature,
            args.hydra_tau,
            args.hydra_k,
            args.hydra_pair,
            out_dim=args.hydra_out_dim or embed_dim,
            filter_negatives=args.hydra_filter_negatives,
        ).cuda()
        if utils.has_batchnorms(hydra):
            hydra = nn.SyncBatchNorm.convert_sync_batchnorm(hydra)
        hydra = nn.parallel.DistributedDataParallel(hydra, device_ids=[args.gpu])
    else:
        teacher_table = None
        hydra = None

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if hydra is not None:
        params_groups = _merge_param_groups(params_groups, utils.get_params_groups(hydra))
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                            args.epochs, len(data_loader))
                  
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    if args.init_ibot:
        _init_from_ibot(student, teacher_without_ddp, args.init_ibot)
    if args.load_from:
        ckpt_kwargs = dict(
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            ibot_loss=ibot_loss,
        )
        if hydra is not None:
            ckpt_kwargs['hydra'] = hydra
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            **ckpt_kwargs,
        )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting iBOT training!")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        data_loader.dataset.set_epoch(epoch)

        # ============ training one epoch of iBOT ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, ibot_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, hydra, teacher_table, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'ibot_loss': ibot_loss.state_dict(),
        }
        if hydra is not None:
            save_dict['hydra'] = hydra.state_dict()
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and (epoch + 1) % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch + 1:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if not args.no_wandb:
                wandb.log({**{f'train/{k}': v for k, v in train_stats.items()}, 'epoch': epoch})

    if utils.is_main_process() and not args.no_wandb:
        wandb.finish()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def _wandb_log_step(metrics, global_step, epoch_progress, args):
    """Average scalar step metrics across ranks and log one W&B record."""
    if args.no_wandb:
        return
    keys = sorted(metrics)
    values = torch.tensor(
        [float(metrics[key]) for key in keys],
        dtype=torch.float64,
        device='cuda',
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values)
        values /= dist.get_world_size()
    if utils.is_main_process():
        import wandb
        wandb.log({
            **{f'train_step/{key}': value for key, value in zip(keys, values.cpu().tolist())},
            'global_step': global_step,
            'epoch_progress': epoch_progress,
        })



def train_one_epoch(student, teacher, teacher_without_ddp, ibot_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, hydra, teacher_table, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    
    # common params
    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in student.module.named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in teacher_without_ddp.named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
    params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]

    pred_labels, real_labels = [], []
    for local_it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if len(batch) == 4:
            images, labels, masks, idx = batch
        else:
            images, labels, masks = batch
            idx = None
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + local_it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        masks = [msk.cuda(non_blocking=True) for msk in masks]        
        
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # get global views
            crops = images[:args.global_crops_number]
            crop_masks = masks[:args.global_crops_number]
            if args.loss == 'both':
                # First return is the backbone. The head tuple is only for iBOT.
                teacher_backbone, teacher_output = teacher(crops, return_backbone_feat=True)
                student_backbone, student_output = student(
                    crops, mask=crop_masks, return_backbone_feat=True)
            elif args.loss == 'hydra':
                teacher_backbone, teacher_output = teacher(crops, return_backbone_feat=True)
                student_backbone, student_output = student(
                    crops, mask=crop_masks, return_backbone_feat=True)
            else:
                teacher_output = teacher(crops)
                student_output = student(crops, mask=crop_masks)
            
            # get local views
            student.module.backbone.masked_im_modeling = False
            student_local_cls = student(images[args.global_crops_number:])[0] if len(images) > args.global_crops_number else None
            student.module.backbone.masked_im_modeling = args.use_masked_im_modeling

            all_loss = {}
            loss = None
            if args.loss in ('ibot', 'both'):
                all_loss = ibot_loss(student_output, teacher_output, student_local_cls, masks, epoch)
                loss = all_loss.pop('loss')
            if args.loss in ('hydra', 'both'):
                if args.global_crops_number < 2:
                    raise RuntimeError("Hydra contrasts two global views")
                # Backbone [CLS] of both global views. Hydra chunks after the norm.
                if args.hydra_pair == 'label':
                    guide = labels.cuda(non_blocking=True)
                else:
                    if idx is None:
                        raise RuntimeError("Hydra needs the FFCV sample index; use an .ffcv data_path")
                    guide = teacher_table.take(idx).cuda(non_blocking=True)
                cls_s = _cls_token(student_backbone)
                cls_t = _cls_token(teacher_backbone)
                loss_hydra = hydra(cls_s, cls_t, guide)
                all_loss['hydra'] = loss_hydra
                term = args.hydra_weight * loss_hydra
                loss = term if loss is None else loss + term

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # log statistics. The prototype argmax exists only with an iBOT head.
        if isinstance(teacher_output, tuple) and teacher_output[0].ndim == 2:
            probs1 = teacher_output[0].chunk(args.global_crops_number)
            probs2 = student_output[0].chunk(args.global_crops_number)
            pred1 = utils.concat_all_gather(probs1[0].max(dim=1)[1])
            pred2 = utils.concat_all_gather(probs2[1].max(dim=1)[1])
            acc = (pred1 == pred2).sum() / pred1.size(0)
        else:
            pred1 = utils.concat_all_gather(torch.zeros(
                labels.shape[0], dtype=torch.long, device=labels.device if labels.is_cuda else 'cuda'))
            acc = 0.0
        pred_labels.append(pred1)
        real_labels.append(utils.concat_all_gather(labels.to(pred1.device)))

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            gnorm = _grad_norm(student, hydra)
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
                if hydra is not None:
                    utils.clip_gradients(hydra, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.unscale_(optimizer)
            gnorm = _grad_norm(student, hydra)
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
                if hydra is not None:
                    utils.clip_gradients(hydra, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            if hydra is not None:
                hydra.module.update_teacher(m)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        for key, value in all_loss.items():
            metric_logger.update(**{key: value.item()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(acc=acc)
        metric_logger.update(gnorm=gnorm)
        if hydra is not None:
            metric_logger.update(**hydra.module.monitor)
        if (local_it + 1) % 10 == 0:
            step_metrics = {
                'loss': loss.item(),
                **{key: value.item() for key, value in all_loss.items()},
                'lr': optimizer.param_groups[0]['lr'],
                'wd': optimizer.param_groups[0]['weight_decay'],
                'acc': acc,
                'gnorm': gnorm,
            }
            if hydra is not None:
                step_metrics.update(hydra.module.monitor)
            _wandb_log_step(
                step_metrics, it + 1,
                epoch + (local_it + 1) / len(data_loader), args,
            )
        if args.max_steps and local_it + 1 >= args.max_steps:
            break

    pred_labels = torch.cat(pred_labels).cpu().detach().numpy()
    real_labels = torch.cat(real_labels).cpu().detach().numpy()
    nmi, ari, fscore, adjacc = eval_pred(real_labels, pred_labels, calc_acc=False)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("NMI: {}, ARI: {}, F: {}, ACC: {}".format(nmi, ari, fscore, adjacc))
    print("Averaged stats:", metric_logger)
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return_dict.update({"nmi": nmi, "ari": ari, "fscore": fscore, "adjacc": adjacc})
    return return_dict


def _merge_param_groups(student_groups, hydra_groups):
    """Keep the iBOT split: group 0 follows the weight-decay schedule."""
    return [
        {'params': student_groups[0]['params'] + hydra_groups[0]['params']},
        {'params': student_groups[1]['params'] + hydra_groups[1]['params'], 'weight_decay': 0.},
    ]


def _stats_along_samples(table, chunk=65536):
    """Mean and std of each column, reduced over the sample axis."""
    n, d = table.shape
    total = np.zeros(d, dtype=np.float64)
    sq = np.zeros(d, dtype=np.float64)
    for start in range(0, n, chunk):
        x = np.asarray(table[start:start + chunk], dtype=np.float32)
        total += x.sum(axis=0, dtype=np.float64)
        sq += np.square(x, dtype=np.float64).sum(axis=0)
    mean = total / n
    var = sq / n - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def _teacher_column_stats(table, path):
    """Column mean and std. Rank 0 writes a cache; the other ranks only read it."""
    cache = os.path.splitext(path)[0] + '.colstats.npz'
    if os.path.isfile(cache):
        saved = np.load(cache)
        return saved['mean'], saved['std']
    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    if not distributed or dist.get_rank() == 0:
        print(f'teacher table stats: {table.shape[0]} rows -> {cache}', flush=True)
        mean, std = _stats_along_samples(table)
        tmp = cache + '.tmp.npz'
        np.savez(tmp, mean=mean, std=std)
        os.replace(tmp, cache)
        print('teacher table stats ready', flush=True)
    if distributed:
        dist.barrier()
    saved = np.load(cache)
    return saved['mean'], saved['std']


def _stage_teacher_table(path):
    """Copy the table onto this node's tmpfs, then random-read it from there.

    ``/tmp`` is per node, so local rank 0 on each node copies once. A size match
    skips the copy. ponytail: same filename and byte size is treated as the same
    table; hash the file if two different tables can share a name.
    """
    staged = os.path.join("/tmp", os.path.basename(path))
    if os.path.abspath(path) == os.path.abspath(staged):
        return path
    src_size = os.path.getsize(path)
    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    ready = os.path.isfile(staged) and os.path.getsize(staged) == src_size
    if local_rank == 0 and not ready:
        print(f'stage teacher table: {path} -> {staged}', flush=True)
        tmp = staged + ".tmp"
        shutil.copyfile(path, tmp)
        os.replace(tmp, staged)
        print('teacher table staged', flush=True)
    if distributed:
        dist.barrier()
    if not (os.path.isfile(staged) and os.path.getsize(staged) == src_size):
        raise FileNotFoundError(staged)
    return staged


class TeacherTable:
    """Frozen teacher vectors on node tmpfs. Copies one batch per step.

    Rows are z-scored along the sample axis: each dimension has mean 0 and
    std 1 over the full table, so a shared direction does not dominate.
    """

    def __init__(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        staged = _stage_teacher_table(path)
        self.table = np.load(staged, mmap_mode='r')
        self.dim = int(self.table.shape[1])
        self.mean, self.std = _teacher_column_stats(self.table, path)

    def take(self, idx):
        rows = np.asarray(self.table[idx.numpy()], dtype=np.float32)
        rows -= self.mean
        rows /= self.std
        return torch.from_numpy(rows)


def _all_gather(tensor):
    """Concatenate a teacher tensor across ranks. No gradient."""
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return tensor
    tensor = tensor.contiguous()
    parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, tensor)
    return torch.cat(parts, dim=0)


def _label_partner(labels, rows):
    """Another gathered key with the same class. ``has`` is false when this class appears once."""
    b = rows.shape[0]
    same = labels.unsqueeze(0).eq(labels[rows].unsqueeze(1))
    same[torch.arange(b, device=same.device), rows] = False
    has = same.any(dim=1)
    weights = same.float()
    # Empty rows need a positive weight so multinomial is defined. Those anchors are not in the loss.
    empty = torch.arange(b, device=same.device)[~has]
    if empty.numel():
        weights[empty, rows[empty]] = 1
    return torch.multinomial(weights, 1).squeeze(1), has


def _label_monitor(z_s, keys, labels, rows, logits, has):
    """Cross-entropy against a uniform distribution on the same-class keys."""
    with torch.no_grad():
        same = labels.unsqueeze(0).eq(labels[rows].unsqueeze(1))
        same[torch.arange(rows.shape[0], device=same.device), rows] = False
        weights = same.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1)
        log_p = logits.detach().float().log_softmax(dim=1)
        ref_h = -(weights * log_p).sum(dim=1)
        ref_ent = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1)
        if has.any():
            ref_h = float(ref_h[has].mean())
            ref_ent = float(ref_ent[has].mean())
        else:
            ref_h = ref_ent = 0.0
        return {
            'ref_h': ref_h,
            'ref_ent': ref_ent,
            'zs_std': float(z_s.detach().float().std(dim=0, unbiased=False).mean()),
            'zt_std': float(keys.detach().float().std(dim=0, unbiased=False).mean()),
            'hit': float(has.float().mean()),
        }


def _sample_rows(t_all, rows, strategy, tau):
    """Index into ``t_all``. ``self`` is ``rows``; ``sim`` and ``rand`` are not."""
    b = rows.shape[0]
    if strategy == 'self':
        return rows
    n = t_all.shape[0]
    if n < 2:
        raise RuntimeError(f"Hydra partner sampling needs at least 2 keys, got {n}")
    if strategy == 'rand':
        t_idx = torch.randint(0, n - 1, (b,), device=t_all.device)
        return t_idx + (t_idx >= rows).long()
    q = F.normalize(t_all[rows], dim=-1)
    k = F.normalize(t_all, dim=-1)
    m = q @ k.T
    m[torch.arange(b, device=m.device), rows] = float('-inf')
    return torch.multinomial(torch.softmax(m / tau, dim=-1), 1).squeeze(1)


def _hydra_monitor(z_s, keys, t_all, rows, logits, tau_s):
    """Reference cross-entropy, plus how spread the two latents are."""
    with torch.no_grad():
        ref = F.normalize(t_all.float(), dim=-1)
        sample_sim = ref @ ref.T
        # y indexes this rank's queries into the gathered reference.
        graph = sample_sim[rows[:, None], torch.arange(sample_sim.shape[1], device=sample_sim.device)]
        excluded = ~torch.isfinite(logits)
        graph = (graph / tau_s).masked_fill(excluded, float('-inf'))
        log_p = logits.detach().float().log_softmax(dim=1).masked_fill(excluded, 0)
        p_ref = graph.softmax(dim=1)
        ref_h = -(p_ref * log_p).sum(dim=1).mean()
        ref_ent = -(p_ref * p_ref.clamp_min(1e-12).log()).sum(dim=1).mean()
        return {
            'ref_h': float(ref_h),
            'ref_ent': float(ref_ent),
            'zs_std': float(z_s.detach().float().std(dim=0, unbiased=False).mean()),
            'zt_std': float(keys.detach().float().std(dim=0, unbiased=False).mean()),
        }


def _grad_norm(*models):
    sq = 0.0
    for model in models:
        if model is None:
            continue
        for p in model.parameters():
            if p.grad is not None:
                sq += float(p.grad.detach().float().pow(2).sum())
    return sq ** 0.5


def sample_partner(t_p, strategy, tau):
    """Partner index. ``self`` is the row itself; ``sim`` and ``rand`` are not."""
    b = t_p.shape[0]
    return _sample_rows(t_p, torch.arange(b, device=t_p.device), strategy, tau)


def _cls_token(tokens):
    """Backbone tokens are [B, 1+N, C]. The [CLS] token is index 0."""
    return tokens[:, 0] if tokens.ndim == 3 else tokens


class Hydra(nn.Module):
    """Own projector and predictor on backbone [CLS], then gated InfoNCE."""

    def __init__(self, teacher_dim, embed_dim, mlp_dim, temperature, tau, k, strategy,
                 out_dim=None, filter_negatives=False):
        super().__init__()

        out_dim = embed_dim if out_dim is None else out_dim
        if out_dim <= 0:
            raise ValueError(f"Hydra out_dim must be positive, got {out_dim}")

        def mlp(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, out_dim),
                nn.LayerNorm(out_dim),
            )

        # Query uses LN only. The matching EMA projector is followed by key-only BN.
        # Key BN uses batch statistics during training, even with gradients disabled.
        self.projector = mlp(embed_dim)
        self.teacher_projector = copy.deepcopy(self.projector).requires_grad_(False)
        self.key_bn = nn.BatchNorm1d(out_dim, affine=False)
        self.predictor = mlp(out_dim)
        # ImageNet class ids replace the 8192-d teacher row. The gate reads that embedding.
        self.label_embed = nn.Embedding(1000, mlp_dim) if strategy == 'label' else None
        gate_in = mlp_dim if strategy == 'label' else teacher_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_in, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, out_dim),
        )
        self.out_dim = out_dim
        self.temperature = temperature
        self.tau = tau
        self.k = k
        self.filter_negatives = filter_negatives
        self.strategy = strategy
        # Experiment switch: a constant gate ignores t_p when choosing the subspace.
        self.force_ones = False
        self.monitor = {}

    @torch.no_grad()
    def update_teacher(self, momentum):
        """EMA parameters only; key BN keeps its own batch-derived buffers."""
        for online, target in zip(self.projector.parameters(), self.teacher_projector.parameters()):
            target.mul_(momentum).add_(online, alpha=1 - momentum)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Old shared-projector checkpoints have a different normalization contract.
        # Refuse partial restoration, which would silently leave random EMA weights.
        if not any(key.startswith(prefix + 'teacher_projector.') for key in state_dict):
            raise RuntimeError("Hydra checkpoint lacks the EMA teacher projector; "
                               "start this architecture from scratch.")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _direction(self, z_s, z_t, guide):
        """One view as the query, the other view as the keys."""
        b = z_s.shape[0]
        keys = _all_gather(z_t.detach())
        label = self.strategy == 'label'
        gathered = _all_gather(guide.long().view(-1) if label else guide)
        offset = utils.get_rank() * b
        rows = torch.arange(b, device=z_s.device) + offset
        guide_sim = None
        if not label and (self.filter_negatives or self.strategy == 'sim') and self.strategy != 'self':
            with torch.no_grad():
                ref = F.normalize(gathered.float(), dim=-1)
                guide_sim = ref[rows] @ ref.T
        total = 0
        has = None
        filtered_records = []
        gate_records = []
        for _ in range(self.k):
            with torch.no_grad():
                if label:
                    t_idx, has = _label_partner(gathered, rows)
                else:
                    if self.strategy == 'sim':
                        sample_logits = guide_sim.masked_fill(F.one_hot(rows, guide_sim.shape[1]).bool(), float('-inf'))
                        t_idx = torch.multinomial(torch.softmax(sample_logits / self.tau, dim=-1), 1).squeeze(1)
                    else:
                        t_idx = _sample_rows(gathered, rows, self.strategy, self.tau)
            if label:
                feat = self.label_embed(gathered[rows])
            else:
                feat = (gathered[rows] + gathered[t_idx]) * 0.5
            # fp16 squares gates below 2.4e-4 to 0, and its clamp_min(1e-12) is 0 too: 0 * inf = NaN.
            with torch.autocast('cuda', enabled=False):
                raw = self.gate(feat.float()).sigmoid()
                # raw * 0 keeps the gate parameters in the DDP graph when the ablation is on.
                gate = raw if not self.force_ones else raw * 0 + 1
                with torch.no_grad():
                    g = gate.detach()
                    self.gate_stats = {
                        'mean': float(g.mean()),
                        'act': float(g.sum(1).mean()),
                        'std': float(g.std(unbiased=False)),
                        'lo': float((g < 0.05).float().mean()),
                        'hi': float((g > 0.95).float().mean()),
                    }
                    gate_records.append(self.gate_stats)
                logits = fused_gated_contrast(z_s.float(), keys.float(), gate) / self.temperature
            if label:
                term = F.cross_entropy(logits[has], t_idx[has]) if has.any() else logits.sum() * 0
            else:
                excluded = torch.zeros_like(logits, dtype=torch.bool)
                if self.strategy != 'self':
                    # The query's other view is not a negative; SimLAP keeps the anchor out of the denominator.
                    excluded.scatter_(1, rows[:, None], True)
                if self.filter_negatives and guide_sim is not None:
                    pair_sim = guide_sim.gather(1, t_idx[:, None])
                    # A key more similar than the sampled pair is an ambiguous false negative.
                    excluded |= guide_sim > pair_sim
                    excluded.scatter_(1, t_idx[:, None], False)
                    candidates = max(logits.shape[1] - 2, 1)
                    filtered = (excluded.sum(1) - 1).clamp_min(0).float() / candidates
                    filtered_records.append(float(filtered.mean()))
                logits = logits.masked_fill(excluded, float('-inf'))
                term = F.cross_entropy(logits, t_idx)
            total = total + term
        self.gate_stats = {
            key: sum(record[key] for record in gate_records) / len(gate_records)
            for key in gate_records[0]
        }
        if label:
            self.monitor = _label_monitor(z_s, keys, gathered, rows, logits, has)
        else:
            self.monitor = _hydra_monitor(z_s, keys, gathered, rows, logits, self.tau)
        self.monitor.update({
            'act': self.gate_stats['act'],
            'gate_mean': self.gate_stats['mean'],
            'gate_std': self.gate_stats['std'],
            'gate_lo': self.gate_stats['lo'],
            'gate_hi': self.gate_stats['hi'],
            'neg_filtered': (sum(filtered_records) / len(filtered_records) if filtered_records else 0.0),
        })
        return total / self.k

    def forward(self, h_s, h_t, guide):
        # Both views share one norm. Chunk only after that.
        z_s = self.predictor(self.projector(h_s))
        with torch.no_grad():
            z_t = self.key_bn(self.teacher_projector(h_t))
        guide = guide.detach()
        q0, q1 = z_s.chunk(2, dim=0)
        k0, k1 = z_t.chunk(2, dim=0)
        loss0 = self._direction(q0, k1, guide)
        mon0 = dict(self.monitor)
        loss1 = self._direction(q1, k0, guide)
        self.monitor = {key: (mon0[key] + self.monitor[key]) / 2 for key in mon0}
        # Backbone collapse shows here first; zs_std sits behind a norm layer and can hide it.
        with torch.no_grad():
            u = F.normalize(h_t.detach().float(), dim=-1)
            self.monitor['cls_cos'] = float((u @ u.T).mean())
        return (loss0 + loss1) / 2


def _check_hydra():
    raw = np.array([[1., 2., 3.], [3., 4., 5.], [5., 6., 9.]], dtype=np.float32)
    mean, std = _stats_along_samples(raw)
    got = (raw - mean) / std
    assert np.allclose(got.mean(0), 0, atol=1e-5)
    assert np.allclose(got.std(0), 1, atol=1e-5)
    torch.manual_seed(0)
    b, d, c = 4, 8, 6
    t_p = torch.tensor([
        [1., 0., 0., 0., 0., 0.],
        [0.9, 0.1, 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0.],
    ])
    u = F.normalize(t_p, dim=-1)
    m = u @ u.T
    m = m.masked_fill(torch.eye(b, dtype=torch.bool), float('-inf'))
    assert int(torch.softmax(m / 1e-3, dim=-1)[0].argmax()) == 1
    own = torch.arange(b)
    labels = torch.tensor([0, 0, 1, 2])
    partner, has = _label_partner(labels, own)
    assert has.tolist() == [True, True, False, False]
    assert partner[:2].tolist() == [1, 0]
    assert torch.equal(sample_partner(t_p, 'self', 0.1), own)
    for _ in range(20):
        drawn = sample_partner(t_p, 'rand', 0.1)
        assert not torch.equal(drawn, own)
        assert int((drawn == own).sum()) == 0
    hydra = Hydra(c, d, 16, temperature=0.2, tau=0.1, k=1, strategy='rand')
    h_s = torch.randn(2 * b, d, requires_grad=True)
    h_t = torch.randn(2 * b, d, requires_grad=True)
    torch.manual_seed(1)
    loss = hydra(h_s, h_t, t_p)
    torch.manual_seed(1)
    z_s = hydra.predictor(hydra.projector(h_s))
    z_t = hydra.key_bn(hydra.teacher_projector(h_t.detach()))
    q0, q1 = z_s.chunk(2, dim=0)
    k0, k1 = z_t.chunk(2, dim=0)

    def one(q, k):
        t_idx = sample_partner(t_p, 'rand', 0.1)
        gate = hydra.gate((t_p + t_p[t_idx]) * 0.5).sigmoid()
        logits = fused_gated_contrast(q, k, gate) / 0.2
        pair = F.cross_entropy(logits.masked_fill(torch.eye(b, dtype=torch.bool), float('-inf')), t_idx)
        return pair

    ref = (one(q0, k1) + one(q1, k0)) / 2
    assert torch.allclose(loss, ref), (float(loss), float(ref))
    loss.backward()
    assert h_s.grad is not None and float(h_s.grad.abs().sum()) > 0
    assert h_t.grad is None
    assert all(not p.requires_grad and p.grad is None for p in hydra.teacher_projector.parameters())
    assert hydra.projector[0].weight.grad is not None
    assert hydra.predictor[0].weight.grad is not None
    assert hydra.gate[0].weight.grad is not None
    assert t_p.grad is None
    assert math.isfinite(float(loss.detach()))
    hydra_self = Hydra(c, d, 16, temperature=0.2, tau=0.1, k=1, strategy='self')
    hs = torch.randn(2 * b, d, requires_grad=True)
    ht = torch.randn(2 * b, d)
    loss_self = hydra_self(hs, ht, t_p)
    zs = hydra_self.predictor(hydra_self.projector(hs))
    zt = hydra_self.key_bn(hydra_self.teacher_projector(ht))
    s0, s1 = zs.chunk(2, dim=0)
    t0, t1 = zt.chunk(2, dim=0)
    # Partner is this sample, so the gate sees t_p averaged with itself.
    gate_self = hydra_self.gate(t_p).sigmoid()
    ref_self = (
        F.cross_entropy(fused_gated_contrast(s0, t1, gate_self) / 0.2, own)
        + F.cross_entropy(fused_gated_contrast(s1, t0, gate_self) / 0.2, own)
    ) / 2
    assert torch.allclose(loss_self, ref_self), (float(loss_self), float(ref_self))
    z = torch.randn(2 * b, d, requires_grad=True)
    same = hydra(z, z, t_p)
    same.backward()
    assert math.isfinite(float(same.detach()))
    with torch.no_grad():
        shifted = hydra.key_bn(hydra.teacher_projector(h_t.detach()))
        assert float(shifted.std(0).mean()) > 0.5, "key BN should normalize a nonconstant batch"
    print('hydra ok', float(loss.detach()))


class iBOTLoss(nn.Module):
    def __init__(self, out_dim, patch_out_dim, ngcrops, nlcrops, warmup_teacher_temp, 
                 teacher_temp, warmup_teacher_temp2, teacher_temp2, 
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, 
                 center_momentum=0.9, center_momentum2=0.9,
                 lambda1=1.0, lambda2=1.0, mim_start_epoch=0):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp2_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2
        )) if mim_start_epoch == 0 else np.concatenate((
            np.ones(mim_start_epoch) * warmup_teacher_temp2,
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch) * teacher_temp2
        ))

    def forward(self, student_output, teacher_output, student_local_cls, student_mask, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output
        
        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])

        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops)
        
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                if v == q:
                    loss2 = torch.sum(-teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1), dim=-1)
                    mask = student_mask[v].flatten(-2, -1)
                    loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                    total_loss2 += loss2.mean()
                    n_loss_terms2 += 1
                else:
                    loss1 = torch.sum(-teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1), dim=-1)
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1
            
        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2
        total_loss = dict(cls=total_loss1, patch=total_loss2, loss=total_loss1 + total_loss2)
        self.update_center(teacher_cls, teacher_patch)                  
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
        self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (1 - self.center_momentum2)

class DataAugmentationiBOT(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # transformation for the rest of global crops
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('iBOT', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ibot(args)
