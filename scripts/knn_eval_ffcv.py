"""Weighted k-NN (DINO protocol) on ImageNet-1k, features from FFCV betons.

Backbone [CLS] feature of the chosen checkpoint key (default teacher), L2
normalized; train betons as the memory bank, val as queries. Several
checkpoints are evaluated in one job. ``random`` evaluates an untrained
ViT as a floor.

    ./cluster/lumi_ddp.sh python scripts/knn_eval_ffcv.py \
        --checkpoints name=/path/ckpt.pth name2=random \
        --output /scratch/.../knn.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "ibot"))

import models  # noqa: E402

FFCV_TRAIN = "/scratch/project_465003400/datasets/ffcv/IN1K_train_smart500_q90.ffcv"
FFCV_VAL = "/scratch/project_465003400/datasets/ffcv/IN1K_val_smart500_q90.ffcv"
MEAN = np.array([0.485, 0.456, 0.406]) * 255
STD = np.array([0.229, 0.224, 0.225]) * 255


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True, help="name=path|random")
    p.add_argument("--key", default="teacher")
    p.add_argument("--arch", default="vit_base")
    p.add_argument("--patch_size", default=16, type=int)
    p.add_argument("--train_path", default=FFCV_TRAIN)
    p.add_argument("--val_path", default=FFCV_VAL)
    p.add_argument("--nb_knn", nargs="+", default=[10, 20, 100, 200], type=int)
    p.add_argument("--temperature", default=0.07, type=float)
    p.add_argument("--batch_size", default=256, type=int)
    p.add_argument("--num_workers", default=8, type=int)
    p.add_argument("--max_images", default=0, type=int, help="debug cap per split")
    p.add_argument("--output", required=True)
    return p.parse_args()


def build_backbone(path, args, device):
    torch.manual_seed(0)
    model = models.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    if path != "random":
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        state = ckpt[args.key]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        state = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
        incompat = model.load_state_dict(state, strict=False)
        # classification head / fc are absent from the backbone by design
        bad = [k for k in incompat.missing_keys if not k.startswith(("head.", "fc."))]
        if bad or incompat.unexpected_keys:
            raise RuntimeError(f"{path}: missing={bad} unexpected={incompat.unexpected_keys}")
    return model.to(device).eval()


def pipelines(device):
    from ffcv.fields.decoders import CenterCropRGBImageDecoder, IntDecoder
    from ffcv.transforms import NormalizeImage, ToDevice, ToTensor, ToTorchImage, View
    return {
        "image": [
            CenterCropRGBImageDecoder((224, 224), 224 / 256),
            NormalizeImage(MEAN, STD, np.float32),
            ToTensor(), ToTorchImage(), ToDevice(device, non_blocking=True),
        ],
        "label": [IntDecoder(), ToTensor(), ToDevice(device), View(-1)],
    }


def true_labels(path, args, device):
    """Dataset-order labels read without images, single process."""
    from ffcv.fields.decoders import IntDecoder
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import ToDevice, ToTensor, View
    indices = np.arange(args.max_images) if args.max_images else None
    loader = Loader(path, batch_size=1024, num_workers=4, order=OrderOption.SEQUENTIAL,
                    distributed=False, indices=indices, drop_last=False,
                    pipelines={"image": None,
                               "label": [IntDecoder(), ToTensor(), ToDevice(device), View(-1)]})
    # FFCV recycles its output buffers, so copy each batch before the next one.
    return torch.cat([(b[0] if isinstance(b, (tuple, list)) else b).clone() for b in loader])


class Split:
    """One beton: labels in dataset order and a strided shard loader per rank.

    Built once and reused for every checkpoint. Recreating FFCV loaders in one
    process made the shard order check fail intermittently on later checkpoints.
    """

    def __init__(self, path, args, rank, world, device):
        from ffcv.loader import Loader, OrderOption
        self.path = path
        self.truth = true_labels(path, args, device)
        self.n = self.truth.shape[0]
        self.shard = np.arange(self.n, dtype=np.int64)[rank::world]
        self.loader = Loader(path, batch_size=args.batch_size, num_workers=args.num_workers,
                             order=OrderOption.SEQUENTIAL, distributed=False, seed=0,
                             indices=self.shard, pipelines=pipelines(device), drop_last=False)


def extract(model, split, rank, world, device):
    """Return L2-normalized features [N, D] and labels [N] on rank 0 (None elsewhere).

    FFCV's distributed order is not relied on: each rank reads an explicit
    strided shard, and every batch's labels are checked against labels read
    without images.
    """
    n, shard, truth = split.n, split.shard, split.truth
    feats = []
    cursor = 0
    for images, target in split.loader:
        b = images.shape[0]
        want = truth[torch.as_tensor(shard[cursor:cursor + b], device=device)]
        if not torch.equal(target, want):
            raise RuntimeError(f"{split.path}: batch at {cursor} does not follow shard order")
        with torch.no_grad():
            out = model(images)
        out = out[:, 0] if out.ndim == 3 else out
        # clone: out[:, 0] is a view that would pin every batch's full token tensor
        feats.append(out.float().clone())
        cursor += b
    if cursor != len(shard):
        raise RuntimeError(f"{split.path}: read {cursor} of {len(shard)} samples")
    feats = torch.cat(feats)
    d = feats.shape[1]
    m = -(-n // world)  # every shard is at most ceil(n / world)
    feats = torch.cat([feats, feats.new_zeros(m - feats.shape[0], d)])
    if world > 1:
        gathered = [torch.zeros_like(feats) for _ in range(world)]
        dist.all_gather(gathered, feats)
    else:
        gathered = [feats]
    if rank != 0:
        return None, None
    out_f = torch.zeros(n, d, device=device)
    for r, g in enumerate(gathered):
        idx = torch.as_tensor(np.arange(n, dtype=np.int64)[r::world], device=device)
        out_f[idx] = g[:idx.shape[0]]
    f = torch.nn.functional.normalize(out_f, dim=1)
    sub = f[:2000]
    cos = (sub @ sub.T).fill_diagonal_(0).sum().item() / (2000 * 1999)
    print(f"CHECK {Path(split.path).name}: n={n} classes={len(torch.unique(truth))} "
          f"feat_std={out_f.std(0).mean():.4f} mean_pair_cos={cos:.4f}", flush=True)
    return f, truth


@torch.inference_mode()
def knn(train_f, train_l, val_f, val_l, ks, temp, classes=1000, chunk=500):
    kmax = max(ks)
    correct = {k: [0, 0] for k in ks}  # top1, top5
    for s in range(0, val_f.shape[0], chunk):
        q = val_f[s:s + chunk]; y = val_l[s:s + chunk]
        sim, ind = (q @ train_f.T).topk(kmax, dim=1)
        nl = train_l[ind]
        for k in ks:
            w = (sim[:, :k] / temp).exp()
            votes = torch.zeros(q.shape[0], classes, device=q.device)
            votes.scatter_add_(1, nl[:, :k], w)
            top5 = votes.topk(5, dim=1).indices
            correct[k][0] += (top5[:, 0] == y).sum().item()
            correct[k][1] += (top5 == y[:, None]).any(dim=1).sum().item()
    n = val_f.shape[0]
    return {k: {"top1": 100 * c[0] / n, "top5": 100 * c[1] / n} for k, c in correct.items()}


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    results = {}
    train_split = Split(args.train_path, args, rank, world, device)
    val_split = Split(args.val_path, args, rank, world, device)
    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        model = build_backbone(path, args, device)
        tr_f, tr_l = extract(model, train_split, rank, world, device)
        va_f, va_l = extract(model, val_split, rank, world, device)
        if rank == 0:
            res = knn(tr_f, tr_l, va_f, va_l, args.nb_knn, args.temperature)
            results[name] = {"checkpoint": path, "key": args.key,
                             "n_train": tr_f.shape[0], "n_val": va_f.shape[0],
                             "knn": {str(k): v for k, v in res.items()}}
            print(name, json.dumps(results[name]["knn"]), flush=True)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
        del model, tr_f, tr_l, va_f, va_l
        if world > 1:
            dist.barrier()


if __name__ == "__main__":
    main()
