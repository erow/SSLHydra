"""ImageNet embeddings from the iBOT teacher [CLS] head (teacher_cls).

teacher_cls is the projection-head output on the teacher [CLS] token
(out_dim, 8192 for vitb_16_rand_mask), the same tensor the iBOT loss
reads before centering and softmax. Patch-token head outputs are not
computed.

Default checkpoint:
https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_rand_mask/checkpoint.pth

Train images come from the FFCV beton
``IN1K_train_smart500_q90.ffcv`` (center crop 224, ImageNet normalize).
Val uses the matching val beton when it is present. Writes
``{split}_teacher_cls.npy`` float16 ``[N, out_dim]`` and ``{split}_labels.npy``.

Launch (full ImageNet, 8 GPUs):

    ./cluster/lumi_ddp.sh scripts/imagenet_embed.py \
        --output_dir "$OUTPUT_ROOT/imagenet_teacher_cls"
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

IBOT_ROOT = Path(__file__).resolve().parents[1] / "src" / "ibot"
sys.path.insert(0, str(IBOT_ROOT))

import models  # noqa: E402
import utils  # noqa: E402
from models.head import iBOTHead  # noqa: E402

CKPT_URL = (
    "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/"
    "archive/2022/ibot/vitb_16_rand_mask/checkpoint.pth"
)
FFCV_TRAIN = os.environ.get(
    "FFCV_TRAIN",
    "/scratch/project_465003400/datasets/ffcv/IN1K_train_smart500_q90.ffcv",
)
FFCV_VAL = os.environ.get(
    "FFCV_VAL",
    "/scratch/project_465003400/datasets/ffcv/IN1K_val_smart500_q90.ffcv",
)
# wnid list. lumi_ddp.sh rewrites IMAGENET_ROOT to the LMDB path.
FOLDER_ROOT = "/scratch/project_465003400/datasets/IN1K"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255


def parse_args():
    p = argparse.ArgumentParser(description="Dump ImageNet teacher_cls embeddings")
    p.add_argument("--checkpoint", default=CKPT_URL, help="checkpoint path or URL")
    p.add_argument(
        "--cache_dir",
        default=os.environ.get("TORCH_HOME", "/scratch/project_465003400/cache/torch"),
        help="where to store a downloaded checkpoint",
    )
    p.add_argument(
        "--data_path",
        default=FFCV_TRAIN,
        help="Train FFCV beton (.ffcv).",
    )
    p.add_argument(
        "--val_path",
        default=FFCV_VAL,
        help="Val FFCV beton. Used when --splits includes val.",
    )
    p.add_argument(
        "--output_dir",
        default=os.path.join(
            os.environ.get("OUTPUT_ROOT", "/scratch/project_465003400/SSLHydra/outputs"),
            "imagenet_teacher_cls",
        ),
    )
    p.add_argument("--splits", default="train,val", help="comma-separated ImageFolder splits")
    p.add_argument("--batch_size", default=256, type=int)
    p.add_argument("--num_workers", default=8, type=int)
    p.add_argument("--max_images", default=0, type=int, help="cap images per split; 0 = all")
    p.add_argument("--self-check", action="store_true", help="CLS path vs full head, then exit")
    return p.parse_args()


def fetch_checkpoint(url_or_path, cache_dir):
    path = Path(url_or_path)
    if path.is_file():
        return path
    dest = Path(cache_dir) / "ibot" / "vitb_16_rand_mask_checkpoint.pth"
    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(str(url_or_path), str(dest))
    return dest


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def published_args():
    """vitb_16_rand_mask args.txt. The full checkpoint does not store args."""
    return argparse.Namespace(
        arch="vit_base",
        patch_size=16,
        out_dim=8192,
        patch_out_dim=8192,
        norm_in_head=None,
        act_in_head="gelu",
        norm_last_layer=True,
        shared_head_teacher=True,
    )


def build_teacher(ckpt_args):
    arch = ckpt_args.arch.replace("deit", "vit")
    backbone = models.__dict__[arch](
        patch_size=ckpt_args.patch_size,
        return_all_tokens=True,
    )
    teacher = utils.MultiCropWrapper(
        backbone,
        iBOTHead(
            backbone.embed_dim,
            ckpt_args.out_dim,
            patch_out_dim=ckpt_args.patch_out_dim,
            norm=ckpt_args.norm_in_head,
            act=ckpt_args.act_in_head,
            norm_last_layer=ckpt_args.norm_last_layer,
            shared_head=ckpt_args.shared_head_teacher,
        ),
    )
    return teacher


def load_teacher(ckpt):
    teacher = build_teacher(ckpt["args"] if "args" in ckpt else published_args())
    state = {k.replace("module.", ""): v for k, v in ckpt["teacher"].items()}
    incompat = teacher.load_state_dict(state, strict=False)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(
            f"teacher state_dict mismatch: missing={incompat.missing_keys} "
            f"unexpected={incompat.unexpected_keys}"
        )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


@torch.inference_mode()
def teacher_cls(teacher, images):
    """iBOT head on the teacher [CLS] token. Same value as teacher(images)[0]."""
    tokens = teacher.backbone(images)
    cls = tokens[:, 0] if tokens.ndim == 3 else tokens
    return teacher.head(cls)


def self_check():
    ckpt_args = published_args()
    teacher = build_teacher(ckpt_args).eval()
    images = torch.randn(2, 3, 224, 224)
    cls = teacher_cls(teacher, images)
    full_cls, _patch = teacher(images)
    # GEMM shape differs (CLS-only vs all tokens); float32 noise stays under 1e-5.
    ok = cls.shape == (2, ckpt_args.out_dim) and torch.allclose(cls, full_cls, rtol=1e-5, atol=1e-5)
    if not ok:
        raise SystemExit(f"self-check failed: shape={tuple(cls.shape)}")
    print(f"self-check ok {tuple(cls.shape)}")


def init_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0 if torch.cuda.is_available() else None
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local)
    return rank, world, local


def out_dim_of(teacher):
    return teacher.head.last_layer.weight_v.shape[0]


def ffcv_pipelines(device):
    from ffcv.fields.decoders import CenterCropRGBImageDecoder, IntDecoder
    from ffcv.transforms import NormalizeImage, ToDevice, ToTensor, ToTorchImage, View

    return {
        "image": [
            CenterCropRGBImageDecoder((224, 224), 224 / 256),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
            ToTensor(),
            ToTorchImage(),
            ToDevice(device, non_blocking=True),
        ],
        "label": [IntDecoder(), ToTensor(), ToDevice(device), View(-1)],
    }


def extract_ffcv(teacher, path, split, args, rank, world, device):
    from ffcv.loader import Loader, OrderOption

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    indices = None
    if args.max_images:
        indices = np.arange(args.max_images)
    loader = Loader(
        str(path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        order=OrderOption.SEQUENTIAL,
        distributed=world > 1,
        seed=0,
        indices=indices,
        pipelines=ffcv_pipelines(device),
        drop_last=False,
    )
    n = len(loader.indices) if indices is None else len(indices)
    order = np.asarray(loader.traversal_order.sample_order(0))
    if rank == 0:
        print(f"{split}: {n} FFCV images from {path}", flush=True)
    feats, labels, index_batches = [], [], []
    cursor = 0
    seen = 0
    for images, target in loader:
        b = images.shape[0]
        cls = teacher_cls(teacher, images).half().cpu()
        feats.append(cls)
        labels.append(target.cpu())
        index_batches.append(torch.as_tensor(order[cursor:cursor + b], dtype=torch.int64))
        cursor += b
        seen += b
        if rank == 0 and len(feats) % 20 == 0:
            print(f"{split}: {seen} images on rank 0", flush=True)
    finalize(feats, labels, index_batches, split, out_dim_of(teacher), args, rank, world, n)


def finalize(feats, labels, indices, split, dim, args, rank, world, n):
    if feats:
        shard = {
            "teacher_cls": torch.cat(feats),
            "labels": torch.cat(labels),
            "index": torch.cat(indices),
        }
    else:
        shard = {
            "teacher_cls": torch.empty(0, dim, dtype=torch.float16),
            "labels": torch.empty(0, dtype=torch.int64),
            "index": torch.empty(0, dtype=torch.int64),
        }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(shard, out / f".{split}.rank{rank}.pt")
    if world > 1:
        dist.barrier()
    if rank == 0:
        merge_shards(out, split, n, dim, world)
    if world > 1:
        dist.barrier()


def merge_shards(out, split, n, dim, world):
    emb = np.lib.format.open_memmap(
        out / f"{split}_teacher_cls.npy", mode="w+", dtype=np.float16, shape=(n, dim)
    )
    labels = np.lib.format.open_memmap(
        out / f"{split}_labels.npy", mode="w+", dtype=np.int64, shape=(n,)
    )
    filled = np.zeros(n, dtype=bool)
    for r in range(world):
        shard_path = out / f".{split}.rank{r}.pt"
        try:
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        except TypeError:
            shard = torch.load(shard_path, map_location="cpu")
        idx = shard["index"].numpy()
        if len(idx):
            emb[idx] = shard["teacher_cls"].numpy()
            labels[idx] = shard["labels"].numpy()
            filled[idx] = True
        del shard
        shard_path.unlink()
    if not filled.all():
        raise RuntimeError(f"{split}: missing {int((~filled).sum())} embeddings")
    emb.flush()
    labels.flush()
    print(f"wrote {split}_teacher_cls.npy {emb.shape} {emb.dtype}", flush=True)


def split_beton(split, args):
    if split in ("val", "validation"):
        return args.val_path
    return args.data_path


def main():
    args = parse_args()
    if args.self_check:
        self_check()
        return

    rank, world, local = init_distributed()
    device = torch.device("cuda", local) if local is not None else torch.device("cpu")
    if rank == 0:
        ckpt_path = fetch_checkpoint(args.checkpoint, args.cache_dir)
    if world > 1:
        dist.barrier()
    if rank != 0:
        ckpt_path = fetch_checkpoint(args.checkpoint, args.cache_dir)
    ckpt = load_checkpoint(ckpt_path)
    teacher_args = ckpt["args"] if "args" in ckpt else published_args()
    teacher = load_teacher(ckpt).to(device)

    out = Path(args.output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            "checkpoint": str(args.checkpoint),
            "arch": teacher_args.arch,
            "patch_size": teacher_args.patch_size,
            "out_dim": teacher_args.out_dim,
            "embedding": "teacher_cls",
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if world > 1:
        dist.barrier()

    for split in args.splits.split(","):
        split = split.strip()
        if not split:
            continue
        if rank == 0 and not (out / "classes.txt").is_file():
            val_root = Path(FOLDER_ROOT) / "val"
            if val_root.is_dir():
                wnids = sorted(p.name for p in val_root.iterdir() if p.is_dir())
                if wnids:
                    (out / "classes.txt").write_text("\n".join(wnids) + "\n")
        extract_ffcv(teacher, split_beton(split, args), split, args, rank, world, device)


if __name__ == "__main__":
    main()
