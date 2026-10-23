"""Two ranks: IBotFFCVLoader shards the val beton and the indices agree.

NCCL needs one GPU per rank. Each rank uses its LOCAL_RANK device.
"""
import os

import torch
import torch.distributed as dist

from loader import IBotFFCVLoader

VAL = os.environ.get(
    "FFCV",
    "/scratch/project_465003400/datasets/ffcv/IN1K_val_smart500_q90.ffcv",
)
BATCH = 32
BATCHES = 2


def main():
    if not torch.cuda.is_available():
        raise SystemExit("nccl needs a GPU")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print(
        f"rank {os.environ.get('RANK', '?')} init nccl device={device} "
        f"cuda_devices={torch.cuda.device_count()}",
        flush=True,
    )
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()
    print(f"rank {rank}/{world} nccl up", flush=True)

    loader = IBotFFCVLoader(
        VAL,
        batch_size=BATCH,
        num_workers=2,
        global_crops_scale=(0.14, 1.0),
        local_crops_scale=(0.05, 0.4),
        global_crops_number=2,
        local_crops_number=0,
        patch_size=16,
        pred_ratio=[0.3],
        pred_ratio_var=[0],
        pred_aspect_ratio=(0.3, 1 / 0.3),
        pred_shape="block",
        pred_start_epoch=0,
        distributed=True,
        seed=0,
    )
    loader.set_epoch(0)
    seen = []
    for step, (images, labels, masks, idx) in enumerate(loader):
        if step >= BATCHES:
            break
        assert len(images) == 2, len(images)
        assert images[0].shape == (BATCH, 3, 224, 224), images[0].shape
        assert images[1].shape == images[0].shape
        assert labels.shape == (BATCH,), labels.shape
        assert idx.shape == (BATCH,), idx.shape
        assert masks[0].shape == (BATCH, 14, 14), masks[0].shape
        assert int(idx.min()) >= 0 and int(idx.max()) < loader.num_samples
        seen.append(idx.cpu())
        print(
            f"rank {rank} step {step} idx0={int(idx[0])} "
            f"device={images[0].device}",
            flush=True,
        )

    mine = torch.cat(seen).to(device="cuda", dtype=torch.int64)
    gathered = [torch.zeros_like(mine) for _ in range(world)]
    dist.all_gather(gathered, mine)
    if rank == 0:
        ids = [set(t.tolist()) for t in gathered]
        for i in range(world):
            for j in range(i + 1, world):
                overlap = ids[i] & ids[j]
                if overlap:
                    raise SystemExit(
                        f"ranks {i} and {j} share indices {sorted(overlap)[:8]}")
        print(
            f"ffcv dist ok samples={loader.num_samples} "
            f"per_rank={mine.numel()} overlap=0",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(f"rank {os.environ.get('RANK', '?')} failed", flush=True)
        raise
