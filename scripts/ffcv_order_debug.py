"""Does an FFCV Loader with explicit indices return samples in that order, every time?"""
import os, sys
import numpy as np, torch, torch.distributed as dist
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(__file__))
import knn_eval_ffcv as K

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ["LOCAL_RANK"])
dist.init_process_group(backend="nccl", init_method="env://")
torch.cuda.set_device(local); device = torch.device("cuda", local)
args = SimpleNamespace(max_images=0, batch_size=256, num_workers=8)
from ffcv.loader import Loader, OrderOption
path = K.FFCV_VAL
for trial in range(4):
    truth = K.true_labels(path, args, device)
    n = truth.shape[0]; shard = np.arange(n, dtype=np.int64)[rank::world]
    loader = Loader(path, batch_size=256, num_workers=8, order=OrderOption.SEQUENTIAL,
                    distributed=False, seed=0, indices=shard, pipelines=K.pipelines(device), drop_last=False)
    bad = 0; first = None; cursor = 0
    for images, target in loader:
        b = images.shape[0]
        want = truth[torch.as_tensor(shard[cursor:cursor + b], device=device)]
        if not torch.equal(target, want):
            bad += 1
            if first is None:
                head = truth[torch.as_tensor(np.arange(b), device=device)]
                first = (cursor, (target == want).float().mean().item(),
                         (target == head).float().mean().item())
        cursor += b
    print(f"rank{rank} trial{trial} n={n} batches_bad={bad} first={first} truth_hash={int(truth.sum())}", flush=True)
