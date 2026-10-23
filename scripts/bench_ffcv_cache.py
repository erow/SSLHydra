"""Time FFCV reads of a beton that already fits in the OS page cache.

cache_type 0 is OSCacheManager: the beton is mmap'd and served from the
kernel page cache. Numba's cache goes to /tmp because Lustre file locks hang.
"""
import os
import sys
import time

import numpy as np

os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba-ffcv-bench"
os.environ["FFCV_DEFAULT_CACHE_PROCESS"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

PATH = os.environ.get(
    "FFCV",
    "/scratch/project_465003400/datasets/ffcv/IN1K_val_smart500_q90.ffcv",
)
BATCH = int(os.environ.get("BATCH", "128"))
WORKERS = int(os.environ.get("WORKERS", "8"))
WARMUP = int(os.environ.get("WARMUP", "4"))
MEASURE = int(os.environ.get("MEASURE", "200"))


def cached_kb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("Cached:"):
                return int(line.split()[1])
    return 0


def prefetch(path):
    size = os.path.getsize(path)
    before = cached_kb()
    t = time.perf_counter()
    got = 0
    with open(path, "rb") as f:
        while True:
            block = f.read(1 << 26)
            if not block:
                break
            got += len(block)
    dt = time.perf_counter() - t
    after = cached_kb()
    print(
        f"file {size / 1e9:.2f} GB  prefetch {dt:.2f}s ({got / dt / 1e6:.0f} MB/s)  "
        f"Cached {before / 1e6:.1f} -> {after / 1e6:.1f} GB",
        flush=True,
    )


def time_loader(loader, warmup, measure):
    it = iter(loader)
    t = time.perf_counter()
    for _ in range(warmup):
        next(it)
    warm_s = time.perf_counter() - t
    n = 0
    t = time.perf_counter()
    for _ in range(measure):
        batch = next(it)
        images = batch[0]
        n += images.shape[0]
    dt = time.perf_counter() - t
    return warm_s, dt, n


def main():
    import torch

    torch.set_num_threads(1)
    prefetch(PATH)

    from ffcv.fields.decoders import CenterCropRGBImageDecoder, IntDecoder
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import NormalizeImage, ToTensor, ToTorchImage

    mean = np.array([0.485, 0.456, 0.406]) * 255
    std = np.array([0.229, 0.224, 0.225]) * 255
    print("building center-crop loader", flush=True)
    t = time.perf_counter()
    loader = Loader(
        PATH,
        batch_size=BATCH,
        num_workers=WORKERS,
        cache_type=0,
        order=OrderOption.SEQUENTIAL,
        seed=0,
        drop_last=True,
        pipelines={
            "image": [
                CenterCropRGBImageDecoder((224, 224), 224 / 256),
                NormalizeImage(mean, std, np.float32),
                ToTensor(),
                ToTorchImage(),
            ],
            "label": [IntDecoder(), ToTensor()],
        },
    )
    print(
        f"built in {time.perf_counter() - t:.1f}s  "
        f"cache={type(loader.memory_manager).__name__}  "
        f"samples={loader.reader.num_samples} batch={BATCH} workers={WORKERS}",
        flush=True,
    )
    warm_s, dt, n = time_loader(loader, WARMUP, MEASURE)
    print(f"warmup {WARMUP} batches in {warm_s:.2f}s", flush=True)
    print(
        f"pass1 {MEASURE} batches  {n} images  {dt:.2f}s  "
        f"{n / dt:.0f} img/s  {n * 224 * 224 * 3 / dt / 1e6:.0f} MB/s decoded",
        flush=True,
    )
    warm_s, dt, n = time_loader(loader, 0, MEASURE)
    print(
        f"pass2 {MEASURE} batches  {n} images  {dt:.2f}s  "
        f"{n / dt:.0f} img/s  {n * 224 * 224 * 3 / dt / 1e6:.0f} MB/s decoded",
        flush=True,
    )


if __name__ == "__main__":
    main()
