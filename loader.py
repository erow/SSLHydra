# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import math
import os
import random
import time

import numpy as np
import torch

from torchvision.datasets import ImageFolder

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255


def _numba_cache_dir():
    """Node-local Numba cache. Lustre file locks hang the compile."""
    path = os.environ.get("NUMBA_CACHE_LOCAL")
    if not path:
        job = os.environ.get("SLURM_JOB_ID", "local")
        path = f"/tmp/numba-ffcv-{job}"
    os.makedirs(path, exist_ok=True)
    return path


def _local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))

class ImageFolderInstance(ImageFolder):
    def __getitem__(self, index):
        img, target = super(ImageFolderInstance, self).__getitem__(index)
        return img, target, index

class MaskSampler:
    """iBOT patch masks. One sample per crop, independent of the image pixels."""

    def __init__(self, pred_ratio, pred_ratio_var, pred_aspect_ratio,
                 pred_shape='block', pred_start_epoch=0):
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(math.log, pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio

        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __call__(self, H, W):
        high = self.get_pred_ratio() * H * W

        if self.pred_shape == 'block':
            # following BEiT (https://arxiv.org/abs/2106.08254), see at
            # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
            mask = np.zeros((H, W), dtype=bool)
            mask_count = 0
            while mask_count < high:
                max_mask_patches = high - mask_count

                delta = 0
                for _attempt in range(10):
                    low = (min(H, W) // 3) ** 2
                    target_area = random.uniform(low, max_mask_patches)
                    aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                    h = int(round(math.sqrt(target_area * aspect_ratio)))
                    w = int(round(math.sqrt(target_area / aspect_ratio)))
                    if w < W and h < H:
                        top = random.randint(0, H - h)
                        left = random.randint(0, W - w)

                        num_masked = mask[top: top + h, left: left + w].sum()
                        if 0 < h * w - num_masked <= max_mask_patches:
                            for i in range(top, top + h):
                                for j in range(left, left + w):
                                    if mask[i, j] == 0:
                                        mask[i, j] = 1
                                        delta += 1

                    if delta > 0:
                        break

                if delta == 0:
                    break
                else:
                    mask_count += delta

        elif self.pred_shape == 'rand':
            mask = np.hstack([
                np.zeros(H * W - int(high)),
                np.ones(int(high)),
            ]).astype(bool)
            np.random.shuffle(mask)
            mask = mask.reshape(H, W)

        else:
            # no implementation
            assert False

        return mask

    def batch(self, n, H, W):
        masks = np.stack([self(H, W) for _ in range(n)])
        return torch.from_numpy(masks)


class ImageFolderMask(ImageFolder):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio,
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(ImageFolderMask, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self._masks = MaskSampler(
            pred_ratio, pred_ratio_var, pred_aspect_ratio, pred_shape, pred_start_epoch)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._masks.set_epoch(epoch)

    def __getitem__(self, index):
        output = super(ImageFolderMask, self).__getitem__(index)

        masks = []
        for img in output[0]:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except Exception:
                # skip non-image
                continue
            masks.append(self._masks(H, W))

        return output + (masks,)


def _cell_token(value):
    if isinstance(value, (int, float, str, bool, type(None))):
        return repr(value)
    if isinstance(value, tuple) and all(
            isinstance(x, (int, float, str, bool, type(None))) for x in value):
        return repr(value)
    if isinstance(value, np.ndarray):
        return repr((str(value.dtype), value.shape, np.round(value, 6).tolist()))
    py = getattr(value, "py_func", None)
    if py is not None:
        return "njit:" + getattr(py, "__qualname__", py.__name__)
    if callable(value):
        return "fn:" + getattr(value, "__qualname__", type(value).__name__)
    return type(value).__name__ + ":" + repr(value)


def _stamp_closure(fn):
    """Put closed-over crop settings into the qualname Numba uses as its cache key.

    Numba keys a cached function by source and qualname, not by closure cells.
    Two decoders that share one nested function and close over different scales
    would otherwise load each other's machine code.
    """
    cells = fn.__closure__
    if not cells:
        return
    token = "|".join(_cell_token(cell.cell_contents) for cell in cells)
    digest = hashlib.sha1(token.encode()).hexdigest()[:16]
    fn.__qualname__ = fn.__qualname__ + "." + digest


def _enable_ffcv_numba_cache():
    """Compile FFCV's Numba stages once and reuse them from this node's /tmp."""
    cache_dir = _numba_cache_dir()
    os.environ["NUMBA_CACHE_DIR"] = cache_dir
    import numba
    from numba import njit
    from ffcv.pipeline.compiler import Compiler

    # train_ibot imports numba before this runs, so the env var alone is too late.
    numba.config.CACHE_DIR = cache_dir
    if getattr(Compiler, "_cache_patched", False):
        return

    @classmethod
    def compile(cls, code, signature=None):
        parallel = False
        if hasattr(code, "is_parallel"):
            parallel = code.is_parallel and cls.num_threads > 1
        if not cls.is_enabled:
            return code
        _stamp_closure(code)
        return njit(signature, fastmath=True, nogil=True, error_model="numpy",
                    parallel=parallel, cache=True)(code)

    Compiler.compile = compile
    Compiler._cache_patched = True


def _warm_one_batch(loader):
    """Run one batch so this process writes the Numba cache before its peers compile."""
    it = iter(loader)
    try:
        next(it)
    finally:
        # Stop the worker before tearing down its memory. close() drops the
        # cache while the thread may still be inside the pipeline.
        it.terminate_event.set()
        it.join(timeout=300)
        alive = it.is_alive()
        it.close()
        if alive:
            raise RuntimeError("FFCV warmup thread did not stop")


def _warmup_marker():
    return os.path.join(_numba_cache_dir(), "warmup.done")


def _wait_warmup():
    """Wait until rank 0 has finished compiling. A NCCL barrier dies after 10 minutes."""
    path = _warmup_marker()
    # ponytail: 3h ceiling so a dead rank 0 does not hold the node until the job walltime
    deadline = time.time() + 3 * 60 * 60
    while not os.path.exists(path):
        if time.time() >= deadline:
            raise TimeoutError("Numba warmup did not finish: " + path)
        time.sleep(5)


def _mark_warmup():
    path = _warmup_marker()
    tmp = path + "." + str(os.getpid())
    with open(tmp, "w") as fh:
        fh.write("ok\n")
    os.replace(tmp, path)


def _crop_ops(size, scale, blur_p, solar_p):
    from ffcv.fields.decoders import RandomResizedCropRGBImageDecoder
    from ffcv.pipeline.operation import Operation
    from ffcv.pipeline.state import State
    from ffcv.transforms import RandomHorizontalFlip
    from ffcv.transforms.color_jitter import RandomColorJitter
    from ffcv.transforms.grayscale import RandomGrayscale
    from ffcv.transforms.solarization import RandomSolarization
    from dataclasses import replace
    import cv2

    class CvBlur(Operation):
        def __init__(self, p):
            super().__init__()
            self.p = p

        def generate_code(self):
            p = self.p

            def blur(images, _dst):
                for i in range(images.shape[0]):
                    if np.random.random() < p:
                        sigma = float(np.random.uniform(0.1, 2.0))
                        images[i] = cv2.GaussianBlur(images[i], (0, 0), sigmaX=sigma, sigmaY=sigma)
                return images

            return blur

        def declare_state_and_memory(self, previous_state: State):
            # cv2 is not numba-safe, so this stage stays in Python.
            return replace(previous_state, jit_mode=False), None

    ops = [
        RandomResizedCropRGBImageDecoder((size, size), scale=tuple(scale)),
        RandomHorizontalFlip(),
        RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        RandomGrayscale(p=0.2),
        CvBlur(blur_p),
    ]
    if solar_p:
        ops.append(RandomSolarization(p=solar_p))
    return ops


def _normalize_ops(device):
    from ffcv.transforms import NormalizeImage, ToDevice, ToTensor, ToTorchImage
    return [
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
        ToTensor(),
        ToTorchImage(channels_last=False),
        ToDevice(device, non_blocking=True),
    ]


class IBotFFCVLoader:
    """FFCV beton -> (crops, labels, masks), same batch layout as ImageFolderMask."""

    def __init__(self, fname, batch_size, num_workers, global_crops_scale, local_crops_scale,
                 global_crops_number, local_crops_number, patch_size, pred_ratio, pred_ratio_var,
                 pred_aspect_ratio, pred_shape, pred_start_epoch, distributed, seed):
        from ffcv.fields.decoders import IntDecoder
        from ffcv.loader import Loader, OrderOption
        from ffcv.pipeline import PipelineSpec
        from ffcv.reader import Reader
        from ffcv.transforms import ToDevice, ToTensor
        from ffcv.transforms.ops import View

        _enable_ffcv_numba_cache()
        
        if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            world = torch.distributed.get_world_size()
        else:
            world = 1
        # Local rank 0 compiles into this node's /tmp. The others poll a file.
        # A global rank is wrong here: another node's /tmp is invisible.
        # dist.barrier is an ALLREDUCE and the NCCL watchdog kills it after 10 minutes.
        if world > 1 and _local_rank() != 0:
            _wait_warmup()

        if torch.cuda.is_available():
            device = torch.device('cuda', torch.cuda.current_device())
        else:
            device = torch.device('cpu')
        self.patch_size = patch_size
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number
        self._masks = MaskSampler(
            pred_ratio, pred_ratio_var, pred_aspect_ratio, pred_shape, pred_start_epoch)
        self.sampler = self
        self.dataset = self
        self.epoch = 0

        crop_specs = []
        crop_specs.append((224, global_crops_scale, 1.0, 0.0))
        for _ in range(global_crops_number - 1):
            crop_specs.append((224, global_crops_scale, 0.1, 0.2))
        for _ in range(local_crops_number):
            crop_specs.append((96, local_crops_scale, 0.5, 0.0))
        self._grids = [size // patch_size for size, _, _, _ in crop_specs]

        fields = list(Reader(fname).handlers)
        pipelines = {name: None for name in fields}
        pipelines['label'] = [
            IntDecoder(), ToTensor(), ToDevice(device, non_blocking=True), View(-1),
        ]
        for i, (size, scale, blur_p, solar_p) in enumerate(crop_specs):
            ops = _crop_ops(size, scale, blur_p, solar_p) + _normalize_ops(device)
            pipelines[f'crop{i}'] = PipelineSpec('image', decoder=ops[0], transforms=ops[1:])

        self.loader = Loader(
            fname,
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.RANDOM,
            distributed=distributed,
            seed=seed,
            pipelines=pipelines,
            drop_last=True,
        )
        if world > 1 and _local_rank() == 0:
            epoch = self.loader.next_epoch
            _warm_one_batch(self.loader)
            # The warmup batch advances next_epoch. Training sets it again,
            # but a caller that does not would walk a different order than its peers.
            self.loader.next_epoch = epoch
            _mark_warmup()
        self.num_samples = int(self.loader.reader.num_samples)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._masks.set_epoch(epoch)
        self.loader.next_epoch = epoch

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        # Same order FFCV is about to walk. sample_order reseeds from the epoch,
        # and Loader.__iter__ calls it again before incrementing next_epoch.
        order = np.asarray(
            self.loader.traversal_order.sample_order(self.loader.next_epoch))
        order = order[:len(self.loader) * self.loader.batch_size]
        cursor = 0
        for batch in self.loader:
            labels = batch[0]
            images = list(batch[1:])
            n = images[0].shape[0]
            if cursor + n > len(order):
                raise RuntimeError(
                    f"FFCV batch exceeds sampled order ({cursor + n} > {len(order)})")
            idx = torch.as_tensor(order[cursor:cursor + n].astype(np.int64))
            cursor += n
            masks = [self._masks.batch(n, g, g) for g in self._grids]
            yield images, labels, masks, idx


def _check_masks():
    block = MaskSampler(0.3, 0, (0.3, 1 / 0.3), 'block', 0)
    block.set_epoch(0)
    masks = block.batch(4, 14, 14)
    assert masks.shape == (4, 14, 14) and masks.dtype == torch.bool
    assert masks.any() and not masks.all()
    rand = MaskSampler(0.3, 0, (0.3, 1 / 0.3), 'rand', 0)
    rand.set_epoch(0)
    rand_masks = rand.batch(2, 14, 14)
    assert rand_masks.shape == (2, 14, 14) and int(rand_masks[0].sum()) == int(0.3 * 14 * 14)
    held = MaskSampler(0.3, 0, (0.3, 1 / 0.3), 'block', 5)
    held.set_epoch(0)
    assert not held.batch(2, 8, 8).any()


def _check_closure_stamp():
    def make(scale):
        def decode(x):
            return x + scale
        return decode

    first = make((0.14, 1.0))
    second = make((0.05, 0.4))
    again = make((0.14, 1.0))
    _stamp_closure(first)
    _stamp_closure(second)
    _stamp_closure(again)
    assert first.__qualname__ != second.__qualname__
    assert first.__qualname__ == again.__qualname__


def _check_warmup_marker():
    import tempfile
    old = os.environ.get("NUMBA_CACHE_LOCAL")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["NUMBA_CACHE_LOCAL"] = tmp
        try:
            assert _numba_cache_dir() == tmp
            assert not os.path.exists(_warmup_marker())
            _mark_warmup()
            assert os.path.exists(_warmup_marker())
            assert _warmup_marker().startswith(tmp)
        finally:
            if old is None:
                os.environ.pop("NUMBA_CACHE_LOCAL", None)
            else:
                os.environ["NUMBA_CACHE_LOCAL"] = old


if __name__ == '__main__':
    _check_masks()
    _check_closure_stamp()
    _check_warmup_marker()
    print('mask sampler ok')