# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from multiprocessing import cpu_count
from typing import Mapping, Sequence, Type, Union

import numpy as np
import torch
import torchvision.transforms.v2 as tfms
from torch import nn
from torchvision.datasets import ImageFolder

from ffcv.fields.base import Field
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.loader.loader import DEFAULT_PROCESS_CACHE, ORDER_MAP, ORDER_TYPE
from ffcv.memory_managers import MemoryManager, OSCacheManager, ProcessCacheManager
from ffcv.pipeline import Compiler, PipelineSpec
from ffcv.pipeline.graph import Graph
from ffcv.pipeline.operation import Operation
from ffcv.reader import Reader
from ffcv.transforms import (
    NormalizeImage, RandomHorizontalFlip, ToDevice, ToTensor, ToTorchImage,
)
from ffcv.transforms.color_jitter import RandomColorJitter
from ffcv.transforms.ops import View
from ffcv.transforms.solarization import RandomSolarization
from ffcv.traversal_order.base import TraversalOrder

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255

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


class MultiLoader(Loader):
    """Several pipelines reading one beton field. Same wiring as SimLAP."""

    def __init__(self,
                 fname: str,
                 batch_size: int,
                 num_workers: int = -1,
                 cache_type: int = DEFAULT_PROCESS_CACHE,
                 order: Union[ORDER_TYPE, TraversalOrder] = OrderOption.SEQUENTIAL,
                 distributed: bool = False,
                 seed: int = None,
                 indices: Sequence[int] = None,
                 pipelines: Mapping[str, Sequence[Union[Operation, torch.nn.Module]]] = {},
                 custom_fields: Mapping[str, Type[Field]] = {},
                 drop_last: bool = True,
                 batches_ahead: int = 3,
                 recompile: bool = False,
                 ):
        if distributed and order == OrderOption.RANDOM and seed is None:
            print('Warning: no ordering seed was specified with distributed=True. '
                  'Setting seed to 0 to match PyTorch distributed sampler.')
            seed = 0
        elif seed is None:
            tinfo = np.iinfo('int32')
            seed = np.random.randint(0, tinfo.max)

        self._args = {
            'fname': fname,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'os_cache': cache_type,
            'order': order,
            'distributed': distributed,
            'seed': seed,
            'indices': indices,
            'pipelines': pipelines,
            'drop_last': drop_last,
            'batches_ahead': batches_ahead,
            'recompile': recompile,
        }
        self.fname = fname
        self.batch_size = batch_size
        self.batches_ahead = batches_ahead
        self.seed = seed
        self.reader = Reader(self.fname, custom_fields)
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.distributed = distributed
        self.code = None
        self.recompile = recompile

        if self.num_workers < 1:
            self.num_workers = cpu_count()
        Compiler.set_num_threads(self.num_workers)

        if indices is None:
            self.indices = np.arange(self.reader.num_samples, dtype='uint64')
        else:
            self.indices = np.array(indices)

        if cache_type == 0:
            self.memory_manager: MemoryManager = OSCacheManager(self.reader)
        elif cache_type == 1:
            self.memory_manager: MemoryManager = ProcessCacheManager(self.reader)
        elif cache_type == 2:
            from ffcv.memory_managers.shared_cache import SharedMemoryManager
            self.memory_manager = SharedMemoryManager(self.reader)
        else:
            raise ValueError("Unknown cache type. Use 0 for process cache, 1 for os cache, or 2 for no cache.")

        if order in ORDER_MAP:
            self.traversal_order = ORDER_MAP[order](self)
        elif isinstance(order, TraversalOrder):
            self.traversal_order = order(self)
        elif issubclass(order, TraversalOrder):
            self.traversal_order = order(self)
        else:
            raise ValueError(f"Order {order} is not a supported order type or a subclass of TraversalOrder")

        memory_read = self.memory_manager.compile_reader()
        self.next_epoch = 0
        self.pipelines = {}
        self.pipeline_specs = {}
        self.field_name_to_f_ix = {}
        custom_pipeline_specs = {}

        for output_name, spec in pipelines.items():
            if isinstance(spec, PipelineSpec):
                pass
            elif isinstance(spec, Sequence):
                spec = PipelineSpec(output_name, decoder=None, transforms=spec)
            elif spec is None:
                continue
            else:
                raise ValueError(
                    f"The pipeline for {output_name} has to be "
                    "either a PipelineSpec or a sequence of operations")
            custom_pipeline_specs[output_name] = spec

        default_name_to_f_ix = {}
        for f_ix, (field_name, _field) in enumerate(self.reader.handlers.items()):
            default_name_to_f_ix[field_name] = f_ix

        for field_name, spec in custom_pipeline_specs.items():
            # Graph looks up the beton field, not the output name.
            self.field_name_to_f_ix[spec.source] = default_name_to_f_ix[spec.source]
            self.field_name_to_f_ix[field_name] = self.field_name_to_f_ix[spec.source]
            if field_name not in self.pipeline_specs:
                self.pipeline_specs[field_name] = spec

        self.graph = Graph(self.pipeline_specs, self.reader.handlers,
                           self.field_name_to_f_ix, self.reader.metadata,
                           memory_read)
        self.generate_code()
        self.first_traversal_order = self.next_traversal_order()


class RandomGaussianBlur(nn.Module):
    """Per-image blur. Runs after ToDevice, so it is not a Numba stage."""

    def __init__(self, p):
        super().__init__()
        self.p = p
        self.op = tfms.GaussianBlur(3, sigma=(0.1, 2.0))

    def forward(self, images):
        if self.p >= 1:
            return self.op(images)
        if self.p <= 0:
            return images
        keep = torch.rand(images.shape[0], device=images.device) >= self.p
        blurred = self.op(images)
        return torch.where(keep.view(-1, 1, 1, 1), images, blurred)


def _crop_pipeline(size, scale, blur_p, solar_p, device):
    transforms = [
        RandomHorizontalFlip(),
        RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
        ToTensor(),
        ToTorchImage(),
        ToDevice(device, non_blocking=True),
        tfms.RandomGrayscale(p=0.2),
        RandomGaussianBlur(blur_p),
    ]
    if solar_p:
        transforms.insert(2, RandomSolarization(128, p=solar_p))
    return PipelineSpec(
        'image',
        RandomResizedCropRGBImageDecoder((size, size), scale=tuple(scale)),
        transforms=transforms,
    )


class IBotFFCVLoader:
    """FFCV beton -> (crops, labels, masks), same batch layout as ImageFolderMask."""

    def __init__(self, fname, batch_size, num_workers, global_crops_scale, local_crops_scale,
                 global_crops_number, local_crops_number, patch_size, pred_ratio, pred_ratio_var,
                 pred_aspect_ratio, pred_shape, pred_start_epoch, distributed, seed):
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

        crop_specs = [(224, global_crops_scale, 1.0, 0.0)]
        for _ in range(global_crops_number - 1):
            crop_specs.append((224, global_crops_scale, 0.1, 0.2))
        for _ in range(local_crops_number):
            crop_specs.append((96, local_crops_scale, 0.5, 0.0))
        self._grids = [size // patch_size for size, _, _, _ in crop_specs]

        pipelines = {
            'label': [IntDecoder(), ToTensor(), ToDevice(device, non_blocking=True), View(-1)],
        }
        for i, (size, scale, blur_p, solar_p) in enumerate(crop_specs):
            pipelines[f'crop{i}'] = _crop_pipeline(size, scale, blur_p, solar_p, device)

        self.loader = MultiLoader(
            fname,
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.RANDOM,
            distributed=distributed,
            seed=seed,
            pipelines=pipelines,
            drop_last=True,
        )
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


if __name__ == '__main__':
    _check_masks()
    print('mask sampler ok')
