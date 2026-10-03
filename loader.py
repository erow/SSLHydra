# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import math
import numpy as np
import torch

from torchvision.datasets import ImageFolder

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