from __future__ import annotations

import functools

import albumentations as A
import torch


def get_preprocessing(
        preprocessing_fn: functools.partial | None,
        img_size: tuple[int, int] = (256, 256)
) -> A.Compose:
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
        img_size: height and width
    Return:
        transform: albumentations.Compose
        :param preprocessing_fn:
        :param img_size:

    """
    preprocessing = A.Lambda(image=preprocessing_fn) if preprocessing_fn else \
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    _transform = [
        A.Resize(*img_size),
        preprocessing,
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)


def to_tensor(img, **kwargs):
    # pylint: disable=unused-argument
    return img.transpose(2, 0, 1).astype('float32')


def denormalize(
        x: torch.Tensor,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225)
):
    ten = x.clone()
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)

    return torch.clamp(ten, 0, 1)
