import albumentations as A
import torch


def get_preprocessing(preprocessing_fn, img_size=(256, 256)):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
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


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def denormalize(
        x,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
):
    ten = x.clone()
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)

    return torch.clamp(ten, 0, 1)
