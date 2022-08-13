import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
from utils import experiment_manager
import cv2


def compose_transformations(augmentation_cfg):
    transformations = []

    if augmentation_cfg.RANDOM_FLIP:
        transformations.append(RandomFlip())

    if augmentation_cfg.RANDOM_ROTATE:
        transformations.append(RandomRotate())

    if augmentation_cfg.COLOR_SHIFT:
        transformations.append(ColorShift())

    if augmentation_cfg.GAMMA_CORRECTION:
        transformations.append(GammaCorrection())

    transformations.append(Numpy2Torch())

    return transforms.Compose(transformations)


class Numpy2Torch(object):
    def __call__(self, img: np.ndarray) -> torch.Tensor:
        img_tensor = TF.to_tensor(img)
        return img_tensor


class RandomFlip(object):
    def __call__(self, img: np.ndarray) -> np.ndarray:
        horizontal_flip = np.random.choice([True, False])
        vertical_flip = np.random.choice([True, False])

        if horizontal_flip:
            img = np.flip(img, axis=1)

        if vertical_flip:
            img = np.flip(img, axis=0)

        img = img.copy()

        return img


class RandomRotate(object):
    def __call__(self, img: np.ndarray) -> np.ndarray:
        k = np.random.randint(1, 4) # number of 90 degree rotations
        img = np.rot90(img, k, axes=(0, 1)).copy()
        return img


class ColorShift(object):
    def __init__(self, min_factor: float = 0.5, max_factor: float = 1.5):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, img: np.ndarray) -> np.ndarray:
        factors = np.random.uniform(self.min_factor, self.max_factor, img.shape[-1])
        img_rescaled = np.clip(img * factors[np.newaxis, np.newaxis, :], 0, 1).astype(np.float32)
        return img_rescaled


class GammaCorrection(object):
    def __init__(self, gain: float = 1, min_gamma: float = 0.25, max_gamma: float = 2):
        self.gain = gain
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    def __call__(self, img: np.ndarray) -> np.ndarray:
        gamma = np.random.uniform(self.min_gamma, self.max_gamma, img.shape[-1])
        img_gamma_corrected = np.clip(np.power(img, gamma[np.newaxis, np.newaxis, :]), 0, 1).astype(np.float32)
        return img_gamma_corrected


class DownSampling(object):
    def __init__(self, down_factor: int):
        self.down_factor = down_factor

    def __call__(self, img: np.ndarray) -> np.ndarray:
        height, width, _, = img.shape
        new_height = height // self.down_factor
        new_width = width // self.down_factor
        img = cv2.pyrDown(img, dstsize=(new_height, new_width))
        return img

