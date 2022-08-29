# modified from https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py
import os
from PIL import Image
import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple

import torch
from torch import Tensor
import json

try:
    import accimage
except ImportError:
    accimage = None

from torchvision.utils import _log_api_usage_once
from torchvision.transforms import functional as F
from torchvision.transforms import *
from torchvision.transforms.functional import _interpolation_modes_from_int, InterpolationMode

import torchvision.transforms as T
from torchvision.transforms import Resize, Pad


class RandomCrop(T.RandomCrop):
    def forward(self, img, i, j):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.
        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        _, height, width = F.get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        # i, j, h, w = self.get_params(img, self.size)
        h, w = self.size

        img = F.crop(img, i, j, h, w)
        return img


class RandomResizedCrop(RandomResizedCrop):
    def forward(self, img, i, j, h, w):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.
        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)


class RandomRotation(RandomRotation):
    def forward(self, img, angle):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.
        Returns:
            PIL Image or Tensor: Rotated image.
        """
        fill = self.fill
        channels, _, _ = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]

        return F.rotate(img, angle, self.resample, self.expand, self.center, fill)


class GaussianBlur(GaussianBlur):
    def forward(self, img: Tensor, sigma) -> Tensor:
        """
        Args:
            img (PIL Image or Tensor): image to be blurred.
        Returns:
            PIL Image or Tensor: Gaussian blurred image
        """
        return F.gaussian_blur(img, self.kernel_size, [sigma, sigma])


def main():

    root = "/Users/bytedance/Workspace/opencv_extra/testdata/cv"
    with open("/Users/bytedance/Workspace/opencv/opencv_contrib/modules/imgaug/test/config.json", "r") as fb:
        config = json.load(fb)

    for test_sample in config["tests"]:
        namespace = test_sample["namespace"]
        method_name = test_sample["method"]
        if namespace == "imgaug":
            method = globals()[method_name](**test_sample["init_args"])
            img_path = os.path.join(root, test_sample["img_path"])
            img = Image.open(img_path)
            # img = method(img, **test_sample["runtime_args"])
            img = Compose([
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])(img)
            import numpy as np
            img = np.asarray(img).transpose([1, 2, 0])

            img = Image.fromarray(np.uint8(np.clip(img * 255, 0, 255)))

            save_path = os.path.join(root, test_sample["save_path"])
            save_path = os.path.join(root, "imgaug/normalize_test_10.jpg")
            img.save(save_path)
            break


if __name__ == '__main__':
    main()
