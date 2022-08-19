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


def main():

    root = "/Users/bytedance/Workspace/opencv_extra/testdata/cv"
    with open("/Users/bytedance/Workspace/opencv/opencv_contrib/modules/aug/test/config.json", "r") as fb:
        config = json.load(fb)

    for test_sample in config["tests"]:
        namespace = test_sample["namespace"]
        method_name = test_sample["method"]
        if namespace == "aug":
            method = globals()[method_name](**test_sample["init_args"])
            img_path = os.path.join(root, test_sample["img_path"])
            img = Image.open(img_path)
            img = method(img, **test_sample["runtime_args"])
            save_path = os.path.join(root, test_sample["save_path"])
            img.save(save_path)


if __name__ == '__main__':
    main()
