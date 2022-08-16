import os
import numpy as np
import torch
from PIL import Image
import cv2
import argparse
from IPython import embed
import transforms as T


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/Users/bytedance/Downloads/PennFudanPed")
    parser.add_argument("--backend", type=str, default="cv2")

    return parser.parse_args()


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, backend='cv2'):
        assert backend in ['cv2', 'pil']
        self.backend = backend
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def _get_boxes(self, mask):
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            yield xmin, ymin, xmax, ymax

    def __getitem__(self, idx):
        # load images and masks
        if self.backend == 'pil':
            img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
            mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
            img = Image.open(img_path).convert("RGB")
            # note that we haven't converted the mask to RGB,
            # because each color corresponds to a different instance
            # with 0 being background
            mask = Image.open(mask_path)
            mask = np.array(mask)
            # convert the PIL Image into a numpy array

            boxes = []
            for x1, y1, x2, y2 in self._get_boxes(mask):
                boxes.append([x1, y1, x2, y2])

            num_objs = len(boxes)
            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.ones((num_objs,), dtype=torch.int64)
            image_id = torch.tensor([idx])

            target = {
                "boxes": boxes,
                "labels": labels,
                "image_id": image_id,
            }

            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target["boxes"], target["labels"]

        else:
            img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
            mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # mask is array of size (H, W), all elements of array are integers
            # background is 0, and each distinct person is represented as a distinct integer starting from 1
            # you can treat mask as grayscale image
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            boxes = []
            for x1, y1, x2, y2 in self._get_boxes(mask):
                # NOTE: in opencv, box is represented as (x, y, width, height)
                boxes.append([x1, x2, x2-x1, y2-y1])
            num_objs = len(boxes)
            labels = torch.ones((num_objs,), dtype=torch.int64)

            if self.transforms is not None:
                img, boxes = self.transforms.call(img, boxes)

            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            return img, boxes, labels

    def __len__(self):
        return len(self.imgs)


def get_transforms(backend='cv2'):
    if backend == 'cv2':
        transforms = cv2.det.Compose([
            cv2.det.RandomFlip(),
            cv2.det.Resize((500, 500)),
        ])
    elif backend == 'pil':
        transforms = T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.Resize((500, 500))
        ])
    return transforms


def main():
    args = get_args()
    if args.backend == 'pil':
        transforms =
    dataset = PennFudanDataset(args.root, backend=args.backend)


if __name__ == '__main__':
    main()





