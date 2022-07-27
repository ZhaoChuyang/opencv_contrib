import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import time
import torchvision.transforms as T
import PIL
import torch

img_path = 'resources/lena.png'

path = "/Users/chuyang/Downloads/cifar-10-batches-py"
data_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
with open(os.path.join(path, data_files[0]), 'rb') as fb:
    raw_data = pickle.load(fb, encoding='bytes')

data = raw_data[b'data']

# for i in range(10):
#     img = data[i, :].reshape((32, 32, 3), order='F')
#     plt.imshow(img)
#     plt.show()


crop_size = np.array([20, 20])
randomCrop = cv2.RandomCrop(crop_size)
randomFlip = cv2.RandomFlip()
transforms = cv2.Compose([randomCrop, randomFlip])

all_images = []
for i in range(len(data)):
    img = data[i, :].reshape((32, 32, 3), order='F')
    all_images.append(img)

start = time.time()
for img in all_images:
    # img = cv2.randomCrop(img, crop_size)
    # img = randomCrop.call(img)
    img = transforms.call(img)
end = time.time()

print("Run Time (OpenCV) : %f" % (end-start))


randomCrop = T.RandomCrop(size=(20, 20))
randomFlip = T.RandomHorizontalFlip()
transforms = T.Compose([randomCrop, randomFlip])
all_images = []
for i in range(len(data)):
    img = data[i, :].reshape((32, 32, 3), order='F')
    img = PIL.Image.fromarray(img)
    # img = torch.tensor(img).permute(2, 0, 1)
    all_images.append(img)

start = time.time()
for img in all_images:
    # img = randomCrop(img)
    img = transforms(img)
end = time.time()

print("Run Time (PyTorch) : %f" % (end-start))
