import cv2
import matplotlib.pyplot as plt
import numpy as np


def main():
    img = cv2.imread("/Users/bytedance/Workspace/opencv/samples/data/lena.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    aug = cv2.RandomFlip()

    for _ in range(5):
        print(img.shape)
        img = aug.call(img)
        # dst = cv2.randomCrop(img, [200, 200], padding=(0, 0, 0, 0))
        print(img.shape)
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    main()
