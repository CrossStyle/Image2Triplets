import os
import torch
import os
from PIL import Image
from xml.dom.minidom import parse
import numpy as np
import cv2
from torch.utils.data import Dataset
import torchvision
import pickle
from torchvision import transforms

'''
Calculate the mean and standard deviation of the images in the dataset
计算数据集中图像的均值和标准差
'''

VOC_ROOT = 'VOCdevkit/VOC2007'


def compute_mean_and_std(dataset):
    # 输入PyTorch的dataset，输出均值和标准差
    mean_r = 0
    mean_g = 0
    mean_b = 0

    for img in dataset:
        img = np.asarray(img)  # change PIL Image to numpy array

        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])

    mean_b /= len(dataset)
    mean_g /= len(dataset)
    mean_r /= len(dataset)

    diff_r = 0
    diff_g = 0
    diff_b = 0

    N = 0

    for img in dataset:
        img = np.asarray(img)

        diff_b += np.sum(np.power(img[:, :, 0] - mean_b, 2))
        diff_g += np.sum(np.power(img[:, :, 1] - mean_g, 2))
        diff_r += np.sum(np.power(img[:, :, 2] - mean_r, 2))

        N += np.prod(img[:, :, 0].shape)

    std_b = np.sqrt(diff_b / N)
    std_g = np.sqrt(diff_g / N)
    std_r = np.sqrt(diff_r / N)

    mean = (mean_b.item() / 255.0, mean_g.item() / 255.0, mean_r.item() / 255.0)
    std = (std_b.item() / 255.0, std_g.item() / 255.0, std_r.item() / 255.0)
    print('Mean 均值：', mean)
    print('Std 标准差：', std)
    return mean, std


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        img = Image.open(img_path)
        return img

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dataset = Dataset(VOC_ROOT)
    mean, std = compute_mean_and_std(dataset)

