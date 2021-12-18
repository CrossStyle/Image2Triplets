import cv2
import torch
from backward_model import BackwardModel
import sys
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import numpy as np
sys.path.append('/')
import hoi_dataset
from hoi_dataset.dataset import ACTION_CLASSES, SEEN_ACTION_CLASSES, UNSEEN_ACTION_CLASSES


def write_bbox(img):
    cv2.namedWindow('image')
    cv2.imshow('image', img)
    show_crosshair = False
    from_center = False
    rect = cv2.selectROI('image', img, show_crosshair, from_center)
    x, y, w, h = rect
    xmin, ymin, xmax, ymax = x, y, x+w, y+h
    rect1 = cv2.selectROI('image', img, show_crosshair, from_center)
    x1, y1, w1, h1 = rect1
    xmin1, ymin1, xmax1, ymax1 = x1, y1, x1 + w1, y1 + h1
    cv2.waitKey(0)
    xmin2 = min(xmin, xmin1)
    ymin2 = min(ymin, ymin1)
    xmax2 = max(xmax, xmax1)
    ymax2 = max(ymax, ymax1)
    return torch.tensor([[xmin, ymin, xmax, ymax]])/416.0, torch.tensor([[xmin1, ymin1, xmax1, ymax1]])/416.0, \
           torch.tensor([[xmin2, ymin2, xmax2, ymax2]])/416.0


def img_to_tensor(img, mean=(0.441, 0.437, 0.425), std=(0.242, 0.244, 0.245)):
    img = img / 255.0
    img -= mean
    img /= std
    img = img[:, :, (2, 1, 0)]
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(dim=0)
    return img


if __name__ == '__main__':
    img = cv2.imread('demo/UseWeldingClamp.jpg')
    img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_LINEAR)
    bb1, bb2, bb3 = write_bbox(img)
    bb1 = bb1.unsqueeze(0)
    bb2 = bb2.unsqueeze(0)
    bb3 = bb3.unsqueeze(0)

    img = img_to_tensor(img).float().cuda()

    vec = torch.load('word_embeddings/vec.pth').cuda()
    raw1 = pd.read_csv('word_embeddings/link.txt', delimiter=' ', header=None)
    edge = torch.zeros(len(raw1), 2)
    for j, row in enumerate(range(len(raw1))):
        edge[j] = torch.tensor([float(i) for i in raw1.iloc[row, :].values])
    edge = edge.long().T
    data = Data(x=vec, edge_index=edge).to(torch.device('cuda'))
    mode = 'gzsl'
    vrd = BackwardModel((416, 416), 768, data, mode)
    vrd.load_state_dict(torch.load('model/500.pth'))
    vrd.eval().cuda()
    out = vrd(img, bb1, bb2, bb3)[0].detach().cpu().numpy()
    out = np.around(out, 2)
    num_list = list(np.reshape(out, -1))  # 纵坐标值
    x = range(len(num_list))
    fig = plt.figure(figsize=(12, 4))
    plt.xlabel("interactions")
    plt.ylabel("confidence")
    if mode == 'test':
        label_list = UNSEEN_ACTION_CLASSES
    elif mode == "train":
        label_list = SEEN_ACTION_CLASSES
    else:
        label_list = ACTION_CLASSES
    plt.bar(label_list, num_list)
    for i in range(len(num_list)):
        plt.text(i, num_list[i]+0.02, "%s" % num_list[i], va='center')
    plt.show()
