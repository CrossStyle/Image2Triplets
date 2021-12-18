import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.poolers import MultiScaleRoIAlign
from sklearn.metrics import average_precision_score
from visdom import Visdom
import time
import collections
import random
import pandas as pd
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from scheduler import WarmupMultiStepLR
import sys
sys.path.append('../')
from Image2Triplets.hoi_dataset import ACTION_CLASSES


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)


class BackwardModel(nn.Module):
    def __init__(self, image_sizes, num_class, data,  phase):
        super(BackwardModel, self).__init__()
        self.backbone = resnet_fpn_backbone('resnet50', True)
        self.num_class = num_class
        self.train_phase = phase
        self.index = [3, 9, 11]
        # [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        # [2, 3, 4, 6, 9, 11]
        self.data = data
        self.vec = None
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(False)

        self.multi_scale_align = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=2, sampling_ratio=2)
        self.image_sizes = image_sizes

        self.fc1 = nn.Sequential(
            nn.Linear(1024*3+8, 1024),
            # nn.Linear(1024*3, 1024),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1))
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1))

        self.predict = nn.Linear(512, self.num_class)

        self.gcn_conv1 = GCNConv(768, 2048)
        self.gcn_conv2 = GCNConv(2048, 1024)
        self.gcn_conv3 = GCNConv(1024, 1024)
        self.gcn_conv4 = GCNConv(1024, self.num_class)

    def forward(self, img, human_box, object_box, union_box):
        self.backbone.eval()
        self.backbone.cuda()
        interactive_class = []

        vec, edge_index = self.data.x, self.data.edge_index
        vec = F.leaky_relu(self.gcn_conv1(vec, edge_index), 0.1)
        vec = F.leaky_relu(self.gcn_conv2(vec, edge_index), 0.1)
        vec = F.leaky_relu(self.gcn_conv3(vec, edge_index), 0.1)
        vec = self.gcn_conv4(vec, edge_index)
        vec = vec[:13]
        self.vec = vec

        if self.train_phase == 'train':
            for idx in sorted(self.index, reverse=True):
                vec = vec[torch.arange(vec.shape[0]) != idx]
        elif self.train_phase == 'test':
            index = torch.tensor([True if i in self.index else False for i in range(13)])
            vec = vec[index]

        for i in range(img.shape[0]):
            x = self.backbone(img[i].unsqueeze(0).cuda())
            b1, b2, b3 = human_box[i].float().cuda(), object_box[i].float().cuda(), union_box[i].float().cuda()
            s = torch.cat((b1, b2), dim=1)
            f_h = self.multi_scale_align(x, [b1], [self.image_sizes]).view(b1.shape[0], -1)
            f_o = self.multi_scale_align(x, [b2], [self.image_sizes]).view(b2.shape[0], -1)
            f_u = self.multi_scale_align(x, [b3], [self.image_sizes]).view(b3.shape[0], -1)
            x = torch.cat((f_h, f_o, f_u, s), dim=1)
            # x = f_u
            # x = torch.cat((f_h, f_o, f_u), dim=1)

            # print(x.shape)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.predict(x)
            result = torch.sigmoid(torch.mm(x, vec.T))
            interactive_class.append(result)
        return interactive_class


def adjust_label(hoi, box, size):
    b1, b2, b3 = [], [], []
    for i in range(len(hoi)):
        tmp1 = torch.zeros(hoi[i].shape[0], 4)
        tmp2 = torch.zeros(hoi[i].shape[0], 4)
        tmp3 = torch.zeros(hoi[i].shape[0], 4)
        for j in range(hoi[i].shape[0]):
            tmp1[j] = box[i][str(int(hoi[i][j, 0].item()))][:4] * size[0]
            tmp2[j] = box[i][str(int(hoi[i][j, 1].item()))][:4] * size[0]
            tmp3[j][0] = min(tmp1[j][0], tmp2[j][0])
            tmp3[j][1] = min(tmp1[j][1], tmp2[j][1])
            tmp3[j][2] = max(tmp1[j][2], tmp2[j][2])
            tmp3[j][3] = max(tmp1[j][3], tmp2[j][3])

        b1.append(tmp1)
        b2.append(tmp2)
        b3.append(tmp3)
    return b1, b2, b3


class BceFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.75, reduction='mean'):
        super(BceFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        self.weight = torch.tensor([[2.8184, 244.5000, 61.1250, 326.0000, 0.5161,81.5000, 108.6667,
                                    54.3333, 69.8571, 88.9091]]).cuda()

    def forward(self, inputs, target):
        pos_id = (target == 1.0).float()
        neg_id = (1 - pos_id).float()
        pos_loss = -self.weight * pos_id * (1.0 - inputs)**self.gamma * self.alpha * torch.log(inputs + 1e-14)
        neg_loss = -self.weight * neg_id * inputs ** self.gamma * (1 - self.alpha) * torch.log(1.0 - inputs + 1e-14)
        if self.reduction == 'mean':
            return torch.mean(torch.sum(pos_loss+neg_loss, 1))
        else:
            return pos_loss+neg_loss


def adjust_loader(loader):
    for i in range(len(loader)):
        indexes = []
        for j in range(len(loader[i]['hoi'])):
            if loader[i]['hoi'][j].shape[0] == 0:
                indexes.append(j)
        for index in sorted(indexes, reverse=True):
            del loader[i]['hoi'][index], loader[i]['box'][index]
            loader[i]['img'] = loader[i]['img'][torch.arange(loader[i]['img'].shape[0]) != index]
    return loader


def delete_null(loader):
    new_loader = []
    for i in range(len(loader)):
        if len(loader[i]['hoi']) != 0:
            new_loader.append(loader[i])
    return new_loader


def cal_map(test_loader):
    model.eval()
    y_true = np.zeros((1, test_loader[0]['hoi'][0].shape[1]-2))
    y_pred = np.zeros((1, test_loader[0]['hoi'][0].shape[1]-2))

    with torch.no_grad():
        for _it, _batch in enumerate(test_loader):
            img = _batch['img']
            hoi = _batch['hoi']
            box = _batch['box']
            human_box, object_box, union_box = adjust_label(hoi, box, size)
            _out = model(img, human_box, object_box, union_box)
            _target = hoi
            for i in range(len(_out)):
                # if _out[i].shape[0] != _target[i].shape[0]:
                #     print(_it)
                y_pred = np.row_stack((y_pred, _out[i].cpu().detach().numpy()))
                y_true = np.row_stack((y_true, _target[i][:, 2:].numpy()))
    ap = []
    for i in range(y_true.shape[1]):
        ap.append(average_precision_score(y_true[1:, i], y_pred[1:, i]))

    map = np.mean(np.nan_to_num(ap))
    print('mAP: ', map)
    print(ap)
    return map, ap


if __name__ == '__main__':
    # python -m visdom.server
    viz = Visdom(env='Learning curve')
    options = collections.namedtuple('Options', ['loss', 'lr', 'test_mAP', 'val_mAP', 'gzsl_mAP'])(
        loss={'xlabel': 'Epoch', 'ylabel': 'Loss', 'showlegend': True, 'title': 'train_loss'},
        lr={'xlabel': 'Epoch', 'ylabel': 'Learning rate', 'showlegend': True, 'title': 'Learning rate'},
        test_mAP={'xlabel': 'Epoch', 'ylabel': 'mAP', 'showlegend': True, 'title': 'test mAP'},
        val_mAP={'xlabel': 'Epoch', 'ylabel': 'mAP', 'showlegend': True, 'title': 'val mAP'},
        gzsl_mAP={'xlabel': 'Epoch', 'ylabel': 'mAP', 'showlegend': True, 'title': 'gzsl mAP'})

    cudnn.benchmark = True
    size = (416, 416)
    batch_size = 6
    # dataset = HoiDataset(get_transform(True, size=size))
    # train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
    #                                                 collate_fn=collate_fn)
    # train_data_loader = list(train_data_loader)
    # adjusted_data_loader = adjust_data_loader(train_data_loader)
    gzsl_loader = torch.load('../hoi_dataset/new_gzsl_loader.pth')
    train_loader = torch.load('../hoi_dataset/new_train_loader.pth')
    test_loader = torch.load('../hoi_dataset/new_test_loader.pth')

    # train_loader = adjust_loader(train_loader)
    # test_loader = adjust_loader(test_loader )
    # gzsl_loader = adjust_loader(gzsl_loader)

    vec = torch.load('word_embeddings/new_vec.pth').cuda()
    raw1 = pd.read_csv('word_embeddings/link.txt', delimiter=' ', header=None)
    edge = torch.zeros(len(raw1), 2)
    for j, row in enumerate(range(len(raw1))):
        edge[j] = torch.tensor([float(i) for i in raw1.iloc[row, :].values])
    edge = edge.long().T

    # Random chain-like
    # edge = torch.zeros(2, 127)
    # for i in range(127):
    #     edge[0, i] = i
    #     edge[1, i] = i + 1
    # edge = edge.long()

    # FC
    # edge = torch.zeros(2, 128*127)
    # for i in range(128):
    #     for j in range(127):
    #         edge[0, i*127+j] = i
    #         edge[1, i*127+j] = j
    # edge = edge.long()

    # PFC
    # edge = edge[:, :128*2]

    data = Data(x=vec, edge_index=edge).to(torch.device('cuda'))
    base_lr, epochs = 0.0000125, 200
    loss_function = nn.BCELoss().cuda()
    # loss_function = BceFocalLoss().cuda()
    num_class = len(ACTION_CLASSES)
    model = BackwardModel(size, 768, data, 'gzsl').cuda()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=4e-5)

    warmup_epochs = 5
    lr_milestones = [200]
    scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=0.5,
                                  warmup_iters=warmup_epochs, warmup_factor=1e-5)
    for epoch in range(epochs):
        start = time.time()
        train_loss = torch.tensor([0.0]).cuda()
        lr = next(iter(optimizer.param_groups))['lr']
        for iter_, sample_batch in enumerate(train_loader):
            img = sample_batch['img']
            hoi = sample_batch['hoi']
            box = sample_batch['box']

            human_box, object_box, union_box = adjust_label(hoi, box, size)
            out = model(img, human_box, object_box, union_box)
            target = hoi
            loss = torch.zeros(1).cuda()
            for i in range(len(hoi)):
                tmp_loss = loss_function(out[i], target[i][:, 2:].float().cuda())
                loss = tmp_loss + loss
            optimizer.zero_grad()
            finial_loss = loss/len(hoi)
            train_loss += finial_loss
            finial_loss.backward()
            optimizer.step()
            print('Epoch[%d/%d]' % (epoch + 1, epochs), 'iter[%d]' % (iter_ + 1), 'loss: %.5f' % finial_loss)

        end = time.time()
        scheduler.step()
        viz.line(torch.stack([train_loss / iter_]), [epoch + 1], win="train loss", update='append', opts=options.loss)
        viz.line(X=torch.Tensor([epoch + 1]), Y=torch.Tensor([lr]), win='Learning rate', update='append', opts=options.lr)
        print('time comsume: ', end - start)
        if (epoch + 1) % 10 == 0:
            print('###################train eval mode####################')
            model.train_phase = 'train'
            train_map, train_ap = cal_map(train_loader)
            viz.line(X=torch.Tensor([epoch + 1]), Y=torch.Tensor([train_map]), update='append', win='val mAP', opts=options.val_mAP)
            print('###################test eval mode####################')
            model.train_phase = 'test'
            test_map, test_ap = cal_map(test_loader)
            viz.line(X=torch.Tensor([epoch + 1]), Y=torch.Tensor([test_map]), update='append', win='test mAP',
                     opts=options.test_mAP)
            print('###################gzsl eval mode####################')
            model.train_phase = 'gzsl'
            gzsl_map, gzsl_ap = cal_map(gzsl_loader)
            viz.line(X=torch.Tensor([epoch + 1]), Y=torch.Tensor([gzsl_map]), update='append', win='gzsl mAP',
                     opts=options.gzsl_mAP)
            model.train_phase = 'train'
            model.train()
        # if (epoch + 1) % 10 == 0 and epoch > 100:
        #     torch.save(model.state_dict(), 'new_%d.pth' % (epoch + 1))

