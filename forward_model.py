import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.poolers import MultiScaleRoIAlign
from sklearn.metrics import average_precision_score
from visdom import Visdom
import time
import collections
import sys
sys.path.append('../')
from Image2Triplets import hoi_dataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)


class ForwardModel(nn.Module):
    # 前向过程
    def __init__(self, image_sizes, num_class):
        super(ForwardModel, self).__init__()
        self.backbone = resnet_fpn_backbone('resnet50', False)
        self.num_class = num_class
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(False)

        self.multi_scale_align = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=2, sampling_ratio=2)
        self.image_sizes = image_sizes

        self.fc1 = nn.Sequential(
            nn.Linear(1024*3+8, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU())

        self.predict = nn.Linear(512, self.num_class)

    def forward(self, img, human_box, object_box, union_box, vec):
        self.backbone.eval()
        self.backbone.cuda()
        interactive_class = []
        for i in range(img.shape[0]):
            x = self.backbone(img[i].unsqueeze(0).cuda())
            b1, b2, b3 = human_box[i].float().cuda(), object_box[i].float().cuda(), union_box[i].float().cuda()
            s = torch.cat((b1, b2), dim=1)
            f_h = self.multi_scale_align(x, [b1], [self.image_sizes]).view(b1.shape[0], -1)
            f_o = self.multi_scale_align(x, [b2], [self.image_sizes]).view(b2.shape[0], -1)
            f_u = self.multi_scale_align(x, [b3], [self.image_sizes]).view(b3.shape[0], -1)
            x = torch.cat((f_h, f_o, f_u, s), dim=1)
            # print(x.shape)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.predict(x)
            result = torch.sigmoid(torch.mm(x, vec))
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


def cal_map(test_loader, word_vectors):
    model.eval()
    y_true = np.zeros((1, word_vectors.shape[1]))
    y_pred = np.zeros((1, word_vectors.shape[1]))
    with torch.no_grad():
        for _it, _batch in enumerate(test_loader):
            img = _batch['img']
            hoi = _batch['hoi']
            box = _batch['box']

            human_box, object_box, union_box = adjust_label(hoi, box, size)
            _out = model(img, human_box, object_box, union_box, word_vectors)
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


class BceFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.75, reduction='mean'):
        super(BceFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        self.weight = torch.tensor([[2.8184, 244.5000, 61.1250, 2.2904, 326.0000, 0.5161,81.5000, 108.6667,
                                    54.3333, 51.4737, 69.8571,195.6000, 88.9091]]).cuda()

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
    train_loader = torch.load('../hoi_dataset/new_train_loader.pth')
    test_loader = torch.load('../hoi_dataset/new_test_loader.pth')
    gzsl_loader = torch.load('../hoi_dataset/new_gzsl_loader.pth')

    # train_loader = adjust_loader(train_loader)
    # test_loader  = adjust_loader(test_loader )
    # gzsl_loader = adjust_loader(gzsl_loader)

    action_embeddings = torch.load('word_embeddings/new_gzsl_vec.pth').T.cuda()
    seen_action_embeddings = torch.load('word_embeddings/new_seen_vec.pth').T.cuda()
    unseen_action_embeddings = torch.load('word_embeddings/new_unseen_vec.pth').T.cuda()

    base_lr, epochs = 0.000001, 200
    loss_function = nn.BCELoss().cuda()
    # loss_function = BceFocalLoss().cuda()
    model = ForwardModel(size, 768).cuda()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=4e-5)

    for epoch in range(epochs):
        start = time.time()
        train_loss = torch.tensor([0.0]).cuda()
        lr = next(iter(optimizer.param_groups))['lr']
        for iter_, sample_batch in enumerate(train_loader):
            img = sample_batch['img']
            hoi = sample_batch['hoi']
            box = sample_batch['box']

            human_box, object_box, union_box = adjust_label(hoi, box, size)
            out = model(img, human_box, object_box, union_box, seen_action_embeddings)
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
        # scheduler.step()
        viz.line(torch.stack([train_loss / iter_]), [epoch + 1], win="train loss", update='append', opts=options.loss)
        viz.line(X=torch.Tensor([epoch + 1]), Y=torch.Tensor([lr]), win='Learning rate', update='append', opts=options.lr)
        print('time comsume: ', end - start)
        if (epoch + 1) % 10 == 0:
            print('###################train eval mode####################')
            train_map, train_ap = cal_map(train_loader, seen_action_embeddings)
            viz.line(X=torch.Tensor([epoch + 1]), Y=torch.Tensor([train_map]), update='append', win='val mAP', opts=options.val_mAP)
            print('###################test eval mode####################')
            test_map, test_ap = cal_map(test_loader, unseen_action_embeddings)
            viz.line(X=torch.Tensor([epoch + 1]), Y=torch.Tensor([test_map]), update='append', win='test mAP',
                     opts=options.test_mAP)
            print('###################gzsl eval mode####################')
            gzsl_map, gzsl_ap = cal_map(gzsl_loader, action_embeddings)
            viz.line(X=torch.Tensor([epoch + 1]), Y=torch.Tensor([gzsl_map]), update='append', win='gzsl mAP',
                     opts=options.gzsl_mAP)
            model.train()
        # if (epoch + 1) % 20 == 0:
        #     torch.save(model.state_dict(), '%d.pth' % (epoch + 1))

