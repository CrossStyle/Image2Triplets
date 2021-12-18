import torch
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import json
import copy


VOC_CLASSES = ('Polisher', 'Forklift', 'Board', 'Glass_glue', 'Peg', 'Brush', 'Crane', 'Glass', 'person', 'Concrete',
               'Cell_phone', 'Iron_hook', 'Spatula', 'Ladder', 'Lron_clamp', 'Ceramic_tile', 'Iron_rod', 'Rebar',
               'Barrel', 'Trolley', 'Pallet_and_concrete', 'Cement', 'Lime_slurry', 'Da_ya_zui', 'Aluminum',
               'Circular_saw', 'Axe', 'Air_conditioner', 'Screws', 'Welding_tongs', 'rubber_hammer', 'Sand',
               'Arc_mask', 'Cold_bending_machine', 'Hammer_head', 'Tuohui_ban', 'Glass_suction_cup',
               'Shovel', 'Brick', 'Tape_measure', 'Laser', 'Window_frame', 'Paint_roller', 'Electric_drill', 'Slab')


ACTION_CLASSES = ('Use', 'Cold bend', 'Install', 'Hold', 'Push or pull', 'Move', 'Knock', 'Apply', 'Step on', 'Transport', 'Shovel', 'Lay', 'Drive')
SEEN_ACTION_CLASSES = ('Use', 'Cold bend', 'Install', 'Push or pull', 'Move', 'Knock', 'Apply', 'Step on', 'Shovel', 'Drive')
UNSEEN_ACTION_CLASSES = ('Hold', 'Transport', 'Lay')


VOC_ROOT = 'VOCdevkit/VOC2007'
Path = 'final.json'


def get_transform(train, size=(416, 416)):
    mean = [0.441, 0.437, 0.425]
    std = [0.242, 0.244, 0.245]
    transform = transforms.Compose([
        transforms.Resize(size=size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    if train:
        transform = transforms.Compose([
            transforms.Resize(size=size),
            # transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    return transform


class HoiDataset(torch.utils.data.Dataset):
    def __init__(self, name, transforms=None, size=(416, 416)):
        self.root = VOC_ROOT
        self.path = Path
        self.name = name
        self.raw_dict = None
        self.transforms = transforms
        self.label_data = self.get_label_data()
        self.voc_class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        if self.name == 'gzsl':
            self.action_class_to_ind = dict(zip(ACTION_CLASSES, range(len(ACTION_CLASSES))))
        elif self.name == 'train':
            self.action_class_to_ind = dict(zip(SEEN_ACTION_CLASSES, range(len(SEEN_ACTION_CLASSES))))
        elif self.name == 'test':
            self.action_class_to_ind = dict(zip(UNSEEN_ACTION_CLASSES, range(len(UNSEEN_ACTION_CLASSES))))
        self.size = size

    def __getitem__(self, idx):
        label_data = self.label_data[idx]
        img_path = os.path.join(self.root, "JPEGImages", label_data['img'])
        img = Image.open(img_path).convert("RGB")
        height = img.height
        width = img.width
        target, finial_boxes = {}, {}
        boxes = copy.deepcopy(label_data['boxes'])
        for index in range(len(boxes)):
            box = boxes[str(index)][1:]
            normalized_box = [float(int(box[i]) / width) if i % 2 == 0 else float(int(box[i]) / height) for i in
                              list(range(4))]
            class_idx = self.voc_class_to_ind[boxes[str(index)][0]]
            finial_boxes[str(index)] = torch.as_tensor(normalized_box + [float(class_idx)], dtype=torch.float32)
        target['boxes'] = finial_boxes
        new_verb_array = np.column_stack((np.zeros((len(label_data['verb']), 2)),
                                          np.zeros((len(label_data['verb']), len(self.action_class_to_ind)))))

        for i, verb in enumerate(label_data['verb']):
            new_verb_array[i, :2] = verb[:2]
            for action in verb[2:]:
                if action in self.action_class_to_ind.keys():
                    action_idx = int(self.action_class_to_ind[action])
                    new_verb_array[i, 2 + action_idx] = 1

        person_list = []
        for i in boxes.keys():
            if boxes[i][0] == 'person':
                person_list.append(int(i))
        delete_list = []
        for i in range(len(new_verb_array)):
            if sum(new_verb_array[i, 2:]) == 0:
                delete_list.append(i)
            if new_verb_array[i, 0] not in person_list and new_verb_array[i, 1] not in person_list:
                delete_list.append(i)
            elif new_verb_array[i, 1] in person_list:
                new_verb_array[i, 0], new_verb_array[i, 1] = new_verb_array[i, 1], new_verb_array[i, 0]
        finial_verb_array = np.delete(new_verb_array, delete_list, 0)
        target['verb'] = torch.from_numpy(finial_verb_array)
        target['img'] = label_data['img']

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.label_data)

    def get_label_data(self):
        tmp = []
        with open(self.path, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                tmp.append(json.loads(line))
        return tmp


def collate_fn(batch):
    return tuple(zip(*batch))


def adjust_data_loader(loader):
    new_data_loader = []
    for index, sample_batch in enumerate(loader):
        new_batch = dict()
        # print(len(sample_batch[0]))
        new_batch['img'] = torch.zeros(len(sample_batch[0]), 3, 416, 416)
        tmp_boxes_list, tmp_label_list = [], []
        for batch_index in range(len(sample_batch[0])):
            new_batch['img'][batch_index] = sample_batch[0][batch_index]
            tmp_boxes_list.append(sample_batch[1][batch_index]['boxes'])
            tmp_label_list.append(sample_batch[1][batch_index]['verb'])
        new_batch['hoi'] = tmp_label_list
        new_batch['box'] = tmp_boxes_list
        new_data_loader.append(new_batch)
    return new_data_loader


if __name__ == '__main__':
    device = torch.device('cuda')
    size = (416, 416)
    num_classes = len(VOC_CLASSES)
    dataset = HoiDataset('test', get_transform(True, size))
    # dataset = HoiDataset('train', get_transform(True, size))
    # dataset = HoiDataset('gzsl', get_transform(True, size))
    train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True,
                                                    collate_fn=collate_fn, drop_last=True)
    train_data_loader = list(train_data_loader)

    adjusted_data_loader = adjust_data_loader(train_data_loader)
    for i in range(len(adjusted_data_loader)):
        indexes = []
        for j in range(len(adjusted_data_loader[i]['hoi'])):
            if adjusted_data_loader[i]['hoi'][j].shape[0] == 0:
                indexes.append(j)
        for index in sorted(indexes, reverse=True):
            del adjusted_data_loader[i]['hoi'][index], adjusted_data_loader[i]['box'][index]
            adjusted_data_loader[i]['img'] = adjusted_data_loader[i]['img'][
                torch.arange(adjusted_data_loader[i]['img'].shape[0]) != index]

    torch.save(adjusted_data_loader, 'test_loader.pth')
    # torch.save(adjusted_data_loader, 'train_loader.pth')
    # torch.save(adjusted_data_loader, 'gzsl_loader.pth')

