import os
import yaml
from PIL import Image
import torch
import detection.transforms as T


def get_data_path():
    data_path = 'data/RecGrapReslutForNet'
    cwd = os.getcwd()
    path  = os.path.join(cwd, data_path)
    return path


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_bbox_and_labels_from_ini_config(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        boxes = []
        labels = []
        for key in config:
            value = config[key]
            labels.append(value['label'])
            boxes.append([value['x1'], value['y1'], value['x2'], value['y2']])
        return boxes, labels


class LeucorrheaDataset(object):
    def __init__(self,train=True):
        root = get_data_path()
        self.root = root
        self.transforms = get_transform(train)
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'Images'))))
        self.anotations = list(sorted(os.listdir(os.path.join(root, "Anotations"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root, "Images", img_name)
        anotation_name = self.anotations[idx]
        anotation_path = os.path.join(self.root, "Anotations", anotation_name)
        img = Image.open(img_path).convert("RGB")

        boxes, labels = get_bbox_and_labels_from_ini_config(anotation_path)
        num_objs = len(labels)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
