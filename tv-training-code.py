from engine import train_one_epoch, evaluate
import utils
import transforms as T
from models import *
from data_set import LeucorrheaDataset
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn

root = "/home/ai/PycharmProjects/TrainNet/data/RecGrapReslutForNet"
net_name = 'fasterrcnn_resnet50_fpn'
version = '0.01'
save_model_path = '/home/ai/PycharmProjects/TrainNet/TrainedModels/{}_{}.pth.tar'.format(net_name, version)
save_model_path_best = '/home/ai/PycharmProjects/TrainNet/TrainedModels/{}_{}_best.pth.tar'.format(net_name, version)

best_mAp = 0
cn_mul_gpu = False


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_writer(name):
    path = 'runs/{}_{}/{}'.format(net_name, version, name)
    if os.path.exists(path):
        shutil.rmtree(path)
    writer = SummaryWriter(path)
    return writer


def main():
    device = torch.device('cuda:0')

    dataset_train = LeucorrheaDataset(train=True)
    dataset_test = LeucorrheaDataset(train=False)

    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = torch.utils.data.Subset(dataset_train, indices[0:-2000])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-2000:])
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=6, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=6, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    model = get_model(net_name)
    if cn_mul_gpu:
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0005,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=50,
                                                   gamma=0.1)

    num_epochs = 200
    writer_train = get_writer('train')
    writer_test = get_writer('test')
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=200, writer=writer_train)
        lr_scheduler.step()
        _, bbox_mAp = evaluate(model, data_loader_test, device, epoch, writer_test)

        if bbox_mAp > best_mAp:
            torch.save(model.state_dict(), save_model_path_best)

        if epoch % 10 == 9:
            torch.save(model.state_dict(), save_model_path)
    print("Done!!!!!!!!!!!!!!!!!!!!!")


if __name__ == "__main__":
    main()
