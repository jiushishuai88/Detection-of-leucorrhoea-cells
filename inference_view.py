from models import *
from data_set import LeucorrheaDataset
import utils_custom.view_tools as view_tools
import time
import  numpy as np
import torch


def main():
    device = torch.device('cuda')
    model = get_model('trained_fasterrcnn_resnet50_fpn')
    model.eval()
    model.to(device)
    data_set = LeucorrheaDataset(train=False)
    indices = torch.randperm(len(data_set))
    for i in range(100):
        index = indices[i]
        img, target = data_set[index]
        img = img.to(device)
        boxes = target['boxes'].numpy()
        labels = target['labels'].numpy()
        img_origin = img.cpu().numpy().transpose((1, 2, 0)).astype(np.float32).copy()
        view_tools.draw_one_image(img_origin, boxes, labels)

        start_time = time.time()
        output = model([img])
        time_elapsed = time.time() - start_time
        print('inference time elapsed:{}'.format(time_elapsed))
        predict_boxes = output[0]['boxes'].cpu().detach().numpy()
        predict_labels = output[0]['labels'].cpu().detach().numpy()
        img_predict = img.cpu().numpy().transpose((1, 2, 0)).astype(np.float32).copy()
        view_tools.draw_one_image(img_predict, predict_boxes, predict_labels)

        view_tools.showImages([img_origin,img_predict],['img_origin','img_predict'])


if __name__ == '__main__':
    main()