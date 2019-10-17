import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.detection.rpn import AnchorGenerator

__all = ['fasterrcnn_resnet50_fpn', 'trained_fasterrcnn_resnet50_fpn']

model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
}


def fasterrcnn_resnet50_fpn():
    model = rcnn_model()

    state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
                                          progress=True)
    model.load_state_dict(state_dict)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 8)

    return model


def trained_fasterrcnn_resnet50_fpn():
    path = './TrainedModels/Trained_faster_rcnn_rpn_50.pth.tar'
    model = rcnn_model()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 8)

    model.load_state_dict(torch.load(path))
    return model


def rcnn_model():
    anchor_sizes = ((52,), (95,), (245,), (348,), (457,))
    aspect_ratios = ((0.58, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )

    cls_weights = torch.tensor([1,1.38, 1.,1.84,6.73,12.55,72.34,56.89]).to(torch.device('cuda'))
    backbone = resnet_fpn_backbone('resnet50', pretrained=False)
    model = FasterRCNN(backbone, 91, rpn_anchor_generator=rpn_anchor_generator,box_nms_thresh=0.3,cross_weights=cls_weights)
    return model
