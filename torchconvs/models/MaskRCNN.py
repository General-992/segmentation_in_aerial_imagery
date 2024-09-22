import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, faster_rcnn, mask_rcnn

def MaskRCNN(n_class, pretrained=False, pretrained_backbone=True):
    # Load a pretrained RCNN
    model = maskrcnn_resnet50_fpn(pretrained=pretrained)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(
        in_features, n_class)
    # Now modify the mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, n_class
    )
    return model