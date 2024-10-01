from .DeepLabv3 import resnet, deeplab, intermediate_layer_getter

# Deeplab v3+ resnet backbone
def Deeplabv3plus_resnet(n_class: int=6, pretrained_backbone: bool=True):
    """Constructs a DeepLabV3Plus model."""
    replace_stride_with_dilation = [False, False, True]
    aspp_dilate = [6, 12, 18]

    backbone = resnet.resnet50(
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation
    )

    inplanes = 2048
    low_level_planes = 256

    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    classifier = deeplab.DeepLabHeadV3Plus(inplanes, low_level_planes, n_class, aspp_dilate)

    backbone = intermediate_layer_getter.IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = deeplab.Deep_SimpleSegmentationModel(backbone, classifier)

    for param in model.parameters():
        param.requires_grad = True

    model.train(True)
    return model