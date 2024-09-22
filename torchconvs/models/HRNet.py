
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
class HRNetSegmentation(nn.Module):
    def __init__(self, n_class=7, pretrained=True):
        super(HRNetSegmentation, self).__init__()
        # Load HRNet backbone from TIMM or torchvision
        self.backbone = create_model('hrnet_w48', pretrained=pretrained, features_only=True)

        # Fully Convolutional Network (FCN) head for segmentation
        self.conv1 = nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(512, n_class, kernel_size=1)

    def forward(self, x):
        # Extract features from HRNet backbone
        features = self.backbone(x)[-1]  # Get the last feature map from HRNet

        # Apply segmentation head (fully convolutional)
        x = self.conv1(features)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)

        # Upsample to the size of the input
        x = F.interpolate(x, size=(x.shape[2] * 4, x.shape[3] * 4), mode='bilinear', align_corners=False)
        return x

def HRNet(n_class=7, pretrained=True):
    """
    Constructs a HRNet model.
    """
    model = HRNetSegmentation(n_class=n_class, pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = True
    model.train()
    return model