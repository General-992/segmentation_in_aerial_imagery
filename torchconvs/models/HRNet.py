import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from torch.cuda.amp import autocast

class HRNetV2Segmentation(nn.Module):
    def __init__(self, n_class=6, pretrained=True):
        super(HRNetV2Segmentation, self).__init__()

        # Load HRNet backbone from timm with features_only=True to get the feature maps at different resolutions
        self.backbone = create_model('hrnet_w32', pretrained=pretrained, features_only=True)

        # 1x1 Convolution after concatenation of upsampled features
        self.conv1x1 = nn.Conv2d(in_channels=1984, out_channels=512, kernel_size=1,
                                 bias=False)  # Adjust in_channels based on concatenated features
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        # Final segmentation head with the number of classes
        self.final_conv = nn.Conv2d(512, n_class, kernel_size=1)

    def forward(self, x):

        features = self.backbone(x)
        # Get the highest resolution shape to upsample lower-resolution maps
        target_size = features[0].shape[2:]  # Shape of the highest resolution feature map (e.g., [8, 64, 256, 256])

        # Upsample lower-resolution feature maps to match the highest resolution
        upsampled_features = [
            F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) for f in features
        ]
        # concat all upsampled feature maps
        combined_features = torch.cat(upsampled_features, dim=1)  # Now it's [batch_size, sum(channels), height, width]
        # 1x1 convolution to mix the combined multi-resolution features
        x = self.conv1x1(combined_features)
        x = self.bn(x)
        x = self.relu(x)
        # final segmentation head
        x = self.final_conv(x)
        # upsample to the input size for pixel-wise segmentation
        x = F.interpolate(x, size=(x.shape[2] * 2, x.shape[3] * 2), mode='bilinear', align_corners=False)
        return x
def HRNet(n_class=7, pretrained=True):
    """
    Constructs a HRNet model.
    """
    model = HRNetV2Segmentation(n_class=n_class, pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = True
    model.train()
    return model


# if __name__ == '__main__':
    # dummy_image = torch.rand([4, 3, 512, 512])
    # dummy_target = torch.rand([4, 512, 512])
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    # dummy_image, dummy_target = dummy_image.to(device), dummy_target.to(device)
    # model = HRNet(n_class=7)
    # model.to(device)
    # a = 2
    # output = model(dummy_image)
    # print(output.shape)