import torch
import torch.nn as nn
import torchvision.models as models

class SegNetBase(nn.Module):
    def __init__(self, num_classes = 6, pretrained=True):
        super(SegNetBase, self).__init__()

        if pretrained:
            vgg16 = models.vgg16_bn(pretrained=True)
        else:
            vgg16 = models.vgg16(pretrained=False)

        self.encoder = nn.Sequential(*list(vgg16.features.children())[:33])

        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def SegNet(n_class: int, pretrained=True):
    """
    Constructs custom SegNet model
    """
    model = SegNetBase(n_class)
    for param in model.parameters():
        param.requires_grad = True
    model.train()
    return model