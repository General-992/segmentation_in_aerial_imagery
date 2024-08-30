import torch
import segmentation_models_pytorch as smp

def UnetResnet50(n_class):
    model = smp.UnetPlusPlus(
            encoder_name="resnet50",     # Encoder model, choose from available models
            encoder_weights="imagenet",  # Use ImageNet pre-trained weights for encoder initialization
            in_channels=3,               # Model input channels (e.g., 3 for RGB images)
            classes=n_class,             # Number of classes
        )

    for param in model.parameters():
        param.requires_grad = True
    return model