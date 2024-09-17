import segmentation_models_pytorch as smp

def UnetPlusPlus(n_class: int):
    """
    Constructs Unet with ResNet50 DeepLabv3.
    UNet++ introduces nested skip connections.
    """
    model = smp.UnetPlusPlus(
            encoder_name="resnet50",     # Encoder model, choose from available models
            encoder_weights="imagenet",  # Use ImageNet pre-trained weights for encoder initialization
            in_channels=3,               # Model input channels (e.g., 3 for RGB images)
            classes=n_class,             # Number of classes
        )
    # Ensure that gradients are on
    for param in model.parameters():
        param.requires_grad = True
    model.train()
    return model


