import segmentation_models_pytorch as smp
def UnetPlusPlus(n_class: int = 6):
    """
    Constructs Unet with ResNet34 pretrained on imagenet.
    UNet++ introduces nested skip connections.
    """
    model = smp.UnetPlusPlus(
            encoder_name="resnet34",     # Encoder backbone model
            encoder_weights="imagenet",  # Use ImageNet pre-trained weights for encoder initialization
            in_channels=3,               # Model input channels (e.g., 3 for RGB images)
            classes=n_class,             # Number of classes
        )
    # Ensure that gradients are on
    for param in model.parameters():
        param.requires_grad = True
    model.train()
    return model


