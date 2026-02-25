import torch.nn as nn
import torchvision.models as models

def get_mobilenet_model():
    model = models.mobilenet_v3_large(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[3] = nn.Linear(1280, 1)

    return model
