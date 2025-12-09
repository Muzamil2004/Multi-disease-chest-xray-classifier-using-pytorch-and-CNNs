import torch.nn as nn
from torchvision import models
from src.config import CFG

def get_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(CFG.LABELS))
    return model
