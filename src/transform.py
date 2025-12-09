from torchvision import transforms
from src.config import CFG

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
