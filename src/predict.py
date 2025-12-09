import torch
from PIL import Image
from torchvision import transforms

def predict_single_image(model, img_path, labels, img_size, device):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = torch.sigmoid(model(image)).cpu().numpy()[0]

    return dict(zip(labels, outputs))
