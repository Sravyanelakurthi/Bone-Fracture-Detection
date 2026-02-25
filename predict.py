import torch
from PIL import Image
from torchvision import transforms
from models.cnn_model import CNNModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNModel().to(DEVICE)
model.load_state_dict(torch.load("saved_models/cnn_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = torch.sigmoid(model(image))
        prediction = (output > 0.5).item()

    return "Fractured" if prediction else "Not Fractured"

print(predict("sample.jpg"))
