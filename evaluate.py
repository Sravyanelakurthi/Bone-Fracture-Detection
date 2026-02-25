import torch
from sklearn.metrics import classification_report, confusion_matrix
from models.cnn_model import CNNModel
from preprocessing.data_loader import get_data_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, _, test_loader = get_data_loaders("dataset")

model = CNNModel().to(DEVICE)
model.load_state_dict(torch.load("saved_models/cnn_model.pth"))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = torch.sigmoid(model(images)).cpu()
        predictions = (outputs > 0.5).int()

        y_true.extend(labels.numpy())
        y_pred.extend(predictions.numpy())

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
