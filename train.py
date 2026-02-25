import torch
import torch.nn as nn
import torch.optim as optim
from preprocessing.data_loader import get_data_loaders
from models.cnn_model import CNNModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10

train_loader, val_loader, test_loader = get_data_loaders("dataset")

model = CNNModel().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.float().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader)}")

torch.save(model.state_dict(), "saved_models/cnn_model.pth")
print("Model Saved Successfully!")
