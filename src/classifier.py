import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
import os

from src.dataset import get_dataloader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))   # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))   # 8x8 -> 4x4
        x = x.view(x.size(0), -1)              # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # raw scores (logits)


if __name__ == "__main__":
    EPOCHS = 5
    BATCH_SIZE = 64
    LR = 1e-3

    model = SimpleClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    dataloader = get_dataloader(batch_size=BATCH_SIZE)

    os.makedirs("outputs", exist_ok=True)
    mlflow.set_experiment("cifar10-classifier")

    with mlflow.start_run():
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("lr", LR)

        step = 0
        for epoch in range(EPOCHS):
            correct, total = 0, 0
            for images, labels in dataloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                loss = loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                if step % 100 == 0:
                    mlflow.log_metric("loss", loss.item(), step=step)
                step += 1

            acc = correct / total
            print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Train Acc: {acc:.4f}")
            mlflow.log_metric("train_accuracy", acc, step=epoch)

    torch.save(model.state_dict(), "outputs/classifier.pt")
    print("Saved outputs/classifier.pt")