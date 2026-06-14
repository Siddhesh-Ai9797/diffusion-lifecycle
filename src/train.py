import torch
import torch.nn as nn
import mlflow
import os
import glob

from src.unet import SimpleUNet
from src.dataset import get_dataloader
from src.noise_schedule import T, alphas_cumprod, add_noise

# ---- Config ----
EPOCHS = 100
BATCH_SIZE = 64
LR = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ---- Setup ----
model = SimpleUNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
alphas_cumprod = alphas_cumprod.to(DEVICE)

dataloader = get_dataloader(batch_size=BATCH_SIZE)

os.makedirs("outputs", exist_ok=True)

# ---- Resume logic ----
start_epoch = 0
checkpoints = glob.glob("outputs/model_epoch*.pt")
if checkpoints:
    latest = max(checkpoints, key=lambda x: int(x.split("epoch")[1].split(".")[0]))
    start_epoch = int(latest.split("epoch")[1].split(".")[0]) + 1
    model.load_state_dict(torch.load(latest, map_location=DEVICE))
    print(f"Resuming from {latest}, starting at epoch {start_epoch}")

# ---- MLflow ----
mlflow.set_experiment("diffusion-cifar10")

with mlflow.start_run():
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("lr", LR)
    mlflow.log_param("T", T)

    step = start_epoch * len(dataloader)
    for epoch in range(start_epoch, EPOCHS):
        for images, _ in dataloader:
            images = images.to(DEVICE)
            batch_size = images.shape[0]

            t = torch.randint(0, T, (batch_size,), device=DEVICE)
            noisy_images, true_noise = add_noise(images, t, alphas_cumprod)
            predicted_noise = model(noisy_images, t)
            loss = loss_fn(predicted_noise, true_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
                mlflow.log_metric("loss", loss.item(), step=step)

            step += 1

        torch.save(model.state_dict(), f"outputs/model_epoch{epoch}.pt")
        print(f"Saved checkpoint: outputs/model_epoch{epoch}.pt")

print("Training complete!")