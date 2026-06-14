import torch
import torch.nn.functional as F
import mlflow
import os

from src.unet import SimpleUNet
from src.noise_schedule import T, alphas_cumprod, add_noise

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BETA = 1.0     # DPO temperature - controls how strongly we push toward preferences
LR = 1e-5      # small LR - we want SMALL controlled updates
EPOCHS = 10

# ---- Load data ----
data = torch.load("outputs/scored_samples.pt")
images = data["images"].to(DEVICE)         # (100, 3, 32, 32)
confidences = data["confidences"].to(DEVICE)  # (100,)

sorted_idx = torch.argsort(confidences, descending=True)
chosen_imgs = images[sorted_idx[:50]]    # top 50 = high confidence
rejected_imgs = images[sorted_idx[50:]]  # bottom 50 = low confidence

print(f"Chosen pairs: {chosen_imgs.shape}, Rejected pairs: {rejected_imgs.shape}")

# ---- Load models ----
policy_model = SimpleUNet().to(DEVICE)
policy_model.load_state_dict(torch.load("outputs/model_epoch99.pt", map_location=DEVICE))

reference_model = SimpleUNet().to(DEVICE)
reference_model.load_state_dict(torch.load("outputs/model_epoch99.pt", map_location=DEVICE))
reference_model.eval()
for p in reference_model.parameters():
    p.requires_grad = False  # frozen

optimizer = torch.optim.Adam(policy_model.parameters(), lr=LR)
alphas_cumprod = alphas_cumprod.to(DEVICE)

os.makedirs("outputs", exist_ok=True)
mlflow.set_experiment("diffusion-dpo")

with mlflow.start_run():
    mlflow.log_param("beta", BETA)
    mlflow.log_param("lr", LR)
    mlflow.log_param("epochs", EPOCHS)

    step = 0
    for epoch in range(EPOCHS):
        t = torch.randint(0, T, (chosen_imgs.shape[0],), device=DEVICE)

        noisy_chosen, noise_chosen = add_noise(chosen_imgs, t, alphas_cumprod)
        noisy_rejected, noise_rejected = add_noise(rejected_imgs, t, alphas_cumprod)

        policy_chosen_pred = policy_model(noisy_chosen, t)
        policy_rejected_pred = policy_model(noisy_rejected, t)

        with torch.no_grad():
            ref_chosen_pred = reference_model(noisy_chosen, t)
            ref_rejected_pred = reference_model(noisy_rejected, t)

        policy_chosen_err = F.mse_loss(policy_chosen_pred, noise_chosen, reduction='none').mean(dim=[1,2,3])
        policy_rejected_err = F.mse_loss(policy_rejected_pred, noise_rejected, reduction='none').mean(dim=[1,2,3])
        ref_chosen_err = F.mse_loss(ref_chosen_pred, noise_chosen, reduction='none').mean(dim=[1,2,3])
        ref_rejected_err = F.mse_loss(ref_rejected_pred, noise_rejected, reduction='none').mean(dim=[1,2,3])

        policy_diff = policy_rejected_err - policy_chosen_err
        ref_diff = ref_rejected_err - ref_chosen_err

        logits = BETA * (policy_diff - ref_diff)
        loss = -F.logsigmoid(logits).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} | DPO Loss: {loss.item():.4f}")
        mlflow.log_metric("dpo_loss", loss.item(), step=step)
        step += 1

torch.save(policy_model.state_dict(), "outputs/model_dpo.pt")
print("Saved outputs/model_dpo.pt")