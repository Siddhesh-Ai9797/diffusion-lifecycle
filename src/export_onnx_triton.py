import torch
import torch.nn as nn
from src.unet import SimpleUNet
from src.noise_schedule import T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Wrapper that reshapes t from 2D (batch, 1) to 1D (batch,)
class UNetTritonWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t):
        # Triton sends t as (batch, 1) — squeeze to (batch,)
        t = t.squeeze(-1)
        return self.model(x, t)

model = SimpleUNet().to(DEVICE)
model.load_state_dict(torch.load("outputs/model_epoch99.pt", map_location=DEVICE))
model.eval()

wrapper = UNetTritonWrapper(model)

# dummy inputs — t is now (batch, 1) for Triton
dummy_x = torch.randn(4, 3, 32, 32, device=DEVICE)
dummy_t = torch.randint(0, T, (4, 1), device=DEVICE)

with torch.no_grad():
    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_t),
        "outputs/unet_triton.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["x", "t"],
        output_names=["noise_pred"],
        dynamic_axes={
            "x": {0: "batch_size"},
            "t": {0: "batch_size"},
            "noise_pred": {0: "batch_size"}
        },
        dynamo=False
    )

size_mb = __import__('os').path.getsize("outputs/unet_triton.onnx") / (1024*1024)
print(f"Exported: outputs/unet_triton.onnx ({size_mb:.2f} MB)")
print("t input shape: (batch, 1) — compatible with Triton batching")