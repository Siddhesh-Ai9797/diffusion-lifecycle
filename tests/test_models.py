import pytest
import torch

from src.noise_schedule import T, add_noise, alphas_cumprod, betas
from src.unet import SimpleUNet
from src.unet_attention import SimpleUNetAttention
from src.unet_linear_attention import SimpleUNetLinearAttention
from src.unet_mamba import SimpleUNetMamba


def test_schedule_shapes_and_range():
    assert betas.shape == alphas_cumprod.shape == (T,)
    assert torch.all(alphas_cumprod[1:] < alphas_cumprod[:-1])
    assert 0.99 < alphas_cumprod[0] <= 1.0
    assert alphas_cumprod[-1] < 1e-3


def test_add_noise_keeps_signal_at_t0_and_destroys_it_at_tmax():
    torch.manual_seed(0)
    image = torch.randn(4, 3, 32, 32)
    early, _ = add_noise(image, torch.zeros(4, dtype=torch.long), alphas_cumprod)
    late, noise = add_noise(image, torch.full((4,), T - 1), alphas_cumprod)
    assert torch.allclose(early, image, atol=0.05)
    assert torch.allclose(late, noise, atol=0.05)


@pytest.mark.parametrize("model_cls", [SimpleUNet, SimpleUNetAttention, SimpleUNetLinearAttention, SimpleUNetMamba])
def test_every_unet_variant_predicts_noise_of_input_shape(model_cls):
    torch.manual_seed(0)
    model = model_cls().eval()
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, T, (2,))
    with torch.no_grad():
        out = model(x, t)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_unet_gradients_flow():
    model = SimpleUNet()
    loss = model(torch.randn(2, 3, 32, 32), torch.randint(0, T, (2,))).pow(2).mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
