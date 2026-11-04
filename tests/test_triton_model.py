"""The ONNX model in the Triton repository must match its config and the PyTorch model."""
import re
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from src.unet import SimpleUNet

MODEL_DIR = Path("triton_models/diffusion_unet")
ONNX_PATH = MODEL_DIR / "1" / "model.onnx"


def test_onnx_io_names_match_triton_config():
    config = (MODEL_DIR / "config.pbtxt").read_text()
    config_names = re.findall(r'name:\s*"(\w+)"', config)
    graph = onnx.load(str(ONNX_PATH), load_external_data=False).graph
    assert [i.name for i in graph.input] == ["x", "t"]
    assert [o.name for o in graph.output] == ["noise_pred"]
    for name in ("x", "t", "noise_pred"):
        assert name in config_names


def test_onnx_batch_dimension_is_dynamic():
    graph = onnx.load(str(ONNX_PATH), load_external_data=False).graph
    for tensor in [*graph.input, *graph.output]:
        assert tensor.type.tensor_type.shape.dim[0].dim_param, f"{tensor.name} has a fixed batch size"


def test_onnx_matches_pytorch_weights():
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    model = SimpleUNet()
    state = torch.load("outputs/model_fp16.pt", map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.float() for k, v in state.items()})
    model.eval()

    x = torch.randn(3, 3, 32, 32)
    t = torch.tensor([[0], [250], [999]])
    with torch.no_grad():
        expected = model(x, t.squeeze(-1)).numpy()

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"x": x.numpy(), "t": t.numpy()})[0]
    # The checked in PyTorch weights are FP16, so allow FP16 rounding error.
    assert np.abs(actual - expected).max() < 0.05
