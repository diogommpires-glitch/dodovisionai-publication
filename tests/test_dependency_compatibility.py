"""Exercise the patched CPU dependencies without model downloads or hardware."""

from importlib.metadata import version
from io import BytesIO

from packaging.version import Version


def test_patched_image_codec_preserves_rgb_pixels():
    from PIL import Image

    assert Version(version("pillow")) >= Version("13.0.0")
    pixels = Image.new("RGB", (4, 3), (12, 34, 56))
    encoded = BytesIO()
    pixels.save(encoded, format="PNG")
    encoded.seek(0)
    with Image.open(encoded) as decoded:
        assert decoded.size == (4, 3)
        assert decoded.convert("RGB").tobytes() == pixels.tobytes()


def test_patched_filelock_excludes_another_holder(tmp_path):
    import pytest
    from filelock import FileLock, Timeout

    assert Version(version("filelock")) >= Version("4.2.1")
    path = tmp_path / "inference.lock"
    with FileLock(path), pytest.raises(Timeout), FileLock(path, timeout=0):
        pass
    with FileLock(path, timeout=0) as released:
        assert released.is_locked


def test_patched_torch_scripts_tensor_arithmetic():
    import torch

    assert Version(version("torch")) >= Version("2.13.0")
    script = torch.jit.CompilationUnit(
        "def twice(x: Tensor) -> Tensor:\n    return x + x\n"
    )
    actual = script.twice(torch.tensor([1.0, -2.0]))
    torch.testing.assert_close(actual, torch.tensor([2.0, -4.0]))


def test_patched_keras_preserves_dense_inference():
    import keras
    import numpy as np

    assert Version(version("keras")) >= Version("3.15.0")
    layer = keras.layers.Dense(1, use_bias=False, kernel_initializer="ones")
    result = layer(np.array([[2.0, 3.0]], dtype=np.float32))
    np.testing.assert_allclose(keras.ops.convert_to_numpy(result), [[5.0]])
