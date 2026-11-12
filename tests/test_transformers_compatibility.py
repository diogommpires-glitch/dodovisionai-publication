"""Check the foundation PAD adapter against the patched Transformers API."""

from importlib.metadata import version

from packaging.version import Version


def test_clip_qv_adapters_preserve_initial_vision_output():
    import torch
    from transformers import CLIPConfig, CLIPModel

    from pad.train_foundation_pad import _inject_lora_on_qv

    assert Version(version("transformers")) >= Version("5.10.1")
    config = CLIPConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "max_position_embeddings": 8,
        },
        vision_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 16,
            "patch_size": 8,
        },
        projection_dim=4,
    )
    model = CLIPModel(config).eval()
    pixels = torch.zeros(1, 3, 16, 16)
    with torch.no_grad():
        expected = model.vision_model(pixels).pooler_output
        adapters = _inject_lora_on_qv(model, rank=2, alpha=2, dtype=torch.float32)
        actual = model.vision_model(pixels).pooler_output
    assert len(adapters) == 4
    assert all(parameter.requires_grad for parameter in adapters)
    torch.testing.assert_close(actual, expected)
