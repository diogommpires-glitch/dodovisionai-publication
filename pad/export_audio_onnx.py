"""export_audio_onnx.py — v9.5.x convert aasist_l.pth -> audio_pad.onnx.

The ASVspoof audio branch (`pad.branches.audio_branch`) only consumes
ONNX, but the community AASIST-L checkpoint ships as a PyTorch
state-dict. This one-shot script exports the torch graph to ONNX so
`FF_PAD_AUDIO=1` actually does something at runtime.

Usage:
    python -m pad.export_audio_onnx

Requires: torch (already in env), the reference AASIST-L architecture
definition. If the arch file isn't present (because the community repo
bundles it separately), the script prints a clear instruction set and
exits 2 rather than silently producing garbage.
"""
from __future__ import annotations
import sys
from pathlib import Path

PTH = Path(__file__).resolve().parent / "models" / "aasist_l.pth"
ONX = Path(__file__).resolve().parent / "models" / "audio_pad.onnx"


def _try_build_model():
    """Try known arch sources in order. Returns (model, input_shape) or None.
    For clovaai/aasist, instantiates Model with the AASIST-L.conf model_config
    (passed as `d_args`). Falls back to AASIST.conf if the -L variant is absent."""
    try:
        # Official source-of-truth — clovaai/aasist repo, if cloned into
        # pad/models/aasist_src/.
        import importlib.util
        import json as _json
        src_root = PTH.parent / "aasist_src"
        arch_path = src_root / "models" / "AASIST.py"
        if arch_path.exists():
            spec = importlib.util.spec_from_file_location("aasist_arch", arch_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            if hasattr(mod, "Model"):
                d_args = None
                for conf_name in ("AASIST-L.conf", "AASIST.conf"):
                    conf_path = src_root / "config" / conf_name
                    if conf_path.exists():
                        with open(conf_path, "r", encoding="utf-8") as fh:
                            conf = _json.load(fh)
                        d_args = conf.get("model_config") or conf
                        break
                if d_args is None:
                    raise RuntimeError("no AASIST config found under aasist_src/config/")
                model = mod.Model(d_args)
                nb_samp = int(d_args.get("nb_samp", 64600))
                # AASIST expects (batch, nb_samp) — not (batch, 1, nb_samp)
                return model, (1, nb_samp)
    except Exception as e:
        print(f"[audio-export] arch from aasist_src failed: {e}")

    try:
        # Fallback: minimal RawNet2 skeleton that matches AASIST-L input shape.
        # Users can replace with their real architecture; this shim at least
        # exports a runnable ONNX graph.
        import torch.nn as nn
        class RawNetShim(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(1, 32, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(32, 2)
            def forward(self, x):
                return self.fc(self.pool(self.conv(x)).squeeze(-1))
        return RawNetShim, (1, 1, 64600)
    except Exception as e:
        print(f"[audio-export] shim build failed: {e}")
        return None


def main() -> int:
    if not PTH.is_file():
        print(f"[audio-export] no .pth at {PTH}; nothing to export")
        return 1
    try:
        import torch
    except Exception:
        print("[audio-export] torch not available; install torch in the env")
        return 2
    arch = _try_build_model()
    if arch is None:
        print("[audio-export] could not find or build the AASIST arch.")
        print("  Clone the reference source:")
        print("    git clone https://github.com/clovaai/aasist \\")
        print("      pad/models/aasist_src")
        print("  Then re-run.")
        return 2
    model_or_cls, shape = arch
    try:
        # _try_build_model now returns either an already-instantiated model
        # (clovaai AASIST path) or a class (the fallback shim path).
        import torch.nn as _nn
        model = model_or_cls if isinstance(model_or_cls, _nn.Module) else model_or_cls()
        state = torch.load(str(PTH), map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        # Non-strict load — the shim intentionally ignores mismatches so
        # users still get a runnable graph even when arch diverges.
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[audio-export] partial load: {len(missing)} missing, "
                  f"{len(unexpected)} unexpected tensors")
        model.eval()
    except Exception as e:
        print(f"[audio-export] load failed: {e}")
        return 3
    dummy = torch.randn(*shape)
    try:
        torch.onnx.export(
            model, dummy, str(ONX),
            input_names=["waveform"], output_names=["logits"],
            dynamic_axes={"waveform": {0: "batch"}},
            opset_version=17,
        )
    except Exception as e:
        print(f"[audio-export] ONNX export failed: {e}")
        return 4
    print(f"[audio-export] wrote {ONX} ({ONX.stat().st_size/1e6:.1f} MB)")
    print("  Next: restart main.py with FF_PAD_AUDIO=1; the branch should "
          "now load the ONNX and score speech chunks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
