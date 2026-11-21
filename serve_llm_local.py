"""serve_llm_local.py -- helper that prints the vLLM (or fallback) start
command for serving Llama 3.1 8B locally on the 5090 Laptop.

API_up v9.8 -- replaces the Together AI / Groq external dependencies in
the deception-narrative path with a fully-offline local LLM. Used by
`llm.py::_call_vllm_local` when `LLM_BACKEND=vllm_local`.

Why two options:
  * vLLM is the highest-throughput open-source server (fastest first-
    token, fastest tok/s on Blackwell). Best for production.
  * Ollama is far easier to install on Windows + has GUI in some forks.
    Use when vLLM build/install bites you.

Memory budget on RTX 5090 Laptop (24 GB GDDR7):
  * Llama 3.1 8B Instruct, FP16:   ~16 GB weights + 2-4 GB KV cache  -> fits
  * Llama 3.1 8B Instruct, INT4:   ~5 GB weights  + 2-4 GB KV cache  -> easy fit, faster
  * Llama 3.1 70B Instruct, INT4:  ~40 GB weights -- DOES NOT FIT on 24 GB

Run:
    python serve_llm_local.py            # prints all options + recommended one
    python serve_llm_local.py --vllm     # just print the vllm command
    python serve_llm_local.py --ollama   # just print the ollama command
    python serve_llm_local.py --check    # probe whether a server is already running
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

PORT_VLLM   = 8001
PORT_OLLAMA = 11434

VLLM_INSTALL = """
# Install vLLM (one-time, ~2 GB):
.venv\\Scripts\\python.exe -m pip install vllm

# Note: vLLM Windows support has historically lagged Linux. If pip install
# fails on Windows + cu128, the proven path is:
#   * WSL2 (Ubuntu) on this same box: pip install vllm works cleanly
#   * Or use Ollama instead -- simpler on Windows
""".strip()

VLLM_START_FP16 = """
# Start vLLM serving Llama 3.1 8B Instruct (FP16, ~16 GB VRAM):
.venv\\Scripts\\python.exe -m vllm.entrypoints.openai.api_server ^
    --model meta-llama/Meta-Llama-3.1-8B-Instruct ^
    --port 8001 ^
    --dtype float16 ^
    --max-model-len 8192 ^
    --gpu-memory-utilization 0.85
""".strip()

VLLM_START_INT4 = """
# Start vLLM serving Llama 3.1 8B Instruct (INT4 AWQ, ~6 GB VRAM):
.venv\\Scripts\\python.exe -m vllm.entrypoints.openai.api_server ^
    --model TheBloke/Meta-Llama-3.1-8B-Instruct-AWQ ^
    --port 8001 ^
    --quantization awq ^
    --max-model-len 8192 ^
    --gpu-memory-utilization 0.50
""".strip()

OLLAMA_INSTALL = """
# Install Ollama (one-time, no Python deps):
#   1. Download installer from https://ollama.com/download/windows
#   2. Install (puts ollama.exe on PATH)
#   3. Pull Llama 3.1 8B (~5 GB): ollama pull llama3.1:8b
""".strip()

OLLAMA_START = """
# Start Ollama (it auto-starts as a Windows service after install).
# Verify it's running:
ollama list   # should show llama3.1:8b
curl http://localhost:11434/api/tags

# DodoVisionAI's `_call_ollama` backend already targets this endpoint;
# point at it via:
set LLM_BACKEND=ollama
""".strip()

DODO_WIRING = """
# Wire DodoVisionAI to the local backend:
set LLM_BACKEND=vllm_local            ;: routes deception narrative to vLLM
set VLLM_URL=http://127.0.0.1:8001
set VLLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
set VLLM_NUM_CTX=8192

# Or for Ollama:
set LLM_BACKEND=ollama
set OLLAMA_MODEL=llama3.1:8b
""".strip()


def check_running():
    """Probe both ports; report which servers are reachable."""
    try:
        import httpx
    except ImportError:
        print("[--] httpx not installed (it's already in requirements.lock; venv issue?)")
        return
    print("Probing local LLM endpoints ...")
    for label, url in (("vLLM",   f"http://127.0.0.1:{PORT_VLLM}/v1/models"),
                        ("Ollama", f"http://127.0.0.1:{PORT_OLLAMA}/api/tags")):
        try:
            r = httpx.get(url, timeout=3.0)
            if r.status_code == 200:
                models_json = r.json()
                if label == "vLLM":
                    models = [m.get("id", "?") for m in models_json.get("data", [])]
                else:
                    models = [m.get("name", "?") for m in models_json.get("models", [])]
                print(f"  [OK]  {label} reachable at {url}")
                for m in models[:5]:
                    print(f"            model: {m}")
            else:
                print(f"  [--]  {label} HTTP {r.status_code}")
        except Exception as e:
            print(f"  [--]  {label} not reachable ({type(e).__name__})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vllm",   action="store_true", help="just print vLLM commands")
    ap.add_argument("--ollama", action="store_true", help="just print Ollama commands")
    ap.add_argument("--check",  action="store_true", help="probe which servers are running")
    args = ap.parse_args()

    if args.check:
        check_running()
        return 0

    if args.vllm:
        print("\n--- vLLM (recommended for max throughput) ---\n")
        print(VLLM_INSTALL); print()
        print(VLLM_START_FP16); print()
        print(VLLM_START_INT4); print()
        return 0
    if args.ollama:
        print("\n--- Ollama (recommended for easy install on Windows) ---\n")
        print(OLLAMA_INSTALL); print()
        print(OLLAMA_START); print()
        return 0

    print("\n" + "=" * 70)
    print("  DodoVisionAI -- local LLM serving options (API_up v9.8)")
    print("=" * 70)
    print("\nRecommendation on this hardware (5090 Laptop, 24 GB):")
    print("  * Use Ollama if you want it working in 10 min on Windows.")
    print("  * Use vLLM (via WSL2 or after pip install vllm succeeds) for")
    print("    2-4x higher throughput in production-like demos.\n")

    print("--- Option A: Ollama (easiest on Windows) ---\n")
    print(OLLAMA_INSTALL); print()
    print(OLLAMA_START); print()

    print("\n--- Option B: vLLM (fastest, possibly WSL2-only on Windows) ---\n")
    print(VLLM_INSTALL); print()
    print(VLLM_START_FP16); print()
    print(VLLM_START_INT4); print()

    print("\n--- Wire DodoVisionAI to the local backend ---\n")
    print(DODO_WIRING); print()
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
