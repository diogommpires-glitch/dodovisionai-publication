# DodoVisionAI

Source-available code for experimental multimodal signal processing and
presentation-attack detection. Original project code is provided for
non-commercial use under [PolyForm Noncommercial 1.0.0](LICENSE).
[LICENSING.md](LICENSING.md) explains the scope and third-party terms.

This is a code-only publication snapshot. It contains application, signal,
fusion, evaluation and training source, a browser interface and a curated
offline regression suite. It contains no recordings, face images, camera or
PRNU arrays, calibration state, model weights, external datasets, accounts,
credentials, logs or previous Git history.

## Supported verification environment

Use Python 3.11 in a new virtual environment. The supported installation is
the patched **CPU dependency profile for offline code review and tests**:

```sh
python -m venv .venv
# Activate .venv, then:
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip check
python -m pytest -q
```

On Windows the interpreter is `.venv\Scripts\python.exe`; on Linux it is
`.venv/bin/python`. See [DEPENDENCIES.md](DEPENDENCIES.md) for the separate
PAD adapter environment. Installation downloads software packages; the tests
use synthetic inputs and do not download pretrained models or train models.

The offline checks exercise rendering and filesystem security boundaries,
dependency compatibility, fusion mathematics, ISO-style metrics, defence
utilities and per-person dynamics. The separately selected Transformers test
constructs a tiny random CLIP model locally and checks adapter output equality.
Passing these checks establishes neither detector accuracy nor sensor,
clinical, forensic or deployment suitability.

## Runtime limits

The CUDA distribution profile is omitted. GPU execution and the complete
application are not supported or validated by this snapshot's install and CI.
Starting `main.py` can initialise hardware and load or fetch external models.
Training, fetch, calibration and orchestration scripts are retained as source
for review and are never invoked by the publication checks. Their external
data, model and hardware prerequisites must be reviewed before deliberate use.
No runtime startup command is supplied as an acceptance procedure.

Only tests that work without missing fixtures, model bytes, application startup
or training are included. Older application wiring, absent ASVspoof helper,
camera fixture and model-dependent tests are omitted. This curated suite does
not claim full upstream test coverage or dataset split integrity.

Authentication, path containment and safe checkpoint boundaries remain in the
source. Local notification helpers require explicit topic and token settings;
no personal destination or populated secret is supplied. Do not publish runtime
outputs generated from personal or biometric inputs.
