# Dependency profiles

Use Python 3.11 in a dedicated environment. The default `requirements.txt`
selects the patched CPU profile in `requirements-cpu.txt`, whose shared
dependencies are pinned in `requirements.lock`. `requirements-dev.txt` supplies
test and audit tools. These files support offline verification, rather than a
qualified hardware application installation.

```sh
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip check
python -m pytest -q
```

The profile uses CPU Torch 2.13.0, torchvision 0.28.0, torchaudio 2.11.1 and
setuptools 84.1.0. The former CUDA 12.8 profile is not distributed here. Its
Torch/setuptools constraints had unresolved advisories, and no replacement GPU
stack or hardware migration has been validated. Omitting that profile changes
this distribution's support scope; it does not repair the older GPU install.

For the separate CPU PAD adapter check, use another Python 3.11 environment:

```sh
python -m pip install torch==2.13.0+cpu torchvision==0.28.0+cpu torchaudio==2.11.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r pad/requirements.txt -r requirements-dev.txt
python -m pip check
python -m pytest tests/test_transformers_compatibility.py -q
```

The PAD profile has its own NumPy constraint and must not be combined with the
runtime lock. No model download, dataset, hardware access or training is needed
for the selected adapter test. Passing synthetic checks does not validate
scientific model performance, CUDA support or the complete application.

The curated default test paths are explicit in `pytest.ini`. The adapter test
is separate because Transformers is installed by the PAD profile. CI uses
these same boundaries. Other source entry points retain their original runtime
and external prerequisites and have not been qualified by these checks.

The unused optional Ultralytics and Ultralytics THOP pins are omitted from
the verification lock. DeepFace 0.0.99 does not require them in its package
metadata; this source selects its `skip` or `retinaface` detector paths.
The offline checks do not invoke DeepFace or a YOLO backend. Installing or
redistributing optional components requires their own licence review. In
particular, the omitted Ultralytics packages use AGPL terms, which this
project's non-commercial licence does not replace. No combined YOLO
application distribution is provided or approved by this snapshot.
