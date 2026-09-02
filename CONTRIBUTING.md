# Contributing

This is a research reference release rather than a maintained OpenGait fork.
Focused corrections to the paper mapping, compatibility notes, tests, and
evaluation protocol are welcome.

For a reproducibility or compatibility report, include:

- the OpenGait commit or release;
- Python, PyTorch, CUDA, and GPU versions;
- the exact configuration diff;
- modality tensor shapes and the selected preprocessing filenames;
- whether `use_student_ce` and `detach_balance` were enabled;
- the full traceback or the evaluation protocol used.

Do not upload datasets, private paths, credentials, or checkpoints without
redistribution permission. In an OpenGait environment where PyTorch is already
installed, run the core tests after installing the development requirements:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests/test_core_math.py
```

By contributing, you agree that your changes may be distributed under this
repository's license.
