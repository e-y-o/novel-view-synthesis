# Novel View Synthesis

Fast single-image novel view synthesis with a web GUI.

Pipeline: depth estimation (EXIF-aware) → analytic reprojection → depth-weighted
splatting → classical background fill (interactive preview) → generative infill.

Status: design phase.

Model weights are never included in this repository. Depth Pro 2 weights are
under Apple's research-only licence and are downloaded separately by each user.

## Setup

Requires Windows/Linux with an NVIDIA GPU + recent driver (`nvidia-smi` works)
and [uv](https://docs.astral.sh/uv/). Developed on an RTX 5070 Ti (sm_120).

```powershell
winget install --id astral-sh.uv   # if uv is missing
uv sync                            # creates .venv (Python 3.12) with PyTorch cu130
```

PyTorch comes from the CUDA 13.0 wheel index (configured in `pyproject.toml`);
the stable cu130 build includes Blackwell `sm_120` kernels.
