[简体中文](README.zh-CN.md) | **English**

<div align="center">
  <img src="https://raw.githubusercontent.com/tile-ai/TileOPs/main/assets/logo.png" width="350"/>
  <h1>TileOPs</h1>
  <p><strong>Spec-driven GPU operator library for LLMs — designed for AI agents to build, evaluate, and optimize</strong></p>
  <p>Built on <a href="https://github.com/tile-ai/tilelang">TileLang</a></p>
  <!-- <p>
    <a href="https://pypi.org/project/tileops/"><img src="https://img.shields.io/badge/PyPI-tileops-1E90FF" alt="PyPI version" height="20"></a>
  </p> -->
  <p>
    <a href="https://github.com/tile-ai/TileOPs/tree/main/tileops/manifest"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftile-ai%2FTileOPs%2Fstats%2Fmanifest-implemented.json" alt="Spec coverage"></a>
    <a href="https://github.com/tile-ai/TileOPs/tree/main/benchmarks"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftile-ai%2FTileOPs%2Fstats%2Fmanifest-benchmark.json" alt="Bench coverage"></a>
  </p>
  <p>
    <a href="#installation"><b>Installation</b></a> |
    <a href="#quick-start"><b>Quick Start</b></a> |
    <a href="#documentation"><b>Docs</b></a>
  </p>
</div>

> **Status**: TileOPs is under active development. APIs may change.

## 2026 Summer Camp

Summer-camp participants should use the `summer-camp-2026` branch and follow the [Chinese migration guide](docs/summer-camp/README.md). The guide defines operator claiming, the two-PR Manifest/implementation workflow, MetaX GPU validation, benchmarking, Roofline evidence, and acceptance criteria; claimable projects are listed in the [operator migration inventory](docs/summer-camp/TileKernels-MACA-operator-migration-inventory.en.md).

We recommend renting online MetaX C500 compute through the [Gitee AI Compute Marketplace](https://ai.gitee.com/compute) and selecting the summer-camp image: `PyTorch-Agent / 2.8.0 / Python 3.12 / MACA 3.7.1.5`. See the [detailed English guide](docs/summer-camp/README.en.md) for setup and environment self-check requirements.

## Overview

TileOPs is a GPU operator library for LLM training and inference, built on [TileLang](https://github.com/tile-ai/tilelang). Beyond providing a growing collection of production-quality operators, TileOPs explores a **spec-driven development model** where AI agents can read declarative operator specifications, generate kernel implementations, and evaluate them against hardware-theoretical performance bounds — with minimal human scaffolding.

### Architecture

Every operator is split into two layers with a strict boundary:

- **Op** (L2) — stateless Python entry point. Handles validation, dtype casting, and memory layout. Compatible with CUDA-Graph and `torch.compile`.
- **Kernel** (L1) — TileLang GPU implementation with hardware-specific optimizations. Upstream TileOPs kernels declare their support range in NVIDIA architecture terms (Ampere, Hopper); this repository adds `*_maca.py` implementations for some operators, dispatched at the Op layer via `is_maca()`. See [Operator availability on MetaX C500](docs/summer-camp/README.en.md#12-operator-availability-on-metax-c500).

This separation keeps user-facing behavior independent of GPU strategy, allowing agents and developers to modify either layer without side effects on the other.

### Key Properties

- **Spec-driven** — each operator is declared in a machine-readable manifest (`tileops/manifest/`) that specifies signatures, workloads, and roofline formulas, serving as the entry point for both agent code generation and automated validation
- **Roofline-evaluated** — kernel performance is measured against Speed-of-Light hardware bounds, not relative baselines
- **Auto-tuning** — built-in search over tile sizes, pipelines, and scheduling parameters
- **Lightweight** — depends only on TileLang, PyTorch, and einops

## Installation

An MXMACA-capable MetaX GPU is required at runtime.

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.1 (MetaX build, e.g. `2.8.0+metax3.7.1.3`)
- MetaX GPU: **C500**
- [TileLang](https://github.com/tile-ai/tilelang): on MetaX, use the pre-built MACA version shipped in the container

> [!WARNING]
> **Inside a MetaX container, do not run `make install`, `pip install tileops`, or any pip
> command that resolves the tilelang dependency.**
>
> The container's TileLang is an in-place source build for MACA (e.g.
> `/opt/tilelang-metax-v0.1.10`), not a pip package — `pip show tilelang` finds nothing.
> pip therefore treats it as "not installed" and pulls the official **CUDA** wheel from the
> index into site-packages, shadowing the MACA build so kernels compile for the wrong backend.
>
> For the same reason, do not use `python3 -m venv` without `--system-site-packages`: it cuts
> off the MetaX PyTorch build and the `apache-tvm-ffi` that `libtilelang.so` is ABI-coupled to.
>
> Do not pass `-c constraints.txt` either. Those pins target the CUDA CI runner and would
> downgrade `apache-tvm-ffi` below what the in-place build was compiled against. Such a
> mismatch is invisible at `import` time and only fails when the first kernel compiles.

### MetaX: use the container's pre-built TileLang

Nothing needs to be installed. Set `PYTHONPATH` and you are ready:

```bash
# Point at the container's pre-built MACA TileLang, plus this repository root
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/path/to/TileOPs-Metax:$PYTHONPATH
```

`tileops` imports without `pip install`; Manifest validation and tests run directly.

If you do need `tileops` registered in the environment (for example to run scripts from outside the repository), `--no-deps` is the only safe form:

```bash
python -m pip install -e . --no-deps --no-build-isolation
```

`--no-deps` is the essential part — it stops pip from resolving `tilelang`. The repository's CI uses exactly this form in [`scripts/ci/install_tileops.sh`](scripts/ci/install_tileops.sh).

Verify:

```bash
# MetaX GPU status. If the output has a Sliced GPU section, the usable memory and compute
# are the slice quota, not the whole-card values shown in the first section
mx-smi
python --version
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
# PyTorch must be the MetaX build (version string contains 'metax')
python -c "import torch; print(f'PyTorch {torch.__version__}')"
# TileLang must come from the container's MACA build, not a pip-installed CUDA wheel.
# The path should be under /opt/tilelang-metax-*; site-packages means it was overwritten
python -c "import tilelang; print(tilelang.__version__); print(tilelang.__file__)"
# The compilation backend must be maca, not cuda
python -c "from tilelang.utils.target import determine_target; print(determine_target('auto'))"
python -c "import einops; print('einops OK')"
```

## Quick Start

```python
import torch
from tileops.ops import GemmOp

M, N, K = 1024, 1024, 512

# GemmOp is input-inferred: m/n/k and dtype come from the forward inputs, so the
# constructor only declares layout. trans_b=False means B is stored [K, N];
# the default True corresponds to [N, K].
gemm = GemmOp(trans_a=False, trans_b=False)

A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(K, N, device="cuda", dtype=torch.float16)

C = gemm(A, B)          # [M, N]
```

> [!NOTE]
> Set `PYTHONPATH` first (see Installation above).
>
> On C500, `GemmOp` dispatches through `is_maca()` to
> `tileops/kernels/gemm_maca.py`. Not every operator has a MACA implementation — before
> picking an operator or a workload, read
> [Operator availability on MetaX C500](docs/summer-camp/README.en.md#12-operator-availability-on-metax-c500).

## Documentation

Design docs and development guides are in [`docs/`](docs/). The full API reference and performance tables are published at [TileOPs.github.io](https://github.com/tile-ai/TileOPs.github.io).

## Contributing

See [docs/](docs/) for design docs. Branch and commit conventions are in [`.claude/conventions/types.sh`](.claude/conventions/types.sh).

## License

TileOPs is released under the [MIT License](LICENSE).
