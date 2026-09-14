# MAE 5110 Code Assignments

Code assignments for MAE 5110.

## Installation

Install [Git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/). After cloning this repository, run the following command from its root directory:

```console
uv sync --python 3.14
```

This creates a local `.venv` and installs the required dependencies. Run Python commands inside the environment with `uv run`, for example:

```console
uv run python assignment_0.py
```

## Assignments

- [Assignment 0](assignments/assignment_0.md)
- [Assignment 1](assignments/assignment_1.md)

## Assignment 1 submission

The [Assignment 1 report](assignment_1/assignment_1.md), simulation code, and figures
are collected in `assignment_1/`. The original assignment instructions remain above.

- `assignment_1/assignment_1.md`: report.
- `assignment_1/codes/`: simulation and analysis scripts, model, and RK4 integrator.
- `assignment_1/figures/`: figures and saved numerical results.

After installing dependencies, run these commands from the repository root:

```console
uv run python assignment_1/codes/sanity_checks.py
uv run python assignment_1/codes/state_space_basins.py
uv run python assignment_1/codes/phase_portrait.py
uv run python assignment_1/codes/compare_attractors.py
uv run python assignment_1/codes/poincare_map.py
uv run python -u assignment_1/codes/parameter_study.py
```

To redraw the parameter-study figures from the saved results:

```console
uv run python assignment_1/codes/parameter_study.py --plot-only
```

The optional slope-sweep visualization is available with
`uv run python assignment_1/codes/gamma_sweep.py`.
All scripts save their results in `assignment_1/figures/`, regardless of the
working directory.
