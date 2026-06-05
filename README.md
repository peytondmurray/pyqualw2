# pyqualw2

`pyqualw2` is a Python toolkit for configuring, running, and analyzing output
for the [CE-QUAL-W2 water quality and hydrodynamic simulation
engine](https://www.ce.pdx.edu/w2/).

> [!WARNING]
> `pyqualw2` is under construction. Not all features are implemented.

## Background

CE-QUAL-W2 is a hydrodynamic and water quality simulation engine that can model
rivers, estuaries, lakes, reservoirs, and river basin systems. It's a widely
used and incredibly useful tool, but it requires great care and niche expertise
to configure and use it for modeling real world systems. Some pain points
include

- **Configuration**: Configuring a CE-QUAL-W2 simulation requires manipulating
  values in a large CSV configuration file by hand, making it both error-prone
  and difficult to understand what each configuration value does.
- **Simulation**: The CE-QUAL-W2 binary expects simulation inputs and outputs
  to exist on certain places on disk. Furthermore, running the engine modifies
  those input files, making it impossible to know what the original
  configuration was for a given simulation.
- **Post-processing**: Cleaning up after a simulation, carrying out
  post-processing on results, and generating data visualizations remains a
  tedious manual task.

`pyqualw2` aims to introduce tooling at each step of the process to ease the
burden of configuring, running, and generating insights from CE-QUAL-W2 output.

## Installation

### Pixi (Recommended)

1. If running on Linux, install [Wine](https://www.winehq.org/) so that the
   CE-QUAL-W2 binary can be run.
2. Install [`pixi`](https://pixi.prefix.dev/latest/installation/).
2. Then install the project with `pixi install`.

### Manual installation

To install `pyqualw2`, ensure you have python installed. Then simply use `pip`:

```bash
pip install git+https://github.com/dwarfstar-dev/pyqualw2
```

## Usage

> [!WARNING]
> The Python interface is still under construction.

`pyqualw2` provides a Python interface for CE-QUAL-W2, allowing simulations to
be configured, run, and analyzed from inside Python.

## Development

Start by cloning the repository:


```bash
git clone git@github.com:dwarfstar-dev/pyqualw2
```

### Pixi (Recommended)

Install all the dependencies:

```bash
pixi install --all
```

### Manual installation

To set up your development environment, install the `dev` and `test` optional
dependency group:

```bash
pip install -e '.[dev,test]'
```

This will install the package along with some other developer tooling.

### Pre-commit hooks

This project uses pre-commit hooks to ensure code quality. To get started,
ensure [`prek`](https://github.com/j178/prek) is installed (it's one of the
dependencies included in the optional `dev` dependency group).

Run

```bash
pixi run install-lint
```

or, if you're not using pixi, just do

```bash
prek install
```

to install the pre-commit hooks.

Whichever way these are installed, they will now be run on every `git commit`.
To ignore the hooks temporarily, do `git commit -n`. To run them manually, run
`pixi run lint` (if you're using pixi) or `prek run --all-files` if not. Read
more about pre-commit hooks on the [`prek` github
page](https://github.com/j178/prek).

## Testing

Make sure the `test` optional dependencies are installed. Then run tests by
calling `pytest` from the root of the repository, or `pixi run pytest` if you're
using pixi. If you want to run end-to-end tests as well, pass the `--e2e` flag;
this requires Wine if you're on Linux.

## Jupyter Notebooks

Some example notebooks are provided in `notebooks/`. To use them, make sure the
`notebook` optional dependencies are installed, then do `pixi run notebook`, or
`jupyter lab` if you're not using pixi.

See the [JupyterLab
docs](https://jupyterlab.readthedocs.io/en/stable/index.html) for more
information.
