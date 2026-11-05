# Contributing to deppro

Thank you for considering a contribution. `deppro` is research software for
reconstructing depth-composition profiles from angle-resolved XPS and HAXPES
data. Bug reports, questions about results, documentation fixes and code
changes are all welcome.

## Reporting a problem

Open an issue at
<https://github.com/stoyoda0012-cyber/depthprofiler/issues>. A useful report
includes:

- the `deppro` version (`python -c "import deppro; print(deppro.__version__)"`),
  the Python version and the operating system, and which extras are installed
  (`fit`, `nn`, `mlx`);
- the smallest script or command that shows the problem, with synthetic data
  if the measured data cannot be shared;
- what you expected and what you got, including the full traceback.

If a reconstruction looks physically wrong rather than crashing, say which
solver, regularizer, weight (`alpha`), prior and starting values you used.
In this package those choices are part of the result (see "What you can do
with it" in the [README](README.md)), so the report cannot be reproduced
without them.

## Proposing a change

1. Open an issue first for anything beyond a small fix, so the approach can
   be agreed before you spend time on it.
2. Fork the repository, create a branch from `main`, and open a pull request
   against `main`. Keep one topic per pull request.
3. Describe what changed and why, and how you checked it. If a change alters
   numerical results, say which results change and by how much.
4. Add an entry under `[Unreleased]` in [`CHANGELOG.md`](CHANGELOG.md) for
   user-visible changes.

## Development setup

Python 3.11 or later.

```bash
git clone https://github.com/stoyoda0012-cyber/depthprofiler.git
cd depthprofiler
pip install -e '.[dev,fit]'
```

`fit` installs `toyomacro` (the compound, binding-energy and cross-section
tables) from a pinned GitHub release. Without it, the tests under
`src/deppro/app/tests/` are not collected. The `nn` and `mlx` extras are only
needed for the denoiser bridge and the Apple-GPU solvers; tests that need them
are skipped when they are not installed.

## Running the tests

```bash
pytest
```

An argument-less `pytest` in the repository root runs the regression suite
and excludes tests marked `slow` (run those with `pytest -m slow`). The CI
workflow ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the
same suite on Linux with Python 3.12, and weekly with 3.11 as well.

A few tests are **acceptance gates**: they state a target the implementation
has not yet reached, and they are expected to fail. CI runs them in a
separate job and deselects them from the regression job, so the regression
job means "nothing got worse" and the acceptance job shows what is still
open. Please do not turn a failing gate green with `skip`, `xfail` or a
looser threshold; if your change makes a gate pass, say so in the pull
request.

## Style

- Follow the style of the surrounding code: the docstring format of the
  module you edit, the existing naming, and comments only where they
  explain why.
- Comments, docstrings and documents use American spelling.
- Physical tables (cross sections, mean-free-path parameters, binding
  energies) come from cited sources. Do not type values in from memory; add
  the source and how the value was read.
- The README and the paper separate what has been shown from what is only
  possible. Please keep that separation when you document a feature.

## Use of generative AI

You may use AI tools to prepare a contribution. Please say in the pull
request which tools you used and for what, and confirm that you have
reviewed and tested the result yourself.

## Project governance

`deppro` is maintained by Satoshi Toyoda, who reviews pull requests and
decides what is merged and released. Releases follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); until 1.0 the
API may change between minor versions. Each tagged release is archived on
Zenodo. Contributors are credited in the release notes and, for substantial
contributions, in [`CITATION.cff`](CITATION.cff).

By contributing you agree that your contribution is licensed under the
project's MIT License ([`LICENSE`](LICENSE)).

For how to get help, see [`SUPPORT.md`](SUPPORT.md).
