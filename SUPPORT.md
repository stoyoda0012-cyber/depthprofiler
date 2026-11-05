# Getting help with deppro

## Where to ask

Use the GitHub issue tracker for questions, bug reports and feature requests:
<https://github.com/stoyoda0012-cyber/depthprofiler/issues>. Questions are
welcome there too; please search the existing issues first. The
[README](README.md), [`docs/api_reference.md`](docs/api_reference.md),
[`docs/data_format.md`](docs/data_format.md) and
[`docs/environment_variables.md`](docs/environment_variables.md) cover most
setup questions.

## What to expect

`deppro` is maintained by one person alongside research work. Support is
given on a best-effort basis: issues are usually answered within two weeks,
and there is no guaranteed response time.

Supported:

- the latest release and `main`;
- Python 3.11 and later on Linux and macOS (the MLX solvers need Apple
  silicon);
- problems in installing, running and interpreting the output of the
  package.

Outside the scope of support:

- analyzing your measured data for you, or choosing a structure model for a
  particular sample;
- older releases, once a newer one exists;
- the sibling packages `toyomacro` and `dnndenoiser`, which have their own
  repositories (see "Related repositories" in the README).

## Reproducing the JVST and SIA papers

What can and cannot be recomputed from the public deposit is listed in
[`docs/REPRODUCTION_JVST_SIA.md`](docs/REPRODUCTION_JVST_SIA.md). If you run
the stored-result check on your own machine, please report the outcome in an
issue, whether or not it matches.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).
