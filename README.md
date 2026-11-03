# depthprofiler (`deppro`)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23877023.svg)](https://doi.org/10.5281/zenodo.23877023)

A Japanese version of this README is at [`docs/ja/README.md`](docs/ja/README.md); this English version is authoritative.

`deppro` reconstructs the depth profile of composition, D(z), of a thin
film from angle-resolved XPS (AR-XPS) or HAXPES peak intensities. It is
written for surface scientists who already have angle series — one
sample, or every pixel of a spatial map — and who want to see how the
composition profile, the film thickness and the interface position they
infer depend on the choices made along the way: the structural
assumption put into the inversion, the regularizer and its strength, and
the preprocessing of the spectra (binning, background, cross sections,
denoising).

**Input**: peak intensities I(θ, element) from any peak-fitting program,
or binned HAXPES frame stacks that the `haxpes` subpackage turns into
intensities. **Output**: D(z) on a chosen depth grid, the fitted film
thickness and interface position where the model has them, the forward
prediction and misfit, and — for maps — per-pixel arrays and metrics.
Python 3.11 or later.

## What it is used for

The three uses below are the ones the package supports today, with the
API that does each. The same `TransferMatrix` (forward model) and the
same observed intensities are shared by every solver, so a different
inversion is a different solver or configuration, not a different
dataset.

**Thickness of a known layer structure.** When the layer sequence and
the stoichiometry of each layer are known, `GenericLayerSolver` fits the
layer thicknesses (optionally also interface widths and stoichiometric
ratios) from the angle series; `python -m deppro.cli_depth solve
--config multilayer.yaml` does the same from a YAML description
(`src/deppro/examples/multilayer.yaml`).

**Changing the structural assumption.** The three solvers put different
amounts of prior structure into the inversion. `FreeSolver` leaves the
composition at every depth free and relies on the regularizer and an
optional prior profile; `SemiFreeSolver` models the substrate as an
error-function step whose position (the film thickness) is fitted, and
leaves the film composition free at every depth under charge-balance or
stoichiometric constraints; `GenericLayerSolver` fixes the layer
sequence. For any of them, `SolverConfig(method=...)` selects
unregularised least squares or an L1, L2 or maximum-entropy regulariser,
and the solver settings expose the remaining assumptions — the interface
width (fixed by default, `optimize_abruptness`), a thickness prior
(`t_prior_center`), the stoichiometric mode, the noise model (WLS,
Anscombe, Poisson likelihood). Re-running the same intensities under
another assumption is how the sensitivity of D(z), the thickness and the
interface position to that assumption is examined.
`src/deppro/examples/freesolver_recommended.py` is the FreeSolver case
on a synthetic three-layer film; `docs/api_reference.md` has the
configuration of each solver.

**Maps, preprocessing and evaluation.** For a HAXPES spatial map,
`deppro.haxpes` goes from frame stacks to a per-pixel depth profile
(`process_single_pixel`, `run_3d_map` with multiprocessing), with the
preprocessing choices — angle binning, background, cross-section source
and non-dipole correction, angle trimming, the polar offset of the
geometry — in `HAXPESSolverConfig` and `HAXPESInstrumentConfig`, and an
optional bridge to the `dnndenoiser` package for denoising frames
(`HaxpesDenoiser`). The batched MLX solvers (`deppro.solvers_mlx`) solve
many pixels of a map at once on Apple-silicon GPUs. For evaluation,
`deppro.simulate` generates synthetic profiles and intensities with a
known truth, and `deppro.depth_profile` has the comparison metrics
(`wasserstein_distance_1d`, `compare_profiles`, `detect_interfaces`);
`deppro.tools.gt_vs_recon` puts a reconstruction next to its ground truth.

What has been done with this so far, and what has not: the two
published studies (below) kept the inversion fixed — `SemiFreeSolver`
with L1 regularization and a fixed interface width — and compared
denoising methods and exposure times through the per-pixel profiles,
thicknesses and the Wasserstein-1 distance to a pseudo-ground truth. The
sensitivity of the result to the fixed interface width (2.5–10 Å) is
reported in §III.A of the JVST A paper. Comparing the three solver
families, or the regularizers, on the same measured map has been done
only on synthetic data in the test suite and benchmarks, not in a
published study. The Monte Carlo and bootstrap error estimation in
`deppro.batch_processor` is not available in this release (see "Known
limitations"); `deppro.haxpes.bootstrap` has frame and block bootstraps
for the HAXPES pipeline.

```
AR-XPS / HAXPES spectra
    │  (denoising)           … dnndenoiser (separate repository)
    │  (Voigt peak fitting)  … toyomacro (separate repository)
    ▼  depth-profile inverse problem  … deppro (this repository)
depth profile of composition D(z)
```

## Installation

Python 3.11 or later.

```bash
git clone https://github.com/stoyoda0012-cyber/depthprofiler.git
cd depthprofiler
pip install -e .                 # from the repository root (src layout, where pyproject.toml is)
pip install -e '.[fit]'          # compound and cross-section tables and Voigt fitting (toyomacro, pinned release from git)
pip install -e '.[nn]'           # the DNN-denoiser bridge (torch + dnndenoiser, pinned release from git)
pip install -e '.[mlx]'          # Apple GPU batch solvers and the NN inverse solver (mlx)
```

`[fit]` and `[nn]` fetch the sibling packages, which are not on PyPI, at GitHub
release tags pinned in `pyproject.toml` (version and commit recorded there). The
core alone gives `import deppro` and the classical solvers, but looking up a
compound table by name needs `[fit]` or a CSV given through
`DEPPRO_COMPOUND_TABLE_PATH`
([`docs/environment_variables.md`](docs/environment_variables.md)).

The example scripts (`deppro/examples/*.py`) ship with the package, but the data
they read (YAML, CSV) is not in the wheel or sdist; take it from the repository at
the same tag. The tests are not shipped either: run an argument-less `pytest` in a
cloned tree (the configuration lives in `pyproject.toml`).

## Try it

Generate a synthetic three-layer film and reconstruct its depth profile with
FreeSolver:

```bash
pip install -e '.[fit]'
python -m deppro.examples.freesolver_recommended
```

The entry points of the API are `TransferMatrix` (the forward model) and
`FreeSolver` / `SemiFreeSolver` / `GenericLayerSolver` (the inverse problem).
Arguments and return values are described in
[`docs/api_reference.md`](docs/api_reference.md), the H5 data format in
[`docs/data_format.md`](docs/data_format.md). The locations of external data are
passed through `DEPPRO_*` environment variables (listed in
[`docs/environment_variables.md`](docs/environment_variables.md)).

## Reproducing the JVST / SIA papers

For the two papers that used this package (JVST A and SIA),
[`docs/REPRODUCTION_JVST_SIA.md`](docs/REPRODUCTION_JVST_SIA.md) states what can be
recomputed from public material alone, what needs material from the authors, and
what was actually run. The stored results of the Zenodo deposit can be checked with
`python -m deppro.haxpes.deposit_replay` (pass the extracted deposit through
`--sia-dir` / `--jvst-h5`; the steps are in the same document). That check
recomputes the published metrics from the deposited profiles; it does not re-run
the denoiser, the peak fit or the inversion.

## Layout

| Location | Contents |
|------|------|
| `src/deppro/` | transfer matrix and the inverse solvers (LLS / MEM / L1 / L2), the NN inverse solver (`inference/`), the MLX solvers (`solvers_mlx/`), HAXPES preprocessing (`haxpes/`), simulation (`simulate/`), examples (`examples/`) |
| `src/deppro/tests/`, `src/deppro/app/tests/` | the regression suite and the acceptance gates (the CI configuration is `.github/workflows/ci.yml`) |
| `docs/` | environment variables, the H5 data format, the solver API, reproduction of the JVST/SIA papers |
| `paper/` | the software paper (for JOSS; to be submitted after publication) |

## Related repositories

| Repository | Role |
|------|------|
| **depthprofiler** (this repository) | depth analysis (classical, NN and MLX inverse solvers) |
| [**dnndenoiser**](https://github.com/stoyoda0012-cyber/dnndenoiser) | deep-learning denoiser for XPS spectra (including self-supervised training) |
| [**toyomacro**](https://github.com/stoyoda0012-cyber/toyomacro) | Voigt peak fitting, and the compound, binding-energy and cross-section tables |

## Status

Version 0.1.0. The API may change before 1.0.

## Known limitations

- `SemiFreeSolverConfig(gradient_mode=True)` is not available in this
  release and raises `NotImplementedError`.
- The Monte Carlo and bootstrap paths of `deppro.batch_processor`
  (`process_batch`, `monte_carlo_batch`, `bootstrap_batch` and the
  `parallel_*` functions, and the `batch` / `monte-carlo` commands of
  `cli_depth`) were written for a solver that has been removed and raise
  `NotImplementedError` until they are ported.
- The MLX solvers are tested against the CPU reference on defined
  synthetic cases. One acceptance test of the semi-free MLX solver
  (`test_mlx_semifree_batch.py::test_one_sided_objective_and_thickness[LLS]`,
  with the default toyomacro tables) does not pass: on that case the MLX
  objective value differs from the reference by more than the test allows. Numerical equivalence of the
  MLX and CPU solvers is not claimed for all settings.
- Two acceptance tests of the Anscombe transform
  (`test_wls_poisson.py::test_anscombe_improves_poisson` and
  `::test_anscombe_similar_for_gaussian`) pass on the Linux CI but fail
  on the author's macOS machine; no improvement is claimed across
  platforms.
- A plain `pytest` therefore reports these acceptance tests as failures
  on macOS. They are kept failing on purpose rather than relaxed: the
  regression suite (what CI runs on every pull request) deselects them,
  and separate acceptance jobs show their state (the MLX one is run by
  hand on macOS). Skipped tests (for example when an optional dependency or
  a user-supplied table set through `DEPPRO_*` is absent) are not counted
  as verified.

## Citation

If you use this software, please cite it via [CITATION.cff](CITATION.cff).
Archived on Zenodo:

- **All versions** (concept DOI; resolves to the latest):
  [10.5281/zenodo.23877023](https://doi.org/10.5281/zenodo.23877023)
- **v0.1.0**:
  [10.5281/zenodo.23877024](https://doi.org/10.5281/zenodo.23877024)

When a result depends on this software, cite the version DOI of the
release you used, not the all-versions DOI: the concept DOI resolves to
whatever release is latest, so it does not say which code produced a
number.

## Use of generative AI

Anthropic Claude, through Claude Code, was used for code, tests,
documentation and repository maintenance; OpenAI ChatGPT and Codex were
used mainly to review changes and designs. The author made the design
decisions, checked the physics and the numbers, decided which changes to
accept, and is responsible for the published content. Individual files
do not record AI involvement.

## License

MIT License ([`LICENSE`](LICENSE)), the same terms as toyomacro and dnndenoiser.
