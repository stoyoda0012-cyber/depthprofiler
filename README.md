# depthprofiler (`deppro`)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23877023.svg)](https://doi.org/10.5281/zenodo.23877023)

A Japanese version of this README is at [`docs/ja/README.md`](docs/ja/README.md); this English version is authoritative.

`deppro` reconstructs depth profiles of composition, D(z), in thin films
from angle-resolved XPS and HAXPES intensities. It is written for surface
scientists who analyse angle-resolved depth series, from one sample
position or from many pixels or frames, and who need to know how much a
reconstructed profile depends on the choices made on the way to it.

- **Input**: for each pixel, the peak intensity of each element at each
  emission angle (an `n_angles × n_elements` array, background-subtracted
  and corrected for cross sections), the photon energy and geometry, the
  compound of each element for the inelastic mean free paths, and a
  structure description saying which elements belong to the film and
  which to the substrate.
- **Output**: D(z) on a depth grid for each pixel and, depending on the
  structure model, the film thickness (the depth of the substrate
  interface), the interface width and the layer thicknesses.
- **Research use**: invert the same intensities under different
  structure assumptions, regularisers and preprocessing, and follow how
  the composition profile, the film thickness and the interface position
  change, pixel by pixel.

```
AR-XPS / HAXPES spectra
    │  (denoising)           … dnndenoiser (separate repository)
    │  (Voigt peak fitting)  … toyomacro (separate repository)
    ▼  depth-profile inverse problem  … deppro (this repository)
depth profile of composition D(z)
```

## What you can do with it

The three classical solvers build their predictions from the same
transfer matrix of I = T·D (`TransferMatrix`) and take the same intensity
array, so the physics of the forward model is shared when the structure
assumption is changed. They still differ in how the intensities are
normalised, in the optimiser and in the default prior, so a difference
between two reconstructions should be read together with these settings.

**Fit a known layer structure.** `GenericLayerSolver` takes a stack of
layers of given stoichiometry (`GenericLayerConfig`, `LayerDefinition`)
and fits the layer thicknesses, optionally the interface widths and the
stoichiometry within bounds. It is configured in Python;
the YAML configurations in `src/deppro/examples/` drive `SemiFreeSolver`
through `python -m deppro.cli_depth solve --config …`, not this solver.

**Change the structure assumption on the same data.** The three solvers
impose different prior knowledge:

| Solver | What it assumes | What it returns |
|---|---|---|
| `FreeSolver` | no layers; concentrations are non-negative and sum to one at each depth (and charge-neutral, by default, when oxidation states are given) | D(z) |
| `SemiFreeSolver` | one buried substrate interface of error-function shape; the film composition is free at each depth | D(z), interface depth (film thickness), substrate element ratios; the interface width is fixed (5 Å FWHM by default) unless fitting it is switched on |
| `GenericLayerSolver` | a layer stack of given stoichiometry | D(z), layer thicknesses, optionally interface widths and stoichiometry |

`FreeSolver` and `SemiFreeSolver` take the regulariser as a setting
(`SolverConfig.method`: unregularised least squares `LLS`, `L1`, `L2` or
maximum entropy `MEM`). `LLS` has no penalty. For the others, the prior
that the penalty pulls towards depends by default on the solver and the
method: in `FreeSolver` it is uniform for `L2`/`MEM` and near zero for
`L1`, where it equals the lower bound, so under the sum-to-one constraint
the default `L1` term is constant and does not favour sparse profiles; in
`SemiFreeSolver` it is near zero, and the penalty also pulls the thickness
weakly towards its starting value. To change only the penalty, pass the
prior (`D_prior`) explicitly. `GenericLayerSolver` is unregularised. `StructureAnalyzer`
suggests a solver from the intensities.

To see what a choice does where the truth is known, the synthetic round
trip generates a map of profiles, adds Poisson noise, inverts it and
reports RMSE and the Wasserstein distance W1 per element:

```bash
python -m deppro.simulate.d2s2d --preset 7elem_checkered --nx 3 --ny 3 --dt 1 \
    --noise 0.03 --inverse free
python -m deppro.simulate.d2s2d --preset 7elem_checkered --nx 3 --ny 3 --dt 1 \
    --noise 0.03 --inverse semifree --semifree-init base_fv
```

The starting values of `SemiFreeSolver` are part of the assumption. The
round trip does not fit the interface width, so the width that
`--semifree-init` picks is imposed: the default `gt_fv` uses each pixel's
true thickness and width (information a measurement does not have);
`base_fv` uses the scene's nominal values (210 Å, 20 Å); `default` uses
30 Å and 5 Å, the 30 Å start is clipped to the round trip's 50 Å lower
bound, and on this scene the fitted thickness stays at that bound. The
simulator weights the mean free paths by each pixel's true composition,
while the inversion uses a transfer matrix for a uniform composition, so
the printed errors include that difference. They are for this synthetic
scene and noise level, not a benchmark.

**Compare many pixels and preprocessing conditions.** For a map or a
series of frames, every pixel is inverted independently:

- `deppro.haxpes.pipeline.process_single_pixel` takes one pixel's binned
  spectra and does background subtraction, peak integration,
  cross-section correction and an L1 `SemiFreeSolver` fit; its arguments
  set the elements and their roles, the depth grid, the regularisation
  weight, the starting thickness and an optional non-dipole correction.
  `run_3d_map` runs it over all pixels and returns the D(z) map, the
  thickness map and a per-pixel residual map; each pixel can carry its
  own regularisation weight and starting thickness, but the element
  roles (Si/Ti/Al/C/O of the JVST/SIA stack), the depth grid and the
  non-dipole correction stay at their defaults. Other settings need a
  short loop around `process_single_pixel`.
- `deppro.haxpes.config.HAXPESSolverConfig` holds the preprocessing and
  solver choices (angle binning, background method, cross-section table,
  non-dipole correction, regulariser) that `HAXPESPipeline` applies to
  one averaged data set; `run_depth_profiling` reads only its solver
  fields, and `run_3d_map` does not read it.
- `deppro.haxpes.bootstrap` averages frames into pixels
  (`build_frame_pixels`, `average_pixels`) and solves bootstrap resamples
  of the frames (`frame_bootstrap`, `block_bootstrap`), and
  `deppro.haxpes.denoiser.HaxpesDenoiser` applies a trained `dnndenoiser`
  network to frames before the inversion (needs `[nn]` and weights).
- `python -m deppro.streaming_pipeline` solves an H5 map chunk by chunk
  with `SemiFreeSolver`, or with a trained network, and stores D(z) and
  the thickness of each pixel ([`docs/data_format.md`](docs/data_format.md));
  the interface width is held at `--abruptness-init` there, not fitted.
  The MLX classes in `deppro.solvers_mlx` solve batches of pixels on an
  Apple-silicon GPU.
- `deppro.depth_profile.wasserstein_distance_1d` and
  `deppro.app.contract.metrics` (`evaluate`, `thickness_half_crossing`)
  score reconstructions against a reference profile.

For measured data there is no ready-made runner that sweeps
preprocessing or solver settings and tabulates the result; the loop over
conditions is a short script around these functions. For simulated
stacks, a source checkout (not the installed package) has
`python -m deppro.benchmarks.run_solver_landscape`, which runs the three
solvers over noise, thickness, interface width and number of angles with
one regulariser per run. Pixels are not coupled laterally, except through an optional
prior rebuilt from neighbouring pixels' solutions between passes (the
second pass of the streaming pipeline, `--mem-loops`, and
`IterativeMEMSolver.solve_batch_with_neighbor`), so a map is a set of
depth profiles, not a three-dimensional tomographic reconstruction.

### What has been done with it, and what has not

- **Done, in two papers**: in
  [JVST A](https://doi.org/10.1116/6.0005585), thousands of single frames
  recorded at one fixed position on a C/Al₂O₃/TiO₂/Si stack were inverted
  one by one with `SemiFreeSolver` (L1, interface width fixed at 5 Å FWHM)
  to compare frame averaging, bin-pool resampling and self-supervised
  denoising at two exposure times; in
  [SIA](https://doi.org/10.1002/sia.70123), the same per-frame inversion
  was used to test denoising networks trained at one exposure time and
  applied to another, over three exposure times, against a frame-averaging
  ladder. Each reconstruction was scored against the pseudo-ground truth of
  its own data set, the reconstruction of the spectra averaged over all
  its frames, by depth RMSE, thickness deviation and W1. The pixel grids in
  those papers are frame ensembles, not lateral maps. No independent
  structural measurement was used as the truth; JVST A checks only that
  the profiles show the layer sequence of an RBS reference of the same
  sample and a total thickness consistent with stylus profilometry.
- **Possible with the current code, not yet published**: the same
  per-pixel comparison on a laterally resolved map, and comparisons of
  structure models and regularisers on measured data.
- **Still to be verified**: a quantitative comparison of the
  reconstructions with an independent depth measurement (for example RBS
  or SIMS on the same sample), and whether the conclusions of the two
  papers carry over to other samples, instruments and noise regimes.

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

Arguments and return values of `TransferMatrix` and the solvers are described in
[`docs/api_reference.md`](docs/api_reference.md), the H5 data format in
[`docs/data_format.md`](docs/data_format.md). The locations of external data are
passed through `DEPPRO_*` environment variables (listed in
[`docs/environment_variables.md`](docs/environment_variables.md)).

## Reproducing the JVST / SIA papers

[`docs/REPRODUCTION_JVST_SIA.md`](docs/REPRODUCTION_JVST_SIA.md) separates three
things: recomputing metrics from the stored results, re-running the denoiser
and the inversion with the same inputs and weights, and retraining. Only the
first has been done, on the author's development machine; no one has
reproduced it on another machine. `python -m deppro.haxpes.deposit_replay`
recomputes metrics from the depth profiles stored in the Zenodo deposit (pass
the extracted deposit through `--sia-dir` / `--jvst-h5`); it does not re-run
the denoiser or the inversion. For SIA it checks the Table 1 values, the
depth-RMSE and W1 values quoted for Figs. 4 and 5 and the seed-pilot values
against the paper (all matched; other supplementary figures and fits were
not recomputed); for JVST A it checks the deposit's
metric files against the deposited arrays, not against the paper, and the
stored W1 and depth-RMSE values of the 24 s data set do not reproduce. The same
document lists, figure by figure, what can be recomputed from public material
and what needs material requested from the authors.

## Related repositories

| Repository | Role |
|------|------|
| **depthprofiler** (this repository) | depth analysis (classical, NN and MLX inverse solvers) |
| [**dnndenoiser**](https://github.com/stoyoda0012-cyber/dnndenoiser) | deep-learning denoiser for XPS spectra (including self-supervised training) |
| [**toyomacro**](https://github.com/stoyoda0012-cyber/toyomacro) | Voigt peak fitting, and the compound, binding-energy and cross-section tables |

## Status

Version 0.1.0. The API may change before 1.0. Questions and bug reports go
to the [issue tracker](https://github.com/stoyoda0012-cyber/depthprofiler/issues);
see [`SUPPORT.md`](SUPPORT.md) for what to expect and
[`CONTRIBUTING.md`](CONTRIBUTING.md) for how to propose changes and run the
tests.

## Known limitations

- `SemiFreeSolverConfig(gradient_mode=True)` is not available in this
  release and raises `NotImplementedError`.
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
- The forward model uses straight-line attenuation and has no
  elastic-scattering term.
- `BatchProcessor` and `python -m deppro.cli_depth batch` / `monte-carlo`
  belong to a removed solver and are not available in this release; use
  the functions listed above for many pixels.

## Layout

| Location | Contents |
|------|------|
| `src/deppro/` | transfer matrix and the inverse solvers (LLS / MEM / L1 / L2), the NN inverse solver (`inference/`), the MLX solvers (`solvers_mlx/`), HAXPES preprocessing (`haxpes/`), simulation (`simulate/`), examples (`examples/`) |
| `src/deppro/tests/`, `src/deppro/app/tests/` | the regression suite and the acceptance gates (the CI configuration is `.github/workflows/ci.yml`) |
| `docs/` | environment variables, the H5 data format, the solver API, reproduction of the JVST/SIA papers |
| `paper/` | the software paper for JOSS (not yet submitted) |

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
