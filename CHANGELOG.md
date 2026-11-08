# Changelog

All notable changes to `deppro` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Releases are
tagged `vX.Y.Z` on `main`; each tagged release is archived on Zenodo for a
citable DOI. Until 1.0 the API may change between minor versions.

## [Unreleased]

### Added

- `CONTRIBUTING.md` (reporting problems, proposing changes, development
  setup, tests and acceptance gates, style, use of AI tools, governance) and
  `SUPPORT.md` (where to ask, what is supported, expected response),
  linked from the README.

### Fixed

- `python -m deppro.streaming_pipeline` failed on its first pixel: Phase 1
  unpacked eight values from `solver_workers.worker_phase1`, which returns
  nine (chi2 and objective are separate). The worker's docstring said eight.
  An end-to-end Phase 1 test on a 2 × 2 synthetic map was added
  (`tests/integration/test_streaming_pipeline_phase1.py`).
- `deppro.simulate.H5Writer` wrote the binding energies only as
  `ezdeprof/BindingEnergy`; `streaming_io.StreamingDepthReader` (and the
  MATLAB dialect) read `ezdeprof/bindingenergy`. The writer now writes both,
  so a simulated map can be fed to the streaming pipeline.
- `GenericLayerSolver._build_profile` reset the linked-element column after
  the substrate had been added, so a substrate that contains the linked
  element (an oxide substrate with `linked_element_index` = O) lost its share
  of it and came out as the cation alone.
- `VectorizedBatchSolver._gradient_batch` returned the gradient of the
  unnormalised least squares while the residual and the reported chi2 use
  per-angle-normalised intensities; it now takes the normalisation Jacobian
  into account, as `FreeSolver._gradient` does (`check_grad` relative error
  ~80 % before, < 1e-4 after; `tests/solver/test_vectorized_batch_gradient.py`).

## [0.1.0] - 2026-11-02

First public release. Development before this release took place in a
private repository; the public history starts with a single commit that
imports the files selected for publication, and the earlier history is not
part of this repository.

### Added

- **Forward model.** `TransferMatrix` maps a discretized depth profile of
  composition to angle-resolved intensities, from inelastic mean free paths
  (TPP-2M) and the emission geometry. Compound, element binding-energy and
  cross-section tables come from user-supplied files named by `DEPPRO_*`
  environment variables, or otherwise from the `toyomacro` package
  (extra `fit`). The NIST compound binding-energy table has no default and
  must be supplied through `DEPPRO_COMPOUND_BE_TABLE_PATH`.
- **Classical inverse solvers** (NumPy/SciPy, the reference
  implementations): `FreeSolver` (free composition at every depth),
  `SemiFreeSolver` (adds the film thickness) and `GenericLayerSolver`
  (layer thicknesses, optionally interface widths and stoichiometry).
  `FreeSolver` and `SemiFreeSolver` accept unregularised least squares or
  an L1, L2 or maximum-entropy regulariser.
- **Batched solvers on Apple-silicon GPUs** (`deppro.solvers_mlx`, extra
  `mlx`) for the same three families, and a neural-network inverse solver
  on MLX (`deppro.inference`).
- **Synthetic forward simulator** for round-trip tests from a profile to
  intensities and back.
- **HAXPES preprocessing** (`deppro.haxpes`): frame stacks, angle binning,
  cross-section correction, and a bridge to the `dnndenoiser` package for
  frame denoising (extra `nn`).
- **Stored-result check of the JVST and SIA data deposit**
  (`python -m deppro.haxpes.deposit_replay`) and a document listing what
  can and cannot be recomputed from public material
  (`docs/REPRODUCTION_JVST_SIA.md`).

### Known limitations

- `SemiFreeSolverConfig(gradient_mode=True)` is not available and raises
  `NotImplementedError`.
- One acceptance test of the semi-free MLX solver does not pass; numerical
  equivalence of the MLX and CPU solvers is not claimed for all settings.
- Two acceptance tests of the Anscombe transform pass on the Linux CI but
  fail on the author's macOS machine.
- Most of the documentation is in Japanese; English versions will follow.

See the README for details.
