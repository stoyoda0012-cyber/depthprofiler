# Changelog

All notable changes to `deppro` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Releases are
tagged `vX.Y.Z` on `main`; each tagged release is archived on Zenodo for a
citable DOI. Until 1.0 the API may change between minor versions.

## [Unreleased]

### Fixed

- `deppro.simulate.run_simulation` failed with a bare `ZeroDivisionError`
  from the compiled generation kernel when the configured photon energy was
  below the preset's `min_photon_energy` (the default 1486.6 eV with the
  1s-orbital presets): the kinetic energies turned negative, TPP-2M returned
  NaN IMFPs, and nothing checked them. It now raises a `ValueError` naming
  the photon energy and the preset's minimum before any work, and
  `RandomProfileSampler.generate_batch` refuses a `lambda_basic` with
  non-finite or non-positive entries, naming the emitting elements. The CLI
  (`python -m deppro.simulate`), which already raised the energy
  automatically with a warning, is unchanged.

## [0.2.0] - 2026-11-05

### Added

- **Ground truth beside a reconstruction.** `deppro.tools.gt_vs_recon`
  (`python -m deppro.tools.gt_vs_recon --gt GT.h5 --recon EST.h5`, or
  `render_gt_vs_recon` from Python) draws a 2x2 figure. The top row holds
  the 3D isosurfaces of the ground truth and of the reconstruction, from the
  same view, with the same crop and element colors (the view and crop
  conventions of `deppro.tools.triptych`). The bottom row holds their
  (x, y)-averaged depth profiles. A heading gives the frame's RMSE (%, over
  all pixels, depths and elements) and W1 (Å, the median over pixels,
  averaged over elements). `load_profile` reads either `/D_est/data` or
  `/Dstk` files. `check_compatible` refuses a pair whose grid, depth axis or
  element order differ, and it lists every difference. One frame gives a
  PNG. Several frames give an MP4 and a poster PNG. A relative output path
  goes under `gt_vs_recon/` in the outputs directory (`DEPPRO_OUTPUTS`, else
  `outputs/` in a checkout or in the current directory).
  `--no-depth-peeling` keeps panels from coming out blank under software
  OpenGL. Rendering needs pyvista and imageio (imageio-ffmpeg for video).
  No extra installs them. The metrics and checks run without them.
- **Citation metadata for the archive.** `CITATION.cff` carries the Zenodo
  concept DOI (10.5281/zenodo.23877023) and the v0.1.0 version DOI
  (10.5281/zenodo.23877024). The README has a DOI badge and a citation
  section that asks you to cite the version DOI of the release you used.
- `CONTRIBUTING.md` (reporting problems, proposing changes, development
  setup, tests and acceptance gates, style, use of AI tools, governance) and
  `SUPPORT.md` (where to ask, what is supported, expected response),
  linked from the README.
- **English-only check.** A repository test fails if a shipped module,
  `README.md`, `CITATION.cff`, `CHANGELOG.md`, `pyproject.toml` or a
  Markdown file under `docs/` contains kana or kanji. `docs/ja/` is exempt
  because it holds the Japanese translations. The test does not scan
  `CONTRIBUTING.md`, `SUPPORT.md`, `paper/` or the READMEs inside
  `src/deppro/`, and it does not match full-width CJK punctuation.

### Changed

- **Documentation is in English.** The README is English only. The
  Japanese README is now at `docs/ja/README.md`, and the English one is
  authoritative. `docs/api_reference.md`, `docs/data_format.md` and
  `docs/environment_variables.md` are translated. One file is still in
  Japanese: `src/deppro/examples/theory/README.md`. Comments, docstrings
  and most documents now use American spelling. Identifiers keep their
  spelling (`default_colours`, `_forward_normalised`, `analyser_mode` and
  others). The README and the JOSS paper still have some British spellings.
- **Messages, printed text and CLI help are in English.** This covers
  every shipped module. Function, class and argument names and CLI options
  did not change. The strings that changed are:
  - exception messages (`ValueError`, `RuntimeError`, `NotImplementedError`,
    `GenerationError`, `MatlabGenerationError`, `ManifestRejected`,
    `UnsupportedLayout`);
  - the warnings and suggestions that `StructureAnalyzer` and
    `ARXPSPipeline` return;
  - the names and messages of the physics checks (`composition sum`,
    `non-negativity`, `substrate convergence`);
  - the verdicts and notes from `deppro.producer.gui_check`;
  - the verbose output of `GenericLayerSolver`;
  - the log lines of `deppro.app.pipeline_bridge` (`bridge A:` … `bridge D:`);
  - the test-case descriptions in `deppro.benchmark`;
  - the printed output and plot labels of the examples, and the printed
    output of `deppro.io.H5Writer`;
  - the free-text `note` and `role` fields that the producer writes to
    `manifest.json`;
  - the `--help` text of `deppro.tools.triptych`.

  If your code matches the old Japanese text or a physics-check `name`,
  update it.
- **README and JOSS paper.**
  - Both say who the package is for, what goes in and what comes out.
  - The README has a new section, "What you can do with it". It includes a
    list of what has been done with the package, what the current code can
    do but has not been published, and what is still to be verified.
  - The README's note that there is no ready-made runner now refers to
    measured data only. It also points to
    `python -m deppro.benchmarks.run_solver_landscape` for simulated stacks,
    which needs a source checkout.
  - The paper adds a state-of-the-field comparison with instrument MEM
    software, QUASES-ARXPS and SESSA. It also cites Murdoch & McCulloch
    (2023) and PROPHESY. It now has the SIA volume, issue and pages
    (58(12), 1112–1128).
  - Two corrections, also made in `docs/api_reference.md`:
    - In `FreeSolver`, the default L1 prior equals the lower bound. Under
      the sum-to-one constraint the L1 term is then constant, so it does
      not favor sparse profiles.
    - In `SemiFreeSolver`, only L1, L2 and MEM pull the thickness toward
      its starting value. LLS has no penalty.

### Fixed

- **`deppro.visualize_4d.load_gt_4d` can read files from
  `deppro.simulate.writer.H5Writer`.** Before this fix it failed with
  `KeyError` on every such file. `H5Writer.finalize` now writes
  `ezdeprof/element`, the element names with charges (`'Si(+4)'`). It also
  writes `ezdeprof/dxdydt` as `[[nx, ny, nt]]` (shape (1, 3), float64), the
  layout that `deppro.app.write_depth_estimate` uses. `finalize` takes two
  new optional arguments, `nx` and `ny`. Pass both or neither; passing only
  one raises `ValueError`. With neither, the file records the samples as one
  row, `[[n_pixels, 1, 1]]`. If `n_pixels` is not a multiple of `nx * ny`,
  `finalize` raises `ValueError` before it writes anything. Files written by
  earlier versions still lack both datasets. `docs/data_format.md` (dialect
  D3, §4.2 and the §6 dialect test) now describes the new layout.

### Known limitations

- The limitations of 0.1.0 about `SemiFreeSolverConfig(gradient_mode=True)`,
  the semi-free MLX acceptance test and the Anscombe acceptance tests still
  apply.

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
