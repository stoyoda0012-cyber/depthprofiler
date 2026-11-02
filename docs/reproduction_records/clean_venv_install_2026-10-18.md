# Clean-environment install check of the public candidate — 2026-10-18

What was checked: that the public candidate, built into a wheel and an sdist,
installs into an empty virtual environment with only its declared
dependencies, that the public modules import, that the shipped examples and
the deposit replay run, and that the optional extras resolve from their pinned
Git tags. This is the implementer's own check in an isolated environment on
the development machine; it is **not** an independent third-party verification
(brief §4.5 item 6), and it does not touch raw data, weights or GPU code.

## Inputs

| Item | Value |
|---|---|
| Source tree | the export of the public allowlist from private `main` `cc549ae`, plus the 2026-10-18 working-tree files added to the allowlist that day, with the 2026-10-18 edits to `pyproject.toml`, `README.md`, `src/deppro/imfp.py`, two unit tests and `docs/REPRODUCTION_JVST_SIA.md` copied in; the candidate's provenance note (not part of the public set) and `__pycache__` removed |
| Build | `python -m build` (setuptools), host Python 3.12.11 (conda-forge), macOS 27.0.1 arm64 |
| Artifacts | `deppro-0.1.0-py3-none-any.whl` (114 entries, all `.py` + dist-info), `deppro-0.1.0.tar.gz` (134 entries) |
| Environments | two fresh `python -m venv` (3.12.11), `pip` upgraded, no `DEPPRO_*` variable, working directory outside any checkout |
| Resolved core deps | numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, h5py 3.16.0, PyYAML 6.0.3, numba 0.68.0, pandas 3.0.6, openpyxl 3.1.5 |
| Extras | `fit` → toyomacro 0.3.1 from `git+…toyomacro@v0.3.1` (`c7ecdbb4…`), pulling lmfit 1.3.4; `nn` → torch 2.14.1 (PyPI, CPU) + dnndenoiser 0.1.2 from `git+…dnndenoiser@v0.1.2` (`c375d3d8…`) |

## Distribution contents

- Wheel: 110 `.py` files under `deppro/`; **none** from `tests`, `benchmarks`,
  `diagnostics`, `experiments`, `_debug`, `_extensions`; no `.npz`, `.h5`,
  `.pt`, `.local` files; `haxpes/denoiser.py` and `haxpes/deposit_replay.py`
  present. `deppro/examples/` (7 files, including `examples/theory/`, 5 files)
  and `deppro/tools/` (5) are shipped.
- sdist: `README.md` and `pyproject.toml` included (README now declared as
  `readme`); no `docs/`, `paper/`, `conftest.py`, tests or `.local` files.
  Installing the sdist yields the same 110-file set as the wheel.

## Core-only environment

- `import deppro` succeeds; neither torch, toyomacro nor mlx is imported.
- Import sweep over all 110 modules: **89 import**. The 21 that do not:
  - 16 need `mlx` (`deppro.inference.*`, `deppro.solvers_mlx.*`,
    `deppro.tools.mlx_reconstruct_drlt`) — optional extra, as documented;
  - 2 import `toyomacro` at module level (`deppro.depth_generator`,
    `deppro.app.fitpara_reader`) — `ModuleNotFoundError` on import rather than
    a named error at use time;
  - 2 `deppro.examples.theory.freesolver_{2,3}layer` run their analysis **on
    import** and stop with the compound-table `FileNotFoundError`;
  - 1 `deppro.simulate.__main__` (argparse `SystemExit`, expected for a
    `__main__`).
- **Side effect found:** importing `deppro.examples.theory.*` executes the
  scripts (no `__main__` guard) and, with `DEPPRO_OUTPUTS` unset, wrote three
  PNGs to `<venv>/lib/python3.12/outputs/`, because `deppro.paths.outputs_dir()`
  falls back to two directories above `paths.py`, which inside an installed
  package is not a repository root. Two consequences to decide on: whether
  `examples/theory/` ships at all (the 2026-10-05 ruling excludes it from the
  first version; the allowlist still lists it as P), and what `outputs_dir()`
  should do when `deppro` is installed rather than checked out.
- Shipped examples: `freesolver_recommended.py` and
  `auto_substrate_detection.py` stop with
  `FileNotFoundError: No compound parameter source available. Either install
  toyomacro (pip install 'deppro[fit]', or …@v0.3.1 …), or point
  DEPPRO_COMPOUND_TABLE_PATH at a CompoundTable.csv` — the intended behaviour.
- Deposit replay (`python -m deppro.haxpes.deposit_replay`, the local copy of
  the Zenodo v1.0.0 archives): SIA 21/21 of the values pinned at that time
  matched (two more printed values were pinned later the same day and give
  23/23 on the development machine; the venv run was not repeated); JVST
  fix23 47/47 rows consistent, fix25 `w1` 0/45 (same finding as the
  development-machine run). All SIA cell values agree with the
  development-machine run (numpy 2.3.3 / scipy 1.16.3) to a relative
  difference of 5.6e-9.

## With `[fit]`

- `pip install "deppro[fit] @ file://…whl"` resolves the Git direct reference
  and installs toyomacro 0.3.1.
- Both shipped examples complete; `freesolver_recommended.py` prints
  `RMSE: 0.1291`, the value recorded on 2026-10-03 for the toyomacro table.
- `deppro.depth_generator`, `deppro.app.fitpara_reader`,
  `deppro.haxpes.cross_section_loader`, `deppro.haxpes.denoiser` import.
- toyomacro v0.3.1 versus the development checkout (`v0.5.1-5-gf2da135`):
  the 109 compounds have identical `Nv`, `density`, `Mw`, `Eg`; the
  cross-section photon-energy grid, element count (103) and the C 1s / O 1s
  rows are identical. The pin to v0.3.1 therefore does not change the tables
  the HAXPES route reads.
- Test suite of the candidate run **inside this environment** (core + fit,
  no mlx, no torch, no pyvista, no git in the tree; argument-less `pytest`,
  `-m 'not slow'` from `pyproject.toml`): 5 failed / 578 passed / 22 skipped /
  5 deselected. Failures: the 3 known acceptance-gate reds (anscombe ×2,
  narrow_angle); `test_haxpes_denoiser_present::test_without_dnndenoiser_…`
  (the torch `ImportError` fired first — test fixed the same day, see below);
  `test_haxpes_sample_tilt::test_no_tilt_is_written_as_a_number_anywhere_in_src`
  (`git ls-files` on a non-git tree — test given a non-git fallback the same
  day). Skips: 10 for `mlx`, 2 for `pyvista`, 3 torch-dependent bridge tests,
  4 `DEPPRO_COMPOUND_TABLE_PATH` unset (CSV conformance), 2 cross-section
  source files not present, 1 optional-extensions test needing mlx. **Skips
  are unverified, not passed.** The 3 known MLX reds cannot appear here.
- After the two test fixes, in the rebuilt tree: `test_haxpes_sample_tilt.py`,
  `test_haxpes_denoiser_present.py`, `test_deposit_replay.py` → 23 passed,
  3 skipped (the torch-dependent bridge tests, core-only environment).

## With `[nn]`

- `pip install "deppro[nn] @ file://…whl"` installs torch 2.14.1 and
  dnndenoiser 0.1.2 from the pinned tag.
- `HaxpesDenoiser` constructs from the installed packages; the ResNet-FCNN
  at the papers' shape has 658,177 parameters; a random-weight checkpoint plus
  `norm_*.json` round-trips through `denoise_spectrum` with the expected
  normalise → network → de-normalise → resample chain (max |diff| 0.0).
- `test_haxpes_denoiser_present.py` + `test_haxpes_optional_denoiser.py` in
  this environment: 9 passed, 0 skipped.
- Note: dnndenoiser v0.1.2 carries the network class the bridge needs, but
  its CLI `infer` on a measured frame stack was added later (2026-09-28, not in
  any release tag at the time of writing). A fixed-weight re-run that uses the
  dnndenoiser CLI for inference will need a later dnndenoiser release and a
  corresponding pin.

## Changes made to the private tree as a result

Split over the pull requests of 2026-10-18: the first four items below are in
the pull request that carries this record (pinned extras, README, imfp
messages, tilt-test fallback); the denoiser test change is in the denoiser
bridge pull request.

- `pyproject.toml`: `readme = "README.md"`; extras `nn` and `fit` pin
  `dnndenoiser@v0.1.2` and `toyomacro@v0.3.1` as Git direct references, with
  the tag commits in a comment.
- `README.md`: install section lists the extras separately and says what the
  core alone can do.
- `src/deppro/imfp.py`: the two "no table source" messages now name
  `pip install 'deppro[fit]'` as well as the explicit Git install.
- `src/deppro/tests/unit/test_haxpes_denoiser_present.py`: the
  "without dnndenoiser" probe stubs torch; a "without torch" probe added.
- `src/deppro/tests/unit/test_haxpes_sample_tilt.py`: falls back to scanning
  every `.py` under `src/` (nothing excluded) when `git ls-files` is
  unavailable.

## Not covered here

- Python 3.11 (the declared minimum): only 3.12 was available.
- `mlx` extra and the MLX test set; `pyvista`/napari visualisation extras
  (not declared in `pyproject.toml`).
- CI (`.github/workflows/ci.yml` still installs toyomacro from `main`).
- Independent verification by someone other than the implementer.
