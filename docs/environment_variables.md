# Environment variables (external paths)

The locations of external data and external tools are passed through `DEPPRO_*`
environment variables. `import deppro` succeeds with **every variable unset**. What
happens when a variable is unset differs per variable; see the "When unset" column
of the tables below.

**An empty string (`export DEPPRO_X=`) is treated as unset.** Every place that uses a
value as a path follows this rule. A value made only of whitespace gets no special
treatment. The exception is a few diagnostic scripts whose purpose is to record the
raw value in their results (they write the raw value of `DEPPRO_COMPOUND_TABLE_PATH`,
so an empty string is recorded as `""`).

New readers use `deppro.paths.env_value()` (when the value itself is needed) or
`deppro.paths.env_path_or_placeholder()` (when the `-not-set` form is needed). They do
not read `os.environ.get(name, default)`, `os.getenv(name, default)` or
`os.environ[name]` directly: these return an empty string as is, bypass the default,
and end up as `Path("")`, which is the current directory.

`tests/unit/test_env_paths.py` walks `src/deppro/` with the AST and fails on the
following forms (readers whose purpose is to record the raw value are excepted through
the allowlist `_ALLOWED_RAW_READERS`, which carries the number of occurrences per
file).

- Reads with a default: `os.environ.get(name, d)` (also `default=`),
  `os.getenv(name, d)` (also the keyword forms `key=` / `default=`)
- `os.environ.setdefault(name, d)` (whether or not the result is used; with an empty
  string present the default is not stored, so it cannot inject a default either)
- `os.environ.pop(name, ...)` whose value is used (a call used as a bare statement,
  discarding the result, counts as cleanup and is not flagged)
- Subscript reads `os.environ[name]`
- Membership tests `name in os.environ` / `name not in os.environ`
- None checks on a value read without a default: comparing `os.environ.get(name) is None`
  directly, and storing the value in a variable **within the same function (or module)**
  and then comparing it with `is None` / `is not None` (via `=`, an annotated `=`, or `:=`)

`os.environ` includes `environ` (`from os import environ`), aliases in the same file
(`env = os.environ`, `from os import environ as E`, and aliases of aliases),
`os.environ.copy()` and `dict(os.environ)`. `os.getenv` includes `getenv` and its
aliases in the same file. `name` is a `"DEPPRO_…"` string literal, the leading part of
an f-string, or a name assigned in the same file as `NAME = "DEPPRO_…"` (annotated or
not).

**What is not caught (not an exhaustive list):**

- Variable names built at run time (string concatenation, `%` or `.format`,
  `getattr(os, "getenv")`, a constant `NAME = f"DEPPRO_{n}"`)
- Constants or aliases defined in another module, and aliases stored on attributes
  (`self.env = os.environ`)
- Reads through `{**os.environ}` or `dict(os.environ, A=1)`
- Passing the value to another function (closures included) and checking for None there
- Checks with `== None`, None checks on values received through tuple unpacking, and
  chained comparisons (`a is None is b`)
- Reads in default arguments of functions or lambdas followed by a None check, None
  checks on values bound by `for` / `with`, and values read at module level but
  checked inside a function
- Assigning to an attribute and then checking it (`self.v = os.environ.get(...)` followed
  by `self.v is None`)

For these, keeping the rule is the writer's responsibility.

**On false positives:** aliases and constants are followed by name only, ignoring scope
and assignment order. Variables used in None checks are followed within the same
function (or module), again ignoring assignment order. Reusing a name for something
else can therefore be flagged even when it is not a read (the detector errs toward
red, not toward missing a case).

Class bodies and comprehensions are also treated as the same scope as the enclosing
function (or module). Cases that are separate variables in Python can therefore be
flagged, for example a value read in a class body and checked for None at module level
or in another class body, or a comprehension variable that shadows a variable read
outside (`v = os.getenv(...)` followed by `[v is None for v in ...]`).

Forms with an explicit `None` default (`os.environ.get(name, None)`,
`os.getenv(name, None)`, `default=None`) are also flagged as reads with a default.

On a false positive, rename the variable or go through `deppro.paths`.

Matching against the allowlist counts the detected position (the read; for a None
check, the position of the comparison) by line and column. Two reads on one line, or
one value checked for None twice, are separate findings.

The "`-not-set` form" means that the path becomes the string `<VARIABLE>-not-set/...`,
so the variable name appears verbatim in load errors and logs
(`env_path_or_placeholder()`).

`~` in a value works only where the reader applies `expanduser` (see "Expansion of `~`"
at the end).

## Where data and tools live

The tables in this document list only the readers in public modules. Variables read
only by modules outside the public release, and the private readers of the variables
below, are listed in `environment_variables_private.md` (private repository only).

| Variable | Points to | Read by | When unset |
|---|---|---|---|
| `DEPPRO_FIX_IBW_DIR` | Directory of fix-mode IBW frames | `deppro.haxpes.config.FIX_IBW_DIR` (for example the `HAXPESPipeline.run_fix_ibw` example) | `-not-set` form |
| `DEPPRO_FIX_PXT_DIR` | Directory of fix-mode pre-binned PXT files | `deppro.haxpes.config.FIX_PXT_DIR` | `-not-set` form |
| `DEPPRO_MAP_H5` | HDF5 file of the fitted map | `deppro.haxpes.config.MAP_H5` | `-not-set` form |
| `DEPPRO_SIMULATION_ROOT` | Root of the MATLAB-generated benchmark H5 files (the parent of `Standard/` and `Checkered/`) | `deppro.paths.simulation_root()` and its callers, `deppro.paths.legacy_simulation_root()` (`BENCHMARK_ROOT` / `GT_*` in `benchmarks/paths.py`) | `simulation_root()` gives the `-not-set` form. `legacy_simulation_root()` gives the earlier home-relative location (`~/Documents/MATLAB/Simulation/DepthProfiler`). Files under `Simulation/Checkered` (`GT_500x500`) are resolved against the directory one level above this root |
| `DEPPRO_TESTDATA_ROOT` | Test-data directory of the MATLAB DepthProfiler (one `map_*.h5` and one MATLAB result `YYMMDD*.h5` directly inside) | `tools/roundtrip_compare.py` | `-not-set` form |
| `DEPPRO_DEPTHPROFILER_CLI` | The MATLAB generation CLI (`_cli/dp`) | `deppro.app.matlab_generation` | `-not-set` form |
| `DEPPRO_DATASET_REGISTRY` | Registry JSON of the benchmark datasets (`schema: deppro-dataset-registry/1`) | `deppro.app.contract.registry.default_registry()` (when `entry` / `benchmark_targets` / `excluded` / `explain` are called without a registry) | An empty registry (every dataset is "not in the registry"). If the configured path is not a file, a `FileNotFoundError` naming the variable is raised when the registry is used. Callers that pass a registry explicitly (`load_registry(path)`) do not read this variable |

## Reference tables

| Variable | Points to | Read by | When unset |
|---|---|---|---|
| `DEPPRO_COMPOUND_TABLE_PATH` | Compound table CSV | `deppro.imfp.CompoundDatabase`, `deppro.producer.materials` | `imfp`: the table bundled with toyomacro (without toyomacro, a `FileNotFoundError` naming the variable and how to install). `producer.materials`: `-not-set` form |
| `DEPPRO_BINDING_ENERGY_TABLE_PATH` | Elemental binding-energy CSV | `deppro.imfp` | The table bundled with toyomacro (without toyomacro, a `FileNotFoundError` naming the variable and how to install) |
| `DEPPRO_COMPOUND_BE_TABLE_PATH` | Compound binding-energy CSV (NIST) | `deppro.imfp`, `deppro.producer.binding` | `imfp`: there is no default table. Constructing `CompoundBindingEnergyDatabase` raises a `FileNotFoundError` naming the variable (nothing is bundled and sibling checkouts are not searched). `producer.binding`: treated as not given |
| `DEPPRO_ELEMENT_BE_TABLE_PATH` | Elemental binding-energy CSV | `deppro.producer.binding` | Treated as not given |
| `DEPPRO_YEH_LINDAU_TABLE_PATH` | Yeh–Lindau photoionization cross-section CSV (`CrossSectionTable_Yeh=Lindau.csv`) | `deppro.haxpes.cross_section_extrap` (the Z > 12 extrapolation of `apply_cross_section_correction(..., source='yeh_lindau_extrap')`) | If set, that CSV is read. If unset, **the Yeh–Lindau table bundled with toyomacro** (generated from the same CSV). Without toyomacro as well, a `FileNotFoundError` naming the variable and toyomacro (sibling checkouts are not searched). The extrapolated values of the pinned production cases agree between the two sources (same test file) |
| `DEPPRO_CROSS_SECTION_DIR` | Directory holding the Scofield (1973) and Yeh–Lindau (1985) photoionization cross-section workbooks (`excel_scofield_1973_pics.xlsx`, `excel_yeh_lindau_1985_pics-4.xlsx`) | `deppro.haxpes.cross_section_loader` (used by `apply_cross_section_correction`: for every element with `source='scofield'` / `'yeh_lindau'`, and for elements with Z ≤ 12 under the default `'yeh_lindau_extrap'`. Also used by `deppro.cli_depth`) | If set, those Excel workbooks are read. If unset, **the Scofield / Yeh–Lindau tables bundled with toyomacro** (toyomacro owns the completeness and rights review of the tables: `docs/DATA_SOURCES.md`; interpolation stays on the deppro side). Without toyomacro as well, a `FileNotFoundError` naming the variable and toyomacro (sibling checkouts are not searched). **Scope:** on the pinned production cases (`tests/unit/test_cross_section_sources.py`) the two sources agree. They differ for Scofield above 30 keV (the toyomacro table ends there), for some non-production Yeh–Lindau cells (where the workbook and toyomacro transcribe differently, the toyomacro table, checked against the primary source, is authoritative), and in the reported `energy_range`. Queries below threshold give NaN on both paths |

## Output location

| Variable | Points to | Read by | When unset |
|---|---|---|---|
| `DEPPRO_OUTPUTS` | Where figures and generated files go | `deppro.paths.outputs_dir()` / `ensure_outputs_dir()` | In a checkout, `outputs/` at the repository root (ignored). From an installed package, `outputs/` in the **current directory** (warned once) |

## Expansion of `~`

- **Expanded**: `DEPPRO_FIX_IBW_DIR` / `DEPPRO_FIX_PXT_DIR` / `DEPPRO_MAP_H5` / `DEPPRO_SIMULATION_ROOT` / `DEPPRO_TESTDATA_ROOT` /
  `DEPPRO_DEPTHPROFILER_CLI` / `DEPPRO_DATASET_REGISTRY` / `DEPPRO_OUTPUTS`, the `DEPPRO_COMPOUND_TABLE_PATH` /
  `DEPPRO_COMPOUND_BE_TABLE_PATH` / `DEPPRO_ELEMENT_BE_TABLE_PATH` read by `deppro.producer`, `DEPPRO_YEH_LINDAU_TABLE_PATH`,
  and `DEPPRO_CROSS_SECTION_DIR`
- **Not expanded**: `DEPPRO_COMPOUND_TABLE_PATH` / `DEPPRO_BINDING_ENERGY_TABLE_PATH` /
  `DEPPRO_COMPOUND_BE_TABLE_PATH` as read by `deppro.imfp`. Pass absolute paths to these

The same variable is handled differently depending on the reader
(`DEPPRO_COMPOUND_TABLE_PATH` and `DEPPRO_COMPOUND_BE_TABLE_PATH`), so use an absolute
path when both readers must see it.

New code writes its output under `deppro.paths.outputs_dir()`. When a new external path
is needed, add a row here.
