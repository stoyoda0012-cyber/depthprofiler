# H5 data format: one dialect per producer

## 0. How to read this document

The H5 files in this repository have **no single schema**. Five producers write them,
and they differ in group layout, in where the grid is recorded, and in whether
`dxdydt` exists and what it means.

Before touching a file, **establish which producer wrote it**. Blanket assumptions
such as "everything is under `/ezdeprof`" or "`dxdydt` is not the display grid" hold
for some dialects and not for others.

The text separates two kinds of statement:

- **Common guarantees** (§3): what you may rely on in every dialect
- **Implementation conventions** (§4): what you may rely on only for that producer's output

Concrete numbers (the `stack` or time dimension of a particular file) are not pinned
here. §6 shows how to check your own file.

## 1. Dialects by producer

| ID | Producer | Top level | Main contents of `/ezdeprof` | `stack` | `dxdydt` | Where the grid is recorded |
|---|---|---|---|---|---|---|
| **D1** benchmark simple | `benchmarks/generate_checkered.py::save_to_h5` | `Dstk`, `Xstk`, `Drlt` | `depth` (1,-1), `thetaobs`, `stepsize`, `gammaeff`, `photonenergy`, `workfunction`, `charge`, `free`/`subst`/`linked`/`linkedconst`, `TransMatObs`, `totaldepth`, `m`, `noise` | written, `[n_pixels]` | **not written** | `ez.attrs["nx"]`, `ez.attrs["ny"]` |
| **D2** MATLAB-compatible full | `benchmarks/generate_checkered.py::save_to_matlab_h5` | `Dstk`, `Drlt`, `Dtrn`, `Xstk`, `Xrlt`, and more | D1 plus the control and quality metadata of the MATLAB GUI (`loop`, `currentloop`, `currentstack`, `Qini`, `Qopt`, the `*value` flags, `prpties`, ...) | written, `[n_pixels]` uint64 | **written** `[[nx],[ny],[1]]`, shape (3,1) uint16 | both `dxdydt[0:2]` (= `config.nx`, `config.ny`) and `ez.attrs["nx"/"ny"]` |
| **D3** simulate writer | `simulate/writer.py` | `Dstk`, `Xstk` (gzip) | `depth` (-1,1), `thetaobs`, `stepsize` (1,1), `gammaeff`, `photonenergy`, `workfunction`, `elementsymbol`/`orbital`/`compound`, **`element`** (charge-labeled names, `Si(+4)`), `charge`, `linkedconst`, `free`/`subst`/`linked`, `TransMatObs`, `loop`, `currentloop`, `BindingEnergy` | written, `[[n_pixels]]` f64 | **written** `[[nx, ny, nt]]`, shape (1,3) f64 (same layout as D4). Without `finalize(nx=, ny=)` it is `[[n_pixels, 1, 1]]` | `dxdydt[0:2]` (the `nx`/`ny` given to `finalize`; a `ValueError` before writing if `n_pixels` is not a multiple of `nx*ny`) |
| **D4** app result | `app/depth_result_writer.py` | the **`/D_est`** group (`data`, `solver`, `timestamp`) and **`/Xstk`**. **No `Dstk`** | `depth` (-1,1), `stepsize` (1,1), `element`, `dxdydt`, `thetaobs` (optional), `TransMatObs` (optional) | **not written** | **written** `[[n_x, n_y, n_t]]`, shape (1,3) f64, with `n_t = n_px // (n_x*n_y)` | `dxdydt` is the display grid itself |
| **D5** legacy io writer | `io/h5_writer.py` | many MATLAB-conforming datasets | `SimulationOrExperiment`, `depth`, `totaldepth`, `stepsize`, `m`, `slice`, `thetaobs`, `thetaopt`, `angle`, `elementsymbol`, `orbital`, `compound`, `bindingenergy`, `elementnumber`, `orbitalnumber`, `charge`, `stringcharge`, `element`, `subst`/`free`/`linked`, and more | written (`1` or `n_stack`) | **written** (the fixed value `[1, 1, 100]`, **legacy metadata** matching the reference file) | not derivable from `dxdydt` |
| *(post-processing)* metric writer | `benchmark_h5_writer.py` | reads and writes an existing file | **appends** `Dopt` and similar | — | — | — |

## 2. Consumer-side heuristics (the loader in `visualize_4d.py`)

| Item | Implementation |
|---|---|
| Pixel count `n_px` | taken from **`shape[0]` of the top-level `Dstk`** (`/ezdeprof/stack` is not read) |
| `dxdydt` | `f["ezdeprof/dxdydt"]` is read **without an existence check** |
| `ez.attrs["nx"/"ny"]` | **not read** |
| Grid resolution order | ① explicit `nx`/`ny` arguments → ② `dxdydt[0]`, `dxdydt[1]` → ③ a regular expression on the parent folder name `{Nx}x{Ny}x{Nt}` → ④ `ValueError` if `n_px` is still not divisible |
| Time dimension | `nt = n_px // (nx * ny)` (`dxdydt[2]` is not used) |
| Reshape | MATLAB column-major (a Fortran-order pixel map) |
| The only dataset with an existence check | `ezdeprof/Col` |

### Compatibility with the current loader

| Dialect | Compatibility |
|---|---|
| **D5** | **closest to what the current loader expects** (has both `Dstk` and `dxdydt`) |
| **D2** | has both `Dstk` and `dxdydt`, and `dxdydt[0:2]` matches the actual grid |
| **D3** | has `Dstk`, `element` and `dxdydt`, and `dxdydt[0:2]` matches the grid given to `finalize` (by default one row, `n_pixels × 1`) |
| **D1** | `KeyError`, because `dxdydt` is absent. The grid information it does carry (`ez.attrs`) is not used by the loader |
| **D4** | **not readable by the current loader**, because there is no top-level `Dstk` (it does carry `dxdydt`) |

These incompatibilities are known. This document describes the current state; it does
not call for a change to the loader.

## 3. Common guarantees

Only the following hold in every dialect.

1. The `/ezdeprof` group exists.
   **The main data is not necessarily there** (D4 keeps its main data under `/D_est`).
2. Arrays are stored in MATLAB layout
   (`(pixels, elements, depth)` / `(pixels, elements, angles)`).
3. The depth axis `depth` and `stepsize` are under `/ezdeprof`.
   **Their shapes depend on the dialect** (`(1,-1)` / `(-1,1)` / `(1,1)`) and are not aligned.
4. **Where the index sets (`free` / `subst` / `linked`) exist, they are 1-based, as in
   MATLAB.** Not every dialect has them (D4 does not).

## 4. Implementation conventions

What you may rely on only for that producer's output. **Do not carry it over to another
dialect.**

### 4.1 `stack`

`stack` is "the total number of pixels at the time of writing".

- **`stack = nx * ny * dt * n_phases` is the generation rule of `SimulationConfig.n_pixels`
  in `benchmarks/generate_checkered.py`** (`n_phases` defaults to 1; 2 means forward plus
  backward). It applies only to D1 / D2 output; there is no guarantee that the `stack` of
  another dialect factors this way.
- D4 does not write `stack`.
- Nothing in the runtime package reads it (the loader uses `Dstk.shape[0]`). New code
  that depends on `stack` must name the dialect it targets.

### 4.2 `dxdydt`

**Its meaning differs by producer.**

- **D4**: the display grid itself (`[n_x, n_y, n_t]`).
- **D3**: `[[nx, ny, nt]]`, the grid given to `finalize(nx=, ny=)` and the time
  dimension `n_pixels // (nx*ny)` (same layout as D4). Without a grid it is
  `[[n_pixels, 1, 1]]`.
- **D2**: `[nx, ny, 1]`. The first two entries are the generation grid. The third is
  fixed at 1 and is not the time dimension.
- **D5**: the fixed value `[1, 1, 100]`. This is **legacy metadata** written for
  compatibility with the reference file; neither the grid nor the time structure can be
  read from it.
- **D1**: absent.

The `dx`, `dy` in the spatial-modulation formula of `generate_checkered.py` (`eff_dx` /
`eff_dy` fall back to `nx` / `ny` when not given) are **modulation-rate parameters on
the generation side**, not the general meaning of `dxdydt` in H5.

### 4.3 `Dopt`

Only the post-processing `benchmark_h5_writer.py` creates `/ezdeprof/Dopt`. Do not
expect it in a freshly generated file.

## 5. Benchmark folder naming

```
Simulation/DepthProfiler/{Standard,Checkered}/{Nx}x{Ny}x{Nt}_{N}elem[_variant]/
```

- `Standard/` = 6 elements (OCAlTiSi), `Checkered/` = 7 elements (OBCAlTiSi, with B added)
- **This is not a naming habit; the code parses it.**
  `visualize_4d._infer_grid_from_path()` reads `{Nx}x{Ny}` from the parent folder name
  with a regular expression and uses it to infer the grid when `dxdydt` is unusable.
- Because some folders have an intermediate level and `Nt=1` is an exception, **the
  authoritative path constants are in `deppro.benchmarks.paths`** (`GT_6ELEM` and so
  on). Do not hard-code individual paths in scripts.
- `paths.BENCHMARK_ROOT` is currently relative to the home directory and is not
  resolved through a `DEPPRO_*` environment variable.

## 6. Determining the dialect of your own file

```python
import h5py

with h5py.File(path, "r") as f:
    print("top-level:", list(f.keys()))
    print("ezdeprof :", list(f["ezdeprof"].keys()) if "ezdeprof" in f else "(none)")
    print("attrs    :", dict(f["ezdeprof"].attrs) if "ezdeprof" in f else {})
    for key in ("ezdeprof/stack", "ezdeprof/dxdydt"):
        print(key, "=", f[key][()] if key in f else "(absent)")
    for key in ("Dstk", "D_est/data"):
        print(key, "shape =", f[key].shape if key in f else "(absent)")
```

Rules of thumb:

- `D_est` present → **D4**
- `Dtrn` / `Xrlt` / `currentstack` present → **D2**
- `Dstk` present and `dxdydt` absent → **D1** (carries `ez.attrs["nx"]`)
- `Dstk`, `elementsymbol` and `dxdydt` present → **D3** (shape (1,3); older D3 files have
  no `dxdydt`)
- `dxdydt == [1, 1, 100]` → most likely **D5** (a fixed value, unusable as the grid)
