# deppro - ARXPS Depth Profile Analysis Package

A Python reimplementation of the ARXPS depth-profile analysis method that was developed
and validated in MATLAB. The project moves the method off MATLAB and toward practical
use.

## Quick Start

```python
from deppro import (
    create_multilayer_profile,
    TransferMatrix,
    GenericLayerSolver, GenericLayerConfig, LayerDefinition,
    SolverConfig,
)

# Transfer matrix
tm = TransferMatrix(
    depth=np.arange(0, 200, 1.0),
    step_size=1.0,
    angles=[0, 15, 30, 45, 60],
    photon_energy=1486.6,
)

# GenericLayerSolver (thickness estimation for a known structure)
config = GenericLayerConfig(
    layers=[
        LayerDefinition(name="SiO2", element_indices=[0, 1], stoichiometry={0: 1, 1: 2}),
    ],
    substrate=LayerDefinition(name="Si", element_indices=[2], stoichiometry={2: 1}),
    initial_thicknesses=[30.0],
    initial_widths=[5.0],
    linked_element_index=1,
)
solver = GenericLayerSolver(tm, SolverConfig(method='LLS'), config)
result = solver.solve(I_obs)
print(f"Thickness: {result.layer_thicknesses[0]:.2f} Å")
```

## The solver family

```
┌─────────────────────────────────────────────────────────────────────┐
│  GenericLayerSolver                                                 │
│    └─ any number of layers; optimizes thicknesses (and optionally   │
│       stoichiometry) under stoichiometric constraints               │
│    └─ use: thickness estimation for a known structure               │
│                                                                     │
│  FreeSolver                                                         │
│    └─ free optimization of the concentration at every depth         │
│    └─ MEM + exponential prior + optional constraints                │
│    └─ use: exploratory analysis of an unknown structure             │
│                                                                     │
│  SemiFreeSolver                                                     │
│    └─ substrate: fixed erf; surface: free optimization              │
│    └─ use: the intermediate approach                                │
└─────────────────────────────────────────────────────────────────────┘
```

## Test-inferred capabilities (src/deppro/tests)

> **Reorganization of 2026-07-29**: 46 research scripts that were named `test_*.py` but
> defined no tests were moved to homes by role. **pytest now collects only real tests.**
> Read "tests" and "scripts" in the table below as different things.
>
> | Kind | Location | How to run |
> |---|---|---|
> | Tests | `src/deppro/tests/`, `src/deppro/app/tests/` | `pytest` (slow tests excluded by default) |
> | Benchmarks | `src/deppro/benchmarks/{accuracy,noise,optimization,regularization,robustness}/` | `python <file>` |
> | Diagnostic scripts | `src/deppro/diagnostics/` | `python <file>` |

Capabilities that can be read off the tests (excerpt):

- Automatic interface detection (consensus of gradient, 50 % crossing and inflection point)
    - tests/unit/test_interface_detection.py
- Profile comparison metrics (Wasserstein-1, centroid, moments, layer-wise comparison)
    - tests/unit/test_metrics.py
- Pipeline integration (structure estimation → solver selection → thickness estimation)
    - tests/integration/test_pipeline.py
- Inverse reconstruction with FreeSolver (model-free) and comparison of regularizations
    - tests/solver/test_free_solver.py
- Layer-thickness estimation with SemiFreeSolver (semi-constrained) and comparison of regularizations
    - tests/solver/test_semifree_solver.py
- GenericLayerSolver (thickness estimation for a known structure, stoichiometric constraints)
    - tests/solver/test_generic_layer_solver.py
- Noise and robustness evaluation (Poisson / WLS)
    - tests/benchmark/noise/test_wls_poisson.py
    - tests/benchmark/robustness/test_narrow_angle.py
- Regularization-parameter selection (L-curve / GCV)
    - tests/benchmark/regularization/test_alpha_selection.py
- Performance regression of the MLX batch solvers
    - tests/benchmark/test_mlx_perf_regression.py
- Repository hygiene (a file named `test_` must define tests, enforced mechanically)
    - tests/unit/test_repo_hygiene.py

Scripts (not tests; the old paths were all `.../test_<name>.py`):

- Uniform mixture versus layered structure → `diagnostics/uniform_detection.py`
- Behaviour of the structure estimator → `diagnostics/structure_analyzer.py`
- Thickness accuracy and a 5-layer stack → `diagnostics/thickness_accuracy.py`, `diagnostics/five_layer_free.py`
- Film structure with GenericLayer → `diagnostics/film_structure_solver.py`
- Failure rate over 100 trials → `benchmarks/robustness/failure_rate_100trials.py`
- Comparison of regularizations such as L1 → `benchmarks/regularization/l1_regularization.py`
- Numba JIT and solver comparison → `benchmarks/optimization/numba_jit.py`, `benchmarks/accuracy/solver_comparison.py`

Note: the benchmark and diagnostic scripts are exploratory checks. Their results depend
on conditions when applied to measured data.

### Solver comparison

| Solver | Degrees of freedom | What is optimized | Use |
|---------|--------|-------------|------|
| `GenericLayerSolver` | low | the thickness of each layer (and the stoichiometry with `optimize_stoichiometry=True`) | films with a known layer structure |
| `FreeSolver` | high | the concentration of every element at every depth | exploring a film of unknown structure |
| `SemiFreeSolver` | medium | the concentration of the free elements at each depth, and the position of the substrate interface expressed as an erf | films whose substrate interface is one erf (multilayers included) |

`SemiFreeSolver` was used in the two published papers (JVST A, doi:10.1116/6.0005585;
SIA, doi:10.1002/sia.70123) for the analysis of C/Al₂O₃/TiO₂/Si multilayers (L1
regularization, interface width fixed at 5 Å FWHM).

## Handling of the interface width

The angle-resolved intensity depends on both the position and the width of an
interface through the same exponential attenuation, and the position is determined
far more strongly. The interface width is therefore not estimated but fixed
(`optimize_abruptness=False` and the like). The fixed width is a regularization choice,
not a measurement. The explanation of this treatment, and the sensitivity to the width
(2.5–10 Å) in the JVST analysis, are in §III.A of JVST A, doi:10.1116/6.0005585.

## Examples and feature descriptions

### FreeSolver example (`src/deppro/examples/freesolver_recommended.py`)

A synthetic 3-layer film (Al₂O₃ 0–15 Å / SiO₂ 15–45 Å / Si, erf interfaces with σ = 3 Å,
depth grid 0–98 Å in 2 Å steps), angles 0/30/45/60°, Al Kα, 2 % relative Gaussian
noise on the intensities (seed 42; clipped at 1e-6 after adding, then normalized per
angle), `SolverConfig.recommended(alpha=0.0003)` (MEM), exponential prior, and
constraints. RMSE is the RMSE of the concentrations over all depths and elements.

| Compound table | RMSE | χ² | Al³⁺ (0 Å) | Si⁰ (80 Å) | O²⁻ (0 Å) |
|---|---|---|---|---|---|
| Truth | – | – | 0.400 | 1.000 | 0.600 |
| Bundled with toyomacro v0.3.1 (default) | 0.1291 | 1.67e-03 | 0.385 | 0.890 | 0.602 |
| External `CompoundTable.csv` via `DEPPRO_COMPOUND_TABLE_PATH` | 0.1210 | 1.38e-03 | 0.387 | 0.912 | 0.602 |

Measured 2026-10-04 with Python 3.12.11 / NumPy 2.3.3 / SciPy 1.16.3. The result
depends on the source of the compound table.

The "near-perfect reconstruction (RMSE 0.056, 2026-01-29)" that used to be stated
here has been withdrawn. The same example in the first version of this repository
(2026-07-16), with the same CSV table, also gives RMSE 0.1210, the same value as the
current version. Code changes since 2026-07-16 are therefore not the cause of the
difference from 0.056. The code and conditions of 2026-01-29, when 0.056 was written
down, cannot be verified and are not recorded in this repository.

**Settings of this example**:
1. **Exponential prior** (Livesey 1994): `d_penetration = λ_substrate`
2. **MEM regularization**: α = 0.0003 (the default of `SolverConfig.recommended()`. The
   value was chosen on the synthetic example with 2 % noise; a sweep over α has not been
   checked in the current version. Adjust it to the sample and the noise)
3. **Constraints**: `use_sum_constraint=True`, `use_charge_constraint=True`

### Films of variable composition

Giving `stoichiometry_bounds` to a `LayerDefinition` makes the stoichiometry an
optimization parameter (oxygen-deficient SiOx, nitrided layers, and so on).

**Example**:
```python
# Oxygen-deficient SiOx
layer = LayerDefinition(
    name="SiOx",
    element_indices=[0, 1],  # Si, O
    stoichiometry={0: 1, 1: 2.0},  # initial value: SiO2
    stoichiometry_bounds={1: (1.5, 2.0)},  # optimize the O ratio within 1.5-2.0
)

config = GenericLayerConfig(
    layers=[layer],
    substrate=...,
    optimize_stoichiometry=True,  # enable stoichiometry optimization
)
```

**Inspecting the result**:
```python
result = solver.solve(I_obs)
print(result.layer_stoichiometries)  # optimized stoichiometry of each layer
print(result.assumptions)            # the assumptions used (for display)
print(result.summary())              # summary
```

**Note**: the interface width stays fixed even when the stoichiometry is variable
(see "Handling of the interface width" above).

### Automatic substrate detection

`StructureAnalyzer.detect_substrate()` detects the substrate from angle-resolved data:

**Detection algorithms**:
1. **Slope method**: the element with the most negative `d(ln I)/d(1/cosθ)` is the deepest (the substrate)
2. **Ratio method**: the element with the smallest `I(60°)/I(0°)` is the deepest
3. **Compound substrate**: elements with similar slopes are grouped

**Fallback conditions**:
- all elements have a similar angle dependence → possibly a uniform mixture
- all elements have a positive slope → no substrate signal (possibly a thick film)

## Constraint system

SemiFreeSolver builds the following constraints in **algebraically**:

| Constraint | Formula | Implementation |
|------|------|------|
| Sum of concentrations | Σ D_e = 1 | algebraic |
| Charge conservation | Σ (oxidation state × D_e) = 0 | `free_linked_const` |

## Usage examples

### GenericLayerSolver (multilayer)

```python
config = GenericLayerConfig(
    layers=[
        LayerDefinition(name="C", element_indices=[0], stoichiometry={0: 1}),
        LayerDefinition(name="Al2O3", element_indices=[1, 4], stoichiometry={1: 2, 4: 3}),
        LayerDefinition(name="TiO2", element_indices=[2, 4], stoichiometry={2: 1, 4: 2}),
        LayerDefinition(name="SiO2", element_indices=[3, 4], stoichiometry={3: 1, 4: 2}),
    ],
    substrate=LayerDefinition(name="Si", element_indices=[5], stoichiometry={5: 1}),
    initial_thicknesses=[5.0, 15.0, 35.0, 30.0],
    initial_widths=[2.0, 3.0, 4.0, 5.0],
    linked_element_index=4,  # O
)
solver = GenericLayerSolver(tm, SolverConfig(method='LLS'), config)
```

### FreeSolver (using `SolverConfig.recommended()`)

```python
from deppro import (
    FreeSolver, SolverConfig, create_exponential_prior, IMFP
)

# IMFP of the substrate
imfp_si = IMFP.tpp2m(KE=1383, Nv=4, density=2.33, Aw=28.09, Eg=1.1)

# Exponential prior (Livesey 1994)
D_prior = create_exponential_prior(
    depth, n_elements=4, substrate_idx=2,
    d_penetration=imfp_si  # use the IMFP of the substrate
)

# MEM + α=0.0003 (default) + constraints
config = SolverConfig.recommended()

# Run the solver
solver = FreeSolver(tm, config, D_prior, elements=elements)
result = solver.solve(I_obs)
```

### SemiFreeSolver

#### Configuration example (multilayer, L1 regularization)

The values below show the form of the call; they are not the values used in the
papers' analyses. The settings of the JVST/SIA analyses (α and the form of the
objective) are in §III.A of JVST A, doi:10.1116/6.0005585.

```python
from deppro import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig

config = SolverConfig(
    method='L1',      # L1 regularization (sparsity)
    alpha=1.0,        # regularization strength (example value); adjust to the data and the noise
    max_iter=100,
)

semi_free_config = SemiFreeSolverConfig(
    thickness_init=70.0,
    abruptness_init=5.0,         # interface width (FWHM, Å); fixed by default (optimize_abruptness=False)
    thickness_bounds=(10.0, 90.0),
    free_indices=[0, 1, 2, 3],   # C, Al, Ti, Si(ox)
    linked_indices=[4],          # O (determined by charge conservation)
    substrate_indices=[5],       # Si
)

solver = SemiFreeSolver(
    tm, config, semi_free_config,
    elements=elements,
    D_prior=None,     # no prior (a uniform value at the lower bound min_conc); the L1 term is then close to sum|D|
    use_numba=True,   # use the Numba JIT
)
result = solver.solve(I_obs)
```

#### Basic configuration (single layer)

```python
from deppro import SemiFreeSolver, SemiFreeSolverConfig

config = SemiFreeSolverConfig(
    thickness_init=30.0,
    abruptness_init=5.0,        # the interface width is kept fixed (see "Handling of the interface width" above)
    optimize_abruptness=False,
    free_indices=[0],           # optimized freely
    linked_indices=[1],         # determined by charge conservation
    substrate_indices=[2],      # substrate (erf)
)
solver = SemiFreeSolver(tm, SolverConfig(method='LLS'), config)
```

### SemiFreeSolver + stoichiometric mode (NON / ONO analysis)

For systems in which O and N vary independently, such as NON (Si₃N₄/SiO₂/Si₃N₄/Si)
or ONO structures, use `stoichiometric_mode='stoichiometric'`.

```python
from deppro import SemiFreeSolver, SemiFreeSolverConfig

# Configuration for NON / ONO analysis
# Element indices: 0=Si-O, 1=O, 2=N, 3=Si-B
config = SemiFreeSolverConfig(
    free_indices=[1, 2],       # optimize O and N independently
    linked_indices=[],          # no linked elements (unlike charge_balance)
    substrate_indices=[3],      # Si-B

    stoichiometric_mode='stoichiometric',  # for NON / ONO
    stoichiometric_rules={
        0: ([1, 2], [0.5, 0.75])  # Si-O = 0.5*D_O + 0.75*D_N (stoichiometry)
    },
    stoichiometric_constraint_coeffs=[1.5, 1.75],  # (3/2)*O + (7/4)*N = 1 - SiB

    thickness_init=25.0,
    optimize_abruptness=False,
)
solver = SemiFreeSolver(tm, SolverConfig(method='LLS'), config)
result = solver.solve(I_obs)
```

### Choosing the stoichiometric mode

| System | Mode | Reason |
|---|--------|------|
| SiO₂/Si | `charge_balance` (default) | fixed O/Si ratio |
| C/Al₂O₃/TiO₂/SiO₂/Si | `charge_balance` | each layer has a fixed composition |
| NON (Si₃N₄/SiO₂/Si₃N₄) | `stoichiometric` | O and N vary independently at different depths |
| ONO (SiO₂/Si₃N₄/SiO₂) | `stoichiometric` | O and N vary independently at different depths |
| Oxygen-deficient SiOx | `stoichiometric` | the O ratio changes with depth |
| Graded SiGe | `stoichiometry_bounds` recommended | the composition changes continuously |

**Limits of automatic detection**:
- `StructureAnalyzer.use_stoichiometric_mode` is advisory
- multilayer oxides (fixed composition per layer) can be misdetected
- **choose the mode explicitly, based on an understanding of the system**

**How to tell them apart**:
1. **the same pair of elements (O/N) varies at different depths** → `stoichiometric`
2. **different compounds in different layers (fixed within each layer)** → `charge_balance`

**How to choose a mode**:
1. distinguishable by chemical shift → treat as separate elements → GenericLayerSolver
2. not distinguishable, and the composition differs between layers → stoichiometric_mode

### ARXPSPipeline (automatic analysis)

```python
from deppro import ARXPSPipeline

pipeline = ARXPSPipeline()
result = pipeline.analyze(
    I_obs, tm, elements,
    element_labels=['Si_ox', 'O', 'Si_bulk'],
)

if result.success:
    print(f"Thicknesses: {result.layer_thicknesses}")
```

### Automatic substrate detection + FreeSolver

```python
from deppro import (
    StructureAnalyzer, FreeSolver, SolverConfig,
    create_exponential_prior, IMFP
)

# Step 1: detect the substrate
analyzer = StructureAnalyzer()
result = analyzer.analyze_with_substrate_detection(I_obs, angles, element_labels)

if result.substrate_detection.has_substrate:
    substrate_idx = result.substrate_detection.substrate_indices[0]
    print(f"Detected substrate: {element_labels[substrate_idx]}")

    # Step 2: IMFP and exponential prior
    imfp_substrate = IMFP.tpp2m(KE=1383, Nv=4, density=2.33, Aw=28.09, Eg=1.1)
    D_prior = create_exponential_prior(
        depth, n_elements=4, substrate_idx=substrate_idx,
        d_penetration=imfp_substrate
    )

    # Step 3: run FreeSolver
    solver = FreeSolver(tm, SolverConfig.recommended(), D_prior, elements)
    result = solver.solve(I_obs)
else:
    print(f"⚠️ {result.substrate_detection.fallback_reason}")
    print("Give substrate_idx explicitly")
```

## Module Structure

```
deppro/
├── __init__.py              # package exports
├── depth_profile.py         # depth-profile generation
├── transfer_matrix.py       # transfer matrix, angle-resolved calculation
├── imfp.py                  # TPP-2M IMFP calculation
├── inverse_solver.py        # the inverse solvers
├── structure_analyzer.py    # automatic structure analysis
├── pipeline.py              # automatic analysis pipeline
├── benchmark.py             # benchmark system
├── cli_depth.py             # CLI tool
└── tests/                   # test suite
```

## CLI Commands

```bash
# Solve the inverse problem
python -m deppro.cli_depth solve --input intensity.csv --config model.yaml

# Generate a configuration-file template
python -m deppro.cli_depth init-config --model multilayer --output config.yaml

# Benchmark
python -m deppro.benchmark --quick
```

## Notes on high emission angles

- The larger the emission angle, the weaker the substrate signal. For a film that is
  thick compared with the IMFP, the signal is close to zero above about 75° (the angle
  at which it fades depends on the thickness and the IMFP)
- SemiFreeSolver normalizes by the substrate signal by default, so including angles
  with a weak substrate signal can make it unstable

## Regularization methods

### FreeSolver configuration

```python
# Preset
config = SolverConfig.recommended()  # MEM + α=0.0003 (default) + constraints

# Or by hand
config = SolverConfig(
    method='MEM',
    alpha=0.0003,              # the default of recommended(); adjust to the sample and the noise
    use_sum_constraint=True,   # Σconc = 1
    use_charge_constraint=True # charge conservation
)
```

**Note**: LLS does not use α, so the prior term has no effect. The exponential prior
needs MEM.

### Comparison of regularization methods

| Method | Character |
|------|------|
| `LLS` | no regularization (α unused) |
| `L1` | sparse solutions (minimizes sum\|D − D_prior\|) |
| `L2` | smooth solutions |
| `MEM` | maximum entropy; combined with the exponential prior |

The default α is 1.0 in `SolverConfig` and 0.0003 in `SolverConfig.recommended()`.
Neither suits every sample; adjust it to the data and the noise.

## Noise models

### Gaussian versus Poisson noise

⚠️ **Poisson X % and Gaussian X % are not directly comparable**

| Noise | Definition | Behaviour at low intensity |
|--------|------|----------------|
| Gaussian X % | σ = X % at every intensity | constant |
| Poisson X % | σ = X % **only at the maximum intensity I = 1.0** | σ ≈ 3X % at I = 0.1 |

**RMS-equivalent comparison** (the ratio depends on the distribution of intensities;
the values below are a guide):
- Poisson 0.5 % ≈ Gaussian 2 % (RMS basis)
- Poisson 2 % ≈ Gaussian 8 % (RMS basis)

**Recommendation**: compare benchmark results at RMS-equivalent noise

## Physical background

### TPP-2M IMFP calculation

```python
from deppro import IMFP

imfp = IMFP.tpp2m(
    KE=1383.1,      # kinetic energy (eV)
    Nv=16,          # number of valence electrons
    density=2.2,    # g/cm³
    Aw=60.08,       # molecular weight (g/mol)
    Eg=9.0,         # band gap (eV)
)
# → 38.6 Å (SiO2, the value of the TPP-2M formula)
```

### Reference material parameters

| Compound | Density | Molecular weight | Valence electrons | Band gap |
|--------|------|--------|----------|---------------|
| SiO2 | 2.2 | 60.08 | 16 | 9.0 eV |
| Si | 2.33 | 28.09 | 4 | 1.1 eV |

## Dependencies

- numpy
- scipy
- matplotlib (for visualization)

## Related files

- `../tests/`: test suite
