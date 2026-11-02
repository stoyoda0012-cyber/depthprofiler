---
title: 'deppro: depth-composition reconstruction from angle-resolved X-ray photoelectron spectroscopy'
tags:
  - Python
  - XPS
  - HAXPES
  - ARXPS
  - depth profiling
  - inverse problem
  - photoelectron spectroscopy
authors:
  - name: Satoshi Toyoda
    orcid: 0009-0007-1091-1579
    affiliation: "1, 2"
affiliations:
  - name: Office of Institutional Advancement and Communications, Kyoto University, Kyoto, Japan
    index: 1
  - name: Advanced Equipment Division, Vacuum Products Corporation, Tokyo, Japan
    index: 2
date: "to be entered on submission"
bibliography: paper.bib
---

# Summary

Angle-resolved X-ray photoelectron spectroscopy (ARXPS), including its
hard-X-ray variant (HAXPES), measures how the photoelectron intensity of
each element changes with emission angle. Because electrons from deeper
layers are attenuated more strongly at grazing emission, these angle
dependences carry information on how composition varies with depth.
`deppro` reconstructs a depth-composition profile $D(z)$ from such
intensities. It builds a transfer matrix $T$ from inelastic mean free
paths (computed with the TPP-2M formula [@Tanuma1994]) and the
measurement geometry, so that the intensities are modelled as
$I = T\,D$, and it inverts this model under physical constraints and
regularisation. The package provides reference CPU solvers, batched
solvers for Apple-silicon GPUs built on MLX, a neural-network inverse
solver, a synthetic forward simulator for round-trip tests, and a
preprocessing layer for multi-frame HAXPES data.

# Statement of need

Sputter depth profiling consumes the sample as it measures it. ARXPS
is non-destructive, but recovering $D(z)$ from a few emission angles is
an ill-posed inverse problem:
the depth resolution it can reach is limited, and the answer depends on
how the inversion is regularised [@Cumpson1995; @Livesey1994]. Studies
that report ARXPS or HAXPES depth profiles therefore need the forward
model, the regularisation and the solver settings to be explicit and
re-runnable, so that a reader can check how a profile followed from the
measured intensities. `deppro` makes these choices explicit in code: the
same solver configuration selects unregularised least squares or an L1,
L2 or maximum-entropy regulariser. It is written for surface scientists
who analyse ARXPS or HAXPES depth series and want the inversion to be
inspectable and re-runnable. It provides an open, scriptable
implementation of the forward model and of the inversion, complementary
to the general-purpose XPS processing programs described below.

# State of the field

Maximum-entropy inversion of angle-dependent XPS [@Livesey1994] and the
analysis of depth-resolution limits for ARXPS [@Cumpson1995] established
the methodological basis that `deppro` builds on.

General-purpose XPS processing programs such as CasaXPS
[@Fairley2021], and the software supplied with spectrometers, cover peak
fitting, background subtraction and quantification, including for
angle-resolved data series. A different route to depth information is
the analysis of the inelastic background of a single spectrum developed
by Tougaard [@Tougaard1996], which infers how an element is distributed
in depth from the shape of the peak and its loss tail rather than from
the angle dependence of its intensity.
`deppro` is complementary to both. It takes peak intensities from any
fitting program, performs the multi-angle transfer-matrix inversion
itself, makes the regularisation explicit, and solves the pixels of a
map in batches.

# Software design

The forward model is a transfer matrix that maps a discretised depth
profile to angle-resolved intensities. It combines inelastic mean free
paths (TPP-2M, with compound parameters from tabulated data), the
emission geometry and the photon energy. Compound, binding-energy and
cross-section tables are read from user-supplied tables named by
environment variables when those are set, and otherwise from the
`toyomacro` package [@toyomacro031]; the order is fixed, and no
neighbouring directory is searched.

On top of the forward model, three solver families each take a
`SolverConfig` together with their own settings:

- `FreeSolver` treats the composition at every depth as free;
- `SemiFreeSolver` adds the film thickness as a parameter;
- `GenericLayerSolver` fits the layer thicknesses of a layered model;
  interface widths and layer stoichiometry can optionally be fitted too.

`FreeSolver` and `SemiFreeSolver` accept unregularised least squares or
an L1, L2 or maximum-entropy regulariser; `GenericLayerSolver`
minimises the least-squares misfit of row-normalised intensities. The CPU implementations, built on NumPy [@numpy2020] and
SciPy [@scipy2020], are the reference. Batched implementations of the
same three families on MLX (<https://github.com/ml-explore/mlx>) solve
many pixels of a map at once on Apple-silicon GPUs. A neural-network
inverse solver, also on MLX, maps intensities directly to profiles and
can provide starting points for the iterative solvers.

A synthetic forward simulator generates intensities from known
profiles, so that every solver can be checked in a round trip from a
profile to intensities and back. The `haxpes` subpackage handles
multi-frame HAXPES data (frame stacks, angle binning, cross-section
correction) and connects to the separate `dnndenoiser` package for
denoising individual frames.

Two design choices shape the package. The CPU solvers serve as the
reference implementations. The MLX implementations are tested against
them on defined synthetic cases, comparing objective values and
reconstructed parameters; one acceptance case of the semi-free MLX
solver remains open, so numerical equivalence is not claimed across all
supported settings.
The physical tables live in a separate package rather than in this one,
so that one reviewed copy serves both the peak-fitting and the
depth-inversion code. Neural-network architectures and depth-grid
types are registered through small hooks (`register_architecture`,
`register_grid_type`), so that further neural-network inverse models and
depth grids can be added without changing the core.

# Research impact statement

`deppro` performed the depth inversions in two studies of
self-supervised denoising for angle-resolved HAXPES depth profiling
[@Toyoda2026JVST; @Toyoda2026SIA]. A module in the package uses their
public data deposit [@Toyoda2026deposit] to recompute the published
metrics and tables of the SIA study and to check that the deposited
JVST metric files agree with the deposited arrays; an accompanying
document lists, figure by figure, what can be recomputed from public
material alone and what cannot, including the deposited values that do
not reproduce. Results obtained with the package have also been
presented at meetings of the Japan Society of Applied Physics and the
Japan Society of Vacuum and Surface Science.

# AI usage disclosure

Generative AI tools were used throughout development. Anthropic Claude,
through Claude Code, was used for code generation, refactoring, tests,
documentation, repository maintenance and drafting this paper. OpenAI
ChatGPT and Codex were used mainly to review changes and designs. The
current model of each tool was used as new models were released during
development, so no single version applies. Changes were also checked by
separate AI sessions acting as independent auditors, with the test suite
and the published data as references. The author made the core design
decisions, decided which changes and scientific claims to accept, and
reviewed, edited and validated all AI-assisted output.

# Acknowledgements

This work received no dedicated funding.

# References
