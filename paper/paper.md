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
$I = T\,D$, and it inverts this model under physical constraints and a
chosen regulariser. Three solvers impose different structural
assumptions on the same forward model, and each pixel of a map or each
frame of a time series is inverted independently.

# Statement of need

Sputter depth profiling consumes the sample as it measures it. ARXPS
is non-destructive, but recovering $D(z)$ from a few emission angles is
an ill-posed inverse problem: the depth resolution it can reach is
limited, and the answer depends on how the inversion is regularised and
on what is assumed about the structure [@Cumpson1995; @Livesey1994].
A reported thickness or interface position is therefore a property of
the data together with these choices.

This matters more as datasets grow. A mapping instrument, or a HAXPES
measurement recorded as thousands of short frames, yields one angle
series per pixel or frame, and the question becomes how the measurement
(exposure per frame), the preprocessing (angle binning, background,
cross-section table, denoising) and the inversion (structure model,
regulariser, starting values) each move the reported composition
profile, film thickness and interface position, and their spread across
pixels. Answering it means
re-analysing the same data many times with one choice changed at a time,
and scoring the results against a reference.

`deppro` is written for surface scientists who analyse ARXPS or HAXPES
depth series and want to do that. The forward model, the constraints and
the regulariser are explicit in code, three solvers change the
structural assumption on one forward model, the same calls run over
every pixel of a map, and a simulator with known profiles lets the
effect of each choice be measured where the truth is known before it is
applied to measured data.

# State of the field

Maximum-entropy inversion of angle-dependent XPS
[@Smith1992; @Livesey1994] and the analysis of depth-resolution limits
for ARXPS [@Cumpson1995] established the methodological basis that
`deppro` builds on.

General-purpose XPS processing programs such as CasaXPS [@Fairley2021]
cover peak fitting, background subtraction and quantification, including
for angle-resolved data series. Spectrometer software includes
maximum-entropy reconstruction of concentration depth profiles from
ARXPS data, as in the application note of one manufacturer
[@KratosMEM2024]. QUASES-ARXPS fits a layered structure, with adjustable
concentration and coverage for each layer, to angle-resolved XPS or AES
data, with automatic optimisation of the structure and an optional
correction for elastic scattering [@QUASESARXPS]; practical correction
procedures of this kind were assessed by its developers [@Lassen2001].
SESSA simulates XPS and Auger spectra, including at HAXPES energies, for
multilayers and, in its current version, nanostructures; the user
adjusts the sample until simulated and measured spectra agree, and
command sequences run series of simulations automatically [@Smekal2005; @Werner2013; @SESSA].
A different route to depth information is the analysis of the inelastic
background of a single spectrum [@Tougaard1996]. Depth reconstruction,
compositional constraints and automation are therefore all available in
existing software.

`deppro` addresses a different need. In the programs that invert ARXPS data, the model and
the objective are set by the program and reached through its own
interface, and SESSA simulates spectra rather than inverting them. The re-analysis described above needs an
inversion whose objective, constraints and parameterisation can be
changed, several parameterisations on one forward model, per-pixel
batch processing, and a direct connection to the preprocessing and the
scoring metrics. The closest public tools we know of are a spreadsheet
for Tikhonov-regularised ARXPS profiles from up to five angles and five
components [@Murdoch2023] and PROPHESY, which inverts
photon-energy-resolved XPS across liquid interfaces [@Ozon2023]; neither
is a package to which these features could readily be added, so
`deppro` was built as a separate open package. Its forward model is simpler than that of SESSA or
QUASES-ARXPS: it uses straight-line attenuation and has no
elastic-scattering term, so intensities simulated with SESSA could be
used to test that approximation; this has not yet been done.

# Software design

The forward model maps a discretised depth profile to angle-resolved
intensities. It combines inelastic mean free paths (TPP-2M, with
compound parameters from tabulated data), the emission geometry, the photon energy and an
optional non-dipole correction. It is computed for an assumed
composition, uniform by default, held fixed within each optimisation,
and can be rebuilt from the previous solution between outer passes. The physical tables come from the
`toyomacro` package [@toyomacro031] unless the user names others.

The three solvers differ in the prior knowledge they impose. They share
the transfer matrix and the input arrays but not the intensity
normalisation or the optimiser, which a comparison has to take into
account.

- `FreeSolver` assumes no layer structure. The composition of every
  element at every depth is a free variable, constrained to be
  non-negative and to sum to one at each depth (and optionally to be
  charge-neutral). What is assumed enters only through the regulariser
  and its prior profile: by default the L2 and maximum-entropy terms
  pull towards a uniform prior, while the L1 prior equals the lower
  bound, so under the sum-to-one constraint the L1 term is constant; a
  user-supplied prior replaces either.
- `SemiFreeSolver` assumes one buried substrate interface. The
  substrate elements share a single error-function step; its midpoint,
  reported as the film thickness, and the ratios between the substrate
  elements are fitted, and its width is fixed by default (5 Å FWHM),
  because angle-resolved data constrain the position of an interface
  much more strongly than its width. On the film side, the declared
  film elements remain free at every depth; elements declared as linked,
  typically anions, follow from charge balance, and elements declared as
  constant are fixed. The thickness is fitted together with the film
  composition, with the same choice of penalty as in `FreeSolver`; by
  default the penalty is taken relative to a near-zero profile and, for
  L1, L2 and maximum entropy, also pulls the thickness weakly towards its
  starting value, so the starting thickness is part of the assumption.
- `GenericLayerSolver` assumes a stack of layers of given stoichiometry
  with error-function interfaces. It fits the layer thicknesses and,
  optionally, the interface widths and the stoichiometry within bounds,
  by unregularised least squares on row-normalised intensities.

The CPU implementations, built on NumPy [@numpy2020] and SciPy [@scipy2020],
are the reference. Batched implementations on MLX
(<https://github.com/ml-explore/mlx>) solve many pixels at once on
Apple-silicon GPUs with their own optimisation algorithms; they are
tested against the CPU reference on defined synthetic cases, and one
acceptance case of the semi-free solver remains open, so numerical
equivalence is not claimed for all settings. A neural-network inverse
solver on MLX can supply starting points for the iterative solvers.

Each pixel is inverted independently. The only lateral link is an
optional prior rebuilt from neighbouring pixels' solutions between
passes; there is no joint inversion across pixels, so a map is a set of
depth profiles rather than a tomographic reconstruction. The `haxpes`
subpackage turns multi-frame HAXPES data into per-pixel intensities
(frame averaging and resampling, angle binning, background subtraction,
cross-section correction) and connects to the separate `dnndenoiser`
package for denoising frames. A synthetic simulator generates maps from
known profiles with Poisson noise, and metrics (the Wasserstein distance
between profiles, depth RMSE, thickness deviation) score a
reconstruction against a reference.

# Research impact statement

`deppro` performed the depth inversions in two studies of
self-supervised denoising for angle-resolved HAXPES depth profiling of a
C/Al$_2$O$_3$/TiO$_2$/Si stack [@Toyoda2026JVST; @Toyoda2026SIA]. In
both, thousands of single frames recorded at one fixed position were
each inverted with `SemiFreeSolver`, so that the spread of the
profiles across frames measured how stable each preprocessing route
was. The first study compared frame averaging,
bin-pool resampling and a self-supervised network at two exposure
times, and found that at the lower count level the inversion often
failed to locate the film–substrate interface. The second applied
networks trained at one exposure time to data recorded at another, over
three exposure times, and identified a regime in which denoised frames
invert to a wrong, constraint-saturated profile that the summary metrics
reflect only weakly. Each reconstruction was scored against the
pseudo-ground truth of its own data set, the reconstruction of the
spectra averaged over all of its frames, by depth RMSE, thickness
deviation and the Wasserstein distance. No independent structural
measurement served as the truth (the first study checked only the layer
sequence against an RBS reference and the total thickness against
stylus profilometry), and the frame grids are not
lateral maps. Laterally resolved maps and comparisons of structure
models on measured data are possible with the current code but
unpublished; a quantitative comparison with an independent depth
measurement, and transfer to other samples and instruments, remain to
be verified.

A module in the package, `deposit_replay`, recomputes metrics from the
depth profiles stored in the public data deposit of these studies
[@Toyoda2026deposit]. For the SIA study it reproduces the Table 1 values,
the depth-error values quoted for Figs. 4 and 5 and the seed-pilot
values, while other supplementary figures were not recomputed; for the JVST study it checks the
deposited metric files against the deposited arrays rather than against
the paper, and the stored W1 and depth-RMSE values of the shorter
exposure do not reproduce. It does not re-run the denoiser or the
inversion; re-analysis from the raw frames needs undeposited data and
weights and has not been done. Results obtained with the package have also been
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
