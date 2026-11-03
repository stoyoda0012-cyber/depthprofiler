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
regularisation. The package provides three inverse solvers that differ
in how much structure they assume, reference CPU implementations and
batched implementations for Apple-silicon GPUs built on MLX, a
neural-network inverse solver, a synthetic forward simulator with
comparison metrics, and a preprocessing layer for multi-frame HAXPES
data.

# Statement of need

Sputter depth profiling consumes the sample as it measures it. ARXPS
is non-destructive, but recovering $D(z)$ from a few emission angles is
an ill-posed inverse problem: the depth resolution it can reach is
limited, and the answer depends on how the inversion is regularised
[@Cumpson1995; @Livesey1994]. The profile a study reports is therefore
the product of three sets of choices — the structural assumption put
into the inversion (free composition, a parametrised substrate
interface, or a fixed layer sequence), the regulariser and its strength,
and the preprocessing of the spectra (angle binning, background,
cross-section correction, denoising) — and a reader cannot tell how much
of a reported thickness or interface position came from the data and
how much from these choices unless the choices are explicit and the
analysis can be re-run with them varied.

`deppro` is written for surface scientists who have ARXPS or HAXPES
angle series — one sample, or every pixel of a spatial map — and want to
examine that dependence. The same forward model and the same measured
intensities are shared by all solvers, so re-solving under another
structural assumption, another regulariser or another preprocessing is
a change of configuration, not of dataset, and the effect can be
followed to the structural quantities of interest: the composition
profile, the film thickness and the interface position, per pixel. For
multi-pixel data the package carries this through to the level of maps,
with batched solvers, per-pixel arrays and profile metrics, so that a
measurement condition (exposure time), a preprocessing step (a denoiser)
or an inversion setting (the fixed interface width) can be compared by
its effect on the reconstructed structure rather than only on the
spectra. The two published studies that used the package did this for
denoising methods and exposure times at a fixed inversion; the
comparison of structural assumptions and regularisers on measured maps
is what the design makes possible and has so far been exercised on
synthetic data.

# State of the field

Maximum-entropy inversion of angle-dependent XPS [@Livesey1994] and the
analysis of depth-resolution limits for ARXPS [@Cumpson1995] established
the methodological basis that `deppro` builds on, and software for
non-destructive depth reconstruction from ARXPS exists in several forms.
General-purpose XPS processing programs such as CasaXPS [@Fairley2021]
cover peak fitting, background subtraction and quantification, including
for angle-resolved series. Spectrometer software includes
maximum-entropy reconstruction of concentration depth profiles from
ARXPS series [@KratosMEM]. QUASES-ARXPS [@QUASESARXPS]
is a dedicated program for determining the depth distribution of
elements from the angle dependence of peak intensities, and the same
group's analysis of the inelastic background [@Tougaard1996] infers depth
distributions from the shape of a peak and its loss tail rather than from
the angle dependence. SESSA [@Smekal2005; @SESSA] simulates XPS and
Auger spectra for layered and other model samples from a user-defined
model, so that depth information is obtained by comparing simulated and
measured intensities under that model. Non-destructive depth
reconstruction, maximum-entropy regularisation, composition constraints
and automated processing are thus available in existing tools; `deppro`
does not claim to be first with any of them, nor to be more accurate or
faster.

What `deppro` adds is an independent, open implementation of the
multi-angle transfer-matrix inversion in which the pieces that a
methodological study needs to vary are exposed as code: three structural
parametrisations under one forward model, a choice of regulariser
(unregularised, L1, L2, maximum entropy) and noise model, batched solving
over the pixels of a map, and the preprocessing and the evaluation
metrics in the same package as the inversion. It takes peak intensities
from any fitting program and sits beside, not in place of, the programs
above.

# Software design

The forward model is a transfer matrix that maps a discretised depth
profile to angle-resolved intensities from inelastic mean free paths
(TPP-2M, with compound parameters from tabulated data), the emission
geometry and the photon energy. Compound, binding-energy and
cross-section tables live in a separate package, `toyomacro`
[@toyomacro031], so that one reviewed copy serves both the peak-fitting
and the depth-inversion code; user-supplied tables can be substituted
through environment variables.

Three solver families share the forward model and a common
`SolverConfig`, and differ in the prior knowledge they put into the
inversion:

- `FreeSolver` leaves the composition of every element at every depth
  free. Structure enters only through the regulariser, an optional prior
  profile (for example an exponential substrate prior in the
  maximum-entropy case) and the sum and charge-balance constraints.
- `SemiFreeSolver` parametrises the substrate as an error-function step
  whose position (the film thickness) is fitted and whose width is fixed
  by default, because the angle dependence constrains the position far
  more strongly than the width. The film side stays free: the
  concentrations of the free elements are optimised at every depth, the
  linked elements follow from charge balance or from explicit
  stoichiometric rules, and the substrate ratios may be fitted. An
  optional thickness prior can be added to the objective.
- `GenericLayerSolver` fixes the layer sequence and the stoichiometry of
  each layer and fits the layer thicknesses; interface widths and
  stoichiometric ratios can optionally be fitted too.

`FreeSolver` and `SemiFreeSolver` accept unregularised least squares or
an L1, L2 or maximum-entropy regulariser; `GenericLayerSolver` minimises
the least-squares misfit of row-normalised intensities. Because a
different assumption is a different solver on the same inputs, the
sensitivity of a profile, thickness or interface position to the
assumption can be examined by re-solving. The CPU implementations,
built on NumPy [@numpy2020] and SciPy [@scipy2020], are the reference.
Batched implementations of the same three families on MLX
(<https://github.com/ml-explore/mlx>) solve many pixels of a map at
once on Apple-silicon GPUs; they are tested against the CPU reference on
defined synthetic cases, and one acceptance case of the semi-free MLX
solver remains open, so numerical equivalence is not claimed across all
settings. A neural-network inverse solver, also on MLX, maps intensities
directly to profiles and can provide starting points for the iterative
solvers; architectures and depth grids can be registered without
changing the core.

A synthetic forward simulator generates intensities from known profiles,
so that every solver can be checked in a round trip from a profile to
intensities and back, and profile metrics (Wasserstein-1 distance,
interface detection, layer-wise comparison) quantify the result. The
`haxpes` subpackage handles multi-frame HAXPES data — frame stacks,
angle binning, background subtraction, cross-section and non-dipole
correction, the geometry of the instrument — runs the inversion per
pixel with multiprocessing, and connects to the separate `dnndenoiser`
package for denoising individual frames. Keeping the preprocessing, the
inversion and the metrics in one package is what lets a study vary one
condition and follow its effect to the reconstructed structure.

# Research impact statement

`deppro` performed the per-pixel depth inversions in two studies of
self-supervised denoising for angle-resolved HAXPES depth profiling of a
C/Al$_2$O$_3$/TiO$_2$/Si multilayer [@Toyoda2026JVST; @Toyoda2026SIA].
In both, the inversion was held fixed — `SemiFreeSolver` with L1
regularisation and a fixed interface width — and the quantity compared
across conditions was the reconstructed structure: per-pixel profiles,
film thickness and its spread over the map, and the Wasserstein-1
distance of each pixel's profile from a pseudo-ground truth obtained from
the average over all frames of the same dataset.
The JVST A study compared denoising strategies (frame averaging, per-bin
resampling and the self-supervised network) on maps of 50 × 50 pixels at
24 s and 20 × 20 pixels at 120 s exposure, and examined the sensitivity
of the result to the fixed interface width (2.5–10 Å). The SIA study
compared denoisers trained at one exposure and applied at another
(4.8 s, 24 s, 120 s) through the depth RMSE and Wasserstein-1 distance
of the per-pixel profiles and the equivalent number of averaged frames.

A module in the package, `deposit_replay`, uses the studies' public data
deposit [@Toyoda2026deposit] to recompute the published metrics and
tables from the deposited profiles and pseudo-ground truths and to check
the deposited JVST metric files against the deposited arrays. This is a
check of stored results: it does not re-run the denoiser, the peak fit
or the inversion, and the comparison is against a pseudo-ground truth
from the same data, not against an independently known structure. An
accompanying document lists, figure by figure, what can be recomputed
from public material alone, what would need the authors' weights and raw
frames, and which deposited values do not reproduce. Results obtained
with the package have also been presented at meetings of the Japan
Society of Applied Physics and the Japan Society of Vacuum and Surface
Science.

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
