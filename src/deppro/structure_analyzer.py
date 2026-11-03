"""
Structure Analyzer - automatic solver selection
==============================================

Chooses an appropriate solver from the character of the ARXPS data.

Case classification:
1. Well-defined layers → GenericLayerSolver
2. Mixed / gradient → SemiFreeSolver
3. Completely unknown structure → FreeSolver

Criteria:
- strength of the angle dependence (presence of a layer structure)
- correlation between elements (whether stoichiometry holds)
- shape of the intensity profiles (step versus gradient)

Usage:
    from deppro.structure_analyzer import StructureAnalyzer

    analyzer = StructureAnalyzer()
    recommendation = analyzer.analyze(I_obs, tm)
    print(recommendation.solver_name)
    print(recommendation.confidence)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

from .transfer_matrix import TransferMatrix


class StructureType(Enum):
    """Structure type"""
    WELL_DEFINED_LAYERS = "well_defined_layers"    # well-defined layers
    MIXED_GRADIENT = "mixed_gradient"              # mixed / gradient
    UNKNOWN = "unknown"                            # completely unknown
    SUBSTRATE_ONLY = "substrate_only"              # substrate only (no film)
    UNIFORM_MIXED = "uniform_mixed"                # uniformly mixed (no depth resolution possible)


class SolverRecommendation(Enum):
    """Recommended solver"""
    GENERIC_LAYER = "GenericLayerSolver"
    SEMI_FREE = "SemiFreeSolver"
    FREE = "FreeSolver"


@dataclass
class SubstrateDetectionResult:
    """Substrate detection result"""
    substrate_indices: List[int]       # indices of the substrate elements
    substrate_group: List[int]         # substrate group (several for a compound substrate)
    confidence: float                  # detection confidence (0-1)
    method: str                        # detection method used

    # Detailed indicators
    slopes: np.ndarray                 # slope of each element
    ratios: np.ndarray                 # 60°/0° ratio of each element

    # Fallback information
    has_substrate: bool = True         # whether a substrate was detected
    fallback_reason: Optional[str] = None  # reason for the fallback

    def __repr__(self):
        return (f"SubstrateDetectionResult(\n"
                f"  substrate_indices={self.substrate_indices},\n"
                f"  substrate_group={self.substrate_group},\n"
                f"  confidence={self.confidence:.2f},\n"
                f"  has_substrate={self.has_substrate}\n"
                f")")


@dataclass
class AnalysisResult:
    """Analysis result"""
    structure_type: StructureType
    solver_recommendation: SolverRecommendation
    confidence: float  # 0-1

    # Detailed indicators
    angle_dependence_score: float      # angle-dependence score
    stoichiometry_score: float         # stoichiometric-consistency score
    layer_sharpness_score: float       # layer-boundary sharpness score
    n_estimated_layers: int            # estimated number of layers

    # Additional information
    element_correlations: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    # Substrate detection result
    substrate_detection: Optional[SubstrateDetectionResult] = None

    # Variable-composition detection (090317 / stoichiometric mode)
    stoichiometry_variability_score: float = 0.0  # composition-variability score
    use_stoichiometric_mode: bool = False         # recommend stoichiometric mode

    def __repr__(self):
        return (f"AnalysisResult(\n"
                f"  structure_type={self.structure_type.value},\n"
                f"  solver={self.solver_recommendation.value},\n"
                f"  confidence={self.confidence:.2f},\n"
                f"  estimated_layers={self.n_estimated_layers}\n"
                f")")


class StructureAnalyzer:
    """
    Structure analyzer

    Analyzes ARXPS data and recommends a solver.
    Also provides substrate detection.
    """

    def __init__(
        self,
        stoich_threshold: float = 0.8,
        angle_dep_threshold: float = 0.05,
        layer_sharp_threshold: float = 0.7,
        substrate_slope_margin: float = 0.15,  # grouping margin for a compound substrate
    ):
        """
        Parameters
        ----------
        stoich_threshold : float
            Stoichiometric-consistency threshold (at or above: layered structure)
        angle_dep_threshold : float
            Angle-dependence threshold (at or above: depth structure present). Default 0.05.
        layer_sharp_threshold : float
            Layer-sharpness threshold (at or above: well-defined layers)
        """
        self.stoich_threshold = stoich_threshold
        self.angle_dep_threshold = angle_dep_threshold
        self.layer_sharp_threshold = layer_sharp_threshold
        self.substrate_slope_margin = substrate_slope_margin

    def analyze(
        self,
        I_obs: np.ndarray,
        tm: Optional[TransferMatrix] = None,
        element_labels: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> AnalysisResult:
        """
        Analyze ARXPS data and return a solver recommendation

        Parameters
        ----------
        I_obs : ndarray
            Observed intensities [n_angles, n_elements]
        tm : TransferMatrix, optional
            Transfer matrix (for the angle information)
        element_labels : List[str], optional
            Element labels
        verbose : bool
            Verbose output

        Returns
        -------
        AnalysisResult
        """
        n_angles, n_elements = I_obs.shape

        if verbose:
            print("=" * 60)
            print("Structure Analysis")
            print("=" * 60)
            print(f"Data shape: {n_angles} angles × {n_elements} elements")

        # Normalize (to be safe)
        I_norm = I_obs / np.maximum(I_obs.sum(axis=1, keepdims=True), 1e-10)

        # 1. Angle-dependence score
        angle_dep_score = self._compute_angle_dependence(I_norm)
        if verbose:
            print(f"Angle dependence score: {angle_dep_score:.3f}")

        # 2. Stoichiometric-consistency score
        stoich_score, correlations = self._compute_stoichiometry_score(I_norm, element_labels)
        if verbose:
            print(f"Stoichiometry score: {stoich_score:.3f}")

        # 3. Layer-sharpness score
        sharpness_score = self._compute_layer_sharpness(I_norm)
        if verbose:
            print(f"Layer sharpness score: {sharpness_score:.3f}")

        # 4. Estimated number of layers
        n_layers = self._estimate_layer_count(I_norm)
        if verbose:
            print(f"Estimated layer count: {n_layers}")

        # 5. Variable-composition score (for the stoichiometric-mode decision)
        stoich_var_score = self._compute_stoichiometry_variability(I_norm)
        if verbose:
            print(f"Stoichiometry variability score: {stoich_var_score:.3f}")

        # 6. Structure type
        structure_type = self._determine_structure_type(
            angle_dep_score, stoich_score, sharpness_score, n_layers
        )

        # 7. Solver recommendation
        solver_rec, confidence = self._recommend_solver(
            structure_type, angle_dep_score, stoich_score, sharpness_score
        )

        # 8. Stoichiometric-mode recommendation
        # Condition: angle dependence (layers) + composition variability + signs of compound formation
        # Threshold: recommended when the nonlinearity indicator > 0.12 (relaxed to catch NON/ONO)
        # Note: automatic detection has limits and can misfire on multilayer oxides.
        #       Users should choose explicitly from an understanding of the system.
        use_stoichiometric_mode = (
            angle_dep_score > 0.1 and      # angle dependence present
            stoich_var_score > 0.12 and    # composition varies (nonlinearity)
            stoich_score > 0.5             # signs of compound formation
        )

        # 9. Warnings and suggestions
        warnings, suggestions = self._generate_feedback(
            structure_type, angle_dep_score, stoich_score, sharpness_score,
            n_layers, stoich_var_score
        )

        result = AnalysisResult(
            structure_type=structure_type,
            solver_recommendation=solver_rec,
            confidence=confidence,
            angle_dependence_score=angle_dep_score,
            stoichiometry_score=stoich_score,
            layer_sharpness_score=sharpness_score,
            n_estimated_layers=n_layers,
            element_correlations=correlations,
            warnings=warnings,
            suggestions=suggestions,
            stoichiometry_variability_score=stoich_var_score,
            use_stoichiometric_mode=use_stoichiometric_mode,
        )

        if verbose:
            print()
            print(f"Structure type: {structure_type.value}")
            print(f"Recommended solver: {solver_rec.value}")
            print(f"Confidence: {confidence:.2f}")
            if warnings:
                print(f"Warnings: {warnings}")
            if suggestions:
                print(f"Suggestions: {suggestions}")

        return result

    def _compute_angle_dependence(self, I_norm: np.ndarray) -> float:
        """
        Compute the angle-dependence score

        High → there is structure along depth
        Low → uniform (substrate only, or completely mixed)
        """
        # Angle dependence of each element (coefficient of variation)
        cv_per_element = np.std(I_norm, axis=0) / np.maximum(np.mean(I_norm, axis=0), 1e-10)

        # Overall score (mean coefficient of variation)
        score = np.mean(cv_per_element)

        return min(score, 1.0)

    def _compute_stoichiometry_score(
        self,
        I_norm: np.ndarray,
        element_labels: Optional[List[str]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute the stoichiometric-consistency score

        High → fixed ratios between elements (compound formation)
        Low → random mixing
        """
        n_elements = I_norm.shape[1]
        correlations = {}

        # Correlation coefficients between element pairs
        corr_values = []
        for i in range(n_elements):
            for j in range(i + 1, n_elements):
                corr = np.corrcoef(I_norm[:, i], I_norm[:, j])[0, 1]
                if not np.isnan(corr):
                    key = f"elem{i}-elem{j}"
                    if element_labels and len(element_labels) > max(i, j):
                        key = f"{element_labels[i]}-{element_labels[j]}"
                    correlations[key] = corr
                    corr_values.append(abs(corr))

        # The more strong correlations (positive or negative), the higher the score
        if corr_values:
            score = np.mean(corr_values)
        else:
            score = 0.0

        return score, correlations

    def _compute_stoichiometry_variability(self, I_norm: np.ndarray) -> float:
        """
        Quantify the composition variation within compound layers (variable-composition score)

        High → composition changes with depth (variable composition: SiOx, NO/ON, graded SiGe)
        Low → constant composition (fixed stoichiometry: pure SiO2, Si3N4)

        Method:
        - evaluate the "nonlinearity" of the correlation between film-element pairs
        - fixed composition gives a linear correlation (coefficient ~1)
        - variable composition gives a nonlinear one (coefficient < 1 with structured residuals)

        Rough thresholds:
        - < 0.10: fixed composition (strong linear correlation)
        - 0.10-0.20: slight variation
        - 0.20-0.35: moderate variation (stoichiometric mode recommended)
        - > 0.35: strong variation (stoichiometry_bounds recommended)
        """
        n_angles, n_elements = I_norm.shape

        if n_elements < 2 or n_angles < 3:
            return 0.0

        # Find substrate candidates (signal near 0 at low angles, rising at high angles)
        substrate_candidates = set()
        for e in range(n_elements):
            # Signal at low angles (surface sensitive)
            low_angle_signal = I_norm[0, e]
            # Signal at high angles (depth sensitive)
            high_angle_signal = I_norm[-1, e]

            # Substrate pattern: small at the surface, large deep down
            if low_angle_signal < 0.1 and high_angle_signal > 0.3:
                substrate_candidates.add(e)
            # or an element that increases strongly
            elif low_angle_signal > 0.01 and high_angle_signal / (low_angle_signal + 1e-10) > 3:
                substrate_candidates.add(e)

        # Film elements (everything but the substrate)
        film_elements = [e for e in range(n_elements) if e not in substrate_candidates]

        if len(film_elements) < 2:
            return 0.0

        nonlinearities = []
        for i in range(len(film_elements)):
            for j in range(i + 1, len(film_elements)):
                ei, ej = film_elements[i], film_elements[j]

                xi = I_norm[:, ei]
                xj = I_norm[:, ej]

                # Check that both elements carry enough signal
                if xi.mean() < 0.03 or xj.mean() < 0.03:
                    continue

                # Residual of a linear fit
                try:
                    coeffs = np.polyfit(xi, xj, 1)
                    predicted = np.polyval(coeffs, xi)
                    residuals = xj - predicted

                    # Normalized residual (RMSE / standard deviation)
                    rmse = np.sqrt(np.mean(residuals**2))
                    std_xj = np.std(xj)

                    if std_xj > 0.01:
                        # Nonlinearity indicator: larger residual = more nonlinear
                        nonlinearity = rmse / std_xj
                        if not np.isnan(nonlinearity) and not np.isinf(nonlinearity):
                            nonlinearities.append(nonlinearity)
                except (np.linalg.LinAlgError, ValueError):
                    continue

        return np.mean(nonlinearities) if nonlinearities else 0.0

    def _compute_layer_sharpness(self, I_norm: np.ndarray) -> float:
        """
        Compute the layer-sharpness score

        High → abrupt interfaces (well-defined layers)
        Low → gradual transitions (gradient, mixing)
        """
        # Second derivative (curvature) of the intensity change across angles
        if I_norm.shape[0] < 3:
            return 0.5  # not enough data

        # Second derivative for each element
        curvatures = []
        for e in range(I_norm.shape[1]):
            profile = I_norm[:, e]
            if len(profile) >= 3:
                second_deriv = np.diff(profile, n=2)
                curvature = np.max(np.abs(second_deriv))
                curvatures.append(curvature)

        if curvatures:
            # High curvature → abrupt interface
            max_curv = max(curvatures)
            score = min(max_curv * 10, 1.0)  # scaling
        else:
            score = 0.5

        return score

    def _estimate_layer_count(self, I_norm: np.ndarray) -> int:
        """
        Estimated number of layers

        Estimated from the number of inflection points of the angle profiles
        """
        n_elements = I_norm.shape[1]

        # Detect inflection points for each element
        inflection_counts = []
        for e in range(n_elements):
            profile = I_norm[:, e]
            if len(profile) >= 3:
                # Count sign changes of the first derivative
                diff1 = np.diff(profile)
                sign_changes = np.sum(np.abs(np.diff(np.sign(diff1))) > 0)
                inflection_counts.append(sign_changes)

        if inflection_counts:
            # Maximum number of inflection points + 1 = estimated layers
            n_layers = max(inflection_counts) + 1
        else:
            n_layers = 1

        return max(1, min(n_layers, 10))  # range 1-10

    def _determine_structure_type(
        self,
        angle_dep: float,
        stoich: float,
        sharpness: float,
        n_layers: int,
    ) -> StructureType:
        """Decide the structure type"""

        # Almost no angle dependence → uniform mixture or substrate only
        if angle_dep < self.angle_dep_threshold:
            if stoich > 0.8:
                # Stoichiometry holds → single-compound substrate
                return StructureType.SUBSTRATE_ONLY
            elif stoich > 0.5:
                # Some correlation → uniform mixture
                return StructureType.UNIFORM_MIXED
            else:
                # Low correlation → completely unknown
                return StructureType.UNKNOWN

        # High stoichiometry and sharp boundaries → well-defined layers
        if stoich > self.stoich_threshold and sharpness > self.layer_sharp_threshold:
            return StructureType.WELL_DEFINED_LAYERS

        # High stoichiometry but unclear boundaries → mixed / gradient
        if stoich > self.stoich_threshold * 0.7:
            return StructureType.MIXED_GRADIENT

        # Otherwise → unknown
        return StructureType.UNKNOWN

    def _recommend_solver(
        self,
        structure_type: StructureType,
        angle_dep: float,
        stoich: float,
        sharpness: float,
    ) -> Tuple[SolverRecommendation, float]:
        """Recommend a solver"""

        if structure_type == StructureType.WELL_DEFINED_LAYERS:
            # Well-defined layers → GenericLayerSolver
            confidence = min((stoich + sharpness) / 2, 1.0)
            return SolverRecommendation.GENERIC_LAYER, confidence

        elif structure_type == StructureType.MIXED_GRADIENT:
            # Mixed / gradient → SemiFreeSolver
            confidence = 0.5 + stoich * 0.3
            return SolverRecommendation.SEMI_FREE, confidence

        elif structure_type == StructureType.SUBSTRATE_ONLY:
            # Substrate only → GenericLayerSolver (single layer) for verification
            return SolverRecommendation.GENERIC_LAYER, 0.7

        elif structure_type == StructureType.UNIFORM_MIXED:
            # Uniform mixture → no depth resolution possible; recommend ordinary XPS
            # No solver recommendation (FREE is returned instead of None, with low confidence)
            return SolverRecommendation.FREE, 0.1

        else:  # UNKNOWN
            # Completely unknown → FreeSolver
            confidence = 0.3 + angle_dep * 0.4
            return SolverRecommendation.FREE, confidence

    def _generate_feedback(
        self,
        structure_type: StructureType,
        angle_dep: float,
        stoich: float,
        sharpness: float,
        n_layers: int,
        stoich_var: float = 0.0,
    ) -> Tuple[List[str], List[str]]:
        """Generate warnings and suggestions"""
        warnings = []
        suggestions = []

        # Warnings
        if angle_dep < self.angle_dep_threshold:
            warnings.append(f"Very weak angle dependence (CV={angle_dep:.4f}): depth structure is hard to detect")

        if stoich < 0.3:
            warnings.append("Low stoichiometric consistency: possibly a complex mixed state")

        if n_layers > 5:
            warnings.append(f"Many estimated layers ({n_layers}): watch the convergence of the analysis")

        # Suggestions
        if structure_type == StructureType.UNIFORM_MIXED:
            warnings.append("Possibly a uniform mixture: ARXPS cannot resolve depth")
            suggestions.append("Quantitative analysis with ordinary XPS is recommended")
            suggestions.append("If a depth distribution is needed, consider ion-sputter depth profiling")
            suggestions.append("To confirm uniformity, measure over another angle range as well")

        elif structure_type == StructureType.UNKNOWN:
            suggestions.append("Start with an exploratory FreeSolver analysis")
            suggestions.append("Infer the layer structure from the result, then refine with GenericLayerSolver")

        elif structure_type == StructureType.MIXED_GRADIENT:
            suggestions.append("Set a larger interface width")
            suggestions.append("Check the composition distribution with SemiFreeSolver, then refine if needed")

        if sharpness < 0.3 and structure_type == StructureType.WELL_DEFINED_LAYERS:
            suggestions.append("The interface may be gradual: consider optimizing the interface width too")

        # Suggestions about variable composition
        # Thresholds are based on the nonlinearity indicator (0 = fully linear, 1 = fully nonlinear)
        # Note: automatic detection has limits; multilayer oxides (C/Al2O3/TiO2/SiO2/Si and the like)
        #       can be misdetected, so judge from an understanding of the system.
        if stoich_var > 0.5:
            suggestions.append(
                "High composition variability detected: consider a multilayer oxide too. "
                "For variable composition, give the range with stoichiometry_bounds"
            )
        elif stoich_var > 0.20:
            suggestions.append(
                "Moderate composition variability detected: SemiFreeSolver with stoichiometric_mode='stoichiometric' is recommended. "
                "For multilayer oxides, charge_balance is the right choice"
            )
        elif stoich_var > 0.08 and angle_dep > 0.1:
            suggestions.append(
                "Slight composition variability: consider variable composition (SiOx, NON/ONO, SiGe)"
            )

        return warnings, suggestions

    # ========== Substrate detection ==========

    def detect_substrate(
        self,
        I_obs: np.ndarray,
        angles: np.ndarray,
        element_labels: Optional[List[str]] = None,
        oxidation_states: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> SubstrateDetectionResult:
        """
        Detect the substrate automatically from angle-resolved data

        Detection algorithms:
        1. Slope method: the element with the most negative d(ln I)/d(1/cosθ) is the deepest
        2. Ratio method: the element with the smallest I(60°)/I(0°) is the deepest
        3. Compound substrate: elements with similar slopes are grouped

        Parameters
        ----------
        I_obs : ndarray
            Observed intensities [n_angles, n_elements]
        angles : ndarray
            Measurement angles (degrees)
        element_labels : List[str], optional
            Element labels
        oxidation_states : ndarray, optional
            Oxidation states (used to group a compound substrate)
        verbose : bool
            Verbose output

        Returns
        -------
        SubstrateDetectionResult
        """
        n_angles, n_elements = I_obs.shape

        if verbose:
            print("=" * 60)
            print("Substrate Detection")
            print("=" * 60)

        # Normalize
        I_norm = I_obs / np.maximum(I_obs.sum(axis=1, keepdims=True), 1e-10)

        # 1. Slope method (primary)
        slopes = self._compute_angle_slopes(I_norm, angles)

        # 2. Ratio method (secondary)
        ratios = self._compute_angle_ratios(I_norm, angles)

        if verbose:
            print(f"\nSlopes (d(ln I)/d(1/cosθ)):")
            for i, slope in enumerate(slopes):
                label = element_labels[i] if element_labels and i < len(element_labels) else f"elem{i}"
                print(f"  {label}: {slope:.3f}")
            print(f"\nRatios (I_60/I_0):")
            for i, ratio in enumerate(ratios):
                label = element_labels[i] if element_labels and i < len(element_labels) else f"elem{i}"
                print(f"  {label}: {ratio:.3f}")

        # 3. Identify the substrate candidates
        substrate_idx, substrate_group, confidence, method, has_substrate, fallback_reason = \
            self._identify_substrate(slopes, ratios, oxidation_states, element_labels, verbose)

        result = SubstrateDetectionResult(
            substrate_indices=[substrate_idx] if substrate_idx is not None else [],
            substrate_group=substrate_group,
            confidence=confidence,
            method=method,
            slopes=slopes,
            ratios=ratios,
            has_substrate=has_substrate,
            fallback_reason=fallback_reason,
        )

        if verbose:
            print(f"\nResult:")
            print(f"  substrate indices: {result.substrate_indices}")
            print(f"  substrate group: {result.substrate_group}")
            print(f"  confidence: {result.confidence:.2f}")
            print(f"  method: {result.method}")
            if not has_substrate:
                print(f"  ⚠️ fallback: {fallback_reason}")

        return result

    def _compute_angle_slopes(self, I_norm: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """
        Compute the slope of the angle dependence

        slope = d(ln I) / d(1/cosθ)
        Most negative slope = deepest position (substrate)
        """
        angles_rad = np.deg2rad(angles)
        x = 1.0 / np.cos(angles_rad)
        n_elements = I_norm.shape[1]

        slopes = []
        for e in range(n_elements):
            y = np.log(np.maximum(I_norm[:, e], 1e-10))
            # Linear regression
            slope, _ = np.polyfit(x, y, 1)
            slopes.append(slope)

        return np.array(slopes)

    def _compute_angle_ratios(self, I_norm: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """
        Compute the high-angle / low-angle intensity ratio

        ratio = I(high_angle) / I(low_angle)
        Smaller ratio = deeper position (substrate)
        """
        n_elements = I_norm.shape[1]

        # Indices of the smallest and largest angles
        idx_low = np.argmin(angles)
        idx_high = np.argmax(angles)

        ratios = []
        for e in range(n_elements):
            I_low = max(I_norm[idx_low, e], 1e-10)
            I_high = max(I_norm[idx_high, e], 1e-10)
            ratios.append(I_high / I_low)

        return np.array(ratios)

    def _identify_substrate(
        self,
        slopes: np.ndarray,
        ratios: np.ndarray,
        oxidation_states: Optional[np.ndarray],
        element_labels: Optional[List[str]],
        verbose: bool,
    ) -> Tuple[Optional[int], List[int], float, str, bool, Optional[str]]:
        """
        Identify the substrate candidates

        Returns
        -------
        substrate_idx : int or None
            Index of the main substrate element
        substrate_group : List[int]
            Substrate group (several elements for a compound substrate)
        confidence : float
            Detection confidence
        method : str
            Detection method used
        has_substrate : bool
            Whether a substrate was detected
        fallback_reason : str or None
            Reason for the fallback
        """
        n_elements = len(slopes)

        # ===== Fallback conditions =====

        # 1. All elements have similar slopes (possibly a uniform mixture)
        slope_range = np.max(slopes) - np.min(slopes)
        if slope_range < 0.1:
            # Uniform mixture → take the most concentrated element as a provisional substrate
            # (note: this is provisional; a user-specified substrate is recommended)
            return (
                None,
                list(range(n_elements)),
                0.1,
                "fallback_uniform",
                False,
                "All elements have a similar angle dependence (possibly a uniform mixture). Do not use the exponential prior; FreeSolver with a uniform prior is recommended."
            )

        # 2. All slopes positive (surface layer only, no substrate?)
        if np.all(slopes > -0.05):
            return (
                None,
                [],
                0.2,
                "fallback_no_substrate",
                False,
                "All elements are concentrated at the surface. The substrate signal may not be detected; the film may be thicker than the probing depth."
            )

        # ===== Normal substrate detection =====

        # Slope method: most negative slope = deepest
        slope_order = np.argsort(slopes)  # ascending
        substrate_idx = slope_order[0]  # most negative (deepest)

        # Group detection for a compound substrate
        min_slope = slopes[substrate_idx]
        threshold = min_slope * (1 - self.substrate_slope_margin)  # group the elements within the margin

        substrate_group = []
        for i in range(n_elements):
            if slopes[i] <= threshold:
                substrate_group.append(i)

        # Confidence
        # The larger the slope separation, the higher the confidence
        if len(substrate_group) == 1:
            # Single-element substrate
            second_min = slopes[slope_order[1]] if n_elements > 1 else min_slope
            separation = second_min - min_slope
            confidence = min(separation / 0.5, 1.0)  # a separation of 0.5 gives confidence 1.0
            method = "slope_single"
        else:
            # Compound substrate
            # Confidence from the agreement of the slopes within the group
            group_slopes = slopes[substrate_group]
            group_consistency = 1.0 - np.std(group_slopes) / (np.abs(np.mean(group_slopes)) + 1e-10)
            confidence = max(0.3, min(group_consistency, 0.9))  # range 0.3-0.9
            method = "slope_compound"

        # Cross-check with the ratio method
        ratio_order = np.argsort(ratios)  # ascending
        ratio_substrate = ratio_order[0]

        if ratio_substrate == substrate_idx:
            # Slope and ratio methods agree → higher confidence
            confidence = min(confidence + 0.1, 1.0)
            method += "+ratio_confirmed"

        if verbose:
            print(f"\nSubstrate detection details:")
            print(f"  slope order: {slope_order.tolist()}")
            print(f"  ratio order: {ratio_order.tolist()}")
            print(f"  group threshold: slope <= {threshold:.3f}")
            if element_labels:
                group_labels = [element_labels[i] for i in substrate_group if i < len(element_labels)]
                print(f"  detected group: {group_labels}")

        return substrate_idx, substrate_group, confidence, method, True, None

    def analyze_with_substrate_detection(
        self,
        I_obs: np.ndarray,
        angles: np.ndarray,
        element_labels: Optional[List[str]] = None,
        oxidation_states: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> AnalysisResult:
        """
        Complete structure analysis including substrate detection

        Parameters
        ----------
        I_obs : ndarray
            Observed intensities [n_angles, n_elements]
        angles : ndarray
            Measurement angles (degrees)
        element_labels : List[str], optional
            Element labels
        oxidation_states : ndarray, optional
            Oxidation states
        verbose : bool
            Verbose output

        Returns
        -------
        AnalysisResult
            Analysis result including the substrate detection
        """
        # Basic analysis
        result = self.analyze(I_obs, element_labels=element_labels, verbose=verbose)

        # Substrate detection
        substrate_result = self.detect_substrate(
            I_obs, angles, element_labels, oxidation_states, verbose
        )
        result.substrate_detection = substrate_result

        # Extra suggestions from the substrate detection
        if not substrate_result.has_substrate:
            result.warnings.append(f"Substrate detection failed: {substrate_result.fallback_reason}")
            result.suggestions.append("Specify the substrate index by hand")

        return result


def demonstrate_analyzer():
    """Demonstration"""
    from scipy.special import erf

    print("=" * 70)
    print("Structure Analyzer Demonstration")
    print("=" * 70)

    analyzer = StructureAnalyzer()

    # Case 1: well-defined layers (SiO2/Si)
    print("\n>>> Case 1: Well-defined SiO2/Si layer")
    angles = np.array([15, 30, 45, 60, 75, 85])
    n_angles = len(angles)

    # Simulate the angle dependence (more surface layer → more deep layer)
    surface_ratio = 1 - 0.3 * np.sin(np.radians(angles))  # surface sensitivity
    I_sio2 = np.column_stack([
        0.33 * surface_ratio,      # Si_ox
        0.67 * surface_ratio,      # O
        1 - surface_ratio,         # Si_bulk
    ])
    I_sio2 = I_sio2 / I_sio2.sum(axis=1, keepdims=True)

    result1 = analyzer.analyze(I_sio2, element_labels=['Si_ox', 'O', 'Si_bulk'], verbose=True)

    # Case 2: mixed / gradient
    print("\n>>> Case 2: Mixed/Gradient structure")
    I_mixed = np.column_stack([
        0.2 + 0.1 * np.random.randn(n_angles),  # mixed element 1
        0.3 + 0.15 * np.random.randn(n_angles),  # mixed element 2
        0.5 + 0.1 * np.random.randn(n_angles),   # substrate
    ])
    I_mixed = np.clip(I_mixed, 0.01, None)
    I_mixed = I_mixed / I_mixed.sum(axis=1, keepdims=True)

    result2 = analyzer.analyze(I_mixed, element_labels=['A', 'B', 'Sub'], verbose=True)

    # Case 3: completely unknown (random)
    print("\n>>> Case 3: Unknown/Random structure")
    np.random.seed(123)
    I_random = np.random.rand(n_angles, 4)
    I_random = I_random / I_random.sum(axis=1, keepdims=True)

    result3 = analyzer.analyze(I_random, element_labels=['X', 'Y', 'Z', 'W'], verbose=True)

    # Case 4: 4-layer film
    print("\n>>> Case 4: 4-layer multilayer (C/Al2O3/TiO2/SiO2/Si)")
    # The composition changes from the surface downward
    depth_sensitivity = 1 - 0.4 * np.sin(np.radians(angles))
    I_multilayer = np.column_stack([
        0.15 * depth_sensitivity**2,              # C (surface)
        0.10 * depth_sensitivity * (1-depth_sensitivity*0.3),  # Al
        0.15 * (1-depth_sensitivity*0.5),         # Ti
        0.10 * (1-depth_sensitivity*0.7),         # Si_ox
        0.30 * (1-depth_sensitivity*0.3),         # O
        0.20 * (1-depth_sensitivity),             # Si_bulk
    ])
    I_multilayer = np.clip(I_multilayer, 0.01, None)
    I_multilayer = I_multilayer / I_multilayer.sum(axis=1, keepdims=True)

    result4 = analyzer.analyze(
        I_multilayer,
        element_labels=['C', 'Al', 'Ti', 'Si_ox', 'O', 'Si_bulk'],
        verbose=True
    )

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Case':<20} {'Structure Type':<25} {'Recommended Solver':<20} {'Confidence':<10}")
    print("-" * 75)
    for i, (name, result) in enumerate([
        ("SiO2/Si layer", result1),
        ("Mixed/Gradient", result2),
        ("Unknown/Random", result3),
        ("4-layer multilayer", result4),
    ], 1):
        print(f"{name:<20} {result.structure_type.value:<25} "
              f"{result.solver_recommendation.value:<20} {result.confidence:.2f}")


if __name__ == "__main__":
    demonstrate_analyzer()
