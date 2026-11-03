"""
ARXPS Analysis Pipeline - automatic analysis pipeline
===============================================

A pipeline that chains StructureAnalyzer → automatic run of the appropriate solver →
result verification.

The aim is to run from structure determination to solver execution automatically, without human or LLM judgment in between.

Usage:
    from deppro.pipeline import ARXPSPipeline

    pipeline = ARXPSPipeline()
    result = pipeline.analyze(I_obs, tm, element_labels)

    if result.success:
        print(f"Thicknesses: {result.layer_thicknesses}")
    else:
        print(f"Analysis failed: {result.failure_reason}")

Date: 2026-01-24
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from enum import Enum
import time

from .transfer_matrix import TransferMatrix
from .structure_analyzer import (
    StructureAnalyzer,
    StructureType,
    SolverRecommendation,
    AnalysisResult,
)
from .inverse_solver import (
    FreeSolver,
    SemiFreeSolver,
    SemiFreeSolverConfig,
    GenericLayerSolver,
    GenericLayerConfig,
    LayerDefinition,
    SolverConfig,
    SolverResult,
)
from .depth_profile import ElementConfig


class ValidationStatus(Enum):
    """Verification status"""
    PASSED = "passed"               # every check passed
    CHI2_WARNING = "chi2_warning"   # Chi² somewhat high
    CHI2_FAILED = "chi2_failed"     # Chi² above the threshold
    CONVERGENCE_FAILED = "convergence_failed"  # did not converge
    PHYSICAL_INVALID = "physical_invalid"      # physically implausible


@dataclass
class ValidationResult:
    """Verification result"""
    status: ValidationStatus
    chi2: float
    chi2_threshold: float
    chi2_normalized: float  # Chi² / n_data
    convergence_ok: bool
    physical_checks: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == ValidationStatus.PASSED


@dataclass
class PipelineResult:
    """Pipeline result"""
    # Success / failure
    success: bool
    failure_reason: Optional[str] = None

    # Structure analysis result
    structure_analysis: Optional[AnalysisResult] = None

    # Solver result
    solver_used: Optional[str] = None
    solver_result: Optional[Any] = None  # SolverResult or GenericLayerResult

    # Verification result
    validation: Optional[ValidationResult] = None

    # Extracted information
    layer_thicknesses: Optional[List[float]] = None
    interface_widths: Optional[List[float]] = None
    D_reconstructed: Optional[np.ndarray] = None
    I_fitted: Optional[np.ndarray] = None

    # Computation time
    elapsed_time: float = 0.0

    # Additional information
    suggestions: List[str] = field(default_factory=list)

    def __repr__(self):
        if self.success:
            chi2_str = f"{self.validation.chi2:.4f}" if self.validation else "N/A"
            return (f"PipelineResult(success=True, "
                    f"solver={self.solver_used}, "
                    f"thicknesses={self.layer_thicknesses}, "
                    f"chi2={chi2_str})")
        else:
            return f"PipelineResult(success=False, reason='{self.failure_reason}')"


@dataclass
class PipelineConfig:
    """Pipeline settings"""
    # Structure analysis
    angle_dep_threshold: float = 0.05
    stoich_threshold: float = 0.8

    # Solver
    solver_method: str = 'LLS'  # 'LLS', 'L2', 'MEM'
    scipy_method: str = 'SLSQP'
    max_iter: int = 100

    # Initial values
    default_thickness: float = 30.0
    default_width: float = 5.0

    # Verification
    chi2_threshold: float = 0.1       # passes when Chi²/n_data < 0.1
    chi2_warning_threshold: float = 0.05  # no warning at or below this
    min_thickness: float = 1.0        # minimum thickness (Å)
    max_thickness: float = 200.0      # maximum thickness (Å)

    # Automatic settings
    auto_detect_layers: bool = True   # detect the layer structure automatically


class ARXPSPipeline:
    """
    Automatic ARXPS analysis pipeline

    1. Analyze the data with StructureAnalyzer
    2. Choose and run the appropriate solver
    3. Verify the result
    4. Generate a report
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Parameters
        ----------
        config : PipelineConfig, optional
            Pipeline settings
        """
        self.config = config or PipelineConfig()
        self.structure_analyzer = StructureAnalyzer(
            angle_dep_threshold=self.config.angle_dep_threshold,
            stoich_threshold=self.config.stoich_threshold,
        )

    def analyze(
        self,
        I_obs: np.ndarray,
        tm: TransferMatrix,
        elements: List[ElementConfig],
        element_labels: Optional[List[str]] = None,
        layer_config: Optional[GenericLayerConfig] = None,
        verbose: bool = False,
    ) -> PipelineResult:
        """
        Run the ARXPS analysis

        Parameters
        ----------
        I_obs : ndarray
            Observed intensities [n_angles, n_elements]
        tm : TransferMatrix
            Transfer matrix
        elements : List[ElementConfig]
            Element configuration
        element_labels : List[str], optional
            Element labels (for display)
        layer_config : GenericLayerConfig, optional
            Layer structure (GenericLayerSolver is used when given)
        verbose : bool
            Verbose output

        Returns
        -------
        PipelineResult
        """
        start_time = time.time()

        if verbose:
            print("=" * 70)
            print("ARXPS Analysis Pipeline")
            print("=" * 70)

        # Labels
        if element_labels is None:
            element_labels = [f"elem{i}" for i in range(I_obs.shape[1])]

        # Step 1: structure analysis
        if verbose:
            print("\n[Step 1] Structure Analysis")
            print("-" * 40)

        analysis = self.structure_analyzer.analyze(
            I_obs, tm, element_labels, verbose=verbose
        )

        # Step 2: choose and run the solver
        if verbose:
            print("\n[Step 2] Solver Execution")
            print("-" * 40)
            print(f"Recommended solver: {analysis.solver_recommendation.value}")

        # Use GenericLayerSolver when a layer structure is given
        if layer_config is not None:
            solver_result, solver_name = self._run_generic_layer_solver(
                I_obs, tm, layer_config, verbose
            )
        else:
            solver_result, solver_name = self._run_recommended_solver(
                I_obs, tm, elements, analysis, verbose
            )

        if solver_result is None:
            elapsed = time.time() - start_time
            return PipelineResult(
                success=False,
                failure_reason="Solver execution failed",
                structure_analysis=analysis,
                elapsed_time=elapsed,
            )

        # Step 3: verify the result
        if verbose:
            print("\n[Step 3] Validation")
            print("-" * 40)

        validation = self._validate_result(I_obs, solver_result, verbose)

        # Step 4: organize the result
        elapsed = time.time() - start_time

        # Extract thicknesses and interface widths
        layer_thicknesses = None
        interface_widths = None
        D_reconstructed = None
        I_fitted = None

        if hasattr(solver_result, 'layer_thicknesses'):
            layer_thicknesses = list(solver_result.layer_thicknesses)
        if hasattr(solver_result, 'interface_widths'):
            interface_widths = list(solver_result.interface_widths)
        if hasattr(solver_result, 'D_opt'):
            D_reconstructed = solver_result.D_opt
        if hasattr(solver_result, 'I_opt'):
            I_fitted = solver_result.I_opt

        # Success
        success = validation.passed or validation.status == ValidationStatus.CHI2_WARNING

        # Suggestions
        suggestions = self._generate_suggestions(
            analysis, validation, solver_name
        )

        result = PipelineResult(
            success=success,
            failure_reason=None if success else f"Validation failed: {validation.status.value}",
            structure_analysis=analysis,
            solver_used=solver_name,
            solver_result=solver_result,
            validation=validation,
            layer_thicknesses=layer_thicknesses,
            interface_widths=interface_widths,
            D_reconstructed=D_reconstructed,
            I_fitted=I_fitted,
            elapsed_time=elapsed,
            suggestions=suggestions,
        )

        if verbose:
            print("\n[Result Summary]")
            print("-" * 40)
            print(f"Success: {success}")
            print(f"Solver: {solver_name}")
            if layer_thicknesses:
                print(f"Layer thicknesses: {[f'{t:.2f}' for t in layer_thicknesses]} Å")
            if interface_widths:
                print(f"Interface widths: {[f'{w:.2f}' for w in interface_widths]} Å")
            print(f"Chi²/n: {validation.chi2_normalized:.4f}")
            print(f"Elapsed time: {elapsed*1000:.1f} ms")
            if suggestions:
                print(f"Suggestions: {suggestions}")

        return result

    def _run_generic_layer_solver(
        self,
        I_obs: np.ndarray,
        tm: TransferMatrix,
        layer_config: GenericLayerConfig,
        verbose: bool,
    ) -> Tuple[Optional[Any], str]:
        """Run GenericLayerSolver"""
        try:
            solver_config = SolverConfig(
                method=self.config.solver_method,
                scipy_method=self.config.scipy_method,
                max_iter=self.config.max_iter,
            )
            solver = GenericLayerSolver(tm, solver_config, layer_config)
            result = solver.solve(I_obs, verbose=verbose)

            if verbose:
                print(f"GenericLayerSolver completed")
                print(f"  Thicknesses: {result.layer_thicknesses}")
                print(f"  Chi²: {result.chi2:.4f}")

            return result, "GenericLayerSolver"

        except Exception as e:
            if verbose:
                print(f"GenericLayerSolver failed: {e}")
            return None, "GenericLayerSolver"

    def _run_recommended_solver(
        self,
        I_obs: np.ndarray,
        tm: TransferMatrix,
        elements: List[ElementConfig],
        analysis: AnalysisResult,
        verbose: bool,
    ) -> Tuple[Optional[Any], str]:
        """Run the recommended solver"""

        recommendation = analysis.solver_recommendation

        if recommendation == SolverRecommendation.GENERIC_LAYER:
            # Infer the layer structure automatically and run GenericLayerSolver
            layer_config = self._auto_detect_layer_config(I_obs, tm, elements, analysis)
            if layer_config:
                return self._run_generic_layer_solver(I_obs, tm, layer_config, verbose)
            # Fall back to SemiFreeSolver when automatic detection fails
            recommendation = SolverRecommendation.SEMI_FREE

        if recommendation == SolverRecommendation.SEMI_FREE:
            return self._run_semi_free_solver(I_obs, tm, elements, verbose)

        if recommendation == SolverRecommendation.FREE:
            return self._run_free_solver(I_obs, tm, elements, verbose)

        return None, "Unknown"

    def _run_semi_free_solver(
        self,
        I_obs: np.ndarray,
        tm: TransferMatrix,
        elements: List[ElementConfig],
        verbose: bool,
    ) -> Tuple[Optional[Any], str]:
        """Run SemiFreeSolver"""
        try:
            # Simplified settings (assumes a two-element system)
            n_elements = I_obs.shape[1]

            # Classify into the oxide (first element) and the rest
            free_indices = [0] if n_elements > 1 else []
            linked_indices = [1] if n_elements > 2 else []
            substrate_indices = [n_elements - 1] if n_elements > 0 else [0]

            semi_config = SemiFreeSolverConfig(
                thickness_init=self.config.default_thickness,
                abruptness_init=self.config.default_width,
                free_indices=free_indices,
                linked_indices=linked_indices,
                substrate_indices=substrate_indices,
            )

            solver_config = SolverConfig(
                method=self.config.solver_method,
                scipy_method=self.config.scipy_method,
                max_iter=self.config.max_iter,
            )

            solver = SemiFreeSolver(tm, solver_config, semi_config, elements=elements)
            result = solver.solve(I_obs, verbose=verbose)

            if verbose:
                print(f"SemiFreeSolver completed")
                print(f"  Thickness: {result.thickness:.2f} Å")
                print(f"  Chi²: {result.chi2:.4f}")

            return result, "SemiFreeSolver"

        except Exception as e:
            if verbose:
                print(f"SemiFreeSolver failed: {e}")
            return None, "SemiFreeSolver"

    def _run_free_solver(
        self,
        I_obs: np.ndarray,
        tm: TransferMatrix,
        elements: List[ElementConfig],
        verbose: bool,
    ) -> Tuple[Optional[Any], str]:
        """Run FreeSolver"""
        try:
            solver_config = SolverConfig(
                method='L2',  # FreeSolver needs regularization
                alpha=0.01,
                scipy_method=self.config.scipy_method,
                max_iter=self.config.max_iter,
            )

            solver = FreeSolver(tm, solver_config, elements=elements)
            result = solver.solve(I_obs, verbose=verbose)

            if verbose:
                print(f"FreeSolver completed")
                print(f"  Chi²: {result.chi2:.4f}")

            return result, "FreeSolver"

        except Exception as e:
            if verbose:
                print(f"FreeSolver failed: {e}")
            return None, "FreeSolver"

    def _auto_detect_layer_config(
        self,
        I_obs: np.ndarray,
        tm: TransferMatrix,
        elements: List[ElementConfig],
        analysis: AnalysisResult,
    ) -> Optional[GenericLayerConfig]:
        """
        Detect the layer structure automatically from the element information

        Handles only simple cases (SiO2/Si and the like).
        Returns None for complex cases (fallback).
        """
        n_elements = len(elements)

        # Infer the layer structure from the oxidation states
        oxide_elements = []
        substrate_elements = []

        for i, elem in enumerate(elements):
            if elem.oxidation_state is not None:
                if elem.oxidation_state > 0:
                    oxide_elements.append((i, elem))
                elif elem.oxidation_state == 0:
                    substrate_elements.append((i, elem))
                else:
                    # Anions (O, Cl, ...)
                    pass

        # Only simple two-layer systems like SiO2/Si
        if len(oxide_elements) == 1 and len(substrate_elements) == 1:
            oxide_idx, oxide_elem = oxide_elements[0]
            sub_idx, sub_elem = substrate_elements[0]

            # Look for oxygen
            o_idx = None
            for i, elem in enumerate(elements):
                if elem.symbol == 'O' and elem.oxidation_state and elem.oxidation_state < 0:
                    o_idx = i
                    break

            if o_idx is not None:
                # SiO2/Si-type settings
                layer_def = LayerDefinition(
                    name=oxide_elem.compound or f"{oxide_elem.symbol}_oxide",
                    element_indices=[oxide_idx, o_idx],
                    stoichiometry={oxide_idx: 1, o_idx: 2},
                )
                substrate_def = LayerDefinition(
                    name=sub_elem.compound or sub_elem.symbol,
                    element_indices=[sub_idx],
                    stoichiometry={sub_idx: 1},
                )

                return GenericLayerConfig(
                    layers=[layer_def],
                    substrate=substrate_def,
                    initial_thicknesses=[self.config.default_thickness],
                    initial_widths=[self.config.default_width],
                    linked_element_index=o_idx,
                )

        # Automatic detection not possible
        return None

    def _validate_result(
        self,
        I_obs: np.ndarray,
        solver_result: Any,
        verbose: bool,
    ) -> ValidationResult:
        """Verify the result"""
        warnings = []
        physical_checks = {}

        # Chi²
        if hasattr(solver_result, 'chi2'):
            chi2 = solver_result.chi2
        else:
            chi2 = float('inf')

        n_data = I_obs.size
        chi2_normalized = chi2 / n_data

        # Convergence
        if hasattr(solver_result, 'success'):
            convergence_ok = solver_result.success
        else:
            convergence_ok = True  # success by default

        # Physical plausibility
        if hasattr(solver_result, 'layer_thicknesses'):
            thicknesses = solver_result.layer_thicknesses
            physical_checks['thickness_positive'] = all(t > 0 for t in thicknesses)
            physical_checks['thickness_reasonable'] = all(
                self.config.min_thickness < t < self.config.max_thickness
                for t in thicknesses
            )
            if not physical_checks['thickness_positive']:
                warnings.append("Negative thickness detected")
            if not physical_checks['thickness_reasonable']:
                warnings.append(f"Thickness outside the plausible range ({self.config.min_thickness}-{self.config.max_thickness} Å)")

        if hasattr(solver_result, 'D_opt'):
            D = solver_result.D_opt
            physical_checks['concentration_bounded'] = np.all(D >= 0) and np.all(D <= 1)
            physical_checks['sum_to_one'] = np.allclose(D.sum(axis=1), 1.0, atol=0.01)
            if not physical_checks['concentration_bounded']:
                warnings.append("Concentration outside 0-1")
            if not physical_checks['sum_to_one']:
                warnings.append("Concentrations do not sum to 1")

        # Status
        if not convergence_ok:
            status = ValidationStatus.CONVERGENCE_FAILED
        elif not all(physical_checks.values()) if physical_checks else False:
            status = ValidationStatus.PHYSICAL_INVALID
        elif chi2_normalized > self.config.chi2_threshold:
            status = ValidationStatus.CHI2_FAILED
        elif chi2_normalized > self.config.chi2_warning_threshold:
            status = ValidationStatus.CHI2_WARNING
            warnings.append(f"Chi²/n = {chi2_normalized:.4f} (somewhat high)")
        else:
            status = ValidationStatus.PASSED

        if verbose:
            print(f"Validation status: {status.value}")
            print(f"Chi²/n: {chi2_normalized:.4f} (threshold: {self.config.chi2_threshold})")
            print(f"Convergence: {'OK' if convergence_ok else 'FAILED'}")
            if physical_checks:
                print(f"Physical checks: {physical_checks}")
            if warnings:
                print(f"Warnings: {warnings}")

        return ValidationResult(
            status=status,
            chi2=chi2,
            chi2_threshold=self.config.chi2_threshold,
            chi2_normalized=chi2_normalized,
            convergence_ok=convergence_ok,
            physical_checks=physical_checks,
            warnings=warnings,
        )

    def _generate_suggestions(
        self,
        analysis: AnalysisResult,
        validation: ValidationResult,
        solver_name: str,
    ) -> List[str]:
        """Generate suggestions"""
        suggestions = []

        # Suggestions from the structure analysis
        suggestions.extend(analysis.suggestions)

        # Suggestions from the verification
        if validation.status == ValidationStatus.CHI2_FAILED:
            suggestions.append("High Chi²: the model may be inappropriate")
            if solver_name == "GenericLayerSolver":
                suggestions.append("Consider more compositional freedom with SemiFreeSolver")
            suggestions.append("Consider adding or revising the measurement angles")

        if validation.status == ValidationStatus.CHI2_WARNING:
            suggestions.append("The fit succeeded, but there is room for improvement")

        if validation.status == ValidationStatus.CONVERGENCE_FAILED:
            suggestions.append("Revise the initial values and run again")
            suggestions.append("Check the interface-width settings")

        if validation.status == ValidationStatus.PHYSICAL_INVALID:
            suggestions.append("Physically implausible result: the model needs revision")

        return suggestions


def run_pipeline_demo():
    """Pipeline demonstration"""
    from scipy.special import erf

    print("=" * 70)
    print("ARXPS Pipeline Demo")
    print("=" * 70)

    # Test data (SiO2/Si)
    depth = np.arange(0, 150, 1.0)
    n_depth = len(depth)
    thickness = 30.0
    width = 5.0

    sigma = width / (2 * np.sqrt(np.log(2)))
    D = np.zeros((n_depth, 3))

    sio2_profile = 1 - 0.5 * (1 + erf((depth - thickness) / sigma))
    D[:, 0] = (1/3) * sio2_profile  # Si_ox
    D[:, 1] = (2/3) * sio2_profile  # O
    D[:, 2] = 0.5 * (1 + erf((depth - thickness) / sigma))  # Si_bulk

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)

    elements = [
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    angles = [15, 30, 45, 60, 75, 85]
    from .transfer_matrix import TransferMatrix
    tm = TransferMatrix(depth=depth, step_size=1.0, angles=angles, photon_energy=1486.6)
    tm.compute(D, elements)

    I_true = tm.forward(D, normalize=True)

    # Add noise
    np.random.seed(42)
    noise = 0.02 * np.random.randn(*I_true.shape)
    I_obs = I_true + noise * I_true
    I_obs = np.clip(I_obs, 1e-6, None)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    # Run the pipeline
    print("\n>>> Case 1: Auto-detection (SiO2/Si)")
    pipeline = ARXPSPipeline()
    result = pipeline.analyze(
        I_obs, tm, elements,
        element_labels=['Si_ox', 'O', 'Si_bulk'],
        verbose=True
    )

    print(f"\n>>> True thickness: {thickness:.1f} Å")
    if result.layer_thicknesses:
        print(f">>> Estimated thickness: {result.layer_thicknesses[0]:.2f} Å")
        print(f">>> Error: {abs(result.layer_thicknesses[0] - thickness):.2f} Å")

    # Explicit layer structure
    print("\n" + "=" * 70)
    print(">>> Case 2: Explicit layer config")
    print("=" * 70)

    layer_config = GenericLayerConfig(
        layers=[
            LayerDefinition(
                name="SiO2",
                element_indices=[0, 1],
                stoichiometry={0: 1, 1: 2},
            ),
        ],
        substrate=LayerDefinition(
            name="Si",
            element_indices=[2],
            stoichiometry={2: 1},
        ),
        initial_thicknesses=[25.0],  # slightly off initial value
        initial_widths=[5.0],
        linked_element_index=1,
    )

    result2 = pipeline.analyze(
        I_obs, tm, elements,
        element_labels=['Si_ox', 'O', 'Si_bulk'],
        layer_config=layer_config,
        verbose=True
    )

    print(f"\n>>> True thickness: {thickness:.1f} Å")
    if result2.layer_thicknesses:
        print(f">>> Estimated thickness: {result2.layer_thicknesses[0]:.2f} Å")
        print(f">>> Error: {abs(result2.layer_thicknesses[0] - thickness):.2f} Å")


if __name__ == "__main__":
    run_pipeline_demo()
