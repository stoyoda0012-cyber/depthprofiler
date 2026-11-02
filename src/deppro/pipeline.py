"""
ARXPS Analysis Pipeline - 自動解析パイプライン
===============================================

StructureAnalyzer → 適切なソルバー自動実行 → 結果検証
の一連の流れを統合したパイプライン。

人手や LLM の判断を挟まずに、構造の判定からソルバーの実行までを自動で回すことを目指す。

Usage:
    from deppro.pipeline import ARXPSPipeline

    pipeline = ARXPSPipeline()
    result = pipeline.analyze(I_obs, tm, element_labels)

    if result.success:
        print(f"膜厚: {result.layer_thicknesses}")
    else:
        print(f"解析失敗: {result.failure_reason}")

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
    """検証ステータス"""
    PASSED = "passed"               # 全検証を通過
    CHI2_WARNING = "chi2_warning"   # Chi²がやや高い
    CHI2_FAILED = "chi2_failed"     # Chi²が閾値超過
    CONVERGENCE_FAILED = "convergence_failed"  # 収束失敗
    PHYSICAL_INVALID = "physical_invalid"      # 物理的に妥当でない


@dataclass
class ValidationResult:
    """検証結果"""
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
    """パイプライン結果"""
    # 成功/失敗
    success: bool
    failure_reason: Optional[str] = None

    # 構造解析結果
    structure_analysis: Optional[AnalysisResult] = None

    # ソルバー結果
    solver_used: Optional[str] = None
    solver_result: Optional[Any] = None  # SolverResult or GenericLayerResult

    # 検証結果
    validation: Optional[ValidationResult] = None

    # 抽出された情報
    layer_thicknesses: Optional[List[float]] = None
    interface_widths: Optional[List[float]] = None
    D_reconstructed: Optional[np.ndarray] = None
    I_fitted: Optional[np.ndarray] = None

    # 計算時間
    elapsed_time: float = 0.0

    # 追加情報
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
    """パイプライン設定"""
    # 構造解析設定
    angle_dep_threshold: float = 0.05
    stoich_threshold: float = 0.8

    # ソルバー設定
    solver_method: str = 'LLS'  # 'LLS', 'L2', 'MEM'
    scipy_method: str = 'SLSQP'
    max_iter: int = 100

    # 初期値設定
    default_thickness: float = 30.0
    default_width: float = 5.0

    # 検証設定
    chi2_threshold: float = 0.1       # Chi²/n_data < 0.1 で合格
    chi2_warning_threshold: float = 0.05  # この値以下で警告なし
    min_thickness: float = 1.0        # 最小膜厚 (Å)
    max_thickness: float = 200.0      # 最大膜厚 (Å)

    # 自動設定
    auto_detect_layers: bool = True   # 層構造を自動検出


class ARXPSPipeline:
    """
    ARXPS自動解析パイプライン

    1. StructureAnalyzerでデータを解析
    2. 適切なソルバーを選択・実行
    3. 結果を検証
    4. レポートを生成
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Parameters
        ----------
        config : PipelineConfig, optional
            パイプライン設定
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
        ARXPS解析を実行

        Parameters
        ----------
        I_obs : ndarray
            観測強度 [n_angles, n_elements]
        tm : TransferMatrix
            転送行列
        elements : List[ElementConfig]
            元素設定
        element_labels : List[str], optional
            元素ラベル（表示用）
        layer_config : GenericLayerConfig, optional
            層構造設定（指定時はGenericLayerSolverを使用）
        verbose : bool
            詳細出力

        Returns
        -------
        PipelineResult
        """
        start_time = time.time()

        if verbose:
            print("=" * 70)
            print("ARXPS Analysis Pipeline")
            print("=" * 70)

        # ラベルの設定
        if element_labels is None:
            element_labels = [f"elem{i}" for i in range(I_obs.shape[1])]

        # Step 1: 構造解析
        if verbose:
            print("\n[Step 1] Structure Analysis")
            print("-" * 40)

        analysis = self.structure_analyzer.analyze(
            I_obs, tm, element_labels, verbose=verbose
        )

        # Step 2: ソルバー選択・実行
        if verbose:
            print("\n[Step 2] Solver Execution")
            print("-" * 40)
            print(f"Recommended solver: {analysis.solver_recommendation.value}")

        # 層構造が指定されている場合はGenericLayerSolverを使用
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

        # Step 3: 結果検証
        if verbose:
            print("\n[Step 3] Validation")
            print("-" * 40)

        validation = self._validate_result(I_obs, solver_result, verbose)

        # Step 4: 結果の整理
        elapsed = time.time() - start_time

        # 膜厚・界面幅の抽出
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

        # 成功判定
        success = validation.passed or validation.status == ValidationStatus.CHI2_WARNING

        # 提案の生成
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
        """GenericLayerSolverを実行"""
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
        """推奨ソルバーを実行"""

        recommendation = analysis.solver_recommendation

        if recommendation == SolverRecommendation.GENERIC_LAYER:
            # 層構造を自動推定してGenericLayerSolverを実行
            layer_config = self._auto_detect_layer_config(I_obs, tm, elements, analysis)
            if layer_config:
                return self._run_generic_layer_solver(I_obs, tm, layer_config, verbose)
            # 自動検出失敗時はSemiFreeSolverにフォールバック
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
        """SemiFreeSolverを実行"""
        try:
            # 簡易的な設定（2元素系を想定）
            n_elements = I_obs.shape[1]

            # 酸化物（最初の元素）とその他を分類
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
        """FreeSolverを実行"""
        try:
            solver_config = SolverConfig(
                method='L2',  # FreeSolverには正則化が必要
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
        元素情報から層構造を自動検出

        単純なケース（SiO2/Si等）のみ対応。
        複雑な場合はNoneを返す（フォールバック）。
        """
        n_elements = len(elements)

        # 酸化状態から層構造を推定
        oxide_elements = []
        substrate_elements = []

        for i, elem in enumerate(elements):
            if elem.oxidation_state is not None:
                if elem.oxidation_state > 0:
                    oxide_elements.append((i, elem))
                elif elem.oxidation_state == 0:
                    substrate_elements.append((i, elem))
                else:
                    # 陰イオン（O, Cl等）
                    pass

        # SiO2/Si のような単純な2層系のみ対応
        if len(oxide_elements) == 1 and len(substrate_elements) == 1:
            oxide_idx, oxide_elem = oxide_elements[0]
            sub_idx, sub_elem = substrate_elements[0]

            # 酸素を探す
            o_idx = None
            for i, elem in enumerate(elements):
                if elem.symbol == 'O' and elem.oxidation_state and elem.oxidation_state < 0:
                    o_idx = i
                    break

            if o_idx is not None:
                # SiO2/Si型の設定
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

        # 自動検出できない場合
        return None

    def _validate_result(
        self,
        I_obs: np.ndarray,
        solver_result: Any,
        verbose: bool,
    ) -> ValidationResult:
        """結果を検証"""
        warnings = []
        physical_checks = {}

        # Chi²の計算
        if hasattr(solver_result, 'chi2'):
            chi2 = solver_result.chi2
        else:
            chi2 = float('inf')

        n_data = I_obs.size
        chi2_normalized = chi2 / n_data

        # 収束状況
        if hasattr(solver_result, 'success'):
            convergence_ok = solver_result.success
        else:
            convergence_ok = True  # デフォルトは成功

        # 物理的妥当性チェック
        if hasattr(solver_result, 'layer_thicknesses'):
            thicknesses = solver_result.layer_thicknesses
            physical_checks['thickness_positive'] = all(t > 0 for t in thicknesses)
            physical_checks['thickness_reasonable'] = all(
                self.config.min_thickness < t < self.config.max_thickness
                for t in thicknesses
            )
            if not physical_checks['thickness_positive']:
                warnings.append("負の膜厚が検出されました")
            if not physical_checks['thickness_reasonable']:
                warnings.append(f"膜厚が妥当な範囲外 ({self.config.min_thickness}-{self.config.max_thickness} Å)")

        if hasattr(solver_result, 'D_opt'):
            D = solver_result.D_opt
            physical_checks['concentration_bounded'] = np.all(D >= 0) and np.all(D <= 1)
            physical_checks['sum_to_one'] = np.allclose(D.sum(axis=1), 1.0, atol=0.01)
            if not physical_checks['concentration_bounded']:
                warnings.append("濃度が0-1の範囲外")
            if not physical_checks['sum_to_one']:
                warnings.append("濃度の合計が1ではない")

        # ステータス判定
        if not convergence_ok:
            status = ValidationStatus.CONVERGENCE_FAILED
        elif not all(physical_checks.values()) if physical_checks else False:
            status = ValidationStatus.PHYSICAL_INVALID
        elif chi2_normalized > self.config.chi2_threshold:
            status = ValidationStatus.CHI2_FAILED
        elif chi2_normalized > self.config.chi2_warning_threshold:
            status = ValidationStatus.CHI2_WARNING
            warnings.append(f"Chi²/n = {chi2_normalized:.4f} (やや高め)")
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
        """提案を生成"""
        suggestions = []

        # 構造解析からの提案を追加
        suggestions.extend(analysis.suggestions)

        # 検証結果に基づく提案
        if validation.status == ValidationStatus.CHI2_FAILED:
            suggestions.append("Chi²が高い: モデルが適切でない可能性")
            if solver_name == "GenericLayerSolver":
                suggestions.append("SemiFreeSolverで組成の自由度を上げることを検討")
            suggestions.append("測定角度の追加または見直しを推奨")

        if validation.status == ValidationStatus.CHI2_WARNING:
            suggestions.append("フィットは成功していますが、改善の余地があります")

        if validation.status == ValidationStatus.CONVERGENCE_FAILED:
            suggestions.append("初期値を見直して再実行することを推奨")
            suggestions.append("界面幅の設定を確認してください")

        if validation.status == ValidationStatus.PHYSICAL_INVALID:
            suggestions.append("物理的に妥当でない結果: モデルの見直しが必要")

        return suggestions


def run_pipeline_demo():
    """パイプラインのデモンストレーション"""
    from scipy.special import erf

    print("=" * 70)
    print("ARXPS Pipeline Demo")
    print("=" * 70)

    # テストデータ生成（SiO2/Si）
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

    # ノイズ追加
    np.random.seed(42)
    noise = 0.02 * np.random.randn(*I_true.shape)
    I_obs = I_true + noise * I_true
    I_obs = np.clip(I_obs, 1e-6, None)
    I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)

    # パイプライン実行
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

    # 明示的な層構造設定
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
        initial_thicknesses=[25.0],  # 少しずれた初期値
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
