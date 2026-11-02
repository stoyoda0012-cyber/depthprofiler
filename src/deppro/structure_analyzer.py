"""
Structure Analyzer - ソルバー自動選択システム
==============================================

ARXPSデータの特徴から、適切なソルバーを自動選択するシステム。

ケース分類:
1. 明確な層構造 (Well-defined layers) → GenericLayerSolver
2. 混合層・グラデーション (Mixed/Gradient) → SemiFreeSolver
3. 完全に未知の構造 (Unknown) → FreeSolver

判定基準:
- 角度依存性の強さ（層構造の存在）
- 元素間の相関（化学量論の成立）
- 強度プロファイルの形状（ステップ vs グラデーション）

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
    """構造タイプ"""
    WELL_DEFINED_LAYERS = "well_defined_layers"    # 明確な層構造
    MIXED_GRADIENT = "mixed_gradient"              # 混合・グラデーション
    UNKNOWN = "unknown"                            # 完全に未知
    SUBSTRATE_ONLY = "substrate_only"              # 基板のみ（薄膜なし）
    UNIFORM_MIXED = "uniform_mixed"                # 完全均一混合（深さ分解不可）


class SolverRecommendation(Enum):
    """推奨ソルバー"""
    GENERIC_LAYER = "GenericLayerSolver"
    SEMI_FREE = "SemiFreeSolver"
    FREE = "FreeSolver"


@dataclass
class SubstrateDetectionResult:
    """基板検出結果"""
    substrate_indices: List[int]       # 基板元素のインデックス
    substrate_group: List[int]         # 基板グループ（化合物基板の場合は複数）
    confidence: float                  # 検出信頼度 (0-1)
    method: str                        # 使用した検出手法

    # 詳細指標
    slopes: np.ndarray                 # 各元素のslope値
    ratios: np.ndarray                 # 各元素の60°/0°比

    # フォールバック情報
    has_substrate: bool = True         # 基板が検出されたか
    fallback_reason: Optional[str] = None  # フォールバック理由

    def __repr__(self):
        return (f"SubstrateDetectionResult(\n"
                f"  substrate_indices={self.substrate_indices},\n"
                f"  substrate_group={self.substrate_group},\n"
                f"  confidence={self.confidence:.2f},\n"
                f"  has_substrate={self.has_substrate}\n"
                f")")


@dataclass
class AnalysisResult:
    """解析結果"""
    structure_type: StructureType
    solver_recommendation: SolverRecommendation
    confidence: float  # 0-1

    # 詳細指標
    angle_dependence_score: float      # 角度依存性スコア
    stoichiometry_score: float         # 化学量論整合性スコア
    layer_sharpness_score: float       # 層境界の明瞭さスコア
    n_estimated_layers: int            # 推定層数

    # 追加情報
    element_correlations: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    # 基板検出結果
    substrate_detection: Optional[SubstrateDetectionResult] = None

    # 可変組成検出 (090317/stoichiometric mode関連)
    stoichiometry_variability_score: float = 0.0  # 組成変動スコア
    use_stoichiometric_mode: bool = False         # stoichiometricモード推奨フラグ

    def __repr__(self):
        return (f"AnalysisResult(\n"
                f"  structure_type={self.structure_type.value},\n"
                f"  solver={self.solver_recommendation.value},\n"
                f"  confidence={self.confidence:.2f},\n"
                f"  estimated_layers={self.n_estimated_layers}\n"
                f")")


class StructureAnalyzer:
    """
    構造解析器

    ARXPSデータを解析し、適切なソルバーを推奨する。
    基板検出機能も提供。
    """

    def __init__(
        self,
        stoich_threshold: float = 0.8,
        angle_dep_threshold: float = 0.05,
        layer_sharp_threshold: float = 0.7,
        substrate_slope_margin: float = 0.15,  # 化合物基板のグループ判定マージン
    ):
        """
        Parameters
        ----------
        stoich_threshold : float
            化学量論整合性の閾値（これ以上なら層構造と判定）
        angle_dep_threshold : float
            角度依存性の閾値（これ以上なら深さ構造あり）。既定値 0.05。
        layer_sharp_threshold : float
            層境界明瞭さの閾値（これ以上なら明確な層）
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
        ARXPSデータを解析し、ソルバー推奨を返す

        Parameters
        ----------
        I_obs : ndarray
            観測強度 [n_angles, n_elements]
        tm : TransferMatrix, optional
            転送行列（角度情報用）
        element_labels : List[str], optional
            元素ラベル
        verbose : bool
            詳細出力

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

        # 正規化（念のため）
        I_norm = I_obs / np.maximum(I_obs.sum(axis=1, keepdims=True), 1e-10)

        # 1. 角度依存性スコア
        angle_dep_score = self._compute_angle_dependence(I_norm)
        if verbose:
            print(f"Angle dependence score: {angle_dep_score:.3f}")

        # 2. 化学量論整合性スコア
        stoich_score, correlations = self._compute_stoichiometry_score(I_norm, element_labels)
        if verbose:
            print(f"Stoichiometry score: {stoich_score:.3f}")

        # 3. 層境界明瞭さスコア
        sharpness_score = self._compute_layer_sharpness(I_norm)
        if verbose:
            print(f"Layer sharpness score: {sharpness_score:.3f}")

        # 4. 推定層数
        n_layers = self._estimate_layer_count(I_norm)
        if verbose:
            print(f"Estimated layer count: {n_layers}")

        # 5. 可変組成スコア（stoichiometricモード判定用）
        stoich_var_score = self._compute_stoichiometry_variability(I_norm)
        if verbose:
            print(f"Stoichiometry variability score: {stoich_var_score:.3f}")

        # 6. 構造タイプ判定
        structure_type = self._determine_structure_type(
            angle_dep_score, stoich_score, sharpness_score, n_layers
        )

        # 7. ソルバー推奨
        solver_rec, confidence = self._recommend_solver(
            structure_type, angle_dep_score, stoich_score, sharpness_score
        )

        # 8. stoichiometricモード推奨判定
        # 条件: 角度依存性あり（層構造）+ 組成変動あり + 化合物形成の兆候
        # 閾値: 非線形性指標 > 0.12 で推奨（NON/ONO検出のため緩和）
        # 注意: 自動判定には限界あり。多層酸化物で誤検出の可能性。
        #       ユーザーが系の特性を理解して明示的に選択することを推奨。
        use_stoichiometric_mode = (
            angle_dep_score > 0.1 and      # 角度依存性あり
            stoich_var_score > 0.12 and    # 組成変動あり（非線形性）
            stoich_score > 0.5             # 化合物形成の兆候
        )

        # 9. 警告・提案
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
        角度依存性スコアを計算

        高い値 → 深さ方向に構造がある
        低い値 → 均一（基板のみ、または完全混合）
        """
        # 各元素の角度依存性（変動係数）
        cv_per_element = np.std(I_norm, axis=0) / np.maximum(np.mean(I_norm, axis=0), 1e-10)

        # 全体スコア（平均変動係数）
        score = np.mean(cv_per_element)

        return min(score, 1.0)

    def _compute_stoichiometry_score(
        self,
        I_norm: np.ndarray,
        element_labels: Optional[List[str]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        化学量論整合性スコアを計算

        高い値 → 元素間に一定の比率関係がある（化合物形成）
        低い値 → ランダムな混合
        """
        n_elements = I_norm.shape[1]
        correlations = {}

        # 元素ペア間の相関係数
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

        # 強い相関（正または負）が多いほどスコアが高い
        if corr_values:
            score = np.mean(corr_values)
        else:
            score = 0.0

        return score, correlations

    def _compute_stoichiometry_variability(self, I_norm: np.ndarray) -> float:
        """
        化合物層内の組成変動を定量化（可変組成検出スコア）

        高い値 → 組成が深さで変化（可変組成：SiOx, NO/ON, SiGe勾配）
        低い値 → 組成が一定（固定化学量論：純粋なSiO2, Si3N4）

        計算方法:
        - 膜層元素ペア間の相関の「非線形性」を評価
        - 固定組成なら線形相関（相関係数~1）
        - 可変組成なら非線形（相関係数<1かつ残差が構造的）

        閾値の目安:
        - < 0.10: 固定組成（強い線形相関）
        - 0.10-0.20: 軽微な変動
        - 0.20-0.35: 中程度変動（stoichiometricモード推奨）
        - > 0.35: 高い変動（stoichiometry_bounds使用推奨）
        """
        n_angles, n_elements = I_norm.shape

        if n_elements < 2 or n_angles < 3:
            return 0.0

        # 基板候補を特定（低角度で信号がほぼ0、高角度で増加）
        substrate_candidates = set()
        for e in range(n_elements):
            # 低角度（表面敏感）での信号
            low_angle_signal = I_norm[0, e]
            # 高角度（深さ敏感）での信号
            high_angle_signal = I_norm[-1, e]

            # 基板パターン: 表面で小さく、深くで大きい
            if low_angle_signal < 0.1 and high_angle_signal > 0.3:
                substrate_candidates.add(e)
            # または、大きく増加する元素
            elif low_angle_signal > 0.01 and high_angle_signal / (low_angle_signal + 1e-10) > 3:
                substrate_candidates.add(e)

        # 膜層元素（基板以外）
        film_elements = [e for e in range(n_elements) if e not in substrate_candidates]

        if len(film_elements) < 2:
            return 0.0

        nonlinearities = []
        for i in range(len(film_elements)):
            for j in range(i + 1, len(film_elements)):
                ei, ej = film_elements[i], film_elements[j]

                xi = I_norm[:, ei]
                xj = I_norm[:, ej]

                # 両方の元素が十分な信号を持つか確認
                if xi.mean() < 0.03 or xj.mean() < 0.03:
                    continue

                # 線形フィットの残差を計算
                try:
                    coeffs = np.polyfit(xi, xj, 1)
                    predicted = np.polyval(coeffs, xi)
                    residuals = xj - predicted

                    # 正規化残差（RMSE / 標準偏差）
                    rmse = np.sqrt(np.mean(residuals**2))
                    std_xj = np.std(xj)

                    if std_xj > 0.01:
                        # 非線形性指標: 残差が大きいほど非線形
                        nonlinearity = rmse / std_xj
                        if not np.isnan(nonlinearity) and not np.isinf(nonlinearity):
                            nonlinearities.append(nonlinearity)
                except (np.linalg.LinAlgError, ValueError):
                    continue

        return np.mean(nonlinearities) if nonlinearities else 0.0

    def _compute_layer_sharpness(self, I_norm: np.ndarray) -> float:
        """
        層境界の明瞭さスコアを計算

        高い値 → 急峻な界面（明確な層構造）
        低い値 → 緩やかな遷移（グラデーション、混合）
        """
        # 角度間の強度変化の二次微分（曲率）
        if I_norm.shape[0] < 3:
            return 0.5  # データ不足

        # 各元素について二次微分を計算
        curvatures = []
        for e in range(I_norm.shape[1]):
            profile = I_norm[:, e]
            if len(profile) >= 3:
                second_deriv = np.diff(profile, n=2)
                curvature = np.max(np.abs(second_deriv))
                curvatures.append(curvature)

        if curvatures:
            # 高い曲率 → 急峻な界面
            max_curv = max(curvatures)
            score = min(max_curv * 10, 1.0)  # スケーリング
        else:
            score = 0.5

        return score

    def _estimate_layer_count(self, I_norm: np.ndarray) -> int:
        """
        推定層数

        角度プロファイルの変曲点数から推定
        """
        n_elements = I_norm.shape[1]

        # 各元素について変曲点を検出
        inflection_counts = []
        for e in range(n_elements):
            profile = I_norm[:, e]
            if len(profile) >= 3:
                # 一次微分の符号変化をカウント
                diff1 = np.diff(profile)
                sign_changes = np.sum(np.abs(np.diff(np.sign(diff1))) > 0)
                inflection_counts.append(sign_changes)

        if inflection_counts:
            # 最大変曲点数 + 1 = 推定層数
            n_layers = max(inflection_counts) + 1
        else:
            n_layers = 1

        return max(1, min(n_layers, 10))  # 1-10の範囲

    def _determine_structure_type(
        self,
        angle_dep: float,
        stoich: float,
        sharpness: float,
        n_layers: int,
    ) -> StructureType:
        """構造タイプを判定"""

        # 角度依存性がほぼない → 均一混合または基板のみ
        if angle_dep < self.angle_dep_threshold:
            if stoich > 0.8:
                # 化学量論が成立 → 単一化合物の基板
                return StructureType.SUBSTRATE_ONLY
            elif stoich > 0.5:
                # ある程度の相関 → 均一混合
                return StructureType.UNIFORM_MIXED
            else:
                # 相関が低い → 完全に未知
                return StructureType.UNKNOWN

        # 化学量論性が高く、層境界が明瞭 → 明確な層構造
        if stoich > self.stoich_threshold and sharpness > self.layer_sharp_threshold:
            return StructureType.WELL_DEFINED_LAYERS

        # 化学量論性は高いが境界が不明瞭 → 混合・グラデーション
        if stoich > self.stoich_threshold * 0.7:
            return StructureType.MIXED_GRADIENT

        # その他 → 未知
        return StructureType.UNKNOWN

    def _recommend_solver(
        self,
        structure_type: StructureType,
        angle_dep: float,
        stoich: float,
        sharpness: float,
    ) -> Tuple[SolverRecommendation, float]:
        """ソルバーを推奨"""

        if structure_type == StructureType.WELL_DEFINED_LAYERS:
            # 明確な層構造 → GenericLayerSolver
            confidence = min((stoich + sharpness) / 2, 1.0)
            return SolverRecommendation.GENERIC_LAYER, confidence

        elif structure_type == StructureType.MIXED_GRADIENT:
            # 混合・グラデーション → SemiFreeSolver
            confidence = 0.5 + stoich * 0.3
            return SolverRecommendation.SEMI_FREE, confidence

        elif structure_type == StructureType.SUBSTRATE_ONLY:
            # 基板のみ → 検証用にGenericLayerSolver（単層）
            return SolverRecommendation.GENERIC_LAYER, 0.7

        elif structure_type == StructureType.UNIFORM_MIXED:
            # 均一混合 → 深さ分解不可、通常XPSを推奨
            # ソルバー推奨なし（Noneの代わりにFREEを返すが低信頼度）
            return SolverRecommendation.FREE, 0.1

        else:  # UNKNOWN
            # 完全に未知 → FreeSolver
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
        """警告と提案を生成"""
        warnings = []
        suggestions = []

        # 警告
        if angle_dep < self.angle_dep_threshold:
            warnings.append(f"角度依存性が非常に弱い (CV={angle_dep:.4f}): 深さ構造の検出が困難")

        if stoich < 0.3:
            warnings.append("化学量論整合性が低い: 複雑な混合状態の可能性")

        if n_layers > 5:
            warnings.append(f"推定層数が多い ({n_layers}層): 解析の収束に注意")

        # 提案
        if structure_type == StructureType.UNIFORM_MIXED:
            warnings.append("均一混合の可能性: ARXPSでは深さ分解不可")
            suggestions.append("通常XPSでの定量分析を推奨")
            suggestions.append("深さ分布が必要な場合はイオンスパッタリング深さプロファイルを検討")
            suggestions.append("本当に均一か確認するため、他の角度範囲でも測定を推奨")

        elif structure_type == StructureType.UNKNOWN:
            suggestions.append("まずFreeSolverで探索的解析を推奨")
            suggestions.append("結果から層構造を推定後、GenericLayerSolverで精密化")

        elif structure_type == StructureType.MIXED_GRADIENT:
            suggestions.append("界面幅を大きめに設定することを推奨")
            suggestions.append("SemiFreeSolverで組成分布を確認後、必要に応じて精密化")

        if sharpness < 0.3 and structure_type == StructureType.WELL_DEFINED_LAYERS:
            suggestions.append("界面が緩やかな可能性: 界面幅の最適化も検討")

        # 可変組成に関する提案
        # 閾値: 非線形性指標に基づく（0=完全線形、1=完全非線形）
        # 注意: 自動判定には限界あり。多層酸化物（C/Al2O3/TiO2/SiO2/Si等）で
        #       誤検出の可能性があるため、系の特性を理解した上で判断すること。
        if stoich_var > 0.5:
            suggestions.append(
                "高い組成変動を検出: 多層酸化物の可能性も検討。"
                "可変組成の場合は stoichiometry_bounds で組成範囲を指定"
            )
        elif stoich_var > 0.20:
            suggestions.append(
                "中程度の組成変動を検出: SemiFreeSolverの stoichiometric_mode='stoichiometric' を推奨。"
                "ただし多層酸化物では charge_balance が適切"
            )
        elif stoich_var > 0.08 and angle_dep > 0.1:
            suggestions.append(
                "軽微な組成変動: 可変組成（SiOx, NON/ONO, SiGe）の可能性を検討"
            )

        return warnings, suggestions

    # ========== 基板検出機能 ==========

    def detect_substrate(
        self,
        I_obs: np.ndarray,
        angles: np.ndarray,
        element_labels: Optional[List[str]] = None,
        oxidation_states: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> SubstrateDetectionResult:
        """
        角度分解データから基板を自動検出

        検出アルゴリズム:
        1. Slope法: d(ln I)/d(1/cosθ) が最も負の元素 = 最も深い
        2. Ratio法: I(60°)/I(0°) が最も小さい元素 = 最も深い
        3. 化合物基板: 類似した傾きを持つ元素をグループ化

        Parameters
        ----------
        I_obs : ndarray
            観測強度 [n_angles, n_elements]
        angles : ndarray
            測定角度（度）
        element_labels : List[str], optional
            元素ラベル
        oxidation_states : ndarray, optional
            酸化状態（化合物基板のグループ判定に使用）
        verbose : bool
            詳細出力

        Returns
        -------
        SubstrateDetectionResult
        """
        n_angles, n_elements = I_obs.shape

        if verbose:
            print("=" * 60)
            print("Substrate Detection")
            print("=" * 60)

        # 正規化
        I_norm = I_obs / np.maximum(I_obs.sum(axis=1, keepdims=True), 1e-10)

        # 1. Slope法（メイン手法）
        slopes = self._compute_angle_slopes(I_norm, angles)

        # 2. Ratio法（補助手法）
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

        # 3. 基板候補の特定
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
            print(f"\n結果:")
            print(f"  基板インデックス: {result.substrate_indices}")
            print(f"  基板グループ: {result.substrate_group}")
            print(f"  検出信頼度: {result.confidence:.2f}")
            print(f"  検出手法: {result.method}")
            if not has_substrate:
                print(f"  ⚠️ フォールバック: {fallback_reason}")

        return result

    def _compute_angle_slopes(self, I_norm: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """
        角度依存性の傾きを計算

        slope = d(ln I) / d(1/cosθ)
        最も負の傾き = 最も深い位置（基板）
        """
        angles_rad = np.deg2rad(angles)
        x = 1.0 / np.cos(angles_rad)
        n_elements = I_norm.shape[1]

        slopes = []
        for e in range(n_elements):
            y = np.log(np.maximum(I_norm[:, e], 1e-10))
            # 線形回帰
            slope, _ = np.polyfit(x, y, 1)
            slopes.append(slope)

        return np.array(slopes)

    def _compute_angle_ratios(self, I_norm: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """
        高角度/低角度の強度比を計算

        ratio = I(high_angle) / I(low_angle)
        小さい比 = より深い位置（基板）
        """
        n_elements = I_norm.shape[1]

        # 最小・最大角度のインデックス
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
        基板候補を特定

        Returns
        -------
        substrate_idx : int or None
            主要基板元素のインデックス
        substrate_group : List[int]
            基板グループ（化合物基板の場合は複数元素）
        confidence : float
            検出信頼度
        method : str
            使用した検出手法
        has_substrate : bool
            基板が検出されたか
        fallback_reason : str or None
            フォールバック理由
        """
        n_elements = len(slopes)

        # ===== フォールバック条件のチェック =====

        # 1. 全元素の傾きが類似（均一混合の可能性）
        slope_range = np.max(slopes) - np.min(slopes)
        if slope_range < 0.1:
            # 均一混合 → 最も濃度が高い元素をデフォルト基板とする
            # (注: これは仮の設定。ユーザー指定が推奨)
            return (
                None,
                list(range(n_elements)),
                0.1,
                "fallback_uniform",
                False,
                "全元素の角度依存性が類似（均一混合の可能性）。Exponential Priorは使用せず、FreeSolverのuniformを推奨。"
            )

        # 2. 全て正の傾き（表面層のみ、基板なし？）
        if np.all(slopes > -0.05):
            return (
                None,
                [],
                0.2,
                "fallback_no_substrate",
                False,
                "全元素が表面に集中。基板信号が検出されない可能性。測定深さを超える厚い膜の可能性。"
            )

        # ===== 通常の基板検出 =====

        # Slope法: 最も負の傾き = 最も深い
        slope_order = np.argsort(slopes)  # 小さい順
        substrate_idx = slope_order[0]  # 最も負（最も深い）

        # 化合物基板のグループ検出
        min_slope = slopes[substrate_idx]
        threshold = min_slope * (1 - self.substrate_slope_margin)  # マージン内の元素をグループ化

        substrate_group = []
        for i in range(n_elements):
            if slopes[i] <= threshold:
                substrate_group.append(i)

        # 信頼度計算
        # 傾きの差が大きいほど信頼度が高い
        if len(substrate_group) == 1:
            # 単一元素基板
            second_min = slopes[slope_order[1]] if n_elements > 1 else min_slope
            separation = second_min - min_slope
            confidence = min(separation / 0.5, 1.0)  # 0.5の差で信頼度1.0
            method = "slope_single"
        else:
            # 化合物基板
            # グループ内の傾きの一致度で信頼度を計算
            group_slopes = slopes[substrate_group]
            group_consistency = 1.0 - np.std(group_slopes) / (np.abs(np.mean(group_slopes)) + 1e-10)
            confidence = max(0.3, min(group_consistency, 0.9))  # 0.3-0.9の範囲
            method = "slope_compound"

        # Ratio法での検証
        ratio_order = np.argsort(ratios)  # 小さい順
        ratio_substrate = ratio_order[0]

        if ratio_substrate == substrate_idx:
            # Slope法とRatio法が一致 → 信頼度アップ
            confidence = min(confidence + 0.1, 1.0)
            method += "+ratio_confirmed"

        if verbose:
            print(f"\n基板検出詳細:")
            print(f"  Slope順: {slope_order.tolist()}")
            print(f"  Ratio順: {ratio_order.tolist()}")
            print(f"  グループ閾値: slope <= {threshold:.3f}")
            if element_labels:
                group_labels = [element_labels[i] for i in substrate_group if i < len(element_labels)]
                print(f"  検出グループ: {group_labels}")

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
        基板検出を含む完全な構造解析

        Parameters
        ----------
        I_obs : ndarray
            観測強度 [n_angles, n_elements]
        angles : ndarray
            測定角度（度）
        element_labels : List[str], optional
            元素ラベル
        oxidation_states : ndarray, optional
            酸化状態
        verbose : bool
            詳細出力

        Returns
        -------
        AnalysisResult
            基板検出結果を含む解析結果
        """
        # 基本解析
        result = self.analyze(I_obs, element_labels=element_labels, verbose=verbose)

        # 基板検出
        substrate_result = self.detect_substrate(
            I_obs, angles, element_labels, oxidation_states, verbose
        )
        result.substrate_detection = substrate_result

        # 基板検出結果に基づく追加の提案
        if not substrate_result.has_substrate:
            result.warnings.append(f"基板検出失敗: {substrate_result.fallback_reason}")
            result.suggestions.append("基板インデックスを手動で指定することを推奨")

        return result


def demonstrate_analyzer():
    """デモンストレーション"""
    from scipy.special import erf

    print("=" * 70)
    print("Structure Analyzer Demonstration")
    print("=" * 70)

    analyzer = StructureAnalyzer()

    # ケース1: 明確な層構造 (SiO2/Si)
    print("\n>>> Case 1: Well-defined SiO2/Si layer")
    angles = np.array([15, 30, 45, 60, 75, 85])
    n_angles = len(angles)

    # 角度依存性をシミュレート（表面層が多い→深い層が多い）
    surface_ratio = 1 - 0.3 * np.sin(np.radians(angles))  # 表面敏感度
    I_sio2 = np.column_stack([
        0.33 * surface_ratio,      # Si_ox
        0.67 * surface_ratio,      # O
        1 - surface_ratio,         # Si_bulk
    ])
    I_sio2 = I_sio2 / I_sio2.sum(axis=1, keepdims=True)

    result1 = analyzer.analyze(I_sio2, element_labels=['Si_ox', 'O', 'Si_bulk'], verbose=True)

    # ケース2: 混合・グラデーション
    print("\n>>> Case 2: Mixed/Gradient structure")
    I_mixed = np.column_stack([
        0.2 + 0.1 * np.random.randn(n_angles),  # 混合元素1
        0.3 + 0.15 * np.random.randn(n_angles),  # 混合元素2
        0.5 + 0.1 * np.random.randn(n_angles),   # 基板
    ])
    I_mixed = np.clip(I_mixed, 0.01, None)
    I_mixed = I_mixed / I_mixed.sum(axis=1, keepdims=True)

    result2 = analyzer.analyze(I_mixed, element_labels=['A', 'B', 'Sub'], verbose=True)

    # ケース3: 完全に未知（ランダム）
    print("\n>>> Case 3: Unknown/Random structure")
    np.random.seed(123)
    I_random = np.random.rand(n_angles, 4)
    I_random = I_random / I_random.sum(axis=1, keepdims=True)

    result3 = analyzer.analyze(I_random, element_labels=['X', 'Y', 'Z', 'W'], verbose=True)

    # ケース4: 4層膜
    print("\n>>> Case 4: 4-layer multilayer (C/Al2O3/TiO2/SiO2/Si)")
    # 表面から深くなるにつれて組成が変化
    depth_sensitivity = 1 - 0.4 * np.sin(np.radians(angles))
    I_multilayer = np.column_stack([
        0.15 * depth_sensitivity**2,              # C (表面)
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

    # サマリー
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
