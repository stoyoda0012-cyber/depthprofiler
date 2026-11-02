"""
MATLAB DepthProfiler 初期化ロジックのPython移植

MATLABの Set_Callback で行われる初期化処理をPythonで再現。
これにより、Pythonで生成したH5ファイルがGUIで正しく読み込める。

主要コンポーネント:
- TransformationMatrix: 転送行列の計算（調和平均 + Beer-Lambert則）
- DistributionCalculator: 深さプロファイルの計算（Gaussian等）
- ChargeBalancer: 電荷バランス調整

使用例:
    from deppro.io.initializer import DepthProfileInitializer

    init = DepthProfileInitializer(
        depth=depth,
        angles_obs=thetaobs,
        angles_opt=thetaopt,
        elements=elements,
        compounds=compounds,
        photon_energy=8000.0,
    )

    # 深さプロファイルから初期化
    init.initialize_from_profile(D_true)

    # H5Writerに設定を適用
    init.apply_to_writer(writer)
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

from deppro.imfp import IMFP


@dataclass
class TransformationMatrixParams:
    """転送行列計算のパラメータ"""
    gammaeff: float = 0.8
    depth: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    stepsize: float = 30.0
    thetaobs: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    thetaopt: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    lambdabasic: np.ndarray = field(default_factory=lambda: np.array([[[1.0]]]))
    n_elements: int = 1
    n_energies: int = 1


class TransformationMatrix:
    """
    転送行列の計算クラス

    MATLABの deppro.core.TransformationMatrix と同等の機能を提供。

    転送行列は深さプロファイルD(z)から角度依存強度I(θ)への変換を行う：
        I(θ, element) = Σ_z TransMat(θ, z, element) × D(z, element)

    計算ステップ:
    1. 調和平均によるλ_obs計算: 1/λ = Σ(D_j / λ_ij)
    2. Beer-Lambert則による減衰: exp(-z / (λ × cosθ))
    3. 累積積による転送行列構築
    """

    @staticmethod
    def calculate_lambda_obs(
        D: np.ndarray,
        lambdabasic: np.ndarray,
        gammaeff: float = 0.8
    ) -> np.ndarray:
        """
        調和平均による有効IMFP計算

        MATLAB式:
            lambdaobs(:,:,k) = gammaeff ./ sum(
                repmat(permute(Dmat, [1 3 2]), 1, nElement, 1) .*
                repmat(permute(lambdabasic(:,:,k), [3 2 1]).^(-1), nDepth, 1, 1), 3)

        Args:
            D: 深さプロファイル (n_depth, n_elem)
            lambdabasic: 基本IMFP (n_elem, n_elem) または (1, n_elem, n_elem)
                         lambdabasic[i, j] = 元素iの光電子が化合物jを通過するIMFP
            gammaeff: 有効減衰係数 (default: 0.8)

        Returns:
            lambdaobs: 有効IMFP (n_depth, n_elem)
        """
        n_depth, n_elem = D.shape

        # lambdabasic の形状を正規化
        if lambdabasic.ndim == 3:
            lb = lambdabasic[0]  # (n_elem, n_elem)
        else:
            lb = lambdabasic

        # 調和平均: 1/λ_eff = Σ_j (D_j / λ_ij)
        # lambdaobs[z, i] = gammaeff / Σ_j (D[z, j] / lb[i, j])
        lambdaobs = np.zeros((n_depth, n_elem), dtype=np.float32)

        for z in range(n_depth):
            for i in range(n_elem):
                denom = 0.0
                for j in range(n_elem):
                    if lb[i, j] > 1e-10:
                        denom += D[z, j] / lb[i, j]
                if denom > 1e-10:
                    lambdaobs[z, i] = gammaeff / denom
                else:
                    # フォールバック: 対角成分
                    lambdaobs[z, i] = lb[i, i] if lb[i, i] > 0 else 100.0

        return lambdaobs

    @staticmethod
    def calculate_transfer_matrix(
        lambdaobs: np.ndarray,
        theta: np.ndarray,
        stepsize: float
    ) -> np.ndarray:
        """
        Beer-Lambert則による転送行列計算

        MATLAB式:
            t1 = (-stepsize / lambdaobs) / cos(theta * pi/180)
            TransMat = cumprod(exp(t1), 2) / exp(t1)

        Args:
            lambdaobs: 有効IMFP (n_depth, n_elem)
            theta: 角度配列 [degrees]
            stepsize: 深さステップ [Å]

        Returns:
            TransMat: 転送行列 (n_angles, n_depth, n_elem)
        """
        n_depth, n_elem = lambdaobs.shape
        n_angles = len(theta)

        TransMat = np.zeros((n_angles, n_depth, n_elem), dtype=np.float32)
        cos_theta = np.cos(np.radians(theta))

        for a in range(n_angles):
            if abs(cos_theta[a]) < 1e-6:
                # cos(90°) ≈ 0 の場合はNaN
                TransMat[a, :, :] = np.nan
                continue

            for e in range(n_elem):
                # t1[z] = -stepsize / (lambda[z, e] * cos(theta))
                t1 = -stepsize / (lambdaobs[:, e] * cos_theta[a])

                # cumprod(exp(t1)) / exp(t1)
                exp_t1 = np.exp(t1)
                cumprod_exp = np.cumprod(exp_t1)

                # MATLABの cumprod(exp(t1), 2) ./ exp(t1) と同等
                TransMat[a, :, e] = cumprod_exp / exp_t1

        return TransMat

    @staticmethod
    def calculate_lambda_cos_theta(
        lambdaobs: np.ndarray,
        theta: np.ndarray
    ) -> np.ndarray:
        """
        λ × cos(θ) の計算

        Args:
            lambdaobs: 有効IMFP (n_depth, n_elem)
            theta: 角度配列 [degrees]

        Returns:
            lambda_cos_theta: (n_angles, n_elem)
        """
        # 深さ方向の平均
        lambda_mean = np.mean(lambdaobs, axis=0)  # (n_elem,)
        cos_theta = np.cos(np.radians(theta))[:, np.newaxis]  # (n_angles, 1)

        return (lambda_mean[np.newaxis, :] * cos_theta).astype(np.float32)

    @staticmethod
    def calculate_all(
        D: np.ndarray,
        params: TransformationMatrixParams
    ) -> Dict[str, np.ndarray]:
        """
        全ての転送行列関連値を計算

        Args:
            D: 深さプロファイル (n_depth, n_elem)
            params: パラメータ

        Returns:
            dict with keys:
                - lambdaobs: (n_depth, n_elem)
                - lambdaopt: (n_depth, n_elem)
                - TransMatObs: (n_angles_obs, n_depth, n_elem)
                - TransMatOpt: (n_angles_opt, n_depth, n_elem)
                - lambdacosinethetaobs: (n_angles_obs, n_elem)
                - lambdacosinethetaopt: (n_angles_opt, n_elem)
        """
        # 有効IMFP計算
        lambdaobs = TransformationMatrix.calculate_lambda_obs(
            D, params.lambdabasic, params.gammaeff
        )
        lambdaopt = lambdaobs.copy()

        # 転送行列計算
        TransMatObs = TransformationMatrix.calculate_transfer_matrix(
            lambdaobs, params.thetaobs, params.stepsize
        )
        TransMatOpt = TransformationMatrix.calculate_transfer_matrix(
            lambdaopt, params.thetaopt, params.stepsize
        )

        # λ×cos(θ)計算
        lambdacosinethetaobs = TransformationMatrix.calculate_lambda_cos_theta(
            lambdaobs, params.thetaobs
        )
        lambdacosinethetaopt = TransformationMatrix.calculate_lambda_cos_theta(
            lambdaopt, params.thetaopt
        )

        return {
            'lambdaobs': lambdaobs,
            'lambdaopt': lambdaopt,
            'TransMatObs': TransMatObs,
            'TransMatOpt': TransMatOpt,
            'lambdacosinethetaobs': lambdacosinethetaobs,
            'lambdacosinethetaopt': lambdacosinethetaopt,
        }


class DistributionCalculator:
    """
    深さプロファイルの計算（Gaussian等の分布関数）

    MATLABの CalculateDistribution.m と同等の機能を提供。
    """

    @staticmethod
    def gaussian(
        depth: np.ndarray,
        position: float,
        width_fwhm: float,
        height: float,
        substrate_profile: np.ndarray
    ) -> np.ndarray:
        """
        Gaussian分布関数

        MATLAB式:
            S(k) = width(k) / (2*sqrt(log(2)))  # FWHM→σ変換
            D0(:,k) = height(k)/100 * exp(-(depth-pos(k))^2 / S(k)^2) * |1-D_subst|

        Args:
            depth: 深さ軸 [Å]
            position: ピーク位置 [Å]
            width_fwhm: FWHM [Å]
            height: 最大濃度 [%] (0-100)
            substrate_profile: 基板プロファイル (n_depth,)

        Returns:
            profile: (n_depth,) 濃度分布
        """
        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(np.log(2)))
        sigma = width_fwhm * fwhm_to_sigma

        if sigma < 1e-6:
            sigma = 1e-6

        # Gaussian × (1 - 基板)
        gaussian = np.exp(-((depth - position) ** 2) / (sigma ** 2))
        profile = (height / 100.0) * gaussian * np.abs(1.0 - substrate_profile)

        return profile.astype(np.float32)

    @staticmethod
    def error_function_substrate(
        depth: np.ndarray,
        thickness: float,
        abruptness_fwhm: float,
        substrate_ratio: float = 1.0
    ) -> np.ndarray:
        """
        Error関数による基板プロファイル

        MATLAB式:
            D(:,subst) = subs * 0.5 * |1 + erf((depth - thck) / abrp_sigma)|

        Args:
            depth: 深さ軸 [Å]
            thickness: 膜厚（界面位置）[Å]
            abruptness_fwhm: 界面急峻度（FWHM）[Å]
            substrate_ratio: 基板元素の濃度比率 (default: 1.0)

        Returns:
            profile: (n_depth,) 基板濃度分布
        """
        from scipy.special import erf

        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(np.log(2)))
        sigma = abruptness_fwhm * fwhm_to_sigma

        if sigma < 1e-6:
            sigma = 1e-6

        profile = substrate_ratio * 0.5 * np.abs(1.0 + erf((depth - thickness) / sigma))

        return profile.astype(np.float32)


class ChargeBalancer:
    """
    電荷バランス調整

    MATLABの電荷バランスループと同等の機能を提供。
    各深さで正電荷と負電荷の合計がゼロになるよう調整。
    """

    @staticmethod
    def balance(
        D: np.ndarray,
        charges: np.ndarray,
        max_iterations: int = 30,
        tolerance: float = 1e-6
    ) -> np.ndarray:
        """
        電荷バランス調整

        Args:
            D: 深さプロファイル (n_depth, n_elem)
            charges: 各元素の電荷 (n_elem,)
            max_iterations: 最大反復回数
            tolerance: 収束判定閾値

        Returns:
            D_balanced: 電荷バランス調整済みプロファイル
        """
        D_balanced = D.copy()
        n_depth, n_elem = D.shape

        # 正電荷・負電荷の元素を分類
        positive_idx = np.where(charges > 0)[0]
        negative_idx = np.where(charges < 0)[0]

        if len(positive_idx) == 0 or len(negative_idx) == 0:
            return D_balanced

        for _ in range(max_iterations):
            total_charge = np.zeros(n_depth)
            for j in range(n_elem):
                total_charge += D_balanced[:, j] * charges[j]

            max_imbalance = np.abs(total_charge).max()
            if max_imbalance < tolerance:
                break

            # 電荷バランス調整
            for z in range(n_depth):
                if abs(total_charge[z]) < tolerance:
                    continue

                if total_charge[z] > 0:
                    # 正電荷過剰: 負電荷元素を増やす
                    neg_sum = sum(D_balanced[z, j] for j in negative_idx)
                    if neg_sum > 1e-10:
                        scale = 1.0 + total_charge[z] / (neg_sum * abs(charges[negative_idx[0]]))
                        for j in negative_idx:
                            D_balanced[z, j] *= min(scale, 2.0)
                else:
                    # 負電荷過剰: 正電荷元素を増やす
                    pos_sum = sum(D_balanced[z, j] for j in positive_idx)
                    if pos_sum > 1e-10:
                        scale = 1.0 - total_charge[z] / (pos_sum * charges[positive_idx[0]])
                        for j in positive_idx:
                            D_balanced[z, j] *= min(scale, 2.0)

            # 正規化
            total = D_balanced.sum(axis=1, keepdims=True)
            D_balanced = np.where(total > 1e-10, D_balanced / total, D_balanced)

        return D_balanced.astype(np.float32)


class DepthProfileInitializer:
    """
    DepthProfiler GUI互換の初期化を行うクラス

    MATLABの Set_Callback と同等の初期化処理を実行し、
    H5Writerで使用できる全てのパラメータを計算する。
    """

    def __init__(
        self,
        depth: np.ndarray,
        angles_obs: np.ndarray,
        elements: List[str],
        compounds: List[str],
        angles_opt: Optional[np.ndarray] = None,
        photon_energy: float = 1486.6,
        gammaeff: float = 0.8,
        charges: Optional[List[float]] = None,
        densities: Optional[List[float]] = None,
        molecular_weights: Optional[List[float]] = None,
        valence_charges: Optional[List[float]] = None,
        bandgaps: Optional[List[float]] = None,
        binding_energies: Optional[List[float]] = None,
    ):
        """
        Args:
            depth: 深さ軸 [Å]
            angles_obs: 観測角度 [degrees]
            elements: 元素記号リスト
            compounds: 化合物名リスト
            angles_opt: 最適化角度 (省略時はデフォルト16点)
            photon_energy: 光子エネルギー [eV]
            gammaeff: 有効減衰係数
            charges: 各元素の電荷
            densities: 密度 [g/cm³]
            molecular_weights: 分子量 [g/mol]
            valence_charges: 価電子数
            bandgaps: バンドギャップ [eV]
            binding_energies: 結合エネルギー [eV]
        """
        self.depth = np.asarray(depth, dtype=np.float32)
        self.angles_obs = np.asarray(angles_obs, dtype=np.float32)
        self.elements = elements
        self.compounds = compounds
        self.photon_energy = photon_energy
        self.gammaeff = gammaeff

        self.n_depth = len(depth)
        self.n_elem = len(elements)
        self.stepsize = float(depth[1] - depth[0]) if len(depth) > 1 else 30.0

        # デフォルト角度
        if angles_opt is None:
            self.angles_opt = np.array(
                [-10, 0, 10, 20, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
                dtype=np.float32
            )
        else:
            self.angles_opt = np.asarray(angles_opt, dtype=np.float32)

        # 物性パラメータの設定
        self._setup_physical_properties(
            charges, densities, molecular_weights,
            valence_charges, bandgaps, binding_energies
        )

        # lambdabasic の計算
        self._compute_lambda_basic()

        # 結果格納用
        self.results: Dict[str, Any] = {}

    def _setup_physical_properties(
        self,
        charges, densities, molecular_weights,
        valence_charges, bandgaps, binding_energies
    ):
        """物性パラメータの設定"""
        from deppro.io.h5_writer import (
            COMPOUND_PROPERTIES, get_compound_properties,
            estimate_charge_from_compound
        )

        n = self.n_elem

        # 電荷
        if charges is not None:
            self.charges = np.array(charges, dtype=np.float32)
        else:
            self.charges = np.array([
                estimate_charge_from_compound(c, e)
                for c, e in zip(self.compounds, self.elements)
            ], dtype=np.float32)

        # 物性
        self.densities = np.zeros(n, dtype=np.float32)
        self.molecular_weights = np.zeros(n, dtype=np.float32)
        self.valence_charges = np.zeros(n, dtype=np.float32)
        self.bandgaps = np.zeros(n, dtype=np.float32)

        for i, comp in enumerate(self.compounds):
            vc, dens, mw, bg = get_compound_properties(comp)
            self.valence_charges[i] = valence_charges[i] if valence_charges else vc
            self.densities[i] = densities[i] if densities else dens
            self.molecular_weights[i] = molecular_weights[i] if molecular_weights else mw
            self.bandgaps[i] = bandgaps[i] if bandgaps else bg

        # 結合エネルギー
        if binding_energies is not None:
            self.binding_energies = np.array(binding_energies, dtype=np.float32)
        else:
            from deppro.io.h5_writer import H5Writer
            # 1s軌道を仮定（HXPS用）
            orbitals = ['1s'] * n
            self.binding_energies = H5Writer._estimate_binding_energies(
                self.elements, orbitals
            )

    def _compute_lambda_basic(self):
        """lambdabasic（基本IMFP行列）の計算"""
        n = self.n_elem
        kinetic_energies = self.photon_energy - self.binding_energies

        # lambdabasic[i, j] = 元素iの光電子が化合物jを通過するIMFP
        self.lambdabasic = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            ke = kinetic_energies[i]
            for j in range(n):
                self.lambdabasic[i, j] = IMFP.tpp2m(
                    ke,
                    self.valence_charges[j],
                    self.densities[j],
                    self.molecular_weights[j],
                    self.bandgaps[j]
                )

    def initialize_from_profile(
        self,
        D: np.ndarray,
        apply_charge_balance: bool = True
    ) -> Dict[str, Any]:
        """
        深さプロファイルから全ての初期化パラメータを計算

        Args:
            D: 深さプロファイル (n_depth, n_elem)
            apply_charge_balance: 電荷バランス調整を適用するか

        Returns:
            dict with all computed parameters
        """
        D = np.asarray(D, dtype=np.float32)

        # 電荷バランス調整
        if apply_charge_balance:
            D = ChargeBalancer.balance(D, self.charges)

        # TransformationMatrix パラメータ
        params = TransformationMatrixParams(
            gammaeff=self.gammaeff,
            depth=self.depth,
            stepsize=self.stepsize,
            thetaobs=self.angles_obs,
            thetaopt=self.angles_opt,
            lambdabasic=self.lambdabasic.reshape(1, self.n_elem, self.n_elem),
            n_elements=self.n_elem,
            n_energies=1,
        )

        # 転送行列計算
        tm_results = TransformationMatrix.calculate_all(D, params)

        # 角度プロファイル計算
        Xobs = self._calculate_angle_profile(D, tm_results['TransMatObs'])
        Xopt = self._calculate_angle_profile(D, tm_results['TransMatOpt'])

        # 結果を保存
        self.results = {
            'D': D,
            'Xobs': Xobs,
            'Xopt': Xopt,
            **tm_results,
            'lambdabasic': self.lambdabasic,
        }

        return self.results

    def _calculate_angle_profile(
        self,
        D: np.ndarray,
        TransMat: np.ndarray
    ) -> np.ndarray:
        """
        角度プロファイルの計算

        I(θ, elem) = Σ_z TransMat(θ, z, elem) × D(z, elem)
        X = I / Σ_elem I  (正規化)
        """
        n_angles = TransMat.shape[0]
        n_elem = D.shape[1]

        I = np.zeros((n_angles, n_elem), dtype=np.float32)

        for e in range(n_elem):
            # NaNを含む行列での計算
            tm_e = TransMat[:, :, e]  # (n_angles, n_depth)
            d_e = D[:, e]  # (n_depth,)
            I[:, e] = np.nansum(tm_e * d_e[np.newaxis, :], axis=1)

        # 正規化
        I_sum = np.nansum(I, axis=1, keepdims=True)
        I_sum = np.where(I_sum > 1e-10, I_sum, 1.0)
        X = I / I_sum

        return X.astype(np.float32)

    def apply_to_writer(self, writer: 'H5Writer'):
        """
        計算結果をH5Writerに適用

        Args:
            writer: H5Writer インスタンス
        """
        if not self.results:
            raise ValueError("initialize_from_profile() を先に呼んでください")

        ez = writer._ezdeprof
        r = self.results

        # lambdabasic: (1, n_elem, n_elem)
        ez['lambdabasic'] = r['lambdabasic'].reshape(1, self.n_elem, self.n_elem)

        # lambdaobs/opt: (1, n_elem, n_depth)
        ez['lambdaobs'] = r['lambdaobs'].T.reshape(1, self.n_elem, self.n_depth)
        ez['lambdaopt'] = r['lambdaopt'].T.reshape(1, self.n_elem, self.n_depth)

        # lambdacosinetheta: (n_angles, n_elem) - writeで転置
        ez['lambdacosinethetaobs'] = r['lambdacosinethetaobs']
        ez['lambdacosinethetaopt'] = r['lambdacosinethetaopt']

        # TransMat: (n_angles, n_depth, n_elem)
        ez['TransMatObs'] = r['TransMatObs']
        ez['TransMatOpt'] = r['TransMatOpt']

        # Dini, Dobs, Dmod, Dopt
        ez['Dini'] = r['D']
        ez['Dobs'] = r['D']
        ez['Dmod'] = r['D']
        ez['Dopt'] = r['D']

        # Xini, Xobs, Xmod, Xopt
        ez['Xini'] = r['Xopt']
        ez['Xmod'] = r['Xopt']
        ez['Xobs'] = r['Xobs']
        ez['Xopt'] = r['Xopt']


# 便利関数
def create_initialized_h5(
    filepath: str,
    depth: np.ndarray,
    angles_obs: np.ndarray,
    elements: List[str],
    compounds: List[str],
    D_true: np.ndarray,
    D_result: Optional[np.ndarray] = None,
    angles_opt: Optional[np.ndarray] = None,
    photon_energy: float = 1486.6,
    subst: Optional[List[int]] = None,
    free: Optional[List[int]] = None,
    linked: Optional[List[int]] = None,
    charges: Optional[List[float]] = None,
    **kwargs
) -> str:
    """
    初期化済みのGUI互換H5ファイルを作成

    Args:
        filepath: 出力パス
        depth: 深さ軸 [Å]
        angles_obs: 観測角度 [degrees]
        elements: 元素記号リスト
        compounds: 化合物名リスト
        D_true: 真の深さプロファイル (n_depth, n_elem)
        D_result: 結果プロファイル（省略時はD_trueにノイズ追加）
        angles_opt: 最適化角度
        photon_energy: 光子エネルギー [eV]
        subst: 基板元素インデックス (1-based)
        free: 自由元素インデックス (1-based)
        linked: リンク元素インデックス (1-based)
        charges: 各元素の電荷
        **kwargs: H5Writer.set_basic_params への追加引数

    Returns:
        出力ファイルパス
    """
    from deppro.io.h5_writer import H5Writer

    # 初期化計算
    init = DepthProfileInitializer(
        depth=depth,
        angles_obs=angles_obs,
        elements=elements,
        compounds=compounds,
        angles_opt=angles_opt,
        photon_energy=photon_energy,
        charges=charges,
    )
    init.initialize_from_profile(D_true)

    # 結果プロファイル
    if D_result is None:
        noise = np.random.normal(0, 0.02, D_true.shape)
        D_result = np.clip(D_true + noise, 0, 1)
        D_result = D_result / D_result.sum(axis=1, keepdims=True)

    # H5Writer
    writer = H5Writer(filepath)
    writer.set_basic_params(
        depth=depth,
        angles=angles_obs,
        angles_opt=init.angles_opt,
        elements=elements,
        compounds=compounds,
        photon_energy=photon_energy,
        subst=subst,
        free=free,
        linked=linked,
        charges=charges,
        **kwargs
    )

    # 初期化結果を適用
    init.apply_to_writer(writer)

    # プロファイル設定
    writer.set_profiles(
        Drlt=D_result,
        Dtrn=D_true,
        Dstk=D_true.copy(),
        Xobs=init.results['Xobs'],
        Xopt=init.results['Xopt'],
    )

    # 目的関数
    writer.set_objectives(Qrlt=np.array([[1000.0, 0.0, 1000.0]]))

    writer.write()

    return filepath
