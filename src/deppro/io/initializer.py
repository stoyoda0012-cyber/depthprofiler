"""
Python port of the MATLAB DepthProfiler initialization logic

Reproduces in Python the initialization that MATLAB's Set_Callback performs,
so that an H5 file generated from Python loads correctly in the GUI.

Main components:
- TransformationMatrix: transfer-matrix calculation (harmonic mean + Beer-Lambert law)
- DistributionCalculator: depth-profile calculation (Gaussian and others)
- ChargeBalancer: charge-balance adjustment

Example:
    from deppro.io.initializer import DepthProfileInitializer

    init = DepthProfileInitializer(
        depth=depth,
        angles_obs=thetaobs,
        angles_opt=thetaopt,
        elements=elements,
        compounds=compounds,
        photon_energy=8000.0,
    )

    # Initialize from a depth profile
    init.initialize_from_profile(D_true)

    # Apply the settings to an H5Writer
    init.apply_to_writer(writer)
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

from deppro.imfp import IMFP


@dataclass
class TransformationMatrixParams:
    """Parameters of the transfer-matrix calculation"""
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
    Transfer-matrix calculator

    Provides the same functionality as MATLAB's deppro.core.TransformationMatrix.

    The transfer matrix maps the depth profile D(z) to the angle-dependent intensity I(θ):
        I(θ, element) = Σ_z TransMat(θ, z, element) × D(z, element)

    Calculation steps:
    1. λ_obs from the harmonic mean: 1/λ = Σ(D_j / λ_ij)
    2. Beer-Lambert attenuation: exp(-z / (λ × cosθ))
    3. Transfer matrix from the cumulative product
    """

    @staticmethod
    def calculate_lambda_obs(
        D: np.ndarray,
        lambdabasic: np.ndarray,
        gammaeff: float = 0.8
    ) -> np.ndarray:
        """
        Effective IMFP from the harmonic mean

        MATLAB formula:
            lambdaobs(:,:,k) = gammaeff ./ sum(
                repmat(permute(Dmat, [1 3 2]), 1, nElement, 1) .*
                repmat(permute(lambdabasic(:,:,k), [3 2 1]).^(-1), nDepth, 1, 1), 3)

        Args:
            D: depth profile (n_depth, n_elem)
            lambdabasic: basic IMFP (n_elem, n_elem) or (1, n_elem, n_elem)
                                      lambdabasic[i, j] = IMFP of a photoelectron of element i travelling through compound j
            gammaeff: effective attenuation coefficient (default: 0.8)

        Returns:
            lambdaobs: effective IMFP (n_depth, n_elem)
        """
        n_depth, n_elem = D.shape

        # Normalize the shape of lambdabasic
        if lambdabasic.ndim == 3:
            lb = lambdabasic[0]  # (n_elem, n_elem)
        else:
            lb = lambdabasic

        # Harmonic mean: 1/λ_eff = Σ_j (D_j / λ_ij)
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
                    # Fallback: diagonal element
                    lambdaobs[z, i] = lb[i, i] if lb[i, i] > 0 else 100.0

        return lambdaobs

    @staticmethod
    def calculate_transfer_matrix(
        lambdaobs: np.ndarray,
        theta: np.ndarray,
        stepsize: float
    ) -> np.ndarray:
        """
        Transfer matrix from the Beer-Lambert law

        MATLAB formula:
            t1 = (-stepsize / lambdaobs) / cos(theta * pi/180)
            TransMat = cumprod(exp(t1), 2) / exp(t1)

        Args:
            lambdaobs: effective IMFP (n_depth, n_elem)
            theta: angle array [degrees]
            stepsize: depth step [Å]

        Returns:
            TransMat: transfer matrix (n_angles, n_depth, n_elem)
        """
        n_depth, n_elem = lambdaobs.shape
        n_angles = len(theta)

        TransMat = np.zeros((n_angles, n_depth, n_elem), dtype=np.float32)
        cos_theta = np.cos(np.radians(theta))

        for a in range(n_angles):
            if abs(cos_theta[a]) < 1e-6:
                # NaN when cos(90°) ≈ 0
                TransMat[a, :, :] = np.nan
                continue

            for e in range(n_elem):
                # t1[z] = -stepsize / (lambda[z, e] * cos(theta))
                t1 = -stepsize / (lambdaobs[:, e] * cos_theta[a])

                # cumprod(exp(t1)) / exp(t1)
                exp_t1 = np.exp(t1)
                cumprod_exp = np.cumprod(exp_t1)

                # Equivalent to MATLAB's cumprod(exp(t1), 2) ./ exp(t1)
                TransMat[a, :, e] = cumprod_exp / exp_t1

        return TransMat

    @staticmethod
    def calculate_lambda_cos_theta(
        lambdaobs: np.ndarray,
        theta: np.ndarray
    ) -> np.ndarray:
        """
        Compute λ × cos(θ)

        Args:
            lambdaobs: effective IMFP (n_depth, n_elem)
            theta: angle array [degrees]

        Returns:
            lambda_cos_theta: (n_angles, n_elem)
        """
        # Average over depth
        lambda_mean = np.mean(lambdaobs, axis=0)  # (n_elem,)
        cos_theta = np.cos(np.radians(theta))[:, np.newaxis]  # (n_angles, 1)

        return (lambda_mean[np.newaxis, :] * cos_theta).astype(np.float32)

    @staticmethod
    def calculate_all(
        D: np.ndarray,
        params: TransformationMatrixParams
    ) -> Dict[str, np.ndarray]:
        """
        Compute every transfer-matrix related quantity

        Args:
            D: depth profile (n_depth, n_elem)
            params: parameters

        Returns:
            dict with keys:
                - lambdaobs: (n_depth, n_elem)
                - lambdaopt: (n_depth, n_elem)
                - TransMatObs: (n_angles_obs, n_depth, n_elem)
                - TransMatOpt: (n_angles_opt, n_depth, n_elem)
                - lambdacosinethetaobs: (n_angles_obs, n_elem)
                - lambdacosinethetaopt: (n_angles_opt, n_elem)
        """
        # Effective IMFP
        lambdaobs = TransformationMatrix.calculate_lambda_obs(
            D, params.lambdabasic, params.gammaeff
        )
        lambdaopt = lambdaobs.copy()

        # Transfer matrix
        TransMatObs = TransformationMatrix.calculate_transfer_matrix(
            lambdaobs, params.thetaobs, params.stepsize
        )
        TransMatOpt = TransformationMatrix.calculate_transfer_matrix(
            lambdaopt, params.thetaopt, params.stepsize
        )

        # λ×cos(θ)
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
    Depth-profile calculation (distribution functions such as the Gaussian)

    Provides the same functionality as MATLAB's CalculateDistribution.m.
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
        Gaussian distribution function

        MATLAB formula:
            S(k) = width(k) / (2*sqrt(log(2)))  # FWHM → σ
            D0(:,k) = height(k)/100 * exp(-(depth-pos(k))^2 / S(k)^2) * |1-D_subst|

        Args:
            depth: depth axis [Å]
            position: peak position [Å]
            width_fwhm: FWHM [Å]
            height: maximum concentration [%] (0-100)
            substrate_profile: substrate profile (n_depth,)

        Returns:
            profile: (n_depth,) concentration distribution
        """
        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(np.log(2)))
        sigma = width_fwhm * fwhm_to_sigma

        if sigma < 1e-6:
            sigma = 1e-6

        # Gaussian × (1 - substrate)
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
        Substrate profile from the error function

        MATLAB formula:
            D(:,subst) = subs * 0.5 * |1 + erf((depth - thck) / abrp_sigma)|

        Args:
            depth: depth axis [Å]
            thickness: film thickness (interface position) [Å]
            abruptness_fwhm: interface abruptness (FWHM) [Å]
            substrate_ratio: concentration ratio of the substrate element (default: 1.0)

        Returns:
            profile: (n_depth,) substrate concentration distribution
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
    Charge-balance adjustment

    Provides the same functionality as MATLAB's charge-balance loop:
    at every depth, the positive and negative charges are adjusted to sum to zero.
    """

    @staticmethod
    def balance(
        D: np.ndarray,
        charges: np.ndarray,
        max_iterations: int = 30,
        tolerance: float = 1e-6
    ) -> np.ndarray:
        """
        Adjust the charge balance

        Args:
            D: depth profile (n_depth, n_elem)
            charges: charge of each element (n_elem,)
            max_iterations: maximum number of iterations
            tolerance: convergence threshold

        Returns:
            D_balanced: charge-balanced profile
        """
        D_balanced = D.copy()
        n_depth, n_elem = D.shape

        # Classify the elements by the sign of their charge
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

            # Charge-balance adjustment
            for z in range(n_depth):
                if abs(total_charge[z]) < tolerance:
                    continue

                if total_charge[z] > 0:
                    # Excess positive charge: increase the negative elements
                    neg_sum = sum(D_balanced[z, j] for j in negative_idx)
                    if neg_sum > 1e-10:
                        scale = 1.0 + total_charge[z] / (neg_sum * abs(charges[negative_idx[0]]))
                        for j in negative_idx:
                            D_balanced[z, j] *= min(scale, 2.0)
                else:
                    # Excess negative charge: increase the positive elements
                    pos_sum = sum(D_balanced[z, j] for j in positive_idx)
                    if pos_sum > 1e-10:
                        scale = 1.0 - total_charge[z] / (pos_sum * charges[positive_idx[0]])
                        for j in positive_idx:
                            D_balanced[z, j] *= min(scale, 2.0)

            # Normalize
            total = D_balanced.sum(axis=1, keepdims=True)
            D_balanced = np.where(total > 1e-10, D_balanced / total, D_balanced)

        return D_balanced.astype(np.float32)


class DepthProfileInitializer:
    """
    Initializer compatible with the DepthProfiler GUI

    Runs the same initialization as MATLAB's Set_Callback and computes
    every parameter that H5Writer needs.
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
            depth: depth axis [Å]
            angles_obs: observation angles [degrees]
            elements: list of element symbols
            compounds: list of compound names
            angles_opt: optimization angles (default: 16 points)
            photon_energy: photon energy [eV]
            gammaeff: effective attenuation coefficient
            charges: charge of each element
            densities: densities [g/cm³]
            molecular_weights: molecular weights [g/mol]
            valence_charges: numbers of valence electrons
            bandgaps: band gaps [eV]
            binding_energies: binding energies [eV]
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

        # Default angles
        if angles_opt is None:
            self.angles_opt = np.array(
                [-10, 0, 10, 20, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
                dtype=np.float32
            )
        else:
            self.angles_opt = np.asarray(angles_opt, dtype=np.float32)

        # Material parameters
        self._setup_physical_properties(
            charges, densities, molecular_weights,
            valence_charges, bandgaps, binding_energies
        )

        # lambdabasic
        self._compute_lambda_basic()

        # Storage for the results
        self.results: Dict[str, Any] = {}

    def _setup_physical_properties(
        self,
        charges, densities, molecular_weights,
        valence_charges, bandgaps, binding_energies
    ):
        """Set the material parameters"""
        from deppro.io.h5_writer import (
            COMPOUND_PROPERTIES, get_compound_properties,
            estimate_charge_from_compound
        )

        n = self.n_elem

        # Charges
        if charges is not None:
            self.charges = np.array(charges, dtype=np.float32)
        else:
            self.charges = np.array([
                estimate_charge_from_compound(c, e)
                for c, e in zip(self.compounds, self.elements)
            ], dtype=np.float32)

        # Material properties
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

        # Binding energies
        if binding_energies is not None:
            self.binding_energies = np.array(binding_energies, dtype=np.float32)
        else:
            from deppro.io.h5_writer import H5Writer
            # Assume the 1s orbital (for HXPS)
            orbitals = ['1s'] * n
            self.binding_energies = H5Writer._estimate_binding_energies(
                self.elements, orbitals
            )

    def _compute_lambda_basic(self):
        """Compute lambdabasic (the basic IMFP matrix)"""
        n = self.n_elem
        kinetic_energies = self.photon_energy - self.binding_energies

        # lambdabasic[i, j] = IMFP of a photoelectron of element i travelling through compound j
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
        Compute every initialization parameter from a depth profile

        Args:
            D: depth profile (n_depth, n_elem)
            apply_charge_balance: whether to apply the charge-balance adjustment

        Returns:
            dict with all computed parameters
        """
        D = np.asarray(D, dtype=np.float32)

        # Charge-balance adjustment
        if apply_charge_balance:
            D = ChargeBalancer.balance(D, self.charges)

        # TransformationMatrix parameters
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

        # Transfer matrix
        tm_results = TransformationMatrix.calculate_all(D, params)

        # Angle profiles
        Xobs = self._calculate_angle_profile(D, tm_results['TransMatObs'])
        Xopt = self._calculate_angle_profile(D, tm_results['TransMatOpt'])

        # Store the results
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
        Compute the angle profiles

        I(θ, elem) = Σ_z TransMat(θ, z, elem) × D(z, elem)
        X = I / Σ_elem I  (normalization)
        """
        n_angles = TransMat.shape[0]
        n_elem = D.shape[1]

        I = np.zeros((n_angles, n_elem), dtype=np.float32)

        for e in range(n_elem):
            # Calculation with NaN in the matrix
            tm_e = TransMat[:, :, e]  # (n_angles, n_depth)
            d_e = D[:, e]  # (n_depth,)
            I[:, e] = np.nansum(tm_e * d_e[np.newaxis, :], axis=1)

        # Normalize
        I_sum = np.nansum(I, axis=1, keepdims=True)
        I_sum = np.where(I_sum > 1e-10, I_sum, 1.0)
        X = I / I_sum

        return X.astype(np.float32)

    def apply_to_writer(self, writer: 'H5Writer'):
        """
        Apply the computed results to an H5Writer

        Args:
            writer: H5Writer instance
        """
        if not self.results:
            raise ValueError("call initialize_from_profile() first")

        ez = writer._ezdeprof
        r = self.results

        # lambdabasic: (1, n_elem, n_elem)
        ez['lambdabasic'] = r['lambdabasic'].reshape(1, self.n_elem, self.n_elem)

        # lambdaobs/opt: (1, n_elem, n_depth)
        ez['lambdaobs'] = r['lambdaobs'].T.reshape(1, self.n_elem, self.n_depth)
        ez['lambdaopt'] = r['lambdaopt'].T.reshape(1, self.n_elem, self.n_depth)

        # lambdacosinetheta: (n_angles, n_elem) - transposed on write
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


# Convenience function
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
    Create an initialized, GUI-compatible H5 file

    Args:
        filepath: output path
        depth: depth axis [Å]
        angles_obs: observation angles [degrees]
        elements: list of element symbols
        compounds: list of compound names
        D_true: true depth profile (n_depth, n_elem)
        D_result: result profile (default: D_true plus noise)
        angles_opt: optimization angles
        photon_energy: photon energy [eV]
        subst: substrate element indices (1-based)
        free: free element indices (1-based)
        linked: linked element indices (1-based)
        charges: charge of each element
        **kwargs: extra arguments for H5Writer.set_basic_params

    Returns:
        Output file path
    """
    from deppro.io.h5_writer import H5Writer

    # Initialization
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

    # Result profile
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

    # Apply the initialization results
    init.apply_to_writer(writer)

    # Profile settings
    writer.set_profiles(
        Drlt=D_result,
        Dtrn=D_true,
        Dstk=D_true.copy(),
        Xobs=init.results['Xobs'],
        Xopt=init.results['Xopt'],
    )

    # Objective function
    writer.set_objectives(Qrlt=np.array([[1000.0, 0.0, 1000.0]]))

    writer.write()

    return filepath
