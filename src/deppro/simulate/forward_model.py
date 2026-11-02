"""
Forward Model Protocol
======================

Defines the ForwardModel Protocol that abstracts the D(z) -> I(theta)
transformation. The existing TransferMatrix already satisfies this
Protocol. A thin BeerLambertModel wrapper is provided for explicit
conformance and as a template for future SESSA correction.

Usage:
    from deppro.simulate.forward_model import ForwardModel, BeerLambertModel

    # Use existing TransferMatrix directly (it satisfies the Protocol):
    from deppro.transfer_matrix import TransferMatrix
    tm = TransferMatrix(depth, step_size, angles)
    assert isinstance(tm, ForwardModel)  # True

    # Or use the explicit wrapper:
    model = BeerLambertModel(depth, step_size, angles, photon_energy=8000)
    model.compute(D, elements)
    I_obs = model.forward(D)
"""

from typing import List, Optional, Protocol, runtime_checkable

import numpy as np

from ..depth_profile import ElementConfig
from ..imfp import CompoundDatabase
from ..transfer_matrix import TransferMatrix


@runtime_checkable
class ForwardModel(Protocol):
    """Protocol for forward models: D(z) -> I(theta).

    Any class implementing compute() and forward() with these
    signatures can be used as a ForwardModel. This allows future
    SESSA elastic scattering correction injection without changing
    callers.

    Current implementations:
        - TransferMatrix (existing, satisfies Protocol)
        - BeerLambertModel (thin wrapper)

    Future:
        - SESSACorrectedModel (Beer-Lambert + correction table)
    """

    def compute(
        self,
        D: np.ndarray,
        elements: List[ElementConfig],
        compound_db: Optional[CompoundDatabase] = None,
    ) -> 'ForwardModel':
        """Compute internal state from depth profile.

        Args:
            D: Concentration profile [n_depth, n_elements]
            elements: Element configurations
            compound_db: Compound database (optional)

        Returns:
            self (for chaining)
        """
        ...

    def forward(
        self,
        D: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """Apply forward model: D(z) -> I(theta).

        Args:
            D: Concentration profile [n_depth, n_elements]
            normalize: If True, normalize to sum=1 at each angle

        Returns:
            Angle-resolved intensities [n_angles, n_elements]
        """
        ...


class BeerLambertModel:
    """Beer-Lambert forward model wrapping existing TransferMatrix.

    Thin adapter that makes TransferMatrix conform to ForwardModel
    Protocol explicitly. Also serves as a template for future
    SESSACorrectedModel implementation.

    Note: TransferMatrix already satisfies ForwardModel Protocol,
    so this wrapper is optional. Use it when you want explicit
    Protocol conformance or when building a SESSA correction chain.
    """

    def __init__(
        self,
        depth: np.ndarray,
        step_size: float,
        angles: np.ndarray,
        photon_energy: float = 1486.6,
        work_function: float = 4.5,
        gamma_eff: float = 1.0,
    ):
        self._tm = TransferMatrix(
            depth=depth,
            step_size=step_size,
            angles=angles,
            photon_energy=photon_energy,
            work_function=work_function,
            gamma_eff=gamma_eff,
        )

    def compute(
        self,
        D: np.ndarray,
        elements: List[ElementConfig],
        compound_db: Optional[CompoundDatabase] = None,
    ) -> 'BeerLambertModel':
        self._tm.compute(D, elements, compound_db)
        return self

    def forward(
        self,
        D: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        return self._tm.forward(D, normalize=normalize)

    @property
    def trans_mat(self) -> Optional[np.ndarray]:
        """Access underlying transfer matrix [n_angles, n_depth, n_elements]."""
        return self._tm.trans_mat

    @property
    def lambda_basic(self) -> Optional[np.ndarray]:
        """Access basic IMFP matrix [n_elements, n_elements]."""
        return self._tm.lambda_basic

    @property
    def tm(self) -> TransferMatrix:
        """Access underlying TransferMatrix object."""
        return self._tm


# Future stub (not implemented in Phase 1):
#
# class SESSACorrectedModel:
#     """Beer-Lambert + SESSA elastic scattering correction.
#
#     Applies angle-dependent correction factors from SESSA
#     Monte Carlo simulations to the Beer-Lambert transfer matrix.
#
#     correction_table: [n_angles, n_elements] float64
#         Multiplicative correction factors.
#         > 1.0 at low angles (elastic scattering extends effective depth)
#         < 1.0 at high angles (saturation from multiple scattering)
#
#     Usage:
#         beer_lambert = BeerLambertModel(...)
#         sessa_table = load_sessa_table('Si_8keV.csv')
#         model = SESSACorrectedModel(beer_lambert, sessa_table)
#         model.compute(D, elements)
#         I_obs = model.forward(D)
#     """
#
#     def __init__(self, base_model: BeerLambertModel,
#                  correction_table: np.ndarray):
#         self._base = base_model
#         self._correction = correction_table
#
#     def compute(self, D, elements, compound_db=None):
#         self._base.compute(D, elements, compound_db)
#         return self
#
#     def forward(self, D, normalize=True):
#         I_base = self._base.forward(D, normalize=False)
#         I_corrected = I_base * self._correction
#         if normalize:
#             total = I_corrected.sum(axis=1, keepdims=True)
#             total[total == 0] = 1
#             I_corrected = I_corrected / total
#         return I_corrected
