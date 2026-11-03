"""
DNN-based spectrum denoiser for HAXPES pipeline.

Wraps the ResNet-FCNN of the ``dnndenoiser`` package for use in the
``deppro.haxpes`` processing pipeline. The network is the one the JVST A /
SIA papers used: ``dnndenoiser.models.network.DenoisingNetwork`` built with
the papers' constructor arguments (``num_features=256``,
``num_hidden_units=100``, ``layer_type='ResNet-FCNN'``,
``encoder_output_dim=64``; the class's own default ``layer_type`` is
``'FCNN'``; for this layer type only ``num_features`` and ``layer_type``
determine the parameter shapes, the other two are accepted for call-site
compatibility) has 658,177 parameters and is ``state_dict``-compatible with
``arhaxpes_denoise.network.DenoisingNetwork`` of the papers'
Zenodo deposit (10.5281/zenodo.22092109), so weights trained against either
class load here with ``strict=True``.

``dnndenoiser`` is a declared optional dependency (``pip install
'deppro[nn]'`` or ``pip install dnndenoiser``). It is imported as an
installed package only: no sibling checkout is searched and no ``sys.path``
entry is added. Importing this module does not import torch or dnndenoiser;
constructing :class:`HaxpesDenoiser` does, and fails with an ``ImportError``
that names the missing package.

Per-element weights ``model_<Elem>_fold<n>.pt`` and their normalization
``norm_<Elem>.json`` (``global_min`` / ``global_max`` of the training stack)
are not part of this repository.

Usage::

    denoiser = HaxpesDenoiser(model_dir='outputs/denoiser_models')
    denoised_pixel = denoiser.denoise_pixel(pixel_data)
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

TARGET_LENGTH = 256


def _make_interp_matrix(n_old: int, n_new: int) -> np.ndarray:
    """Build (n_new, n_old) linear interpolation matrix."""
    x_old = np.linspace(0, 1, n_old)
    x_new = np.linspace(0, 1, n_new)
    M = np.zeros((n_new, n_old), dtype=np.float32)
    for j, xn in enumerate(x_new):
        idx = np.searchsorted(x_old, xn, side='right') - 1
        idx = np.clip(idx, 0, n_old - 2)
        frac = (xn - x_old[idx]) / (x_old[idx + 1] - x_old[idx])
        M[j, idx] = 1.0 - frac
        M[j, idx + 1] = frac
    return M


_INTERP_CACHE: dict = {}


def _resample(arr: np.ndarray, n_new: int) -> np.ndarray:
    """Resample last axis via matrix multiply. arr: (..., n_old) -> (..., n_new)"""
    n_old = arr.shape[-1]
    if n_old == n_new:
        return arr.astype(np.float32)
    key = (n_old, n_new)
    if key not in _INTERP_CACHE:
        _INTERP_CACHE[key] = _make_interp_matrix(n_old, n_new)
    M = _INTERP_CACHE[key]
    return (arr.astype(np.float32) @ M.T).astype(np.float32)


def _import_network():
    """Import torch and the network class, naming what is missing."""
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "HaxpesDenoiser needs torch: pip install 'deppro[nn]' (or pip install torch)"
        ) from e
    try:
        from dnndenoiser.models.network import DenoisingNetwork
    except ImportError as e:
        raise ImportError(
            "HaxpesDenoiser needs the dnndenoiser package: pip install 'deppro[nn]' "
            "(or pip install dnndenoiser). A sibling checkout is not searched."
        ) from e
    return torch, DenoisingNetwork


class HaxpesDenoiser:
    """DNN-based spectrum denoiser for HAXPES pipeline.

    Loads per-element models and normalization parameters, then applies
    denoising to spectra in the pipeline.

    Parameters
    ----------
    model_dir : str or Path
        Directory containing model_*.pt and norm_*.json files.
    device : str
        PyTorch device ('cpu', 'mps', 'cuda').
    fold : int
        Which CV fold model to use.
    """

    def __init__(
        self,
        model_dir: str,
        device: str = 'cpu',
        fold: int = 0,
    ):
        torch, DenoisingNetwork = _import_network()

        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.fold = fold
        self._models: Dict[str, 'DenoisingNetwork'] = {}
        self._norms: Dict[str, dict] = {}
        self._torch = torch
        self._DenoisingNetwork = DenoisingNetwork

    def _load_element(self, elem: str):
        """Lazy-load model and normalization for one element."""
        if elem in self._models:
            return

        # Map short names to training names
        elem_map = {'Si': 'Si1s', 'Ti': 'Ti1s', 'Al': 'Al1s', 'C': 'C1s', 'O': 'O1s'}
        train_name = elem_map.get(elem, elem)

        model_path = self.model_dir / f"model_{train_name}_fold{self.fold}.pt"
        norm_path = self.model_dir / f"norm_{train_name}.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not norm_path.exists():
            raise FileNotFoundError(f"Norm params not found: {norm_path}")

        # Load normalization
        with open(norm_path) as f:
            self._norms[elem] = json.load(f)

        # Load model
        model = self._DenoisingNetwork(
            num_features=TARGET_LENGTH,
            num_hidden_units=100,
            layer_type='ResNet-FCNN',
            encoder_output_dim=64,
        ).to(self.device)
        state = self._torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        self._models[elem] = model

    def denoise_spectrum(
        self,
        spectrum: np.ndarray,
        element: str,
    ) -> np.ndarray:
        """Denoise a batch of spectra for one element.

        Parameters
        ----------
        spectrum : ndarray
            (n_energy, n_angles) or (n_energy,) raw spectrum.
        element : str
            Element name ('Si', 'Ti', 'Al', 'C', 'O').

        Returns
        -------
        denoised : ndarray
            Same shape as input, denoised spectrum.
        """
        self._load_element(element)
        model = self._models[element]
        norm = self._norms[element]

        n_energy = spectrum.shape[0]

        # Reshape to (N, n_energy) for batch processing
        if spectrum.ndim == 1:
            batch = spectrum[np.newaxis, :]  # (1, n_energy)
        else:
            batch = spectrum.T  # (n_angles, n_energy)

        # Resample to TARGET_LENGTH
        batch_256 = _resample(batch, TARGET_LENGTH)

        # Normalize (element_global)
        g_min = norm['global_min']
        g_max = norm['global_max']
        g_denom = g_max - g_min
        if g_denom < 1e-10:
            g_denom = 1.0
        batch_norm = ((batch_256 - g_min) / g_denom).astype(np.float32)

        # Inference
        with self._torch.no_grad():
            inp = self._torch.tensor(batch_norm, dtype=self._torch.float32).to(self.device)
            out, _ = model(inp)
            denoised_norm = out.cpu().numpy()

        # Denormalize
        denoised_256 = denoised_norm * g_denom + g_min

        # Resample back to original n_energy
        denoised = _resample(denoised_256, n_energy)

        # Reshape back
        if spectrum.ndim == 1:
            return denoised[0]
        else:
            return denoised.T  # (n_energy, n_angles)

    def denoise_pixel(
        self,
        pixel_data: Dict[str, np.ndarray],
        elem_names: Optional[list] = None,
    ) -> Dict[str, np.ndarray]:
        """Denoise all spectra in a pixel.

        An element whose model or normalization file is missing is passed
        through **unchanged** (and a warning is logged), so the returned dict
        is not guaranteed to be denoised in every element. Callers that need
        the guarantee should call :meth:`denoise_spectrum` per element.

        Parameters
        ----------
        pixel_data : dict
            {elem: (n_energy, n_angles)} binned spectra.
        elem_names : list, optional
            Elements to denoise. Default: all keys in pixel_data.

        Returns
        -------
        denoised_pixel : dict
            Same structure, denoised spectra.
        """
        if elem_names is None:
            elem_names = list(pixel_data.keys())

        result = {}
        for elem in elem_names:
            if elem not in pixel_data:
                continue
            try:
                result[elem] = self.denoise_spectrum(pixel_data[elem], elem)
            except FileNotFoundError as e:
                logger.warning("denoise_pixel: %s — %s passed through without denoising", e, elem)
                result[elem] = pixel_data[elem]  # fallback: no denoising
        return result
