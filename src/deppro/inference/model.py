"""
MLX Neural Network Models for Depth Profile Prediction
======================================================

DepthMLP:         Standard MLP mapping I_obs -> D(z)
DepthResMLP:      MLP with residual connections
DepthConv1D:      1D CNN along angle axis
DepthElemNet:     Shared encoder + per-element decoders
DepthTransformer: Self-attention over angle tokens

Further architectures can be added with ``register_architecture``.

All models share the same API:
  model = Model(input_dim, output_dim, ...)
  D_pred = model(I_obs)  # (batch, output_dim), non-negative via softplus
"""

import mlx.core as mx
import mlx.nn as nn


def _count_params(model: nn.Module) -> int:
    """Count total trainable parameters."""
    import mlx.utils
    return sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))


# ---------------------------------------------------------------------------
# 1. DepthMLP — baseline
# ---------------------------------------------------------------------------


class DepthMLP(nn.Module):
    """MLP for ARXPS inverse problem: I_obs -> D(z).

    Parameters
    ----------
    input_dim : int
        I_obs dimension (n_angles * n_elements).
    output_dim : int
        D(z) dimension (n_depth * n_elements).
    hidden_dim : int
        Hidden layer width.
    n_hidden : int
        Number of hidden layers.

    Notes
    -----
    - Softplus output ensures non-negative concentrations.
    - LayerNorm for training stability (invariant to batch size).
    """

    def __init__(
        self,
        input_dim: int = 84,
        output_dim: int = 120,
        hidden_dim: int = 256,
        n_hidden: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden

        layers = []
        in_d = input_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # ReLU after each Linear+LayerNorm pair (not after final Linear)
            if i % 2 == 1 and i < len(self.layers) - 1:
                x = nn.relu(x)
        return nn.softplus(x)

    def n_parameters(self) -> int:
        return _count_params(self)


# ---------------------------------------------------------------------------
# 2. DepthResMLP — MLP with residual (skip) connections
# ---------------------------------------------------------------------------


class _ResBlock(nn.Module):
    """Linear → LayerNorm → ReLU with residual connection."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def __call__(self, x: mx.array) -> mx.array:
        return x + nn.relu(self.norm(self.linear(x)))


class DepthResMLP(nn.Module):
    """MLP with residual connections for better gradient flow.

    Architecture:
        Linear(input→hidden) → [ResBlock(hidden)]×n_hidden → Linear(hidden→output) → softplus
    """

    def __init__(
        self,
        input_dim: int = 84,
        output_dim: int = 120,
        hidden_dim: int = 256,
        n_hidden: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden

        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.blocks = [_ResBlock(hidden_dim) for _ in range(n_hidden)]
        self.proj_out = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.norm_in(self.proj_in(x)))
        for block in self.blocks:
            x = block(x)
        return nn.softplus(self.proj_out(x))

    def n_parameters(self) -> int:
        return _count_params(self)


# ---------------------------------------------------------------------------
# 3. DepthConv1D — 1D CNN treating (angles, elements) as spatial
# ---------------------------------------------------------------------------


class DepthConv1D(nn.Module):
    """1D CNN: treat input as (n_angles, n_elem) image, convolve along angles.

    Architecture:
        Reshape(batch, n_angles, n_elem) → Conv1d stack → Flatten → MLP head → softplus

    Captures local angle-domain patterns that MLP misses.
    """

    def __init__(
        self,
        input_dim: int = 84,
        output_dim: int = 120,
        hidden_dim: int = 256,
        n_hidden: int = 4,
        *,
        n_angles: int = 14,
        n_elem: int = 6,
        n_filters: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.n_angles = n_angles
        self.n_elem = n_elem

        # Conv1d stack: (batch, n_angles, n_elem) → (batch, n_angles, n_filters)
        self.conv1 = nn.Conv1d(n_elem, n_filters, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.LayerNorm(n_filters)
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.LayerNorm(n_filters)

        # MLP head: flatten → hidden → output
        flat_dim = n_angles * n_filters
        self.head_in = nn.Linear(flat_dim, hidden_dim)
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.head_blocks = [_ResBlock(hidden_dim) for _ in range(max(1, n_hidden - 2))]
        self.head_out = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        batch = x.shape[0]
        # Reshape to (batch, n_angles, n_elem)
        x = x.reshape(batch, self.n_angles, self.n_elem)

        # Conv1d along angles
        x = nn.relu(self.norm1(self.conv1(x)))  # (batch, n_angles, n_filters)
        x = nn.relu(self.norm2(self.conv2(x)))  # (batch, n_angles, n_filters)

        # Flatten + MLP head
        x = x.reshape(batch, -1)
        x = nn.relu(self.head_norm(self.head_in(x)))
        for block in self.head_blocks:
            x = block(x)
        return nn.softplus(self.head_out(x))

    def n_parameters(self) -> int:
        return _count_params(self)


# ---------------------------------------------------------------------------
# 4. DepthElemNet — shared encoder + per-element decoders
# ---------------------------------------------------------------------------


class DepthElemNet(nn.Module):
    """Shared encoder + per-element decoders.

    Architecture:
        Shared encoder: Linear(input→hidden) → [ResBlock]×(n_hidden-1)
        Per-element:    Linear(hidden→n_depth) → softplus  (×n_elem)

    Physically motivated: elements have independent depth profiles
    but share the same angle-resolved observation space.
    """

    def __init__(
        self,
        input_dim: int = 84,
        output_dim: int = 120,
        hidden_dim: int = 256,
        n_hidden: int = 4,
        *,
        n_depth: int = 20,
        n_elem: int = 6,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.n_depth = n_depth
        self.n_elem = n_elem

        # Shared encoder
        self.enc_in = nn.Linear(input_dim, hidden_dim)
        self.enc_norm = nn.LayerNorm(hidden_dim)
        n_enc = max(1, n_hidden - 1)
        self.enc_blocks = [_ResBlock(hidden_dim) for _ in range(n_enc)]

        # Per-element decoders: each maps hidden → n_depth
        self.decoders = [nn.Linear(hidden_dim, n_depth) for _ in range(n_elem)]

    def __call__(self, x: mx.array) -> mx.array:
        # Shared encoder
        h = nn.relu(self.enc_norm(self.enc_in(x)))
        for block in self.enc_blocks:
            h = block(h)

        # Per-element decode + concat
        parts = [nn.softplus(dec(h)) for dec in self.decoders]
        # Stack → (batch, n_elem, n_depth) → reshape to (batch, n_depth * n_elem)
        out = mx.stack(parts, axis=1)  # (batch, n_elem, n_depth)
        # Interleave to match flat layout: [d0_e0, d0_e1, ..., d0_eN, d1_e0, ...]
        out = mx.transpose(out, axes=(0, 2, 1))  # (batch, n_depth, n_elem)
        return out.reshape(x.shape[0], -1)

    def n_parameters(self) -> int:
        return _count_params(self)


# ---------------------------------------------------------------------------
# 5. DepthTransformer — self-attention over angle tokens
# ---------------------------------------------------------------------------


class DepthTransformer(nn.Module):
    """Transformer encoder treating each angle as a token.

    Architecture:
        Reshape(batch, n_angles, n_elem)
        → Linear(n_elem → hidden_dim)          (token embedding)
        → + learnable positional embedding
        → [TransformerEncoderLayer]×n_hidden    (self-attention + FFN)
        → mean pool over angles
        → Linear(hidden_dim → output_dim) → softplus

    Motivation: self-attention captures global angle-angle relationships
    (e.g., surface vs bulk sensitivity) without locality bias of CNN.
    """

    def __init__(
        self,
        input_dim: int = 84,
        output_dim: int = 120,
        hidden_dim: int = 256,
        n_hidden: int = 4,
        *,
        n_angles: int = 14,
        n_elem: int = 6,
        n_heads: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.n_angles = n_angles
        self.n_elem = n_elem

        # Token embedding: (n_elem) → (hidden_dim)
        self.token_proj = nn.Linear(n_elem, hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)

        # Learnable positional embedding: (n_angles, hidden_dim)
        self.pos_embed = mx.zeros((n_angles, hidden_dim))

        # Transformer encoder layers
        self.enc_layers = [
            nn.TransformerEncoderLayer(
                dims=hidden_dim,
                num_heads=n_heads,
                mlp_dims=hidden_dim * 4,
            )
            for _ in range(n_hidden)
        ]
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Output head: pool → project → softplus
        self.head = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x: mx.array) -> mx.array:
        batch = x.shape[0]

        # Reshape to tokens: (batch, n_angles, n_elem)
        x = x.reshape(batch, self.n_angles, self.n_elem)

        # Token embedding + positional encoding
        x = nn.relu(self.token_norm(self.token_proj(x)))  # (batch, n_angles, hidden_dim)
        x = x + self.pos_embed  # broadcast (n_angles, hidden_dim)

        # Transformer encoder
        for layer in self.enc_layers:
            x = layer(x, mask=None)

        # Mean pool over angle dimension
        x = self.final_norm(x)
        x = mx.mean(x, axis=1)  # (batch, hidden_dim)

        return nn.softplus(self.head(x))

    def n_parameters(self) -> int:
        return _count_params(self)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    'mlp': DepthMLP,
    'resmlp': DepthResMLP,
    'conv1d': DepthConv1D,
    'elemnet': DepthElemNet,
    'transformer': DepthTransformer,
}

# Weight-file detection for architectures added with register_architecture:
# list of (name, weight_key_prefix, detect_dims)
_REGISTERED_LOADERS = []


def register_architecture(name: str, cls, weight_key_prefix: str,
                          detect_dims) -> None:
    """Add a model architecture to ``create_model`` and ``load_model``.

    Parameters
    ----------
    name : architecture name used with ``create_model``.
    cls : model class, constructed as
          ``cls(input_dim, output_dim, hidden_dim, n_hidden)``.
    weight_key_prefix : prefix of the weight keys that identifies this
          architecture in a saved .npz file (checked after the built-in
          architectures, before the MLP fallback).
    detect_dims : callable(weight_dict) -> (input_dim, output_dim,
          hidden_dim, n_hidden).
    """
    if name in ('mlp', 'resmlp', 'conv1d', 'elemnet', 'transformer'):
        raise ValueError(f"Architecture '{name}' is built in and cannot be replaced")
    MODEL_REGISTRY[name] = cls
    _REGISTERED_LOADERS[:] = [e for e in _REGISTERED_LOADERS if e[0] != name]
    _REGISTERED_LOADERS.append((name, weight_key_prefix, detect_dims))


def create_model(
    arch: str = 'mlp',
    input_dim: int = 84,
    output_dim: int = 120,
    hidden_dim: int = 256,
    n_hidden: int = 4,
    **kwargs,
) -> nn.Module:
    """Create a model by name.

    Parameters
    ----------
    arch : str
        Architecture name: 'mlp', 'resmlp', 'conv1d', 'elemnet', 'transformer',
        or one added with ``register_architecture``.
    """
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown arch '{arch}'. Available: {list(MODEL_REGISTRY.keys())}")
    cls = MODEL_REGISTRY[arch]
    return cls(input_dim=input_dim, output_dim=output_dim,
               hidden_dim=hidden_dim, n_hidden=n_hidden, **kwargs)
