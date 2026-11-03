"""One place that decides where generated files go.

Why
----------
Tests, diagnostics and benchmarks used to write their figures to
``/Users/<name>/Projects/depthprofiler/src/outputs/<name>.png``, **an absolute path
on the developer's machine, hard-coded**. ``src/outputs`` was a Git-tracked symlink
pointing into a personal cloud directory. It resolves on the development machine,
but in a CI checkout the link dangles and gives ``FileNotFoundError``. In the CI run
of 2026-08-02 the whole analysis passed and only the final savefig failed, 4 times.


Resolution order:

1. the environment variable ``DEPPRO_OUTPUTS``
2. when used from a checkout, ``outputs/`` at the repository root (in ``.gitignore``)
3. when used from an installed package (no ``pyproject.toml`` at ``repo_root()``),
      ``outputs/`` in the current directory, with a one-time warning (2026-10-19)

In every case :func:`ensure_outputs_dir` creates it if needed. The default is the
repository root rather than a temporary directory so that written figures can be
found later. In CI the checkout is discarded each time, so nothing accumulates.

As with ``DEPPRO_SIMULATION_ROOT`` and the others, **importing works with it unset**;
the directory is decided only when something is actually written.

⚠️ A confusable name already exists: ``DEPPRO_OUTPUT_ROOT`` (singular + ROOT) is used by
three CUDA benchmarks that are not in the public release. It is where JSON is written,
not figures, so it was left alone. Whether to merge them is a separate decision.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "env_value",
    "env_path_or_placeholder",
    "repo_root",
    "outputs_dir",
    "ensure_outputs_dir",
    "in_checkout",
    "simulation_root",
]


def env_value(name: str) -> str | None:
    """Value of the environment variable ``name``. ``None`` when unset **or an empty string**.

    Every ``DEPPRO_*`` path variable is read by this rule: ``export DEPPRO_X=`` (emptied)
    is treated the same as unset. With ``os.environ.get(name, default)`` an empty string
    bypasses the default and becomes ``Path("")`` (= the current directory), and the
    variable name disappears from the error. A value of only whitespace gets no special treatment.
    """
    value = os.environ.get(name)
    return value if value else None


def env_path_or_placeholder(name: str) -> Path:
    """The path of ``name``. ``<name>-not-set`` when unset or an empty string.

    Importing works with it unset; the error at the point of reading that path names the variable.
    ``~`` is expanded.
    """
    return Path(env_value(name) or f"{name}-not-set").expanduser()


def repo_root() -> Path:
    """The repository root (two levels above ``src/deppro/paths.py``).

    When called from an installed package this is two levels above site-packages, not
    the root of a checkout. :func:`in_checkout` tells them apart.
    """
    return Path(__file__).resolve().parents[2]


def in_checkout() -> bool:
    """Whether ``repo_root()`` is a checkout of this repository (has ``pyproject.toml``)."""
    return (repo_root() / "pyproject.toml").is_file()


_warned_installed_outputs = False


def outputs_dir() -> Path:
    """Return where generated files go (without creating it).

    Use this when you only need the location. To actually write, use
    :func:`ensure_outputs_dir`.

    1. ``DEPPRO_OUTPUTS`` if set.
    2. ``<repo>/outputs`` when used from a checkout.
    3. From an installed package without ``DEPPRO_OUTPUTS``, ``outputs/`` **in the
          current directory**, so that nothing is written inside site-packages.
          A warning is emitted once in that case.
    """
    env = env_value("DEPPRO_OUTPUTS")
    if env:
        return Path(env).expanduser()
    if in_checkout():
        return repo_root() / "outputs"
    global _warned_installed_outputs
    if not _warned_installed_outputs:
        import warnings
        warnings.warn(
            "DEPPRO_OUTPUTS is not set and deppro is not running from a checkout; "
            f"writing outputs under the current directory: {Path.cwd() / 'outputs'}. "
            "Set DEPPRO_OUTPUTS to choose the location.",
            stacklevel=2,
        )
        _warned_installed_outputs = True
    return Path.cwd() / "outputs"


def ensure_outputs_dir(*subdirs: str) -> Path:
    """Create and return where generated files go.

    Subdirectories can be added, as in ``ensure_outputs_dir("semifree", "grid")``.
    Writers call this and then append the file name:

        path = ensure_outputs_dir() / "gradient_profile_effect.png"
        plt.savefig(path, dpi=150)
    """
    path = outputs_dir().joinpath(*subdirs)
    path.mkdir(parents=True, exist_ok=True)
    return path


def simulation_root() -> Path:
    """Location of the MATLAB-generated simulation data (``DEPPRO_SIMULATION_ROOT``).

    The root of the benchmark H5 files generated on the MATLAB side, such as
    ``Checkered/10x10x200_7elem_noisy/...``. Read-only external data, so it is kept
    separate from :func:`outputs_dir`.

    As with the other ``DEPPRO_*`` variables, **importing works with it unset**; the
    error, naming the variable, comes when the data is actually read. No default is
    written here because the location of the real data differs per machine.
    """
    return env_path_or_placeholder("DEPPRO_SIMULATION_ROOT")


def legacy_simulation_root() -> Path:
    """Simulation-data location for the existing benchmark and diagnostic scripts.

    ``DEPPRO_SIMULATION_ROOT`` if set (an empty string counts as unset), otherwise the
    home-relative location that was used before
    (``~/Documents/MATLAB/Simulation/DepthProfiler``).

    The home-relative default was hard-coded in ``benchmarks/paths.py`` and elsewhere and
    is gathered here in one place. The behavior when unset is as before. **New code does
    not use this**; it uses :func:`simulation_root` (which gives a path naming the variable when unset).

    Not a public API (not in ``__all__``). The only callers are ``benchmarks/paths.py``
    (and diagnostic scripts that are not in the public release).
    ``tests/unit/test_env_paths.py`` detects references by name or attribute from
    elsewhere in ``src/deppro`` (including ``tests/``). References that build the name
    from a string, such as ``getattr``, are not detected.
    """
    env = env_value("DEPPRO_SIMULATION_ROOT")
    if env:
        return Path(env).expanduser()
    return Path.home() / "Documents/MATLAB/Simulation/DepthProfiler"
