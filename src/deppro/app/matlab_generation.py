"""Adapter to the MATLAB DepthProfiler CLI.

A thin layer that calls the external MATLAB generator and hands the confirmed
manifest to the analysis-side contract. It interprets neither numbers nor physics: **what the producer declared is carried through as is**.

The producer-side contract (`<DEPPRO_MATLAB_SOURCE_ROOT>/DepthProfiler/_cli/README.md` is authoritative):

* `_cli/dp <cfg.json> [result.json]` generates, and confirms in a **separate process**
* exit code 0 means "generation succeeded and the closed files were re-read and found consistent"
* ``manifest.json`` in the output folder is authoritative; ``result.json`` is a copy with the same ``generationId``
* the ``confirmed`` block is a **structural** confirmation, not a claim of physical correctness

The CLI location is ``DEPPRO_DEPTHPROFILER_CLI``. **Importing works with it
unset**; the error, naming the variable, comes when it is used. The output location
reuses the existing ``DEPPRO_SIMULATION_ROOT``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from deppro.paths import env_path_or_placeholder

__all__ = [
    "MatlabGenerationError",
    "ManifestRejected",
    "UnsupportedLayout",
    "matlab_cli",
    "generate",
    "load_confirmed_manifest",
    "stack_coordinates",
    "grid_coverage",
    "sample_ids",
    "SUPPORTED_LAYOUTS",
    "SCHEMA_MAJOR",
]

SCHEMA_MAJOR = 1
SUPPORTED_LAYOUTS = ("rectangular_grid", "rectangular_grid_partial")
NO_GRID_LAYOUTS = ("flat_samples",)


class MatlabGenerationError(RuntimeError):
    """Starting or running the CLI failed."""


class ManifestRejected(ValueError):
    """A manifest exists but does not meet the acceptance conditions.

    Partial JSON, missing confirmation, generation mismatch and the like are rejected here.
    """


class UnsupportedLayout(NotImplementedError):
    """The declared layout cannot be handled by this layer.

    **The exception that prevents a silent reshape to a rectangle.** Stopping explicitly
    is better than treating an unknown layout as a known one.
    """


def matlab_cli() -> Path:
    """Location of the MATLAB generation CLI (`_cli/dp`) (``DEPPRO_DEPTHPROFILER_CLI``).

    Importing works with it unset; the error, naming the variable, comes when this is called.
    There is no default because the MATLAB side lives in a different place on every machine.
    """
    return env_path_or_placeholder("DEPPRO_DEPTHPROFILER_CLI")


@dataclass(frozen=True)
class GenerationResult:
    """The result of one generation.

    Attributes
    ----------
    manifest:
        Contents of ``manifest.json`` in the output folder (already ``confirmed``).
    folder:
        Where the generated files are.
    generation_id:
        Identifier of this generation; the first half of every sample_id.
    """

    manifest: Dict[str, Any]
    folder: Path
    generation_id: str

    @property
    def main_file(self) -> Path:
        return Path(self.manifest["mainFile"])

    @property
    def layout(self) -> Dict[str, Any]:
        return self.manifest.get("layout", {})


def generate(
    cfg: Dict[str, Any],
    *,
    timeout: Optional[float] = None,
    cli: Optional[Path] = None,
) -> GenerationResult:
    """Pass cfg to the MATLAB CLI, generate, and return the confirmed manifest.

    Parameters
    ----------
    cfg:
        Generation settings. The keys map 1:1 to the table in the producer's ``_cli/README.md``.
        ``outputFolder`` is chosen by the caller (this layer makes no default).
    timeout:
        Seconds. :class:`MatlabGenerationError` when exceeded.
    cli:
        Path of the CLI. Defaults to ``DEPPRO_DEPTHPROFILER_CLI``.

    Raises
    ------
    MatlabGenerationError
        The CLI is missing, cannot be started, exits non-zero, or times out.
    ManifestRejected
        Exit code 0 but the manifest does not meet the acceptance conditions.
    """
    if "outputFolder" not in cfg:
        raise ValueError("cfg needs outputFolder (this layer does not choose a default)")
    exe = Path(cli) if cli is not None else matlab_cli()
    if not exe.exists():
        raise MatlabGenerationError(
            f"MATLAB CLI not found: {exe}. "
            "Set DEPPRO_DEPTHPROFILER_CLI to the path of _cli/dp"
        )

    folder = Path(cfg["outputFolder"]).expanduser()
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "cfg.json"
        result_path = Path(tmp) / "result.json"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        # Start it in a new process group. `_cli/dp` does not exec the MATLAB
        # generation stage but starts it as a child, so subprocess.run(timeout=...)
        # by default kills only the direct child (`sh`). MATLAB keeps running, and
        # a successful output appears after "timeout = failure" was reported (a fixture confirmed the grandchild survives).
        try:
            proc = subprocess.Popen(
                [str(exe), str(cfg_path), str(result_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            raise MatlabGenerationError(f"cannot start the CLI: {exe} ({exc})") from exc

        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            killed = _kill_process_group(proc)
            proc.communicate()
            raise MatlabGenerationError(
                f"CLI did not finish within {timeout} s: {exe}"
                f" (process group terminated: {killed})"
            ) from exc

        if proc.returncode != 0:
            raise MatlabGenerationError(
                f"CLI exited non-zero (returncode={proc.returncode}).\n"
                f"stdout tail:\n{(out or '')[-2000:]}\n"
                f"stderr tail:\n{(err or '')[-2000:]}"
            )

    result = load_confirmed_manifest(folder)
    _check_matches_request(result, cfg, folder)
    return result


def _kill_process_group(proc: "subprocess.Popen") -> str:
    """Kill the CLI and everything it started, not just the direct child."""
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        proc.kill()
        return "process only (no group)"
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return f"pgid {pgid}"
        except OSError:
            break
        try:
            proc.wait(timeout=5)
            return f"pgid {pgid}"
        except subprocess.TimeoutExpired:
            continue
    return f"pgid {pgid} (may not have exited)"


def _check_matches_request(result: GenerationResult, cfg: Dict[str, Any], folder: Path) -> None:
    """Check that the returned manifest is **the product of this cfg**.

    Exit code 0 and the internal consistency of `confirmed` cannot tell the case
    where the CLI ignored cfg and returned an old confirmed manifest as is. Depending
    on how `ifExists` is handled that can really happen, so the point counts and the output folder are compared.
    """
    axes = result.manifest.get("resolved", {}).get("axes", {})
    problems: List[str] = []
    for cfg_key, axis_key in (("gridX", "nx"), ("gridY", "ny"), ("timeSteps", "nt")):
        want = cfg.get(cfg_key)
        if want is None:
            continue
        got = axes.get(axis_key)
        if got is None or int(got) != int(want):
            problems.append(f"cfg.{cfg_key}={want} but resolved.axes.{axis_key}={got}")
    declared_folder = result.manifest.get("outputFolder")
    if isinstance(declared_folder, str) and declared_folder:
        try:
            same = Path(declared_folder).expanduser().resolve() == folder.resolve()
        except OSError:
            same = str(declared_folder) == str(folder)
        if not same:
            problems.append(
                f"manifest.outputFolder={declared_folder!r} differs from the folder read, {str(folder)!r}"
            )
    if problems:
        raise ManifestRejected(
            "manifest is not the product of this request (an old confirmed manifest may have been reused):\n  "
            + "\n  ".join(problems)
        )


def load_confirmed_manifest(folder: Path | str) -> GenerationResult:
    """Read ``manifest.json`` from the output folder, check the acceptance conditions, and return it.

    Exit code 0 alone is not enough. Checked here:

    1. the manifest exists and parses as a whole (no partial JSON is picked up)
    2. the major of ``schemaVersion`` matches what this layer expects
    3. ``status == "ok"``
    4. ``confirmed`` exists and ``confirmed.generationId`` equals
          ``generationId`` (**a confirmation of another generation is not reused**)
    5. ``resolved.files`` is **non-empty**, and every named file exists with the recorded size
    6. ``mainFile`` exists, is a regular file, and points to the same artifact as one
          entry of ``resolved.files``

    5 and 6 were added later. Before that, a missing ``resolved.files`` became ``[]``,
    the loop ran zero times, and **a synthetic manifest without a single H5 was
    accepted**: consistent only inside the manifest, with nothing on disk, yet called
    "confirmed".

    ``confirmed`` is a structural confirmation, not physical correctness and not the
    soundness of every stack. The producer's ``confirmed.scope`` says so.
    """
    folder = Path(folder).expanduser()
    path = folder / "manifest.json"
    if not path.is_file():
        derived = folder / "derived_view.json"
        if derived.is_file():
            raise ManifestRejected(
                f"{folder} is a derived_view (a copy for GUI checks), not a generated dataset."
                " Do not use it as an entry point. The authoritative value is parent.folder in derived_view.json"
            )
        raise ManifestRejected(f"manifest.json is missing: {folder}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestRejected(f"manifest.json cannot be parsed: {path} ({exc})") from exc

    if not isinstance(manifest, dict):
        raise ManifestRejected(
            f"top level of manifest.json is {type(manifest).__name__} (expected an object): {path}"
        )

    schema = str(manifest.get("schemaVersion", ""))
    major = schema.split(".")[0] if schema else ""
    if major != str(SCHEMA_MAJOR):
        raise ManifestRejected(
            f"schemaVersion {schema!r} is outside what this layer expects (major {SCHEMA_MAJOR})"
        )

    status = manifest.get("status")
    if status != "ok":
        raise ManifestRejected(f"status is {status!r} (not ok)")

    generation_id = manifest.get("generationId")
    if not generation_id:
        raise ManifestRejected("generationId is missing")

    confirmed = manifest.get("confirmed")
    if not isinstance(confirmed, dict):
        raise ManifestRejected(
            "confirmed block is missing: the confirmation from outside the generation process has not happened"
        )
    if confirmed.get("status") != "ok":
        # The confirmed block of producer 1.0.0 has only at/by/stack/elements. The
        # version number stayed while the contents grew, so the version alone cannot
        # tell them apart; the shape identifies it and makes the reason concrete.
        looks_old = "status" not in confirmed and "generationId" not in confirmed
        hint = (" (confirmed has neither status nor generationId: the shape of producer versions before 1.1.0;"
                " outputs of that generation have no /stackxyt either)" if looks_old else "")
        raise ManifestRejected(f"confirmed.status is {confirmed.get('status')!r}{hint}")
    if confirmed.get("generationId") != generation_id:
        raise ManifestRejected(
            "confirmed belongs to another generation: "
            f"manifest={generation_id!r} confirmed={confirmed.get('generationId')!r}"
        )

    declared_folder = manifest.get("outputFolder")
    if isinstance(declared_folder, str) and declared_folder:
        try:
            same = Path(declared_folder).expanduser().resolve() == folder.resolve()
        except OSError:
            same = str(declared_folder) == str(folder)
        if not same:
            raise ManifestRejected(
                f"manifest.outputFolder={declared_folder!r} differs from the folder read, "
                f"{str(folder)!r}. A manifest of another output folder was put in the wrong place"
            )

    _check_files(manifest, folder)
    return GenerationResult(manifest=manifest, folder=folder, generation_id=generation_id)


def _check_files(manifest: Dict[str, Any], folder: Path) -> None:
    """Do the files the manifest names exist on disk with the recorded sizes?

    A size match is a cheap check, neither tamper detection nor an integrity
    guarantee (the producer's ``unknown.artifactHashes`` says so). What is checked here
    is only "whether what the manifest names is on disk".

    **An empty list is not a success.** Earlier, a missing ``resolved.files`` became
    ``[]``, the for loop never ran, and the check was passed by default. A synthetic
    manifest without a single H5 was accepted. Zero items checked is not the same as passed.
    """
    resolved = manifest.get("resolved")
    if not isinstance(resolved, dict):
        raise ManifestRejected("resolved block is missing (or not an object)")
    files = resolved.get("files")
    if not isinstance(files, list) or not files:
        raise ManifestRejected(
            "resolved.files is missing or empty. A manifest that names no generated file at all"
            " cannot be called \"confirmed\""
        )

    problems: List[str] = []
    present: Dict[Path, int] = {}
    for entry in files:
        if not isinstance(entry, dict):
            problems.append(f"non-object entry in resolved.files: {entry!r}")
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            problems.append(f"entry of resolved.files without a name: {entry!r}")
            continue
        target = folder / name
        if not target.is_file():
            problems.append(f"{name}: in the manifest but not on disk")
            continue
        size = target.stat().st_size
        expected = entry.get("bytes")
        if expected is not None:
            if not isinstance(expected, (int, float)) or isinstance(expected, bool):
                problems.append(f"{name}: bytes is not a number ({expected!r})")
                continue
            if size != int(expected):
                problems.append(f"{name}: {size} bytes, the manifest says {int(expected)}")
                continue
        try:
            present[target.resolve()] = size
        except OSError:
            present[target] = size

    # mainFile merely existing is not enough. A main H5 that is not in the list
    # means this layer would open an artifact that was never checked.
    main = manifest.get("mainFile")
    if not isinstance(main, str) or not main:
        problems.append("mainFile is missing (or not a string)")
    elif not Path(main).expanduser().is_absolute():
        # A relative path resolves against the cwd, so the same manifest would be
        # accepted or rejected depending on the caller's current directory. The producer
        # always writes absolute paths; stop here instead of resolving silently.
        problems.append(
            f"mainFile is a relative path ({main!r}); it would depend on the cwd, so it is rejected"
            " (the producer writes absolute paths)"
        )
    else:
        mp = Path(main).expanduser()
        if not mp.is_file():
            problems.append(f"mainFile is not on disk: {main}")
        else:
            try:
                key = mp.resolve()
            except OSError:
                key = mp
            if key not in present:
                problems.append(
                    f"mainFile {main} matches none of resolved.files"
                    " (this would open an artifact that the list did not check)"
                )

    if problems:
        raise ManifestRejected("manifest and disk disagree:\n  " + "\n  ".join(problems))


def stack_coordinates(result: GenerationResult, *, allow_partial: bool = False) -> np.ndarray:
    """Return the (x, y, t) of every stack as ``(stack, 3)``.

    In the file (as h5py sees it) the shape is ``(3, stack)``, transposed relative to
    MATLAB's report. **Always returning ``(stack, 3)``** is the contract of this layer.

    This is the **index correspondence** of this layout, not physical coordinates,
    not acquisition order and not time. Those are separate information and are not recorded.

    There are three outcomes (:func:`grid_coverage` reports the facts):

    1. the grid holds and is complete → coordinates are returned
    2. the grid holds but the last frame is incomplete → returned only with an explicit
          ``allow_partial=True``; never handed over silently as complete
    3. no grid / unknown → :class:`UnsupportedLayout`. **Even then
          :func:`sample_ids` works** (identifiers do not depend on the grid)

    ``stack`` need not be a multiple of ``nx*ny``. The generator's scene is a function
    of the flat index and has no notion of a frame boundary.
    """
    cov = grid_coverage(result)
    ltype = cov["type"]
    if ltype in NO_GRID_LAYOUTS:
        raise UnsupportedLayout(
            f"layout {ltype!r} has no grid (for measured data, stack is the number of measurement"
            " points, unrelated to the dxdydt grid). Coordinates cannot be made, but **sample_ids works**"
        )
    if ltype not in SUPPORTED_LAYOUTS:
        raise UnsupportedLayout(
            f"layout {ltype!r} cannot be handled by this layer (supported: {SUPPORTED_LAYOUTS})."
            " Not reshaped as a rectangle. sample_ids works"
        )
    if not cov["consistent"]:
        raise ManifestRejected(
            f"layout {cov['nx']}x{cov['ny']}x{cov['nt']} does not fit the declared stack {cov['stack']}"
            f" ({cov['note']}). Only the grid interpretation fails; as a flat sequence of samples"
            " it is valid: sample_ids works as is"
        )
    if cov["partial"] and not allow_partial:
        raise ManifestRejected(
            f"the last frame has only {cov['last_frame_count']}/{cov['nx']*cov['ny']} points:"
            " a partial grid, not returned silently as complete. To accept it, say so with"
            " stack_coordinates(..., allow_partial=True)"
        )

    nx, ny, nt, stack = cov["nx"], cov["ny"], cov["nt"], cov["stack"]

    import h5py  # heavy, imported lazily

    with h5py.File(result.main_file, "r") as f:
        if "stackxyt" not in f:
            raise ManifestRejected(
                f"{result.main_file} has no /stackxyt: output from before the generator declared"
                " coordinates (the boundary is the generator version, not a date; some outputs"
                " made on the morning of 2026-09-07 lack it too)"
            )
        raw = np.asarray(f["stackxyt"])

    if raw.ndim != 2:
        raise ManifestRejected(f"/stackxyt has {raw.ndim} dimensions (expected 2)")

    # The orientation is **decided by the schema**. Guessing from the shape is
    # ambiguous when stack==3, (3,3), and would silently return a transposed array
    # (it actually did). The stored shape is declared as (3, stack); only that is accepted.
    if raw.shape == (3, stack):
        xyt = raw.T
    elif stack == 3 and raw.shape == (3, 3):
        xyt = raw.T          # rows are (x,y,t) as declared; the schema removes the ambiguity
    elif raw.shape == (stack, 3) and stack != 3:
        raise ManifestRejected(
            f"/stackxyt has shape {raw.shape}. The stored shape is declared as (3, stack) = (3, {stack});"
            " a transposed array is not accepted silently"
        )
    else:
        raise ManifestRejected(
            f"/stackxyt has shape {raw.shape}; for the declared stack={stack} it should be "
            f"(3, {stack})"
        )

    if not np.all(np.isfinite(xyt)):
        raise ManifestRejected("/stackxyt contains non-finite values")
    if not np.array_equal(xyt, np.rint(xyt)):
        raise ManifestRejected("/stackxyt contains non-integer values (grid indices)")

    idx = xyt.astype(np.int64)
    lo = idx.min(axis=0)
    hi = idx.max(axis=0)
    if lo.min() < 1 or hi[0] > nx or hi[1] > ny or hi[2] > nt:
        raise ManifestRejected(
            f"/stackxyt out of range: x∈[{lo[0]},{hi[0]}] y∈[{lo[1]},{hi[1]}] "
            f"t∈[{lo[2]},{hi[2]}], declared 1..{nx} / 1..{ny} / 1..{nt}"
        )

    # Agreement with the declared formula. Coordinates that are merely self-consistent
    # do not prove that the rows are in the declared order (permuted rows are still coordinates).
    # The formula is a bijection for ANY stack, so a truncated last frame is
    # checked exactly like a complete one.
    sv = np.arange(1, stack + 1, dtype=np.int64)
    want = np.stack([(sv - 1) % nx + 1,
                     ((sv - 1) % (nx * ny)) // nx + 1,
                     (sv - 1) // (nx * ny) + 1], axis=1)
    if not np.array_equal(idx, want):
        bad = int(np.flatnonzero((idx != want).any(axis=1))[0])
        raise ManifestRejected(
            f"/stackxyt disagrees with the declared formula (flattenOrder.stackToXYT)."
            f" First mismatch at stack={bad + 1}: got {idx[bad].tolist()} want {want[bad].tolist()}"
        )
    return xyt


def grid_coverage(result: GenerationResult) -> Dict[str, Any]:
    """Report **as facts** whether the grid interpretation holds and, if so, whether it is complete.

    No guessing: only the producer-declared ``layout`` and ``resolved.axes.stack`` are
    compared. So that callers cannot take coordinates without looking at this,
    :func:`stack_coordinates` requires an explicit opt-in for a partial grid.
    """
    layout = result.layout
    ltype = layout.get("type")
    out: Dict[str, Any] = {"type": ltype, "partial": False, "consistent": False,
                           "nx": None, "ny": None, "nt": None,
                           "stack": None, "last_frame_count": None, "note": ""}
    if ltype in NO_GRID_LAYOUTS:
        out["note"] = "no grid is declared"
        return out
    if ltype not in SUPPORTED_LAYOUTS:
        out["note"] = f"unsupported layout {ltype!r}"
        return out

    nx, ny, nt = (_positive_int(layout, k, "layout") for k in ("nx", "ny", "nt"))
    stack = _declared_stack(result)
    ppf = nx * ny
    out.update(nx=nx, ny=ny, nt=nt, stack=stack)

    if ltype == "rectangular_grid":
        out["consistent"] = (ppf * nt == stack)
        out["last_frame_count"] = ppf if out["consistent"] else None
        out["note"] = ("" if out["consistent"] else
                       f"claims a complete grid but {nx}*{ny}*{nt} = {ppf*nt} ≠ stack {stack}")
        return out

    # rectangular_grid_partial: nt counts the truncated frame too
    tail = stack - (nt - 1) * ppf
    out["partial"] = True
    out["last_frame_count"] = tail
    out["consistent"] = 0 < tail <= ppf
    out["note"] = ("" if out["consistent"] else
                   f"claims a partial grid but the last frame has {tail} points (outside 1..{ppf})")
    return out


def _positive_int(d: Dict[str, Any], key: str, where: str) -> int:
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v) or int(v) < 1:
        raise ManifestRejected(f"{where}.{key} is not a positive integer: {v!r}")
    return int(v)


def _declared_stack(result: GenerationResult) -> int:
    axes = result.manifest.get("resolved", {}).get("axes", {})
    if not isinstance(axes, dict) or "stack" not in axes:
        raise ManifestRejected("resolved.axes.stack is missing")
    return _positive_int(axes, "stack", "resolved.axes")


def sample_ids(result: GenerationResult, indices: Sequence[int] | None = None) -> List[str]:
    """Build sample_ids. Default: every stack.

    ``<generationId>#<stackIndex>``, where stackIndex is 1-based and is **the number
    assigned when first written**. An array row number is a position, not an identifier:
    after extraction or reordering, row 7 is a different sample point. Carry the original
    number through extraction and reordering.
    """
    stack = _declared_stack(result)
    if indices is None:
        return [f"{result.generation_id}#{i}" for i in range(1, stack + 1)]

    # Passing 0-based row numbers as is is a common slip. #0 does not exist in 1-based
    # numbering, yet letting it through produces a well-formed but meaningless id.
    # bool and non-integers are rejected here too: through int() they would silently
    # turn True -> 1 and 1.9 -> 1, so **separately specified points get the same id**,
    # turn True -> 1 and 1.9 -> 1, so **separately specified points get the same id**, and a legitimate duplicate could not be told from a type-conversion collision.
    out: List[str] = []
    bad: List[str] = []
    for raw in indices:
        if isinstance(raw, bool):
            bad.append(f"{raw!r} (bool)")
            continue
        if isinstance(raw, (int, np.integer)):
            i = int(raw)
        elif isinstance(raw, (float, np.floating)):
            if not np.isfinite(raw) or float(raw) != int(raw):
                bad.append(f"{raw!r} (not an integer)")
                continue
            i = int(raw)
        else:
            bad.append(f"{raw!r} ({type(raw).__name__})")
            continue
        if i < 1 or i > stack:
            bad.append(f"{i} (out of range)")
            continue
        out.append(f"{result.generation_id}#{i}")
    if bad:
        raise ValueError(
            f"stackIndex must be a 1-based integer in 1..{stack}. Rejected values: {bad[:8]}"
            " (check that array row numbers or bools were not passed)"
        )
    return out
