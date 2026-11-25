"""
Depth profile from one HAXPES-Lab angle-resolved measurement
============================================================

Input: a folder of SES ``.pxt`` files recorded in Fixed mode with the Angular56
lens mode, where each region is stored as (energy, angle slice, ...) with the
angle slices already binned by SES ("Number of Slices", 12 in the measurements
this was checked on). All frames and all files that pass ``--filter`` are summed
into one spectrum per region and angle slice, so the result is one depth
profile for the whole folder.

Steps: sum frames -> linear background and peak area per angle slice ->
divide by the cross section -> keep the emission angles inside
``--angle-min/--angle-max`` -> SemiFreeSolver (L1, interface width fixed).

Run::

    python -m deppro.examples.haxpes_lab_depth_profile DATA_DIR --polar 30

    # another sample: copy haxpes_lab_sample.yaml, edit it, then
    python -m deppro.examples.haxpes_lab_depth_profile DATA_DIR --polar 30 \\
        --sample my_sample.yaml --thickness-init 100   # nominal film thickness [A]

    # only list what the files contain (regions, shapes, unreadable regions)
    python -m deppro.examples.haxpes_lab_depth_profile DATA_DIR --inspect

Needs the ``[fit]`` extra (toyomacro: the PXT reader and the reference tables).

The emission angle of a slice is ``flip(detector angle) + polar``: ``--polar``
is the angle from the analyser's lens axis to the sample normal, and the
order of the slices is reversed as on the instrument this was written for.
SES does not record the manipulator angle in a dedicated field, so the value
has to come from the measurement log.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

DEFAULT_SAMPLE = Path(__file__).with_name("haxpes_lab_sample.yaml")
ROLES = ("substrate", "free", "linked")
# The fitted thickness often stays close to its starting value. The profile
# that is written out is the one started from --thickness-init; these other
# starts are solved only to show how much the answer depends on the start.
DEFAULT_STARTS = (30.0, 60.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0)


# ------------------------------------------------------------------
# Sample description
# ------------------------------------------------------------------

def load_sample(path: Path):
    """Read the sample YAML into (experiment config, ElementConfig list, raw entries)."""
    from deppro import ElementConfig
    from deppro.haxpes import HAXPESExperimentConfig

    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    entries = spec["elements"]
    names = []
    for i, e in enumerate(entries):
        missing = [k for k in ("region", "symbol", "orbital", "compound",
                               "binding_energy", "role") if k not in e]
        if missing:
            raise SystemExit(f"{path}: element #{i + 1} lacks {missing}")
        if e["role"] not in ROLES:
            raise SystemExit(f"{path}: role of {e['region']!r} must be one of {ROLES}")
        # Two regions of one element (e.g. two chemical states) need distinct names.
        name = e.get("name", e["symbol"])
        if name in names:
            raise SystemExit(f"{path}: two entries are called {name!r}; "
                             "give each a distinct 'name:'")
        names.append(name)

    from deppro.imfp import CompoundDatabase
    known = CompoundDatabase().list_compounds()
    unknown = sorted({e["compound"] for e in entries if e["compound"] not in known})
    if unknown:
        raise SystemExit(f"{path}: compound {unknown} is not in the compound table; "
                         "see CompoundDatabase().list_compounds()")

    idx = {r: [i for i, e in enumerate(entries) if e["role"] == r] for r in ROLES}
    if not idx["substrate"]:
        raise SystemExit(f"{path}: one element must have role: substrate")
    if not idx["free"]:
        raise SystemExit(f"{path}: at least one element must have role: free")

    elements = [
        ElementConfig(e["symbol"], str(e["orbital"]), e["compound"],
                      binding_energy=float(e["binding_energy"]),
                      oxidation_state=e.get("oxidation_state"))
        for e in entries
    ]
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#333333", "#ff7f0e",
               "#9467bd", "#8c564b", "#17becf"]
    experiment = HAXPESExperimentConfig(
        name=spec.get("name", Path(path).stem),
        elem_names=names,
        elem_colors={n: palette[i % len(palette)] for i, n in enumerate(names)},
        region_to_elem={e["region"]: n for e, n in zip(entries, names)},
        free_indices=idx["free"],
        linked_indices=idx["linked"],
        substrate_indices=idx["substrate"],
        substrate_element=names[idx["substrate"][0]],
    )
    return experiment, elements, entries


# ------------------------------------------------------------------
# What is in the files
# ------------------------------------------------------------------

def inspect_files(pxt_dir: Path, file_filter: str):
    """List every region of every matching file. Returns (rows, photon energies)."""
    from toyomacro.io.readers.pxt_reader import PXTReader, _parse_wave_notes

    files = sorted(f for f in Path(pxt_dir).glob("*.pxt") if file_filter in f.name)
    if not files:
        raise SystemExit(f"no *.pxt file containing {file_filter!r} in {pxt_dir}")
    rows, energies = [], set()
    for f in files:
        reader = PXTReader(f)
        reader._ensure_parsed()
        for region in reader._regions:
            notes = _parse_wave_notes(region.wave_notes)
            name = notes.get("Region Name")
            if "Excitation Energy" in notes:
                energies.add(float(notes["Excitation Energy"]))
            rows.append({
                "file": f.name,
                "region": name,
                "shape": tuple(int(n) for n in region.data.shape),
                "lens_mode": notes.get("Lens Mode"),
                "acquisition": notes.get("Acquisition Mode"),
                "step_time_ms": notes.get("Step Time"),
            })
    return rows, sorted(energies)


def print_inspection(rows):
    print(f"{'file':40s} {'region':10s} {'shape':24s} lens / mode / step time [ms]")
    for r in rows:
        region = r["region"] if r["region"] else "(no header)"
        print(f"{r['file'][:40]:40s} {region:10s} {str(r['shape']):24s} "
              f"{r['lens_mode']} / {r['acquisition']} / {r['step_time_ms']}")
    bad = sorted({r["file"] for r in rows if not r["region"]})
    if bad:
        print("\nRegions without a readable header are skipped; the file is probably "
              "incomplete (for example a measurement stopped before its last frame):")
        for name in bad:
            print(f"  {name}")


# ------------------------------------------------------------------
# Cross sections
# ------------------------------------------------------------------

def cross_sections(elements, entries, photon_energy):
    """One cross section per element; returns (values, where each came from)."""
    from deppro.haxpes import apply_cross_section_correction

    given = [e.get("sigma") for e in entries]
    if any(g is not None for g in given) and not all(g is not None for g in given):
        # The package's own values are in Mb; a mixture is only right if the
        # given ones are in Mb as well, which this script cannot check.
        print("WARNING: sigma is given for some elements only. The others use the "
              "package's Yeh-Lindau extrapolation (Mb); the given values must be in "
              "Mb and consistent with it.", file=sys.stderr)
    sigmas, sources = [], []
    for el, g in zip(elements, given):
        if g is not None:
            sigmas.append(float(g))
            sources.append("sample file")
            continue
        try:
            _, s = apply_cross_section_correction(np.ones((1, 1)), [el], photon_energy)
        except KeyError:
            raise SystemExit(
                f"No built-in cross section for {el.symbol} {el.orbital}. "
                f"Add 'sigma:' for it to the sample file.") from None
        if el.orbital != "1s":
            raise SystemExit(
                f"The built-in cross sections cover 1s levels only; add 'sigma:' "
                f"for {el.symbol} {el.orbital} to the sample file.")
        if not np.isfinite(s[0]) or s[0] <= 0:
            raise SystemExit(f"Cross section of {el.symbol} {el.orbital} at "
                             f"{photon_energy} eV came out as {s[0]}.")
        sigmas.append(float(s[0]))
        sources.append("yeh_lindau_extrap")
    return np.array(sigmas), sources


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pxt_dir", type=Path, help="folder with the SES .pxt files")
    ap.add_argument("--filter", default="Fixed", help="use files whose name contains this (default: Fixed)")
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE, help="sample YAML (default: the C/Al2O3/TiO2/Si example)")
    ap.add_argument("--polar", type=float, default=None, help="angle from the lens axis to the sample normal [deg] (required to solve)")
    ap.add_argument("--angle-min", type=float, default=5.0, help="lowest emission angle used [deg]")
    ap.add_argument("--angle-max", type=float, default=55.0, help="highest emission angle used [deg]")
    ap.add_argument("--photon-energy", type=float, default=None, help="eV; default: read from the files")
    ap.add_argument("--thickness-init", type=float, default=240.0,
                    help="starting film thickness [A]: the nominal (design) thickness of the film")
    ap.add_argument("--scan", type=float, nargs="*", default=list(DEFAULT_STARTS),
                    help="other starting thicknesses solved for comparison only "
                         "(default: %(default)s; give --scan with no value to skip)")
    ap.add_argument("--thickness-max", type=float, default=500.0, help="upper bound of the film thickness [A]")
    ap.add_argument("--total-depth", type=float, default=600.0, help="depth range of the profile [A]")
    ap.add_argument("--n-depth", type=int, default=20, help="number of depth points")
    ap.add_argument("--alpha", type=float, default=0.01, help="L1 regularisation weight")
    ap.add_argument("--out", type=Path, default=Path("depth_profile_out"), help="output folder")
    ap.add_argument("--inspect", action="store_true", help="list the regions in the files and stop")
    args = ap.parse_args(argv)

    rows, energies = inspect_files(args.pxt_dir, args.filter)
    print_inspection(rows)
    sys.stdout.flush()  # keep the listing ahead of any error message when piped
    if args.inspect:
        return 0
    if args.polar is None:
        raise SystemExit("--polar is required (angle from the lens axis to the sample normal)")

    import deppro
    from deppro.haxpes import (HAXPESSolverConfig, integrate_peak_areas,
                               load_prebinned_pxt, run_depth_profiling)

    experiment, elements, entries = load_sample(args.sample)
    names = experiment.elem_names

    if args.photon_energy is not None:
        photon_energy = args.photon_energy
    elif len(energies) == 1:
        photon_energy = energies[0]
    else:
        raise SystemExit(f"photon energy in the files is {energies}; pass --photon-energy")

    # ---- load: sum all frames and files ----
    data = load_prebinned_pxt(args.pxt_dir, experiment=experiment,
                              file_filter=args.filter, polar_offset=args.polar)
    missing = [e["region"] for e, n in zip(entries, names) if n not in data]
    if missing:
        found = sorted({r["region"] for r in rows if r["region"]})
        raise SystemExit(f"regions {missing} of the sample file are not in the data; "
                         f"the files contain {found}")
    angle_sets = [data[n]["emission_angles"] for n in names]
    if any(a.shape != angle_sets[0].shape or not np.allclose(a, angle_sets[0])
           for a in angle_sets):
        raise SystemExit("the regions do not share one set of angle slices")
    angles = angle_sets[0]

    # ---- peak areas ----
    Ismp, _ = integrate_peak_areas({n: data[n]["spectra"] for n in names},
                                   {n: data[n]["energy"] for n in names}, names)
    sigmas, sigma_src = cross_sections(elements, entries, photon_energy)
    Xobs = Ismp / sigmas[:, None]

    mask = (angles >= args.angle_min) & (angles <= args.angle_max)
    if mask.sum() < 4:
        raise SystemExit(f"only {int(mask.sum())} angle slices lie in "
                         f"{args.angle_min}-{args.angle_max} deg (slices: "
                         f"{np.round(angles, 1).tolist()}); check --polar")
    if np.any(Ismp[:, mask] <= 0):
        bad = [n for n, row in zip(names, Ismp[:, mask]) if np.any(row <= 0)]
        print(f"WARNING: non-positive peak area for {bad}; check the energy window "
              "and the background of these regions.", file=sys.stderr)

    # ---- solve: the nominal start, then the comparison starts ----
    if not 10.0 < args.thickness_init < args.thickness_max:
        raise SystemExit(f"--thickness-init must lie inside 10-{args.thickness_max:g} A")

    def solve(t0):
        solver_config = HAXPESSolverConfig(
            alpha=args.alpha, thickness_init=t0,
            thickness_bounds=(10.0, args.thickness_max),
            n_depth=args.n_depth, totaldepth=args.total_depth)
        return run_depth_profiling(
            Xobs[:, mask], angles[mask], elements, solver_config=solver_config,
            experiment=experiment, photon_energy=photon_energy, verbose=False)

    result, depth, angles_used, rrmse = solve(args.thickness_init)
    scan = [(args.thickness_init, float(result.thickness), float(rrmse))]
    for t0 in args.scan:
        if 10.0 < t0 < args.thickness_max and t0 != args.thickness_init:
            r = solve(t0)
            scan.append((t0, float(r[0].thickness), float(r[3])))
    scan.sort()

    # ---- report ----
    sub = experiment.substrate_indices[0]
    obs = Xobs[:, mask].T
    obs_n = obs / obs[:, [sub]]
    fit_n = result.I_opt / result.I_opt[:, [sub]]

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "depth_profile.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["depth_A"] + names)
        for z, row in zip(depth, result.D_opt):
            w.writerow([f"{z:g}"] + [f"{v:.4f}" for v in row])
    with open(args.out / "angle_fit.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["emission_deg"] + [f"{n}_obs" for n in names] + [f"{n}_fit" for n in names])
        for a, o, f in zip(angles_used, obs_n, fit_n):
            w.writerow([f"{a:.2f}"] + [f"{v:.5g}" for v in o] + [f"{v:.5g}" for v in f])
    summary = {
        "deppro_version": deppro.__version__,
        "data_dir": str(args.pxt_dir),
        "files": sorted({r["file"] for r in rows if r["region"]}),
        "sample": experiment.name,
        "photon_energy_eV": photon_energy,
        "polar_deg": args.polar,
        "angles_all_deg": np.round(angles, 3).tolist(),
        "angles_used_deg": np.round(angles_used, 3).tolist(),
        "sigma": {n: {"value": float(s), "source": src}
                  for n, s, src in zip(names, sigmas, sigma_src)},
        "alpha": args.alpha,
        "thickness_init_A": args.thickness_init,
        "other_starts": [{"thickness_init_A": t0, "thickness_A": t, "rrmse": e}
                         for t0, t, e in scan],
        "thickness_bounds_A": [10.0, args.thickness_max],
        "interface_width_A_fixed": float(result.abruptness),
        "thickness_A": float(result.thickness),
        "rrmse_substrate_normalised": float(rrmse),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, n in enumerate(names):
        c = experiment.elem_colors[n]
        ax1.step(depth, result.D_opt[:, i], where="pre", color=c, label=n)
        if i != sub:
            ax2.plot(angles_used, obs_n[:, i], "o", color=c, label=f"{n} measured")
            ax2.plot(angles_used, fit_n[:, i], "-", color=c)
    ax1.axvline(result.thickness, color="0.5", ls="--", lw=0.8)
    ax1.set(xlabel="depth [Å]", ylabel="atomic fraction",
            title=f"{experiment.name}: thickness {result.thickness:.0f} Å")
    ax1.legend()
    ax2.set(xlabel="emission angle [deg]", yscale="log",
            ylabel=f"intensity / {names[sub]} (points: measured, lines: fit)",
            title=f"rRMSE {rrmse:.0%}")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "depth_profile.png", dpi=150)

    print(f"\nphoton energy {photon_energy} eV, polar {args.polar} deg")
    print(f"angle slices [deg]: {np.round(angles, 1).tolist()}")
    print(f"used ({args.angle_min}-{args.angle_max}): {np.round(angles_used, 1).tolist()}")
    print("cross sections: " + ", ".join(
        f"{n} {s:.3e} ({src})" for n, s, src in zip(names, sigmas, sigma_src)))
    print(f"\nfilm thickness {result.thickness:.1f} Å "
          f"(start {args.thickness_init:g}, bounds 10-{args.thickness_max:g}), "
          f"interface width fixed at {result.abruptness:g} Å")
    print(f"rRMSE (intensities normalised to {names[sub]}) {rrmse:.1%}")
    print(f"\n{'depth[Å]':>9s} " + " ".join(f"{n:>6s}" for n in names))
    for z, row in zip(depth, result.D_opt):
        print(f"{z:9.0f} " + " ".join(f"{v:6.3f}" for v in row))
    if len(scan) > 1:
        print("\nDependence on the starting thickness (comparison only):")
        print("  start [Å]  thickness [Å]   rRMSE")
        for t0, t, e in scan:
            mark = "   <- written out" if t0 == args.thickness_init else ""
            print(f"  {t0:9g}  {t:13.1f}  {e:6.1%}{mark}")
        lowest = min(scan, key=lambda x: x[2])
        if lowest[2] < 0.8 * rrmse:
            print(f"  NOTE: the start {lowest[0]:g} Å fits better (rRMSE {lowest[2]:.1%}, "
                  f"thickness {lowest[1]:.1f} Å). A lower residual alone does not make "
                  "it the right thickness; compare with what is known about the film.")
    print(f"\nwritten: {args.out}/depth_profile.csv, angle_fit.csv, summary.json, depth_profile.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
