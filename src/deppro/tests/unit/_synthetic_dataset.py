"""テスト用の合成データセット（MATLAB writer と同じ H5 レイアウト）。

``test_*.py`` ではないので pytest は収集しない。preflight と contract の両テストが使う。
Dstk は (px, elem, depth)、Xstk は (px, elem, angle)、TransMatObs は (elem, depth, angle)。
X は保存 T から厳密に作る（T·D を角度ごとに総和 1 へ正規化）ので、stored_T 閉包は丸め誤差のみ。
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

SYMS = ["Si", "Si", "Ti", "Al", "C", "B", "O"]
CHARGE = [0.0, 4.0, 4.0, 3.0, 0.0, 0.0, -2.0]
# Producer-style stem (six digits + letters, as contract.dataset_files expects),
# visibly synthetic: no real dataset uses this name.
STEM = "000000SyntheticStack"


def layout_for(nx: int, ny: int, stack: int) -> dict:
    """生成側（MATLAB dpLayout / Python producer._layout）と同じ 3 状態のうち格子 2 状態。

    倍数なら rectangular_grid、余りがあれば rectangular_grid_partial で nt は切れたフレームも数える。
    """
    ppf = nx * ny
    n_full, rem = divmod(stack, ppf)
    if rem == 0:
        return {"type": "rectangular_grid", "nx": nx, "ny": ny, "nt": n_full, "complete": True}
    return {"type": "rectangular_grid_partial", "nx": nx, "ny": ny, "nt": n_full + 1,
            "complete": False, "lastFrameCount": rem}


def producer_manifest(nx: int, ny: int, stack: int, *, time_size: int,
                      generation_id: str = "20260909T000000000-000000") -> dict:
    """preflight / アダプタが読む最小限の manifest 1.1.0（schemaVersion, generationId, mainFile, confirmed,
    layout, resolved.axes）。役割宣言は含めない（roles.declared は UNRESOLVED のまま）。"""
    return {"schemaVersion": "1.1.0", "generationId": generation_id, "mainFile": f"{STEM}.h5",
            "confirmed": {"status": "ok", "generationId": generation_id, "stack": stack},
            "layout": layout_for(nx, ny, stack),
            "resolved": {"axes": {"nx": nx, "ny": ny, "stack": stack, "timeSize": time_size}}}


def make_dataset(root: Path, *, family="Checkered", nx=2, ny=2, nt=3, variant="noisy",
                 n_depth=4, n_angle=3, dxdydt=None, n_px_override=None, nan_at=None,
                 zero_px=None, write_prj=True, extra_main=False, manifest=None,
                 spectra=True, seed=0, noise=1.0, fencepost=False, currentstack=None,
                 extra_main_name="other_result.h5", d_scale=None, zero_all_D=False,
                 write_obs=False, x_from_composition=False, stack=None,
                 declare_layout=False, declare_flat=False, sim_or_exp=None,
                 lambda_transposed=False) -> Path:
    """合成データセットを書く。

    stack:          総 stack 数を明示（省略時 nx*ny*nt）。nx*ny の倍数でなくてもよい（素数 stack の
                    fixture）。フォルダ名の nt は「要求 nt」のまま（生成側 CLI が cfg の timeSteps で
                    名付けるのと同じ）。
    declare_layout: 生成側と同じ形の manifest 1.1.0 を同居させる（layout は producer の規則で
                    rectangular_grid / rectangular_grid_partial を自動判定）。manifest= が与えられた
                    ときはそちらが優先。
    declare_flat:   格子なし（実験データ）の宣言 layout.type = flat_samples を持つ manifest を同居させる。
    sim_or_exp:     ezdeprof/SimulationOrExperiment に書く値（1 = simulation, 0 = experiment）。
    lambda_transposed: ezdeprof/lambdabasic を転置して書く（forward は正しい向きで計算）。Python producer
                    が MATLAB dialect と逆向きに保存した 2026-09-09 の実例を再現する fixture。
    """
    E = len(SYMS)
    name = f"{nx}x{ny}x{nt}_{E}elem" + (f"_{variant}" if variant else "")
    folder = root / family / name
    folder.mkdir(parents=True)
    P = stack or n_px_override or nx * ny * nt
    if manifest is None and declare_layout:
        manifest = producer_manifest(nx, ny, P, time_size=(dxdydt[2] if dxdydt is not None else nt))
    if manifest is None and declare_flat:
        manifest = producer_manifest(nx, ny, P, time_size=(dxdydt[2] if dxdydt is not None else nt))
        manifest["layout"] = {"type": "flat_samples", "stack": P,
                              "gridNote": "SimOrExp=0: stack is the number of measured points; the dxdydt grid is unrelated"}
    Z, A = n_depth, n_angle
    rng = np.random.default_rng(seed)
    D = rng.random((P, E, Z))
    D /= D.sum(axis=1, keepdims=True)
    T = rng.random((E, Z, A)) + 0.1
    lam_b = (rng.random((E, E)) * 20 + 10)                      # (E, E) as h5py sees it
    gamma = 1.0
    step = 600.0 / (Z - 1) if fencepost else 30.0
    theta = np.linspace(0.0, 60.0, A)
    if x_from_composition:
        # producer-declared/1 forward: per-pixel T(D) from lambda_basic / gamma, angle-normalised
        from deppro.app.contract.metrics import composition_transfer_matrix, forward as _fwd
        Dp = np.transpose(D, (0, 2, 1))                          # (P, Z, E)
        Xp = _fwd(composition_transfer_matrix(Dp, lam_b, theta, step, gamma), Dp)   # (P, A, E)
        X = np.transpose(Xp, (0, 2, 1))                          # (P, E, A) MATLAB layout
    else:
        Xp = np.einsum("eza,pez->pea", T, D)
        X = Xp / Xp.sum(axis=1, keepdims=True)
    if nan_at is not None:
        D[nan_at, 0, 0] = np.nan
    if zero_px is not None:
        D[zero_px] = 0.0
        X[zero_px] = 0.0
    if d_scale is not None:                       # (depth index, factor): Σ_elem D != 1 there
        D[:, :, d_scale[0]] *= d_scale[1]
    if zero_all_D:
        D[:] = 0.0
    depth = np.arange(Z) * step
    with h5py.File(folder / f"{STEM}.h5", "w") as f:
        f.create_dataset("Dstk", data=D.astype(np.float32))
        f.create_dataset("Xstk", data=X.astype(np.float32))
        f.create_dataset("Drlt", data=np.zeros_like(D, dtype=np.float32))
        g = f.create_group("ezdeprof")
        g["dxdydt"] = np.asarray(dxdydt if dxdydt is not None else [nx, ny, nt],
                                 dtype=np.uint16).reshape(3, 1)
        g["stack"] = np.array([P], dtype=np.uint64)
        g["depth"] = depth.reshape(1, Z).astype(np.float32)
        g["stepsize"] = np.array([step], dtype=np.float32)
        g["totaldepth"] = np.array([600.0 if fencepost else Z * step], dtype=np.float32)
        if currentstack is not None:
            g["currentstack"] = np.array([currentstack], dtype=np.uint64)
        if sim_or_exp is not None:
            g["SimulationOrExperiment"] = np.array([sim_or_exp], dtype=np.uint8)
        g["thetaobs"] = theta.reshape(1, A).astype(np.float32)
        g["noise"] = np.array([noise], dtype=np.float32)
        g.create_dataset("elementsymbol", data=np.array(SYMS, dtype=object).reshape(E, 1),
                         dtype=h5py.string_dtype())
        g["charge"] = np.asarray(CHARGE, dtype=np.float32).reshape(E, 1)
        g["free"] = np.array([2, 3, 4, 5, 6, 0], dtype=np.uint8).reshape(6, 1)   # 0 = 番兵
        g["linked"] = np.array([[7]], dtype=np.uint8)
        g["const"] = np.array([[0]], dtype=np.uint8)
        g["subst"] = np.array([[1]], dtype=np.uint8)
        g["linkedconst"] = np.array([3.0, 3.0, 2.5, 1.0, 1.0, 0.0], dtype=np.float32).reshape(6, 1)
        g["TransMatObs"] = T.astype(np.float32)
        g.create_dataset("orbital", data=np.array(["1s"] * E, dtype=object).reshape(E, 1),
                         dtype=h5py.string_dtype())
        g.create_dataset("compound", data=np.array(SYMS, dtype=object).reshape(E, 1),
                         dtype=h5py.string_dtype())
        g["bindingenergy"] = np.linspace(100.0, 700.0, E).astype(np.float32).reshape(E, 1)
        g["photonenergy"] = np.full((E, 1), 8000.0, dtype=np.float32)
        g["workfunction"] = np.array([4.5], dtype=np.float32)
        g["lambdabasic"] = (lam_b.T if lambda_transposed else lam_b).astype(np.float64).reshape(1, E, E)   # float64 so the gate test is exact
        g["gammaeff"] = np.array([gamma], dtype=np.float32)
        if write_obs:
            g["Xobs"] = X[0].astype(np.float32)
            g["Dobs"] = D[0].astype(np.float32)
    if write_prj:
        (folder / f"{STEM}.prj").write_text("")
    if spectra:
        for s in dict.fromkeys(SYMS):            # distinct symbols, order kept
            h5py.File(folder / f"{s}1s_{STEM}.h5", "w").close()
    if extra_main:
        h5py.File(folder / extra_main_name, "w").close()
    if manifest is not None:
        (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder
