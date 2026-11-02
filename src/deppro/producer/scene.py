"""場面 — stack ごとの FeatureVector 変調。

MATLAB 側の対応（`+deppro/+simulation/SimulationUpdater.m::updateFeatureVector`）:

    FV = FV0
    bottom layer  (Linear | Exponential | Trigonometric)   -> FV(sk, dx, dy)
    top layer     (Checkered)                              -> FV(sk, dx, dy)
    interf redox  (Linear | Exponential | Trigonometric)   -> FV(sk, dxdy, dxdydt)
    etch/grow     (Linear | Exponential | Trigonometric)   -> FV(sk, dxdy, dxdydt)

**場面は flat な stack 番号 sk の関数**で、三重ループもフレーム境界も無い。
`dx`,`dy` は模様の周期を決めるためだけに使われる。したがって `stack` が
`nx*ny` の倍数である必要は無く、素数でもそのまま定義される。

FeatureVector の並び（1-based, MATLAB）:
    col 1        thickness
    col 2        abruptness
    col 3k       position of the k-th free component
    col 3k+1     width
    col 3k+2     height
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

_BOTTOM = ("Linear", "Exponential", "Trigonometric")
_REDOX = ("Linear", "Exponential", "Trigonometric")
_ETCH = ("Linear", "Exponential", "Trigonometric")


@dataclass(frozen=True)
class SceneFlags:
    """cfg の分布関数の選択。空文字列 / None は「その群を適用しない」。"""

    bottom_layer: Optional[str] = None
    top_layer: Optional[str] = None       # "Checkered" のみ
    interf_redox: Optional[str] = None
    etch_grow: Optional[str] = None

    def validate(self) -> None:
        """未移植の分岐は**ここで断る**。既存の分岐へ黙って置き換えない。

        MATLAB 側は 3 群 × 3 形（Linear/Exponential/Trigonometric）を持つが、
        この producer が写したのは各群 1 つだけ。名前として存在することと、
        ここで作れることは別なので、両方を区別してメッセージに出す。
        """
        for name, value, known, table in (
            ("bottom_layer", self.bottom_layer, _BOTTOM, _BOTTOM_FN),
            ("interf_redox", self.interf_redox, _REDOX, _REDOX_FN),
            ("etch_grow", self.etch_grow, _ETCH, _ETCH_FN),
        ):
            if not value:
                continue
            if value not in known:
                raise NotImplementedError(
                    f"{name}={value!r} such a branch does not exist in MATLAB either "
                    f"(MATLAB has {known}); implemented here: {tuple(table)}")
            if value not in table:
                raise NotImplementedError(
                    f"{name}={value!r} exists in MATLAB (+deppro/+simulation) but is NOT ported "
                    f"to this producer. Implemented here: {tuple(table)}. "
                    f"Generate it with the MATLAB CLI, or port the branch first — "
                    f"do not substitute another branch.")
        if self.top_layer and self.top_layer != "Checkered":
            raise NotImplementedError(
                f"top_layer={self.top_layer!r} is not ported (implemented here: ('Checkered',))")


def _n_free(fv: np.ndarray) -> int:
    return (fv.shape[1] - 2) // 3


def distrib_bottom_layer_trigonometric(fv: np.ndarray, sk: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = fv.copy()
    w = (2 - np.cos(2 * np.pi * (sk - 1) / (dx / 3)) - np.cos(2 * np.pi * np.floor((sk - 1) / dy) / (dy / 3))) / 4
    out[:, [0, 2]] = fv[:, [0, 2]] + 50 * w[:, None]
    out[:, 3] = fv[:, 3] + 75 * w
    return out


def distrib_top_layer_checkered(fv: np.ndarray, sk: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """市松。free の最後から 2 番目と最後の **height** を入れ替える。

    周期は速い方向に `dx/6`、遅い方向に `dx*dy/6`（= 6 ブロック/辺）。
    """
    out = fv.copy()
    nfree = _n_free(fv)
    c1 = 3 * (nfree - 1) + 1          # MATLAB 3*(SizeofFree-1)+2 -> 0-based
    c2 = 3 * nfree + 1                # MATLAB 3*SizeofFree+2     -> 0-based
    id1 = np.mod(np.ceil(6 * sk / dx), 2) == 0
    id2 = np.mod(np.ceil(6 * sk / (dx * dy)), 2) == 0
    same = (id1 & id2) | (~id1 & ~id2)
    out[same, c1] = 0
    out[same, c2] = 100
    out[~same, c1] = 100
    out[~same, c2] = 0
    return out


def phenom_interf_redox_exponential(fv: np.ndarray, sk: np.ndarray, dxdy: int, dxdydt: int) -> np.ndarray:
    out = fv.copy()
    cyc = np.floor((sk - 1) / dxdydt)
    rising = np.mod(cyc, 2) == 0
    within = sk - 1 - cyc * dxdydt
    up = 1 - np.exp(-np.floor(within / dxdy) * dxdy / (dxdydt / 5))
    down = np.exp(np.ceil((-sk + 1 + cyc * dxdydt) / dxdy) * dxdy / (dxdydt / 5))
    w = np.where(rising, up, down)
    out[:, [0, 2]] = fv[:, [0, 2]] + 100 * w[:, None]
    out[:, 3] = fv[:, 3] + 150 * w
    return out


def phenom_etch_grow_linear(fv: np.ndarray, sk: np.ndarray, dxdy: int, dxdydt: int) -> np.ndarray:
    out = fv.copy()
    nfree = _n_free(fv)
    cols = [0] + [3 * k - 1 for k in range(1, nfree + 1)]   # MATLAB [1, 3*(1:SizeofFree)]
    cyc = np.floor((sk - 1) / dxdydt)
    rising = np.mod(cyc, 2) == 0
    within = sk - 1 - cyc * dxdydt
    shift = np.where(rising,
                     -np.floor(within / (2 * dxdy)),
                     -50 - np.ceil((-sk + 1 + cyc * dxdydt) / (2 * dxdy)))
    out[:, cols] = fv[:, cols] + shift[:, None]
    return out


_BOTTOM_FN = {"Trigonometric": distrib_bottom_layer_trigonometric}
_REDOX_FN = {"Exponential": phenom_interf_redox_exponential}
_ETCH_FN = {"Linear": phenom_etch_grow_linear}


def feature_vector_for_stacks(
    fv0: np.ndarray, sk: np.ndarray, nx: int, ny: int, time_size: int, flags: SceneFlags
) -> np.ndarray:
    """flat な stack 番号 `sk`（1-based）に対する FeatureVector を返す。

    `sk` はベクトルでよい。MATLAB 側も sk をベクトルで扱う実装なので、
    chunk 分割しても結果は変わらない（stack ごとに独立）。
    """
    flags.validate()
    sk = np.asarray(sk, dtype=float).ravel()
    fv = np.repeat(np.asarray(fv0, dtype=float).reshape(1, -1), sk.size, axis=0)
    dxdy = nx * ny
    dxdydt = nx * ny * time_size

    if flags.bottom_layer:
        fn = _BOTTOM_FN.get(flags.bottom_layer)
        if fn is None:
            raise NotImplementedError(f"bottom_layer={flags.bottom_layer!r} not implemented yet")
        fv = fn(fv, sk, nx, ny)
    if flags.top_layer:
        fv = distrib_top_layer_checkered(fv, sk, nx, ny)
    if flags.interf_redox:
        fn = _REDOX_FN.get(flags.interf_redox)
        if fn is None:
            raise NotImplementedError(f"interf_redox={flags.interf_redox!r} not implemented yet")
        fv = fn(fv, sk, dxdy, dxdydt)
    if flags.etch_grow:
        fn = _ETCH_FN.get(flags.etch_grow)
        if fn is None:
            raise NotImplementedError(f"etch_grow={flags.etch_grow!r} not implemented yet")
        fv = fn(fv, sk, dxdy, dxdydt)
    return fv


#: 時間現象が向きを変えるフレーム間隔。`+deppro/+simulation` の 3 形すべてで同じ。
#:   *Linear / *Exponential : id = ~mod(floor((sk-1)/(nx*ny*dt)),2) の偶奇が変わるところ
#:   *Trigonometric         : (1-cos(pi*f/dt))/2 が折り返すところ
#: どちらも 0 起点フレーム f = dt, 2dt, ... （1-based では dt+1, 2dt+1, ...）。
TURNING_RULE = "per-phenomenon direction reverses at 1-based frame dt+1, 2dt+1, ... (dt = timeSize)"

#: 合成後の極値を数える差分ラグ。`*Linear` は「2 フレームごとに 1 段」の階段なので
#: ラグ 1 の符号では全フレームが折り返しに見える。**素朴な sign(diff) を使わない。**
_EXTREMA_LAG = 2


def scene_timing(fv: np.ndarray, nx: int, ny: int, n_stack: int, time_size: int,
                 column: int = 0) -> dict:
    """時間軸の記録。**各変調の反転**と**合成後の山谷**を別項目として返す。

    前者は規則から決まる（`TURNING_RULE`）。後者は現象の和なので一致しない —
    例えば etch(下降) + redox(上昇) では、各現象の反転は規則どおりのフレームだが、
    合成した膜厚の極値は別のフレームに来ることがある。**混同しないこと。**

    `column` は見る FeatureVector 列（既定 0 = thickness）。各フレームの先頭画素だけを
    見るので、空間変調は固定されて時間方向だけが残る。
    """
    fv = np.atleast_2d(np.asarray(fv, dtype=float))
    dxdy = int(nx) * int(ny)
    dt = int(time_size)
    nt_full, rem = divmod(int(n_stack), dxdy) if dxdy else (0, 0)
    nt = nt_full + (1 if rem else 0)
    head = fv[::dxdy, int(column)] if dxdy else fv[:, int(column)]

    turning = list(range(dt + 1, nt + 1, dt)) if dt > 0 else []
    if dt > 0 and nt <= dt:
        kind_run = "one-way"
    elif dt > 0:
        kind_run = f"round-trip x {nt / (2 * dt):g}"
    else:
        kind_run = "undefined (timeSize <= 0)"

    # 合成後の山谷。**単一フレーム番号では書かない。**
    # `*Linear` の階段（2 フレームで 1 段）が滑らかな成分に重なるため、極値の近傍は
    # 0.5 A 程度のこぎり歯を持つ平坦部になり、「どのフレームが頂点か」は ±1 決まらない。
    # 移動平均（窓 2）で周期 2 の成分を消して極値を出し、生の系列で許容幅に入る
    # フレームの帯として返す。
    extrema: list = []
    tol = 0.0
    if head.size >= 4:
        sm = 0.5 * (head[:-1] + head[1:])                 # 窓 2 の移動平均
        p2p = float(sm.max() - sm.min())
        tol = max(1e-3 * p2p, 1e-9)
        d = np.diff(sm)
        d[np.abs(d) < 1e-12] = 0.0
        nz = np.flatnonzero(d)
        if nz.size > 1:
            sgn = np.sign(d[nz])
            for k in np.flatnonzero(np.diff(sgn) != 0):
                c = int(nz[k]) + 1                        # sm の 0-based 極値位置
                kind = "max" if sgn[k] > 0 else "min"
                lo = hi = c
                while lo > 0 and abs(sm[lo - 1] - sm[c]) <= tol:
                    lo -= 1
                while hi < sm.size - 1 and abs(sm[hi + 1] - sm[c]) <= tol:
                    hi += 1
                extrema.append({"kind": kind,
                                "frameBand": [lo + 1, hi + 2],   # 1-based, 移動平均の 2 フレーム分
                                "value": float(head[c:c + 2].max() if kind == "max"
                                               else head[c:c + 2].min())})
    return {
        "nt": int(nt),
        "timeSize": dt,
        "lastFrameCount": int(rem) if rem else None,
        "halfCycles": (nt / dt) if dt > 0 else None,
        "runKind": kind_run,
        "perPhenomenonTurningFrames": turning,
        "perPhenomenonTurningRule": TURNING_RULE,
        "compositeExtrema": extrema,
        "compositeExtremaColumn": int(column),
        "compositeExtremaTolerance": tol,
        "compositeExtremaMethod":
            "window-2 moving average of the first pixel of each frame (kills the 2-frame "
            "Linear staircase exactly), then sign changes of its difference. Reported as a "
            "frame BAND, not a single frame: near an extremum the raw series is a plateau "
            "with a ~0.5 A sawtooth, so the peak frame is not defined to better than +/-1.",
        "note": "perPhenomenonTurningFrames and compositeExtrema are DIFFERENT things. "
                "The first is where each modulation reverses (a rule from the source); the "
                "second is where the summed profile actually peaks (a measurement). They "
                "coincide only for a single-phenomenon scene; with several phenomena the "
                "composite extrema can fall at other frames than the turning frames.",
        "headRange": [float(head.min()), float(head.max())] if head.size else None,
    }
