"""SemiFree の ``x -> D`` 写像を、solver インスタンスから切り出した純関数。

段階 -1 の抽出。目的は **`SemiFreeSolver` を組まずに、実行可能な x から D を作れるようにする**
こと（GT との距離の上限を測るため）。**挙動は変えない。**
`SemiFreeSolver._build_substrate_profile` / `_build_full_profile` はここへ委譲する。

抽出の範囲（意図的に最小）
--------------------------
* 入る: 基板 erf プロファイル、`normalize_free_conc`、charge-balance の本体（numba / NumPy
  両経路）、最終クリップ。
* 入らない: gradient mode、stoichiometric mode、bounds、制約。これらは `SemiFreeSolver` 側に
  残す（複製を作らないため）。bounds と制約は実験側から `scipy.optimize.minimize` の引数を
  捕捉して読む。

抽出時に見つけた既知の経路差（**ここでは直さない**。直すと挙動が変わる）
---------------------------------------------------------------------
* F1: ``numba_kernels.build_full_profile_numba`` は ``substrate_cutoff`` を持たない。
  NumPy fallback は持つ（``film_scale = where(substrate_sum >= cutoff, 0, film_scale)``）。
  したがって ``substrate_cutoff`` を設定すると **2 経路は異なる D を返す**。
  これは丸め誤差ではない。既定は ``None`` なので通常経路では発生しない。
  この関数は現行挙動をそのまま写すので、numba 経路では cutoff を無視する。
* F2: 両 kernel は ``@jit(..., fastmath=True)``、erf は numba=``math.erf`` /
  NumPy=``scipy.special.erf``。**経路間のビット一致は期待できない**（許容差で照合する）。

経路選択（F6 の修正、2026-09-09）
-----------------------------------
``SemiFreeSolver.__init__`` は ``normalize_free_conc=True`` のとき ``use_numba`` を強制的に
False にし、さらに ``use_numba and NUMBA_AVAILABLE`` で可用性も見る。抽出した入口にも同じ規則を
``resolve_use_numba`` として置き、**solver を経由しない呼び出しでも経路選択が一致する**ようにした。
solver 経由の呼び出しは既に coerce 済みの値を渡してくるので、この規則は no-op であり
**挙動は変わらない**。可用性フラグは ``numba_kernels.NUMBA_AVAILABLE`` を**そのまま**使う
（この module の import 成否から立てると、kernel 側の fallback decorator のせいで numba 不在でも
True になり、solver と逆方向へ食い違う）。

**この規則が実際に何を防ぐか**: solver 側のコメントは
「勾配・forward が非正規化変数を前提とする」と言うが、それは勾配・forward の話であって
``build_full_profile_numba`` 自身は内部で正規化しない。したがって x→D の写像に限れば、
``substrate_cutoff=None`` のもとで両経路の差は**丸め水準**（実測 ~1e-16）にとどまる。
**実害が出るのは ``normalize_free_conc=True`` かつ ``substrate_cutoff`` が非 None のとき**で、
numba kernel が cutoff を持たないぶん D が食い違う（**差の大きさは条件依存**: seed 0-5 ×
cutoff {0.3, 0.5, 0.7} × t {150, 210, 300} の 54 条件で min 0.0001 / median 0.2187 / max 0.3074、
回帰テストと同条件で 0.2532）。この規則は
(1) solver との経路一貫性と (2) その組合せでの食い違い、の 2 つを防ぐ。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.special import erf

try:  # numba は任意依存
    # NOTE: `numba_kernels` は numba 不在でも identity decorator へ fallback するので、
    # **この import は失敗しない**。可用性は kernel 側のフラグを見ないと分からない
    # （以前はここで True を立てており、
    # numba 不在時に solver と逆方向へ食い違っていた）。
    from .numba_kernels import (
        NUMBA_AVAILABLE as NUMBA_KERNELS_AVAILABLE,
        build_full_profile_numba,
        build_substrate_profile_numba,
    )
except ImportError:  # pragma: no cover - 環境依存
    NUMBA_KERNELS_AVAILABLE = False


__all__ = [
    "SemiFreeGeometry",
    "build_substrate_profile",
    "build_full_profile_core",
    "resolve_use_numba",
    "NUMBA_KERNELS_AVAILABLE",
]


# ---------------------------------------------------------------------------
def resolve_use_numba(normalize_free_conc: bool, use_numba: bool) -> bool:
    """経路選択の規則。``SemiFreeSolver.__init__`` と同一（F6）。

    solver 側 (``__init__``) は ``use_numba and NUMBA_AVAILABLE`` を取ったうえで、
    ``normalize_free_conc=True`` なら ``use_numba = False`` に落とす。抽出した入口を
    solver を経由せずに呼ぶ場合、このガードが無いと経路が食い違うので、規則をここに
    1 か所だけ置く。``NUMBA_KERNELS_AVAILABLE`` は ``numba_kernels.NUMBA_AVAILABLE``
    そのもの（この module 内で import 成否から立てると、fallback decorator のせいで
    常に True になり solver と食い違う）。

    何を防ぐかはモジュール docstring の「経路選択」節を見ること。
    ``substrate_cutoff=None`` では丸め水準の差しか防がないが、非 None との組合せでは
    条件依存の食い違い（54 条件で median 0.2187 / max 0.3074、テスト同条件 0.2532）を防ぐ。
    """
    if normalize_free_conc:
        return False
    return bool(use_numba) and NUMBA_KERNELS_AVAILABLE


# ---------------------------------------------------------------------------
def build_substrate_profile(
    depth: np.ndarray,
    thickness: float,
    abruptness: float,
    substrate_ratios: np.ndarray,
    *,
    use_numba: bool = False,
) -> np.ndarray:
    """``D_sub(z, e) = ratio_e * 0.5 * (1 + erf((z - thickness) / sigma))``、``sigma = FWHM/(2 sqrt(ln2))``。

    `SemiFreeSolver._build_substrate_profile` の本体をそのまま移したもの。
    """
    if use_numba:
        return build_substrate_profile_numba(depth, thickness, abruptness, substrate_ratios)

    # NumPy fallback
    # Convert FWHM to sigma (same as MATLAB InitializePriorModel.m line 2)
    # MATLAB: abrp = handles.abrpmod(1)/(2*sqrt(log(2)));
    sigma = abruptness / (2 * np.sqrt(np.log(2)))
    erf_profile = 0.5 * (1 + erf((depth - thickness) / sigma))
    return np.outer(erf_profile, substrate_ratios)


# ---------------------------------------------------------------------------
def build_full_profile_core(
    free_conc: np.ndarray,
    thickness: float,
    abruptness: float,
    substrate_ratios: np.ndarray,
    *,
    depth: np.ndarray,
    n_elements: int,
    free_idx: Sequence[int],
    sub_idx: Sequence[int],
    linked_idx: Sequence[int],
    free_linked_const: np.ndarray,
    min_conc: float,
    max_conc: float,
    substrate_cutoff: Optional[float] = None,
    normalize_free_conc: bool = False,
    free_idx_array: Optional[np.ndarray] = None,
    sub_idx_array: Optional[np.ndarray] = None,
    linked_idx_array: Optional[np.ndarray] = None,
    use_numba: bool = False,
) -> np.ndarray:
    """charge-balance モードの ``x -> D``（gradient / stoichiometric は含まない）。

    `SemiFreeSolver._build_full_profile` の該当部分をそのまま移したもの。
    `substrate_cutoff` は NumPy 経路にのみ効く（F1）。
    """
    use_numba = resolve_use_numba(normalize_free_conc, use_numba)   # F6

    if normalize_free_conc:
        weighted_sum = np.sum(free_conc * free_linked_const, axis=1, keepdims=True)
        weighted_sum = np.maximum(weighted_sum, min_conc)
        free_conc = free_conc / weighted_sum

    if use_numba:
        return build_full_profile_numba(
            free_conc,
            thickness,
            abruptness,
            substrate_ratios,
            depth,
            n_elements,
            free_idx_array if free_idx_array is not None else np.asarray(free_idx, dtype=np.int64),
            sub_idx_array if sub_idx_array is not None else np.asarray(sub_idx, dtype=np.int64),
            linked_idx_array if linked_idx_array is not None else np.asarray(linked_idx, dtype=np.int64),
            free_linked_const,
            min_conc,
            max_conc,
        )

    # NumPy fallback (charge_balance mode)
    n_depth = len(depth)
    D = np.zeros((n_depth, n_elements))

    # 1. Substrate profile
    D_sub = build_substrate_profile(depth, thickness, abruptness, substrate_ratios, use_numba=False)
    D[:, list(sub_idx)] = D_sub

    # 2. Free element concentrations scaled by (1 - substrate_sum)
    substrate_sum = D_sub.sum(axis=1, keepdims=True)
    film_scale = np.maximum(1.0 - substrate_sum, min_conc)
    if substrate_cutoff is not None:
        film_scale = np.where(substrate_sum >= substrate_cutoff, 0.0, film_scale)
    D[:, list(free_idx)] = free_conc * film_scale

    # 3. Linked element concentrations from charge balance
    linked_idx = list(linked_idx)
    if linked_idx:
        D_free = D[:, list(free_idx)]                 # (n_depth, n_free)
        coeff = np.asarray(free_linked_const) - 1     # (n_free,)
        D_linked = (D_free * coeff).sum(axis=1)       # (n_depth,)
        n_linked = len(linked_idx)
        for linked_e in linked_idx:
            D[:, linked_e] = D_linked / n_linked

    return np.clip(D, min_conc, max_conc)


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SemiFreeGeometry:
    """`build_full_profile_core` の固定引数をまとめたもの（solver の外から使うため）。

    `from_solver` は **読み取りのみ**で、solver を変更しない。
    """

    depth: np.ndarray
    n_elements: int
    free_idx: tuple
    sub_idx: tuple
    linked_idx: tuple
    free_linked_const: np.ndarray
    min_conc: float = 1e-6
    max_conc: float = 1.0
    substrate_cutoff: Optional[float] = None
    normalize_free_conc: bool = False
    use_numba: bool = False

    @classmethod
    def from_solver(cls, solver, *, use_numba: Optional[bool] = None) -> "SemiFreeGeometry":
        return cls(
            depth=np.asarray(solver.depth, dtype=float),
            n_elements=int(solver.n_elements),
            free_idx=tuple(solver.free_idx),
            sub_idx=tuple(solver.sub_idx),
            linked_idx=tuple(solver.linked_idx),
            free_linked_const=np.asarray(solver.free_linked_const, dtype=float),
            min_conc=float(solver.config.min_conc),
            max_conc=float(solver.config.max_conc),
            substrate_cutoff=solver.semi_free_config.substrate_cutoff,
            normalize_free_conc=bool(solver.semi_free_config.normalize_free_conc),
            use_numba=bool(solver.use_numba) if use_numba is None else bool(use_numba),
        )

    @property
    def effective_use_numba(self) -> bool:
        """F6 の規則を適用したあとの経路。``use_numba`` の生値ではなくこちらを見る。"""
        return resolve_use_numba(self.normalize_free_conc, self.use_numba)

    @property
    def n_depth(self) -> int:
        return int(self.depth.size)

    @property
    def n_free(self) -> int:
        return len(self.free_idx)

    @property
    def n_sub(self) -> int:
        return len(self.sub_idx)

    def build(self, free_conc, thickness, abruptness, substrate_ratios,
              *, use_numba: Optional[bool] = None) -> np.ndarray:
        return build_full_profile_core(
            np.asarray(free_conc, dtype=float), float(thickness), float(abruptness),
            np.asarray(substrate_ratios, dtype=float),
            depth=self.depth, n_elements=self.n_elements,
            free_idx=self.free_idx, sub_idx=self.sub_idx, linked_idx=self.linked_idx,
            free_linked_const=self.free_linked_const,
            min_conc=self.min_conc, max_conc=self.max_conc,
            substrate_cutoff=self.substrate_cutoff,
            normalize_free_conc=self.normalize_free_conc,
            use_numba=self.use_numba if use_numba is None else use_numba,
        )
