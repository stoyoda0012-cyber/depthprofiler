"""`deppro.semifree_parametrization` の抽出挙動と F6 の回帰テスト。

段階 -1 (2026-09-09) で `SemiFreeSolver._build_substrate_profile` / `_build_full_profile` の
charge_balance 本体を `deppro.semifree_parametrization` へ切り出した。ここで守るのは 3 つ:

1. **仕様との一致** — 抽出先が仕様どおりの式を計算する。テスト内に**独立に書き下した式**と
   照合するので、抽出先を書き換えれば落ちる（実装同士の写し合いではない）。
2. **solver とモジュールの一致** — 片方だけ変えたら落ちる。
3. **F6** — `normalize_free_conc=True` のとき、抽出入口でも numba 経路を選ばない
   （`SemiFreeSolver.__init__` と同じ規則）。

否定対照（最終版 20 件に対して測り直した赤の件数）:

=================================  ========
変異                                赤の件数
=================================  ========
規則を恒等化（可用性も無視）              6
core の F6 適用を外す                    3
Geometry の規則を外す                    2
最終 clip を外す                         3
linked 分配 x1.01                       3
基板 sigma x1.01                        3
正規化の重み flc を落とす                 1
正規化の除算を no-op                     2
正規化 floor を外す                      1
正規化 floor の値を 1e-3 に               1
正規化 floor の値を 0.5 に                1
Geometry.build の上書きを無視             1
=================================  ========

F1（`build_full_profile_numba` に `substrate_cutoff` が無い）は既知の経路差。
**本件では `substrate_cutoff=None` を実行条件として扱う**。非 None の挙動は
characterization として記録するだけで、一般的な修正は別件。
"""
import numpy as np
import pytest
from scipy.special import erf

from deppro.inverse_solver import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig
from deppro.semifree_parametrization import (
    NUMBA_KERNELS_AVAILABLE,
    SemiFreeGeometry,
    build_full_profile_core,
    build_substrate_profile,
    resolve_use_numba,
)
from deppro.transfer_matrix import TransferMatrix

# 段階 -1 で測定前に固定した許容差（docs/STAGE_MINUS1_... §3）
ATOL, RTOL = 1e-12, 1e-9

DEPTH = np.arange(20, dtype=float) * 30.0
FREE, SUB, LINKED = (1, 2, 3, 4, 5), (0,), (6,)
FLC = np.array([3.0, 3.0, 2.5, 1.0, 1.0])
N_ELEM = 7
MIN_CONC, MAX_CONC = 1e-6, 1.0


def _free_conc(seed=0):
    """Σ flc_j c_j = 1 を満たす free_conc（等式制約の内側）。"""
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(len(FREE)), size=DEPTH.size)
    return w / FLC


def _unnormalised_free_conc(seed=0):
    """**Σ flc_j c_j ≠ 1** の free_conc。

    `_free_conc` は構造的に Σ flc·c ≡ 1 になるので、`normalize_free_conc` が
    恒等写像に退化し、正規化の中身を変えても落ちない。
    正規化を対象にするテストは必ずこちらを使うこと。
    """
    return np.random.default_rng(1000 + seed).random((DEPTH.size, len(FREE)))


def _core(free_conc, t, a, sr, *, use_numba=False, normalize=False, cutoff=None):
    return build_full_profile_core(
        free_conc, t, a, sr, depth=DEPTH, n_elements=N_ELEM, free_idx=FREE, sub_idx=SUB,
        linked_idx=LINKED, free_linked_const=FLC, min_conc=MIN_CONC, max_conc=MAX_CONC,
        substrate_cutoff=cutoff, normalize_free_conc=normalize, use_numba=use_numba)


def _make_solver(normalize=False, use_numba=True, cutoff=None):
    tm = TransferMatrix(depth=DEPTH, step_size=30.0, angles=np.linspace(0.0, 60.0, 14),
                        gamma_eff=0.8)
    tm.n_elements = N_ELEM
    tm.trans_mat = np.ones((14, DEPTH.size, N_ELEM))
    semi = SemiFreeSolverConfig(
        free_indices=list(FREE), linked_indices=list(LINKED), substrate_indices=list(SUB),
        free_linked_const_override=list(FLC), normalize_free_conc=normalize,
        substrate_cutoff=cutoff)
    sc = SolverConfig(method="L1", min_conc=MIN_CONC, max_conc=MAX_CONC)
    return SemiFreeSolver(tm, sc, semi, D_prior=None, elements=None, use_numba=use_numba)


# --------------------------------------------------------------------------- 1
def test_substrate_profile_matches_specification():
    """D_sub(z,e) = ratio_e * 0.5 * (1 + erf((z - t) / sigma)), sigma = FWHM/(2 sqrt(ln2))."""
    t, a, sr = 210.0, 20.0, np.array([0.8])
    sigma = a / (2.0 * np.sqrt(np.log(2.0)))
    expected = np.outer(0.5 * (1.0 + erf((DEPTH - t) / sigma)), sr)
    got = build_substrate_profile(DEPTH, t, a, sr, use_numba=False)
    assert np.allclose(got, expected, atol=ATOL, rtol=RTOL)


def test_full_profile_matches_specification():
    """独立に書き下した charge_balance の式と一致すること。"""
    fc, t, a, sr = _free_conc(1), 210.0, 20.0, np.array([1.0])

    sigma = a / (2.0 * np.sqrt(np.log(2.0)))
    ramp = 0.5 * (1.0 + erf((DEPTH - t) / sigma))
    d_sub = np.outer(ramp, sr)
    film_scale = np.maximum(1.0 - d_sub.sum(axis=1), MIN_CONC)
    expected = np.zeros((DEPTH.size, N_ELEM))
    for j, e in enumerate(SUB):
        expected[:, e] = d_sub[:, j]
    for j, e in enumerate(FREE):
        expected[:, e] = fc[:, j] * film_scale
    expected[:, LINKED[0]] = ((fc * film_scale[:, None]) * (FLC - 1.0)).sum(axis=1) / len(LINKED)
    expected = np.clip(expected, MIN_CONC, MAX_CONC)

    assert np.allclose(_core(fc, t, a, sr), expected, atol=ATOL, rtol=RTOL)


def test_feasible_x_gives_row_sums_at_most_one_plus_slack():
    """等式を満たす x では、clip 後の行和が 1 + 7*min_conc を超えない。

    段階 -1 の下限の導出が依存している性質（docs/STAGE_MINUS1_... §4）。
    """
    for seed in range(4):
        for t in (50.0, 210.0, 500.0):
            D = _core(_free_conc(seed), t, 20.0, np.array([1.0]))
            assert D.sum(axis=1).max() <= 1.0 + N_ELEM * MIN_CONC


# --------------------------------------------------------------------------- 2
def test_solver_and_module_agree():
    """solver 経由と抽出モジュール直呼びが同じ D を返す。

    **守れる範囲の限界**: solver は抽出先へ委譲しているので、共有している式そのものを
    書き換えても両側が同じだけ動き、このテストは落ちない（否定対照で確認済み）。
    ここが守るのは **wrapper と委譲の配線**（引数の取り違え、normalize の適用順序、
    from_solver が拾う値）だけ。式そのものは
    `test_full_profile_matches_specification` が独立式で守る。
    """
    fc, t, a, sr = _free_conc(2), 180.0, 20.0, np.array([1.0])
    for normalize in (False, True):
        s = _make_solver(normalize=normalize)
        got_solver = s._build_full_profile(fc, t, a, sr)
        got_module = SemiFreeGeometry.from_solver(s).build(fc, t, a, sr)
        assert np.array_equal(got_solver, got_module), f"normalize={normalize}"


def test_solver_and_module_agree_on_substrate_profile():
    s = _make_solver()
    for t in (50.0, 210.0, 500.0):
        assert np.array_equal(
            s._build_substrate_profile(t, 20.0, np.array([1.0])),
            build_substrate_profile(DEPTH, t, 20.0, np.array([1.0]),
                                    use_numba=s.use_numba))


# --------------------------------------------------------------------------- 3 (F6)
def test_resolve_use_numba_blocks_numba_when_normalising():
    """F6 の本体: normalize_free_conc=True なら **可用性に関わらず** numba を選ばない。"""
    assert resolve_use_numba(True, False) is False
    assert resolve_use_numba(True, True) is False, "F6: normalize 時に numba を選んではいけない"


def test_resolve_use_numba_matches_the_solver_coercion():
    """normalize=False 側の規則が solver の `use_numba and NUMBA_AVAILABLE` と一致すること。

    可用性フラグと直接比べると恒真になる（旧
    `test_resolve_use_numba_respects_kernel_availability` は単一変異で赤にならなかった）。
    **solver が実際に coerce した値**と突き合わせることで、環境仮定のハードコードも避ける。
    """
    assert resolve_use_numba(False, False) is False
    assert _make_solver(normalize=False, use_numba=False).use_numba is False
    assert resolve_use_numba(False, True) is _make_solver(normalize=False, use_numba=True).use_numba


def test_solver_init_coerces_use_numba_when_normalizing():
    """抽出入口が真似るべき solver 側の規則そのもの（これが変わったら F6 の前提が崩れる）。"""
    assert _make_solver(normalize=True, use_numba=True).use_numba is False


def test_core_ignores_use_numba_when_normalizing():
    """F6: 外から normalize=True + use_numba=True を渡しても NumPy 経路と同じ結果になる。"""
    fc, t, a, sr = _free_conc(3), 210.0, 20.0, np.array([1.0])
    forced = _core(fc, t, a, sr, use_numba=True, normalize=True)
    numpy_path = _core(fc, t, a, sr, use_numba=False, normalize=True)
    assert np.array_equal(forced, numpy_path)


def test_geometry_effective_use_numba():
    g = SemiFreeGeometry(depth=DEPTH, n_elements=N_ELEM, free_idx=FREE, sub_idx=SUB,
                         linked_idx=LINKED, free_linked_const=FLC,
                         normalize_free_conc=True, use_numba=True)
    assert g.effective_use_numba is False


class _StubSolver:
    """`__init__` の coerce を経由しない solver 相当。from_solver 側の規則だけを露出させる。"""

    def __init__(self):
        self.depth, self.n_elements = DEPTH, N_ELEM
        self.free_idx, self.sub_idx, self.linked_idx = list(FREE), list(SUB), list(LINKED)
        self.free_linked_const = FLC
        self.use_numba = True                       # coerce されていない生値
        self.config = SolverConfig(method="L1", min_conc=MIN_CONC, max_conc=MAX_CONC)
        self.semi_free_config = SemiFreeSolverConfig(normalize_free_conc=True)


def test_geometry_from_solver_applies_the_rule_not_just_the_solver_flag():
    """from_solver が **自分で** F6 の規則を適用すること。

    実 solver を渡すと `__init__` が既に use_numba=False に落としているので、
    `resolve_use_numba` を壊しても緑のままになる（単一変異では落ちない）。coerce を経由しない stub を渡し、**規則そのもの**を露出させる。
    """
    assert _StubSolver().use_numba is True, "stub は coerce されていない生値を持つこと"
    assert SemiFreeGeometry.from_solver(_StubSolver()).effective_use_numba is False


def test_geometry_from_solver_inherits_coerced_path():
    """実 solver 経由でも coerce 済み経路を落とさないこと（二重防御の確認）。

    **単体では単一変異で赤にならない**（solver 側 coerce と module 側規則の両方が
    False を返すため）。規則そのものの対照は
    `test_geometry_from_solver_applies_the_rule_not_just_the_solver_flag`。
    """
    s = _make_solver(normalize=True, use_numba=True)
    assert SemiFreeGeometry.from_solver(s).effective_use_numba is False


# --------------------------------------------------------------------------- 経路間
def test_normalize_free_conc_matches_specification():
    """正規化の式 c / max(Σ flc·c, min_conc) を独立に書き下して照合。

    **Σ flc·c ≠ 1 の入力を使う**（feasible な入力だと正規化が恒等に退化して、
    重みを落としても落ちない）。
    """
    fc, t, a, sr = _unnormalised_free_conc(0), 210.0, 20.0, np.array([1.0])
    ws = np.maximum(np.sum(fc * FLC, axis=1, keepdims=True), MIN_CONC)
    expected = _core(fc / ws, t, a, sr, normalize=False)
    assert np.allclose(_core(fc, t, a, sr, normalize=True), expected, atol=ATOL, rtol=RTOL)


def test_normalize_is_not_the_identity_on_this_fixture():
    """上のテストが空虚でないことの担保 — 正規化前後で実際に値が動くこと。"""
    fc = _unnormalised_free_conc(0)
    assert np.abs(np.sum(fc * FLC, axis=1) - 1.0).max() > 0.1
    without = _core(fc, 210.0, 20.0, np.array([1.0]), normalize=False)
    with_ = _core(fc, 210.0, 20.0, np.array([1.0]), normalize=True)
    assert np.abs(with_ - without).max() > 1e-3


def test_normalize_floor_protects_against_zero_rows():
    """正規化の floor `max(Σ flc·c, min_conc)` が 0 除算を防ぐこと。

    一様乱数の入力では floor に触れないので、floor を外しても落ちない
    （否定対照で確認した穴）。**Σ flc·c = 0 の行を明示的に入れる。**
    """
    fc = _unnormalised_free_conc(4).copy()
    fc[0, :] = 0.0                      # Σ flc·c = 0 → floor が無いと 0 除算
    # Σ flc·c ≈ 1e-8 の行。**floor の「値」を観測できる行**。0 の行だけだと任意の正の floor で
    # 同じ結果になり、floor を 1e-3 や 0.5 に変えても落ちない。
    fc[1, :] = 1e-8 / (FLC.sum() * len(FREE))
    D = _core(fc, 210.0, 20.0, np.array([1.0]), normalize=True)
    assert np.all(np.isfinite(D)), "floor が無いと inf/nan になる行がある"
    assert np.allclose(D[0, list(FREE)], np.full(len(FREE), MIN_CONC), atol=ATOL, rtol=RTOL)

    # 行 1: Σ flc·c ≪ min_conc なので分母は floor に張り付く。floor の値が変われば外れる。
    assert float((fc[1] * FLC).sum()) < MIN_CONC, "この行は floor が binding しないと意味がない"
    ramp = build_substrate_profile(DEPTH, 210.0, 20.0, np.array([1.0]), use_numba=False)
    film_scale = np.maximum(1.0 - ramp.sum(axis=1), MIN_CONC)
    expected_1 = np.clip((fc[1] / MIN_CONC) * film_scale[1], MIN_CONC, MAX_CONC)
    assert np.allclose(D[1, list(FREE)], expected_1, atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(not NUMBA_KERNELS_AVAILABLE, reason="numba kernels 未導入")
def test_f6_guard_prevents_the_cutoff_divergence_under_normalisation():
    """F6 ガードが**丸めでない実害**を防ぐ唯一の組合せ。

    `normalize_free_conc=True` かつ `substrate_cutoff` 非 None のとき、numba kernel は
    cutoff を持たないので D が食い違う。F6 の規則が numba を選ばせないことで一致する。
    ガードを外すと差は 1e-1 オーダーになる。
    """
    fc, t, sr, cut = _unnormalised_free_conc(2), 210.0, np.array([1.0]), 0.5
    guarded = _core(fc, t, 20.0, sr, use_numba=True, normalize=True, cutoff=cut)
    numpy_path = _core(fc, t, 20.0, sr, use_numba=False, normalize=True, cutoff=cut)
    assert np.array_equal(guarded, numpy_path)

    # ガードが無ければどれだけ食い違うか（規則を迂回して kernel を直接呼ぶ）
    import deppro.semifree_parametrization as m
    raw = m.build_full_profile_numba(
        fc / np.maximum(np.sum(fc * FLC, axis=1, keepdims=True), MIN_CONC), t, 20.0, sr,
        DEPTH, N_ELEM, np.asarray(FREE), np.asarray(SUB), np.asarray(LINKED),
        FLC, MIN_CONC, MAX_CONC)
    assert np.max(np.abs(raw - numpy_path)) > 1e-2, "ガードが防いでいる差が消えている"


def test_geometry_build_use_numba_override_still_obeys_the_rule():
    """`Geometry.build(use_numba=True)` の明示上書きでも F6 の規則が効くこと。"""
    g = SemiFreeGeometry(depth=DEPTH, n_elements=N_ELEM, free_idx=FREE, sub_idx=SUB,
                         linked_idx=LINKED, free_linked_const=FLC, min_conc=MIN_CONC,
                         max_conc=MAX_CONC, normalize_free_conc=True, use_numba=False)
    fc, sr = _unnormalised_free_conc(3), np.array([1.0])
    forced = g.build(fc, 210.0, 20.0, sr, use_numba=True)
    plain = g.build(fc, 210.0, 20.0, sr, use_numba=False)
    assert np.array_equal(forced, plain)


@pytest.mark.skipif(not NUMBA_KERNELS_AVAILABLE, reason="numba kernels 未導入")
def test_geometry_build_use_numba_override_takes_effect():
    """上書きが**実際に経路を切り替える**こと。

    `..._still_obeys_the_rule` は上書きを無視しても緑なので、上書きが効くことを守っていない。
    ここでは normalize=False で F6 を無効化し、
    **F1 の既知経路差（numba は cutoff を無視する）を観測器として使う**。
    """
    g = SemiFreeGeometry(depth=DEPTH, n_elements=N_ELEM, free_idx=FREE, sub_idx=SUB,
                         linked_idx=LINKED, free_linked_const=FLC, min_conc=MIN_CONC,
                         max_conc=MAX_CONC, substrate_cutoff=0.5,
                         normalize_free_conc=False, use_numba=False)
    fc, sr = _free_conc(5), np.array([1.0])
    via_numba = g.build(fc, 210.0, 20.0, sr, use_numba=True)
    via_numpy = g.build(fc, 210.0, 20.0, sr, use_numba=False)
    assert np.max(np.abs(via_numba - via_numpy)) > 1e-2, "上書きが経路に届いていない"


@pytest.mark.skipif(not NUMBA_KERNELS_AVAILABLE, reason="numba kernels 未導入")
def test_paths_agree_when_substrate_cutoff_is_none():
    """実行条件 substrate_cutoff=None での numba / NumPy 一致（事前登録 atol/rtol）。"""
    for seed in range(3):
        fc = _free_conc(seed)
        for t in (50.0, 210.0, 500.0):
            a_ = _core(fc, t, 20.0, np.array([1.0]), use_numba=True)
            b_ = _core(fc, t, 20.0, np.array([1.0]), use_numba=False)
            assert np.allclose(a_, b_, atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(not NUMBA_KERNELS_AVAILABLE, reason="numba kernels 未導入")
def test_substrate_cutoff_is_a_known_path_divergence():
    """F1 の characterization。**仕様の追認ではなく、既知の食い違いの記録**。

    `build_full_profile_numba` は `substrate_cutoff` を引数に取らないので無視し、
    NumPy fallback だけが film_scale を 0 にする。段階 -1 では
    `substrate_cutoff=None` を実行条件として扱い、一般的な修正は別件。
    **F1 を直したらこのテストは落ちる。そのとき本テストと実行条件を更新すること。**
    """
    fc, t, sr = _free_conc(0), 210.0, np.array([1.0])
    nb = _core(fc, t, 20.0, sr, use_numba=True, cutoff=0.5)
    npth = _core(fc, t, 20.0, sr, use_numba=False, cutoff=0.5)
    assert not np.allclose(nb, npth, atol=ATOL, rtol=RTOL)
    assert np.max(np.abs(nb - npth)) > 1e-3
