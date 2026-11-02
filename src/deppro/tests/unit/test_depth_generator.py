"""``deppro.depth_generator`` の前進モデル（深さ分布 → 角度分解強度 → スペクトル）を縛る。

なぜ要るか
----------
このモジュールは公開シンボル ``DepthSpectraGenerator`` / ``DepthGeneratorConfig`` /
``generate_from_profile`` を出しているのに、テストが 1 本も無かった。
2026-08-03 まで「壊れていない」ことの唯一の担保は、``deppro/__init__.py`` が
無条件に import していたので壊れれば全テストが収集エラーになる、という**副作用**
だった。その無条件 import は任意 extra ``fit`` (toyomacro) を事実上必須にして
いたため commit 00bb75a で遅延解決に変えてある。担保はそこで消えた。

検証の性質（重要）
------------------
ここにある数値は**すべて合成データによる整合性確認**であって、測定値でも
文献値でもない。特に:

- 化合物パラメータ (``TESTMAT``) は架空の値。物性の主張を一切含まない。
- 束縛エネルギー・角度・深さグリッドはテスト用に選んだ値で、実験条件ではない。
- §"閉形式" の突き合わせは、IMFP 層が返す λ を**入力として与えられたもの**
  として扱い、減衰と規格化の構造だけを見る。TPP-2M の妥当性はここでは
  検証していない（それは ``imfp`` 側の責務）。

外部データへの非依存
--------------------
``TransferMatrix.compute()`` は ``compound_db=None`` のとき ``CompoundDatabase()``
を作る = リポジトリ外の ``CompoundTable.csv`` を読みに行く。CI には置かれない
ので、各テストは ``DEPPRO_COMPOUND_TABLE_PATH`` を tmp_path 上の 1 行 CSV に
向けて自己完結させる。
"""

import json

import numpy as np
import pytest

# toyomacro は任意 extra `fit`。入っていない環境ではこのファイルごと skip。
pytest.importorskip("toyomacro.voigtfit.spectra_generator")

from deppro.depth_generator import (  # noqa: E402
    DepthGeneratorConfig,
    DepthSpectraGenerator,
    generate_from_profile,
)
from deppro.depth_profile import DepthProfile, ElementConfig  # noqa: E402


# --------------------------------------------------------------------------
# 合成フィクスチャ
# --------------------------------------------------------------------------

# 架空の母材。TPP-2M が有限の λ を返す程度に妥当なだけの値で、
# 特定の物質を表さないし、物性の主張もしない。
_SYNTHETIC_COMPOUND_TABLE = (
    "CompoundName,ValenceCharge,Density [g/cm3],"
    "MolecularWeight [g/mol],BandGap [eV]\n"
    "TESTMAT,4,2.3,28.1,1.1\n"
)

STEP_SIZE = 2.0     # 深さ刻み (Å)
N_DEPTH = 50        # 深さ点数 → 0-98 Å
INTERFACE_IDX = 10  # 既定の界面位置（インデックス）
ANGLES = np.array([0.0, 30.0, 45.0, 60.0, 75.0])


@pytest.fixture
def synthetic_compound_table(tmp_path, monkeypatch):
    """架空の 1 行 CompoundTable を作り、既定パスの解決をそこへ向ける。"""
    csv_path = tmp_path / "compound_table_synthetic.csv"
    csv_path.write_text(_SYNTHETIC_COMPOUND_TABLE, encoding="utf-8")
    monkeypatch.setenv("DEPPRO_COMPOUND_TABLE_PATH", str(csv_path))
    return csv_path


def make_two_layer_profile(
    interface_idx: int = INTERFACE_IDX,
    be_surface: float = 100.0,
    be_buried: float = 100.0,
) -> DepthProfile:
    """表面層 / 埋没層の 2 元素からなる階段状プロファイル。

    ``Xa`` が 0 から ``interface_idx`` 手前までを、``Xb`` がそれ以深を占める。
    どの深さでも濃度和が厳密に 1 になるようにしてあるので、
    λ_obs = γ_eff / Σ_i D(d,i)/λ_basic(e,i) が深さに依らない定数になり、
    転送行列が厳密な Beer-Lambert 形 exp(-d·Δz/(λ·cosθ)) になる。

    元素記号 ``Xa`` / ``Xb`` は架空。束縛エネルギーを明示で渡しているので
    外部の束縛エネルギー表は引かれない。
    """
    depth = np.arange(N_DEPTH) * STEP_SIZE
    concentrations = np.zeros((N_DEPTH, 2))
    concentrations[:interface_idx, 0] = 1.0
    concentrations[interface_idx:, 1] = 1.0

    elements = [
        ElementConfig("Xa", "1s", "TESTMAT", binding_energy=be_surface),
        ElementConfig("Xb", "1s", "TESTMAT", binding_energy=be_buried),
    ]
    return DepthProfile(
        depth=depth,
        elements=elements,
        concentrations=concentrations,
        step_size=STEP_SIZE,
    )


def make_generator(profile=None, angles=ANGLES, **config_kwargs):
    """既定はノイズ無し・MLX 経路無効（プラットフォーム間で経路を固定する）。"""
    config_kwargs.setdefault("noise_level", 0.0)
    config_kwargs.setdefault("use_mlx", False)
    return DepthSpectraGenerator(
        profile if profile is not None else make_two_layer_profile(),
        angles,
        DepthGeneratorConfig(**config_kwargs),
    )


# --------------------------------------------------------------------------
# 形状・軸・設定の対応
# --------------------------------------------------------------------------

def test_intensities_shape_matches_angles_and_elements(synthetic_compound_table):
    """角度分解強度は [n_angles, n_elements]、各角度で規格化されている。"""
    gen = make_generator()

    assert gen.intensities.shape == (len(ANGLES), 2)
    assert np.all(gen.intensities >= 0)
    # forward(normalize=True) の契約: 各角度で元素方向の和が 1
    np.testing.assert_allclose(gen.intensities.sum(axis=1), 1.0, rtol=1e-12)


def test_generate_all_spectra_covers_every_element_and_angle(synthetic_compound_table):
    """全元素 × 全角度が揃い、エネルギー軸とスペクトルの長さが一致する。"""
    profile = make_two_layer_profile(be_surface=100.0, be_buried=530.0)
    gen = make_generator(profile)

    spectra = gen.generate_all_spectra()

    assert set(spectra) == {"Xa1s", "Xb1s"}
    for label, by_angle in spectra.items():
        assert sorted(by_angle) == list(range(len(ANGLES))), label
        for energy, spectrum in by_angle.values():
            assert energy.shape == spectrum.shape
            assert energy.ndim == 1


def test_energy_axis_follows_config(synthetic_compound_table):
    """エネルギー軸は束縛エネルギー中心・幅 energy_edge・刻み energy_step。"""
    profile = make_two_layer_profile(be_surface=100.0, be_buried=530.0)
    config = DepthGeneratorConfig(noise_level=0.0, use_mlx=False)
    gen = DepthSpectraGenerator(profile, ANGLES, config)

    for elem_idx, elem in enumerate(profile.elements):
        energy, _ = gen.generate_spectrum(elem_idx, 0)

        np.testing.assert_allclose(np.diff(energy), config.energy_step, rtol=1e-9)
        assert energy[0] == pytest.approx(elem.binding_energy - config.energy_edge)
        # np.arange の終端は要求上端を 1 刻み以内で行き過ぎ／取りこぼす
        assert abs(energy[-1] - (elem.binding_energy + config.energy_edge)) < config.energy_step


@pytest.mark.parametrize("edge,step", [(2.5, 0.05), (5.0, 0.1), (1.0, 0.02)])
def test_energy_axis_tracks_config_changes(synthetic_compound_table, edge, step):
    """energy_edge / energy_step を変えると軸が追随する（既定値の焼き付き防止）。"""
    gen = make_generator(energy_edge=edge, energy_step=step)
    energy, _ = gen.generate_spectrum(0, 0)

    np.testing.assert_allclose(np.diff(energy), step, rtol=1e-9)
    span = energy[-1] - energy[0]
    assert abs(span - 2 * edge) < step


def test_peak_sits_at_the_configured_binding_energy(synthetic_compound_table):
    """ノイズ無しスペクトルの最大値は設定した束縛エネルギー上（±1 刻み）。"""
    profile = make_two_layer_profile(be_surface=100.0, be_buried=530.0)
    gen = make_generator(profile)

    for elem_idx, elem in enumerate(profile.elements):
        energy, spectrum = gen.generate_spectrum(elem_idx, 0)
        peak_energy = energy[int(np.argmax(spectrum))]
        assert abs(peak_energy - elem.binding_energy) <= gen.config.energy_step


def test_spectrum_is_nonnegative_and_sits_on_the_background_floor(
    synthetic_compound_table,
):
    """強度は非負で、かつ加算した背景 (bg_level × amplitude_scale) を下回らない。"""
    gen = make_generator()
    floor = gen.config.bg_level * gen.config.amplitude_scale
    assert gen.config.bg_level == pytest.approx(
        gen.config.bg_base * gen.config.bg_level_factor
    )

    for elem_idx in range(2):
        for angle_idx in range(len(ANGLES)):
            _, spectrum = gen.generate_spectrum(elem_idx, angle_idx)
            assert np.all(spectrum >= 0)
            assert spectrum.min() >= floor * (1 - 1e-12)


def test_noise_keeps_shape_and_nonnegativity(synthetic_compound_table):
    """ノイズを載せても形状は保たれ、負の強度は出ない。

    ``add_poisson_noise`` は 1 次元入力を seed 無しで引くため、値そのものは
    再現しない。ここで縛るのは構造的性質だけ（分布の検証は toyomacro 側の責務）。
    """
    gen = make_generator()
    _, clean = gen.generate_spectrum(0, 0)

    noisy_by_angle = gen.generate_all_spectra(noise_level=1000.0)["Xa1s"]
    energy, noisy = noisy_by_angle[0]

    assert noisy.shape == clean.shape == energy.shape
    assert np.all(noisy >= 0)
    assert not np.array_equal(noisy, clean), "noise_level>0 なのに素通りしている"

    # noise_level=0 は完全に素通り（既定の決定論性）
    _, again = gen.generate_all_spectra(noise_level=0.0)["Xa1s"][0]
    np.testing.assert_array_equal(again, clean)


# --------------------------------------------------------------------------
# Beer-Lambert 的な深さ依存の向き
# --------------------------------------------------------------------------

def test_attenuation_matches_closed_form_beer_lambert(synthetic_compound_table):
    """階段プロファイルの規格化強度が等比級数の閉形式と一致する。

    濃度和が各深さで 1 の 2 元素・単一母材なので λ_obs は深さ・元素に依らない
    定数 λ になる。すると重み w_d = exp(-d·Δz/(λ·cosθ)) = r^d の等比数列で、

        X_buried(θ) = Σ_{d≥k} r^d / Σ_{d<n} r^d
        X_surface(θ) = 1 - X_buried(θ)

    λ は ``lambda_basic`` から**読み取って**使う。つまりここで検証しているのは
    減衰の深さ・角度依存と規格化の構造であって、λ の値そのもの (TPP-2M) ではない。
    """
    gen = make_generator()

    lambda_obs = gen.transfer_matrix.lambda_obs
    lam = gen.transfer_matrix.lambda_basic[0, 0]
    # 前提（λ が定数）の明示チェック。崩れていれば以下の閉形式は使えない。
    np.testing.assert_allclose(lambda_obs, lam, rtol=1e-12)

    d = np.arange(N_DEPTH)
    for angle_idx, theta in enumerate(ANGLES):
        r = np.exp(-STEP_SIZE / (lam * np.cos(np.deg2rad(theta))))
        weights = r ** d
        expected_buried = weights[INTERFACE_IDX:].sum() / weights.sum()

        assert gen.intensities[angle_idx, 1] == pytest.approx(
            expected_buried, rel=1e-9
        ), f"θ={theta}° で埋没層の強度が閉形式から外れた"
        assert gen.intensities[angle_idx, 0] == pytest.approx(
            1.0 - expected_buried, rel=1e-9
        )


def test_surface_fraction_increases_with_emission_angle(synthetic_compound_table):
    """斜出射ほど表面敏感 = 表面層の寄与が単調増加、埋没層は単調減少。

    向きの根拠: 重み r^d = exp(-d·Δz/(λ·cosθ)) は cosθ が小さくなるほど
    深い d を強く抑える（単調尤度比）。λ が元素間で共通なこのフィクスチャでは
    厳密な単調性が言える。
    """
    gen = make_generator()

    surface = gen.intensities[:, 0]
    buried = gen.intensities[:, 1]

    assert np.all(np.diff(surface) > 0), f"表面層が単調増加でない: {surface}"
    assert np.all(np.diff(buried) < 0), f"埋没層が単調減少でない: {buried}"


def test_burying_the_interface_deeper_raises_the_surface_signal(
    synthetic_compound_table,
):
    """界面を深くするほど表面層の寄与が増える（全角度で）。

    こちらは 2 元素の束縛エネルギーを変えて λ を元素ごとに変えてある。
    表面層の被覆が増え、埋没層の開始が深くなるので、λ が共通でなくても向きは変わらない。
    """
    surface_fractions = []
    for interface_idx in (5, 10, 20, 30):
        profile = make_two_layer_profile(
            interface_idx=interface_idx, be_surface=100.0, be_buried=530.0
        )
        surface_fractions.append(make_generator(profile).intensities[:, 0])

    stacked = np.array(surface_fractions)  # [n_interface, n_angles]
    assert np.all(np.diff(stacked, axis=0) > 0), (
        f"界面を深くしても表面層の寄与が増えていない:\n{stacked}"
    )


# --------------------------------------------------------------------------
# 出力（H5 / JSON）
# --------------------------------------------------------------------------

def test_generate_writes_one_h5_per_angle_and_element(
    synthetic_compound_table, tmp_path
):
    """角度ディレクトリ × 元素の H5 が書かれ、specdata が生成値と一致する。"""
    h5py = pytest.importorskip("h5py")

    profile = make_two_layer_profile(be_surface=100.0, be_buried=530.0)
    gen = make_generator(profile)

    out_dir = tmp_path / "spectra"
    files = gen.generate(out_dir, project_name="synthetic")

    for angle_idx, angle in enumerate(ANGLES):
        angle_dir = out_dir / f"angle_{int(angle):02d}deg"
        assert angle_dir.is_dir()

        for elem_idx, elem in enumerate(profile.elements):
            path = angle_dir / f"{elem.label}_synthetic.h5"
            assert path.exists()
            assert files[f"{elem.label}_angle{int(angle)}"] == path

            energy, spectrum = gen.generate_spectrum(elem_idx, angle_idx)
            with h5py.File(path, "r") as f:
                specdata = f["specdata"][:]
                assert specdata.shape == (2, len(energy))
                # noise_level=0 なので書き出しは生成値と厳密一致するはず
                np.testing.assert_allclose(specdata[0], energy, rtol=1e-12)
                np.testing.assert_allclose(specdata[1], spectrum, rtol=1e-12)

                assert f["fitpara"].shape == (1, gen.config.max_components, 9)
                assert f["xytdata"][0, 0] == pytest.approx(angle)
                assert f["otherpara"].shape == (10, 1)
                assert f["misc"].attrs["maxcomp"] == gen.config.max_components


def test_generate_writes_project_metadata(synthetic_compound_table, tmp_path):
    """プロジェクト H5 に角度数・元素数・強度・深さが記録される。"""
    h5py = pytest.importorskip("h5py")

    profile = make_two_layer_profile(be_surface=100.0, be_buried=530.0)
    gen = make_generator(profile)

    out_dir = tmp_path / "spectra"
    files = gen.generate(out_dir, project_name="synthetic")

    with h5py.File(files["project"], "r") as f:
        assert f.attrs["n_angles"] == len(ANGLES)
        assert f.attrs["n_elements"] == profile.n_elements
        assert f.attrs["photon_energy"] == pytest.approx(gen.config.photon_energy)
        np.testing.assert_allclose(f.attrs["angles"], ANGLES)

        np.testing.assert_allclose(f["depth"][:], profile.depth)
        np.testing.assert_allclose(f["concentrations"][:], profile.concentrations)
        np.testing.assert_allclose(f["intensities"][:], gen.intensities, rtol=1e-12)

        assert f["ezdeprof"]["Col"].shape == (3, profile.n_elements)

    # 併走して書かれる profile JSON が読み戻せること
    saved = json.loads(files["profile"].read_text(encoding="utf-8"))
    assert len(saved["elements"]) == profile.n_elements

    reloaded = DepthProfile.load(files["profile"])
    np.testing.assert_allclose(reloaded.depth, profile.depth)
    np.testing.assert_allclose(reloaded.concentrations, profile.concentrations)


def test_generate_from_profile_accepts_a_json_path(synthetic_compound_table, tmp_path):
    """便宜関数がファイルパスからプロファイルを読み、同じ出力を出す。"""
    profile = make_two_layer_profile(be_surface=100.0, be_buried=530.0)
    profile_path = tmp_path / "profile.json"
    profile.save_json(profile_path)

    files = generate_from_profile(
        profile_path,
        angles=ANGLES,
        output_dir=tmp_path / "from_path",
        noise_level=0.0,
        project_name="synthetic",
    )

    assert files["project"].exists()
    for angle in ANGLES:
        for label in profile.element_labels:
            assert files[f"{label}_angle{int(angle)}"].exists()
