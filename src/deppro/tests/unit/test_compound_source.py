"""`CompoundDatabase` のパラメータ供給元の解決順を縛る。

なぜ要るか
----------
既定がリポジトリの隣のデータ置き場の `CompoundTable.csv`（git 管理外の兄弟ツリー）
だったため、CI では 56 件が FileNotFoundError で落ち、開発機では気付けなかった
（2026-08-03）。既定を toyomacro に移し、暗黙の兄弟パスは**復活させない**。

ここで縛るのは 3 点:

    1. 環境変数も csv_path も無ければ toyomacro から読む
    2. 環境変数があればそれが勝つ（運用者の明示的な選択）
    3. どちらも無ければ、両方の名前を含むエラーで落ちる
       （黙って空になったり、ホーム相対のパスを探しに行ったりしない）

値そのものの検証はここではしない。両テーブルとも per-entry の出典が無く、
「正しい値」を主張できる一次資料がないため。TPP-2M の式の忠実性は
`test_imfp_tpp2m_conformance.py` が別に見ている。
"""

import os
import re
from pathlib import Path

import pytest

from deppro.imfp import CompoundDatabase

pytestmark = pytest.mark.usefixtures()


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv("DEPPRO_COMPOUND_TABLE_PATH", raising=False)


def test_default_source_is_toyomacro(no_env):
    """既定は toyomacro。リポジトリの隣のデータ置き場を暗黙に探しに行かない。"""
    pytest.importorskip("toyomacro.data")

    db = CompoundDatabase()
    assert db.source == "toyomacro.data.CompoundDB"
    assert db.csv_path is None


def test_default_source_covers_the_compounds_this_package_tests(no_env):
    """テストが実際に使う化合物が既定ソースで引けること。

    Si3N4 は CSV では aSi3N4 / bSi3N4 に分割されていてエイリアス経由だったが、
    toyomacro には素の Si3N4 がある。どちらの経路でも引けることを確かめる。
    """
    pytest.importorskip("toyomacro.data")

    db = CompoundDatabase()
    for name in ("Si", "SiO2", "Al2O3", "Si3N4", "TiO2", "C", "Ti", "Al"):
        params = db.get(name)
        assert params.density > 0.0
        assert params.molecular_weight > 0.0
        assert params.valence_charge > 0.0
        assert params.band_gap >= 0.0


def test_env_var_overrides_the_default(no_env, tmp_path, monkeypatch):
    """環境変数を立てれば CSV が勝ち、source にその旨が出る。"""
    csv = tmp_path / "CompoundTable.csv"
    csv.write_text(
        "CompoundName,ValenceCharge,Density [g/cm3],"
        "MolecularWeight [g/mol],BandGap [eV]\n"
        "Unobtainium,7,1.234,56.78,2.5\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEPPRO_COMPOUND_TABLE_PATH", str(csv))

    db = CompoundDatabase()
    assert db.source == f"csv:{csv}"
    assert db.list_compounds() == ["Unobtainium"]

    params = db.get("Unobtainium")
    assert params.valence_charge == pytest.approx(7.0)
    assert params.density == pytest.approx(1.234)
    assert params.molecular_weight == pytest.approx(56.78)
    assert params.band_gap == pytest.approx(2.5)


def test_explicit_csv_path_argument_still_works(no_env, tmp_path):
    """`csv_path=` を直接渡す既存の呼び方を壊していないこと。"""
    csv = tmp_path / "table.csv"
    csv.write_text(
        "CompoundName,ValenceCharge,Density [g/cm3],"
        "MolecularWeight [g/mol],BandGap [eV]\n"
        "Handwavium,3,9.0,10.0,0\n",
        encoding="utf-8",
    )
    db = CompoundDatabase(csv_path=csv)
    assert db.list_compounds() == ["Handwavium"]


def test_missing_everything_names_both_ways_out(no_env, monkeypatch):
    """toyomacro も env も無いときのエラーが、逃げ道を両方名指しすること。"""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "toyomacro.data" or name.startswith("toyomacro"):
            raise ImportError("blocked by test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    with pytest.raises(FileNotFoundError) as excinfo:
        CompoundDatabase()

    message = str(excinfo.value)
    assert "DEPPRO_COMPOUND_TABLE_PATH" in message
    # toyomacro は PyPI に無い。extra `fit` は固定タグへの直接参照になったが、
    # 案内には実際に通る git の手順も出すこと（extra が使えない配布形態のため）。
    assert "git+https://github.com/stoyoda0012-cyber/toyomacro" in message
    # 案内は固定のリリースタグ（main は動く標的なので案内しない）
    assert re.search(r"toyomacro@v\d+\.\d+\.\d+'", message) and "@main" not in message


def test_binding_energy_missing_source_names_a_pinned_install(no_be_env, monkeypatch):
    import builtins

    from deppro.imfp import BindingEnergyDatabase

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("toyomacro"):
            raise ImportError("blocked by test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(FileNotFoundError) as excinfo:
        BindingEnergyDatabase()
    message = str(excinfo.value)
    assert "DEPPRO_BINDING_ENERGY_TABLE_PATH" in message
    assert re.search(r"toyomacro@v\d+\.\d+\.\d+'", message) and "@main" not in message


def test_no_implicit_sibling_lookup(no_env):
    """リポジトリの隣のデータ置き場を既定に戻していないこと。

    この暗黙既定こそが、CI と開発機で挙動を変えていた当の原因なので、
    パス解決関数が環境変数以外を返さないことを直接縛る。
    """
    assert CompoundDatabase._get_override_csv_path() is None

    os.environ["DEPPRO_COMPOUND_TABLE_PATH"] = "/nonexistent/x.csv"
    try:
        assert CompoundDatabase._get_override_csv_path() == Path("/nonexistent/x.csv")
    finally:
        del os.environ["DEPPRO_COMPOUND_TABLE_PATH"]


# ---------------------------------------------------------------------------
# BindingEnergyDatabase — CompoundDatabase と同じ扱いに揃えてある
# ---------------------------------------------------------------------------

@pytest.fixture
def no_be_env(monkeypatch):
    monkeypatch.delenv("DEPPRO_BINDING_ENERGY_TABLE_PATH", raising=False)


def test_binding_energy_default_source_is_toyomacro(no_be_env):
    """既定は toyomacro。リポジトリの隣のデータ置き場を暗黙に探しに行かない。"""
    pytest.importorskip("toyomacro.data.binding_energy")

    from deppro.imfp import BindingEnergyDatabase

    db = BindingEnergyDatabase()
    assert db.source == "toyomacro.data.BindingEnergy"
    assert db.csv_path is None
    assert BindingEnergyDatabase._get_override_csv_path() is None


def test_binding_energy_values_used_by_this_suite(no_be_env):
    """このスイートが実際に使う準位が、既定ソースで引けること。"""
    pytest.importorskip("toyomacro.data.binding_energy")

    from deppro.imfp import BindingEnergyDatabase

    db = BindingEnergyDatabase()
    expected = {("C", "1s"): 284.0, ("Al", "2p"): 73.0, ("Ti", "2p"): 454.0,
                ("Si", "2p"): 99.0, ("O", "1s"): 531.0}
    for (symbol, orbital), value in expected.items():
        assert db.get(symbol, orbital) == pytest.approx(value)


def test_every_orbital_the_source_holds_is_reachable(no_be_env):
    """供給元が持つ (元素, 軌道) を **全数** 引けること。

    なぜ例示ではなく全数か
    ----------------------
    上のテストは C 1s / Al 2p / Ti 2p / Si 2p / O 1s の 5 例に張っており、
    そのすべてが `ORBITAL_COLUMNS` に別名を持つ軌道だった。ところが
    `get()` は別名が無ければ軌道名をそのまま列名として使う設計なので、
    呼び出し側は別名の無い軌道（4f5/2, 4d3/2, 4p1/2, 5p1/2, 5d3/2,
    6p1/2, 6p3/2, 6s, 7s …）も要求できる。

    loader が `ORBITAL_COLUMNS.values()` の 16 列しか読んでいなかったため、
    それらは **316 組** が KeyError になっていた（1be222e で混入、
    2026-08-03 に発覚）。CSV 経路は元から全列を読んでいたので、旧環境では
    引けていた退行である。Au 4f5/2 は XPS の代表的な参照線。

    例示に張ったテストはこの取りこぼしを 1 件も検出しなかった。
    だから母集団そのものに張る。
    """
    BindingEnergy = pytest.importorskip(
        "toyomacro.data.binding_energy").BindingEnergy

    from deppro.imfp import BindingEnergyDatabase

    db = BindingEnergyDatabase()
    missing, checked = [], 0
    for symbol in BindingEnergy.get_all_elements():
        for orbital in BindingEnergy.get_orbitals(symbol):
            expected = BindingEnergy.lookup(symbol, orbital)
            if expected is None:
                continue
            checked += 1
            try:
                got = db.get(symbol, orbital)
            except KeyError:
                missing.append(f"{symbol} {orbital}")
                continue
            assert got == pytest.approx(float(expected)), f"{symbol} {orbital}"

    assert checked > 1000, f"母集団が小さすぎる（{checked} 組）— 走査が壊れている"
    assert not missing, (
        f"{len(missing)}/{checked} 組が引けない。loader が供給元の一部しか "
        f"読んでいない: {missing[:12]}"
    )


def test_orbitals_without_an_alias_are_reachable(no_be_env):
    """別名を持たない軌道が引けること（上の全数テストの回帰ピン）。

    全数テストが将来また母集団を狭めても、ここが独立に落ちる。
    値は CSV 側と一致することを確認済みの実測値。
    """
    pytest.importorskip("toyomacro.data.binding_energy")

    from deppro.imfp import BindingEnergyDatabase

    db = BindingEnergyDatabase()
    aliased = set(BindingEnergyDatabase.ORBITAL_COLUMNS.values())
    for symbol, orbital, value in [("Au", "4f5/2", 88.0),
                                   ("Au", "4d3/2", 353.0),
                                   ("Au", "4p1/2", 643.0),
                                   ("Au", "5p1/2", 74.0),
                                   ("U", "6s", 44.0)]:
        assert orbital not in aliased, f"{orbital} は別名を持つ — ピンとして無効"
        assert db.get(symbol, orbital) == pytest.approx(value)


def test_element_config_resolves_binding_energy_without_the_sibling(no_be_env):
    """ElementConfig が BE を解決できること。

    ここが None のまま素通りすると、ずっと後段の
    transfer_matrix.calculate_lambda_basic で
    `TypeError: float - NoneType` になる（2026-08-03 の CI がそれ）。
    近いところで気付けるよう、直接縛る。
    """
    pytest.importorskip("toyomacro.data.binding_energy")

    from deppro.depth_profile import ElementConfig

    for symbol, orbital, compound in [("Si", "2p", "SiO2"), ("O", "1s", "SiO2"),
                                      ("Al", "2p", "Al2O3"), ("C", "1s", "C")]:
        element = ElementConfig(symbol, orbital, compound)
        assert element.binding_energy is not None, f"{symbol}{orbital} が None"
        assert element.binding_energy > 0.0


# ---------------------------------------------------------------------------
# CompoundBindingEnergyDatabase — 同梱する供給源が無いので、変数を示して止まる
# ---------------------------------------------------------------------------

def test_compound_be_table_without_the_variable_names_it(monkeypatch):
    """変数が無ければ、リポジトリの隣を探さずに変数名つきで止まる。"""
    from deppro.imfp import CompoundBindingEnergyDatabase
    monkeypatch.delenv("DEPPRO_COMPOUND_BE_TABLE_PATH", raising=False)
    with pytest.raises(FileNotFoundError, match="DEPPRO_COMPOUND_BE_TABLE_PATH"):
        CompoundBindingEnergyDatabase()


def test_compound_be_table_reads_the_variable(tmp_path, monkeypatch):
    """変数で指した表を読む（2 行の合成 CSV）。指した先が無ければそのパスで止まる。"""
    from deppro.imfp import CompoundBindingEnergyDatabase
    table = tmp_path / "be.csv"
    table.write_text("Element,SpectralLine,Formula,BE_eV\n"
                     "Si,2p,SiO2,103.5\n"
                     "Si,2p,Si,99.3\n", encoding="utf-8")
    monkeypatch.setenv("DEPPRO_COMPOUND_BE_TABLE_PATH", str(table))
    db = CompoundBindingEnergyDatabase()
    assert len(db) == 2
    assert [r["formula"] for r in db.lookup("Si", "2p")] == ["SiO2", "Si"]

    missing = tmp_path / "absent.csv"
    monkeypatch.setenv("DEPPRO_COMPOUND_BE_TABLE_PATH", str(missing))
    with pytest.raises(FileNotFoundError, match="absent.csv"):
        CompoundBindingEnergyDatabase()
