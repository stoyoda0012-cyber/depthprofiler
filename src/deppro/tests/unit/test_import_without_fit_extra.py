"""`import deppro` が任意依存 (extra `fit` = toyomacro) 無しで通ることを縛る。

なぜ要るか
----------
pyproject は toyomacro を **任意**の extra `fit` として宣言しているのに、
`deppro/__init__.py` が `depth_generator` を無条件に import し、その
`depth_generator.py` が `toyomacro.voigtfit` を無条件に import していた。
結果として宣言と実装が食い違い、`pip install deppro` で入れたパッケージは
**そもそも import できない**状態だった。

CI ではこれが 2026-08-02 の週次実行で顕在化した。conftest の読み込み時点で
ModuleNotFoundError になるため、ci.yml が当てにしていた
`pytest.importorskip` は一度も評価されず、pytest は収集前に exit 4 で死ぬ。
つまり「依存が無ければ skip される」という設計上の前提が破れていた。

ここで縛るのは 2 点:

    1. toyomacro が import 不能でも `import deppro` は成功する
    2. `fit` 由来の 3 シンボルは参照でき、**使った時点**（生成器の構築）で
       `deppro[fit]` を名指しする ImportError を出す
       （黙って None になったり AttributeError にすり替わったりしない）
"""

import subprocess
import sys
import textwrap

# toyomacro を import 不能にした上で deppro を読む。meta_path で塞ぐのは、
# install の方式（.pth 経由の editable、finder 経由の editable、通常 install）に
# 依存せず確実に塞ぐため。（depth_generator.py が兄弟チェックアウトの src を
# sys.path に挿していた経路は 2026-10-18 に廃止した。）
_BLOCK_TOYOMACRO = """
import sys


class _BlockToyomacro:
    def find_spec(self, name, path=None, target=None):
        if name == "toyomacro" or name.startswith("toyomacro."):
            raise ImportError("blocked by test_import_without_fit_extra")
        return None


sys.meta_path.insert(0, _BlockToyomacro())
"""


def _run_without_toyomacro(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_TOYOMACRO + textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )


def test_import_deppro_succeeds_without_toyomacro():
    """toyomacro が無くてもパッケージ import は通る。"""
    result = _run_without_toyomacro(
        """
        import deppro

        # 中核の公開シンボルはそのまま使えること
        assert deppro.DepthProfile is not None
        assert deppro.IMFP is not None
        print("OK")
        """
    )
    assert result.returncode == 0, (
        "toyomacro 無しで `import deppro` が失敗した。"
        f"\n--- stderr ---\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_fit_exports_resolve_and_fail_by_name_on_use():
    """`fit` 由来のシンボルは toyomacro 無しでも参照でき、**使った時点**で
    `deppro[fit]` を名指しする ImportError になる（先頭の import を
    関数内へ遅延し、無ければ名前付きエラーにする）。黙って None になったり
    AttributeError にすり替わったりしない。"""
    result = _run_without_toyomacro(
        """
        import deppro

        for name in ("DepthSpectraGenerator", "DepthGeneratorConfig",
                     "generate_from_profile"):
            assert getattr(deppro, name) is not None, name
        deppro.DepthGeneratorConfig()          # 設定だけなら toyomacro は要らない
        try:
            deppro.DepthSpectraGenerator(profile=object(), angles=[0.0])
        except ImportError as e:
            assert "deppro[fit]" in str(e) and "toyomacro" in str(e), e
        else:
            raise AssertionError("生成器の構築が toyomacro 無しで通ってしまった")
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_fit_exports_are_reachable_when_toyomacro_is_available():
    """toyomacro があるときは、遅延解決でも従来どおり取得できる。

    遅延化が「常に壊れている」ことで前 2 件を通してしまう抜け穴を塞ぐ。
    """
    import pytest

    pytest.importorskip("toyomacro.voigtfit.spectra_generator")

    import deppro

    assert deppro.DepthSpectraGenerator is not None
    assert deppro.DepthGeneratorConfig is not None
    assert callable(deppro.generate_from_profile)


def test_unknown_attribute_still_raises_attributeerror():
    """__getattr__ を足したことで未知属性の例外型が変わっていないこと。"""
    import pytest

    import deppro

    with pytest.raises(AttributeError):
        deppro.this_symbol_does_not_exist
