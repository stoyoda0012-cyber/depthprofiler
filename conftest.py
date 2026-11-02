"""リポジトリ直下の pytest 設定（全テスト共通の前提を 1 箇所に固める）。

ここで面倒を見るのは 2 つだけ:

1. **matplotlib を Agg に固定** — 図を書くテストがヘッドレスでも落ちないように、
   テストモジュールが matplotlib を import する前に決めておく。

2. **カレントディレクトリを固定** — 素の ``pytest`` を repo root から
   走らせると、``FileNotFoundError: 'outputs/xxx.png'`` で落ちるテストがあった。図を
   ``outputs/`` 相対に savefig するテストがあり、``outputs`` は
   ``src/`` と ``src/deppro/`` にしか無いため。「どこから走らせたかで
   結果が変わる」状態は CI に載せられないので、セッション開始時に
   ``outputs/`` を持つ一時ディレクトリへ chdir する。

   副作用として、テストが吐く図は一時ディレクトリに落ちて作業ツリーを
   汚さない（以前はリポジトリ直下に PNG が散っていた）。
"""

import os
from pathlib import Path

import pytest

# テストモジュールが matplotlib.pyplot を import する前に効かせる必要がある
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(scope="session", autouse=True)
def _stable_cwd(tmp_path_factory):
    """テストの実行位置を固定する（呼び出し元ディレクトリに依存させない）。"""
    workdir: Path = tmp_path_factory.mktemp("cwd")
    (workdir / "outputs").mkdir()
    previous = Path.cwd()
    os.chdir(workdir)
    try:
        yield workdir
    finally:
        os.chdir(previous)
