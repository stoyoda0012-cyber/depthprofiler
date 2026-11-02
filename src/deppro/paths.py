"""生成物の置き場を一箇所で決める。

なぜ要るか
----------
テスト・診断・ベンチマークが図を書く先が
``/Users/<name>/Projects/depthprofiler/src/outputs/<name>.png`` と
**開発者マシンの絶対パス直書き**になっていた。``src/outputs`` は Git 追跡
された symlink で、個人のクラウドディレクトリを指していた。開発機では
解決するので通るが、CI のチェックアウトではリンクがぶら下がるため
``FileNotFoundError`` になる。2026-08-02 の CI では、解析側が全部通った上で
最後の savefig だけが 4 件落ちていた。

解決順:

1. 環境変数 ``DEPPRO_OUTPUTS``
2. チェックアウトから使っていればリポジトリ直下の ``outputs/``（``.gitignore`` 済み）
3. インストール済みパッケージから（``repo_root()`` に ``pyproject.toml`` が無い）なら
   カレントディレクトリの ``outputs/``。一度だけ警告を出す（2026-10-19）

どの場合も :func:`ensure_outputs_dir` が必要なら作る。既定を一時
ディレクトリではなくリポジトリ直下にしてあるのは、書いた図をあとで探せる
場所に置くため。CI では毎回チェックアウトごと捨てられるので溜まらない。

``DEPPRO_SIMULATION_ROOT`` などと同じく、未設定でも **import は通り**、
実際に書く段で初めてディレクトリが決まる。

⚠️ 紛らわしい別名が既にある: ``DEPPRO_OUTPUT_ROOT``（単数 + ROOT）を
公開版に含めない CUDA ベンチ 3 本が使っている。これは JSON を書く先で、
図の話ではないので触っていない。統合するかは別途の判断。
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "env_value",
    "env_path_or_placeholder",
    "repo_root",
    "outputs_dir",
    "ensure_outputs_dir",
    "in_checkout",
    "simulation_root",
]


def env_value(name: str) -> str | None:
    """環境変数 ``name`` の値。未設定 **または空文字列** なら ``None``。

    ``DEPPRO_*`` のパス変数はすべてこの規則で読む: ``export DEPPRO_X=`` と
    空にしたときも未設定と同じに扱う。``os.environ.get(name, default)`` だと
    空文字列が既定値を素通りして ``Path("")``（= カレントディレクトリ）に
    なり、エラーから変数名が消えるため。空白だけの値は特別扱いしない。
    """
    value = os.environ.get(name)
    return value if value else None


def env_path_or_placeholder(name: str) -> Path:
    """``name`` のパス。未設定・空文字列なら ``<name>-not-set``。

    未設定でも import は通り、そのパスを読む段のエラーに変数名が出る。
    ``~`` は展開する。
    """
    return Path(env_value(name) or f"{name}-not-set").expanduser()


def repo_root() -> Path:
    """リポジトリのルート（``src/deppro/paths.py`` から 2 つ上）。

    インストール済みパッケージから呼ばれた場合は site-packages の 2 つ上で、
    チェックアウトのルートではない。:func:`in_checkout` で区別する。
    """
    return Path(__file__).resolve().parents[2]


def in_checkout() -> bool:
    """``repo_root()`` がこのリポジトリのチェックアウト（``pyproject.toml`` を持つ）か。"""
    return (repo_root() / "pyproject.toml").is_file()


_warned_installed_outputs = False


def outputs_dir() -> Path:
    """生成物の置き場を返す（作りはしない）。

    場所を知りたいだけのときはこちら。実際に書くなら
    :func:`ensure_outputs_dir` を使う。

    1. ``DEPPRO_OUTPUTS`` が設定されていればそこ。
    2. チェックアウトから使っていれば ``<repo>/outputs``。
    3. インストール済みパッケージからで ``DEPPRO_OUTPUTS`` が無ければ、
       **カレントディレクトリの** ``outputs/``。site-packages の中に書かないため。
       このときは一度だけ警告を出す。
    """
    env = env_value("DEPPRO_OUTPUTS")
    if env:
        return Path(env).expanduser()
    if in_checkout():
        return repo_root() / "outputs"
    global _warned_installed_outputs
    if not _warned_installed_outputs:
        import warnings
        warnings.warn(
            "DEPPRO_OUTPUTS is not set and deppro is not running from a checkout; "
            f"writing outputs under the current directory: {Path.cwd() / 'outputs'}. "
            "Set DEPPRO_OUTPUTS to choose the location.",
            stacklevel=2,
        )
        _warned_installed_outputs = True
    return Path.cwd() / "outputs"


def ensure_outputs_dir(*subdirs: str) -> Path:
    """生成物の置き場を作って返す。

    ``ensure_outputs_dir("semifree", "grid")`` のようにサブディレクトリを
    足せる。書き込み側はこれを呼んでからファイル名を繋ぐ:

        path = ensure_outputs_dir() / "gradient_profile_effect.png"
        plt.savefig(path, dpi=150)
    """
    path = outputs_dir().joinpath(*subdirs)
    path.mkdir(parents=True, exist_ok=True)
    return path


def simulation_root() -> Path:
    """MATLAB 由来のシミュレーションデータ置き場（``DEPPRO_SIMULATION_ROOT``）。

    ``Checkered/10x10x200_7elem_noisy/...`` のような、MATLAB 側で生成した
    ベンチマーク H5 のルート。読み取り専用の外部データなので
    :func:`outputs_dir` とは別建てにしてある。

    ほかの ``DEPPRO_*`` と同じ作法で、**未設定でも import は通り**、
    実際に読む段で変数名を含むエラーになる。既定値をここに書かないのは、
    実データの置き場が環境ごとに違うため。
    """
    return env_path_or_placeholder("DEPPRO_SIMULATION_ROOT")


def legacy_simulation_root() -> Path:
    """既存のベンチマーク・診断スクリプト用のシミュレーションデータ置き場。

    ``DEPPRO_SIMULATION_ROOT`` が設定されていればそれ（空文字列は未設定と同じ）、
    未設定なら以前から使われていたホーム相対の置き場
    （``~/Documents/MATLAB/Simulation/DepthProfiler``）を返す。

    ホーム相対の既定値は ``benchmarks/paths.py`` などに直書きされていたもので、
    ここに 1 か所にまとめた。未設定のときの挙動は以前と同じ。**新しいコードでは
    使わず** :func:`simulation_root` を使うこと（未設定なら変数名入りのパスになる）。

    公開 API ではない（``__all__`` に入れていない）。呼び出し元は
    ``benchmarks/paths.py``（と、公開版に含めない診断スクリプト）だけに限る。
    ``tests/unit/test_env_paths.py`` が、``src/deppro`` のほかの場所
    （``tests/`` を含む）からの名前や属性による参照を検出する。文字列で名前を
    組み立てる ``getattr`` などは検出できない。
    """
    env = env_value("DEPPRO_SIMULATION_ROOT")
    if env:
        return Path(env).expanduser()
    return Path.home() / "Documents/MATLAB/Simulation/DepthProfiler"
