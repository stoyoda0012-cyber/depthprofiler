"""DEPPRO_* パス変数の読み方: 空文字列は未設定と同じ。

``export DEPPRO_X=`` と空にしたとき、``os.environ.get(name, default)`` は
既定値を素通りして ``Path("")``（= カレントディレクトリ）を返す。そうなると
エラーから変数名が消え、読み書きが黙って cwd に向く。これを
``deppro.paths.env_value`` / ``env_path_or_placeholder`` に集約して防ぐ。
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from deppro import paths

VAR = "DEPPRO_TEST_ENV_PATHS_VAR"


# --- 公開版に含めないファイルへの許可 ---------------------------------------
# private リポジトリだけにあるファイル（記録用の診断など）への許可は
# ``_private_allowances.py`` に置く。そのモジュールが無ければ許可は空
# （公開版ではどの例外も認めない）。
def _private_allowances():
    import importlib

    name = f"{__package__}._private_allowances"
    try:
        m = importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name != name:
            raise
        return {}, set()
    return dict(m.RAW_ENV_READERS), set(m.LEGACY_SIMULATION_ROOT_CALLERS)


_PRIVATE_RAW_READERS, _PRIVATE_LEGACY_ROOT_CALLERS = _private_allowances()


# --- helpers -----------------------------------------------------------------

def test_env_value_unset(monkeypatch):
    monkeypatch.delenv(VAR, raising=False)
    assert paths.env_value(VAR) is None


def test_env_value_empty_is_unset(monkeypatch):
    monkeypatch.setenv(VAR, "")
    assert paths.env_value(VAR) is None


def test_env_value_set(monkeypatch):
    monkeypatch.setenv(VAR, "/some/where")
    assert paths.env_value(VAR) == "/some/where"


def test_env_value_whitespace_is_kept(monkeypatch):
    # 空白だけの値は特別扱いしない（空文字列だけを未設定として扱う）
    monkeypatch.setenv(VAR, " ")
    assert paths.env_value(VAR) == " "


@pytest.mark.parametrize("value", [None, ""])
def test_placeholder_names_variable(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(VAR, raising=False)
    else:
        monkeypatch.setenv(VAR, value)
    p = paths.env_path_or_placeholder(VAR)
    assert p == Path(f"{VAR}-not-set")
    assert p != Path("")          # cwd に化けない


def test_placeholder_set_and_expands_tilde(monkeypatch):
    monkeypatch.setenv(VAR, "~/data")
    assert paths.env_path_or_placeholder(VAR) == Path("~/data").expanduser()


# --- readers -----------------------------------------------------------------

def _set(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize("value", [None, ""])
def test_simulation_root_unset_or_empty(monkeypatch, value):
    _set(monkeypatch, "DEPPRO_SIMULATION_ROOT", value)
    assert paths.simulation_root() == Path("DEPPRO_SIMULATION_ROOT-not-set")


def test_simulation_root_set(monkeypatch, tmp_path):
    monkeypatch.setenv("DEPPRO_SIMULATION_ROOT", str(tmp_path))
    assert paths.simulation_root() == tmp_path


@pytest.mark.parametrize("value", [None, ""])
def test_legacy_simulation_root_unset_or_empty(monkeypatch, value):
    # 未設定・空文字列なら、以前から使われていたホーム相対の置き場
    _set(monkeypatch, "DEPPRO_SIMULATION_ROOT", value)
    assert paths.legacy_simulation_root() == (
        Path.home() / "Documents/MATLAB/Simulation/DepthProfiler")


def test_legacy_simulation_root_set(monkeypatch, tmp_path):
    monkeypatch.setenv("DEPPRO_SIMULATION_ROOT", str(tmp_path))
    assert paths.legacy_simulation_root() == tmp_path


def test_legacy_simulation_root_expands_tilde(monkeypatch):
    monkeypatch.setenv("DEPPRO_SIMULATION_ROOT", "~/sim")
    assert paths.legacy_simulation_root() == Path("~/sim").expanduser()


def test_benchmark_paths_follow_legacy_simulation_root(monkeypatch, tmp_path):
    # benchmarks/paths.py の BENCHMARK_ROOT と全 GT_* が、この関数を通って解決
    # されていること（どれかを Path.home() 直書きに戻すと落ちる）
    import importlib
    import deppro.benchmarks.paths as bp
    root = tmp_path / "Simulation" / "DepthProfiler"
    monkeypatch.setenv("DEPPRO_SIMULATION_ROOT", str(root))
    try:
        bp = importlib.reload(bp)
        assert bp.BENCHMARK_ROOT == root
        gts = {k: Path(v) for k, v in vars(bp).items() if k.startswith("GT_")}
        assert "GT_500x500" in gts and len(gts) > 1
        # Simulation/Checkered 配下の GT_500x500 だけがルートの 1 つ上を基準にし、
        # ほかはすべてルートの下にある（基準を取り違えた配線も落とす）
        assert root.parent / "Checkered" in gts.pop("GT_500x500").parents
        for key, p in gts.items():
            assert root in p.parents, key
    finally:
        monkeypatch.undo()
        importlib.reload(bp)   # 以降のテストのために既定の状態へ戻す


def test_legacy_simulation_root_callers_are_limited():
    # 新しいコードは simulation_root() を使う。legacy_simulation_root() を参照して
    # よいのは、以前のホーム相対の既定値を引き継いだ benchmarks/paths.py と、
    # _private_allowances.py が挙げるファイルだけ。
    # 走査するのは src/deppro 全体（tests/ を含む）。除くのは定義元の paths.py、
    # このテストファイル、_archived だけ。名前（別名での import を含む）と属性
    # による参照を見る。文字列で名前を組み立てる getattr などは検出しない。
    src = Path(paths.__file__).resolve().parent
    this = Path(__file__).resolve().relative_to(src).as_posix()
    callers = set()
    for f in sorted(src.rglob("*.py")):
        rel = f.relative_to(src).as_posix()
        if "_archived" in f.parts or rel in ("paths.py", this):
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if ((isinstance(n, ast.Name) and n.id == "legacy_simulation_root")
                    or (isinstance(n, ast.Attribute) and n.attr == "legacy_simulation_root")
                    or (isinstance(n, ast.alias) and n.name == "legacy_simulation_root")):
                callers.add(rel)
    assert callers == {"benchmarks/paths.py"} | _PRIVATE_LEGACY_ROOT_CALLERS


@pytest.mark.parametrize("value", [None, ""])
def test_outputs_dir_unset_or_empty(monkeypatch, value):
    _set(monkeypatch, "DEPPRO_OUTPUTS", value)
    assert paths.outputs_dir() == paths.repo_root() / "outputs"


def test_outputs_dir_set(monkeypatch, tmp_path):
    monkeypatch.setenv("DEPPRO_OUTPUTS", str(tmp_path))
    assert paths.outputs_dir() == tmp_path


def test_outputs_dir_installed_without_env_uses_cwd_and_warns(monkeypatch, tmp_path):
    """インストール済み（repo_root() に pyproject.toml が無い）で DEPPRO_OUTPUTS が
    未設定なら、site-packages の中ではなくカレントディレクトリの outputs/ を返し、
    一度だけ警告する。"""
    import warnings
    fake_root = tmp_path / "lib" / "python3.12"
    fake_root.mkdir(parents=True)
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.delenv("DEPPRO_OUTPUTS", raising=False)
    monkeypatch.setattr(paths, "repo_root", lambda: fake_root)
    monkeypatch.setattr(paths, "_warned_installed_outputs", False)
    monkeypatch.chdir(workdir)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = paths.outputs_dir()
        second = paths.outputs_dir()
    assert first == second == workdir / "outputs"
    assert not str(first).startswith(str(fake_root))
    assert len(caught) == 1 and "DEPPRO_OUTPUTS" in str(caught[0].message)


def test_outputs_dir_checkout_without_env_uses_repo_root(monkeypatch, tmp_path):
    fake_root = tmp_path / "checkout"
    fake_root.mkdir()
    (fake_root / "pyproject.toml").write_text("[project]\nname='x'\n")
    monkeypatch.delenv("DEPPRO_OUTPUTS", raising=False)
    monkeypatch.setattr(paths, "repo_root", lambda: fake_root)
    assert paths.in_checkout() is True
    assert paths.outputs_dir() == fake_root / "outputs"


@pytest.mark.parametrize("value", [None, ""])
def test_matlab_cli_unset_or_empty(monkeypatch, value):
    from deppro.app.matlab_generation import matlab_cli
    _set(monkeypatch, "DEPPRO_DEPTHPROFILER_CLI", value)
    assert matlab_cli() == Path("DEPPRO_DEPTHPROFILER_CLI-not-set")


def test_matlab_cli_set(monkeypatch, tmp_path):
    from deppro.app.matlab_generation import matlab_cli
    monkeypatch.setenv("DEPPRO_DEPTHPROFILER_CLI", str(tmp_path / "dp"))
    assert matlab_cli() == tmp_path / "dp"


@pytest.mark.parametrize("value", [None, ""])
def test_compound_table_path_unset_or_empty(monkeypatch, value):
    from deppro.producer.materials import compound_table_path
    _set(monkeypatch, "DEPPRO_COMPOUND_TABLE_PATH", value)
    assert compound_table_path() == Path("DEPPRO_COMPOUND_TABLE_PATH-not-set")


def test_compound_table_path_set(monkeypatch, tmp_path):
    from deppro.producer.materials import compound_table_path
    monkeypatch.setenv("DEPPRO_COMPOUND_TABLE_PATH", str(tmp_path / "t.csv"))
    assert compound_table_path() == tmp_path / "t.csv"


# --- guard -------------------------------------------------------------------
#
# DEPPRO_* を ``deppro.paths`` を通さずに読むと、空文字列が「設定済み」として
# 素通りする。AST で次の書き方を検出する（どれも空文字列を未設定として扱わない）:
#
# - 既定値つきの読み取り: ``E.get(name, d)`` / ``E.get(name, default=d)`` /
#   ``G(name, d)`` / ``G(key=name, default=d)``
# - ``E.setdefault(name, d)``（常に。空文字列が入っていると既定値が入らない）
# - 値を使う ``E.pop(name[, d])``（文として結果を捨てる形は後片付けなので対象外）
# - 添字の読み取り ``E[name]``
# - 存在判定 ``name in E`` / ``name not in E``
# - 既定値なしで読んだ値の None 判定: ``E.get(name) is None`` のように直接比べる
#   形と、同じ関数（またはモジュール）の中で変数に入れてから比べる形
#   （``=``、型注釈つきの ``=``、``:=``）
#
# E は ``os.environ`` / ``environ`` / 同じファイル内の別名（``x = os.environ``、
# ``from os import environ as x``、別名の別名も）/ ``E.copy()`` / ``dict(E)``。
# G は ``os.getenv`` / ``getenv`` / 同じファイル内の別名。name は ``"DEPPRO_…"`` の
# 文字列リテラル、f-string の先頭、または同じファイル内で ``NAME = "DEPPRO_…"`` と
# 代入された名前。
#
# 捕まらない主なもの（網羅ではない）: 変数名を実行時に組み立てる書き方（連結、
# ``%``、``.format``、``getattr(os, "getenv")``、``NAME = f"DEPPRO_{n}"`` という
# 定数）、別モジュールで定義された定数や別名、属性に入れた別名
# （``self.env = os.environ``）、``{**os.environ}`` や ``dict(os.environ, A=1)``、
# 値を別の関数（クロージャを含む）に渡してから None 判定する形、``== None`` での
# 判定、タプルのアンパックで受けた値、比較演算子を連ねた形（``a is None is b``）、
# 関数や lambda の既定引数で読んで None 判定する形、``for`` / ``with`` の対象で
# 受けた値の None 判定、モジュールで読んで関数の中で None 判定する形、属性に
# 代入してから None 判定する形（``self.v = ...`` → ``self.v is None``）。
#
# 別名と定数は、スコープも代入の順序も見ずに名前だけで追う。None 判定の変数は
# 同じ関数（またはモジュール）の中に限って追うが、代入の順序は見ない。そのため、
# 同じ名前を別のものに使うと誤検出することがある（赤になる側に倒れる）。
#
# また、クラス本体と内包表記は外側の関数（またはモジュール）と同じ範囲として扱う。
# そのため、Python では別の変数でも検出されることがある。例えば、クラス本体で読んだ
# 値をモジュールや別のクラス本体で None 判定する形や、内包表記の対象変数が外側で
# 読んだ変数を隠す形（``v = os.getenv(...)`` → ``[v is None for v in ...]``）。
#
# 既定値に ``None`` を明示した形（``os.environ.get(name, None)``、
# ``os.getenv(name, None)``、``default=None``）も既定値つきとして検出される。
# 誤検出したら、名前を変えるか ``deppro.paths`` を通す書き方に直すこと。
#
# 件数は検出した位置（読み取り、None 判定では比較）の (行, 列) で数える。同じ行に
# 2 つ書いても、1 つの値を 2 回 None 判定しても、それぞれ別の件になる。
#
# 例示のコードはこのファイルでは文字列リテラルなので、自分自身は検出されない。

# 生の値を記録することが目的の読み場所（空文字列なら "" と記録するのが正しい）。
# (src からの相対パス, 変数名) → そのファイルでの出現数。数がずれたら赤にする。
# 該当するファイルはどれも公開版に含めないので、許可は _private_allowances.py にある。
_ALLOWED_RAW_READERS = dict(_PRIVATE_RAW_READERS)

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


class _EnvReadScanner:
    """1 ファイル分の検出。先にファイル全体から定数と別名を集めてから見る。"""

    def __init__(self, tree: ast.AST):
        self.tree = tree
        self.consts: dict[str, str] = {}      # NAME -> "DEPPRO_…"
        self.environ_aliases: set[str] = set()
        self.getenv_aliases: set[str] = set()
        self._collect()

    # -- 事前収集 --------------------------------------------------------------
    def _collect(self) -> None:
        # 別名の別名（y = x, x = os.environ）を走査順に依存せず拾うため、
        # 集合が変わらなくなるまで繰り返す
        while True:
            before = (len(self.consts), len(self.environ_aliases),
                      len(self.getenv_aliases))
            self._collect_once()
            if before == (len(self.consts), len(self.environ_aliases),
                          len(self.getenv_aliases)):
                return

    def _collect_once(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [t.id for t in targets if isinstance(t, ast.Name)]
                v = node.value
                if (isinstance(v, ast.Constant) and isinstance(v.value, str)
                        and v.value.startswith("DEPPRO_")):
                    for n in names:
                        self.consts[n] = v.value
                elif self._is_environ(v):
                    self.environ_aliases.update(names)
                elif self._is_getenv(v):
                    self.getenv_aliases.update(names)
            elif isinstance(node, ast.ImportFrom) and node.module == "os":
                for a in node.names:
                    if a.name == "environ" and a.asname:
                        self.environ_aliases.add(a.asname)
                    elif a.name == "getenv" and a.asname:
                        self.getenv_aliases.add(a.asname)

    # -- 判定の部品 ------------------------------------------------------------
    def _is_environ(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute) and node.attr == "environ":
            return True
        if isinstance(node, ast.Name):
            return node.id == "environ" or node.id in self.environ_aliases
        if isinstance(node, ast.Call) and not node.keywords:
            f = node.func
            # E.copy() / dict(E)
            if (isinstance(f, ast.Attribute) and f.attr == "copy"
                    and not node.args and self._is_environ(f.value)):
                return True
            if (isinstance(f, ast.Name) and f.id == "dict"
                    and len(node.args) == 1 and self._is_environ(node.args[0])):
                return True
        return False

    def _is_getenv(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute) and node.attr == "getenv":
            return True
        return isinstance(node, ast.Name) and (
            node.id == "getenv" or node.id in self.getenv_aliases)

    def _name(self, node: ast.AST | None) -> str | None:
        """``"DEPPRO_X"`` / ``f"DEPPRO_{...}"`` / DEPPRO 定数名なら変数名を返す。"""
        if node is None:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
        elif (isinstance(node, ast.JoinedStr) and node.values
              and isinstance(node.values[0], ast.Constant)
              and isinstance(node.values[0].value, str)):
            text = node.values[0].value
        elif isinstance(node, ast.Name) and node.id in self.consts:
            text = self.consts[node.id]
        else:
            return None
        return text if text.startswith("DEPPRO_") else None

    def _read_call(self, node: ast.AST) -> tuple[str, bool] | None:
        """``E.get`` / ``G`` の呼び出しなら (変数名, 既定値あり) を返す。"""
        if not isinstance(node, ast.Call):
            return None
        f = node.func
        kw = {k.arg: k.value for k in node.keywords}
        if isinstance(f, ast.Attribute) and f.attr == "get" and self._is_environ(f.value):
            first = node.args[0] if node.args else kw.get("key")
        elif self._is_getenv(f):
            first = node.args[0] if node.args else kw.get("key")
        else:
            return None
        name = self._name(first)
        if name is None:
            return None
        return name, len(node.args) >= 2 or "default" in kw

    # -- 本体 --------------------------------------------------------------------
    def hits(self) -> list[tuple[int, str]]:
        # 位置は (行, 列) で持つ。同じ行に 2 つ書いても 2 件と数える（許可リストの
        # 件数照合をすり抜けないように）
        out: list[tuple[int, int, str]] = []
        discarded = {id(n.value) for n in ast.walk(self.tree) if isinstance(n, ast.Expr)}

        def add(node: ast.AST, name: str) -> None:
            out.append((node.lineno, node.col_offset, name))

        for node in ast.walk(self.tree):
            read = self._read_call(node)
            if read and read[1]:
                add(node, read[0])
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and self._is_environ(node.func.value) and node.args
                  and (node.func.attr == "setdefault"
                       or (node.func.attr == "pop" and id(node) not in discarded))):
                name = self._name(node.args[0])
                if name:
                    add(node, name)
            elif (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)
                  and self._is_environ(node.value)):
                name = self._name(node.slice)
                if name:
                    add(node, name)
            elif isinstance(node, ast.Compare) and len(node.ops) == 1:
                op, right = node.ops[0], node.comparators[0]
                if isinstance(op, (ast.In, ast.NotIn)) and self._is_environ(right):
                    name = self._name(node.left)
                    if name:
                        add(node, name)
        for scope in [self.tree, *(n for n in ast.walk(self.tree) if isinstance(n, _SCOPES))]:
            out += self._none_checks(scope)
        return [(line, name) for line, _, name in sorted(set(out))]

    def _scope_nodes(self, scope: ast.AST):
        """scope の中のノード。入れ子の関数の中には入らない。"""
        stack = list(ast.iter_child_nodes(scope))
        while stack:
            n = stack.pop()
            yield n
            if not isinstance(n, _SCOPES):
                stack.extend(ast.iter_child_nodes(n))

    def _none_checks(self, scope: ast.AST) -> list[tuple[int, int, str]]:
        # 既定値なしの読み取りを代入した変数 → 変数名（DEPPRO_…）
        bound: dict[str, str] = {}
        nodes = list(self._scope_nodes(scope))
        for n in nodes:
            if (isinstance(n, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
                    and n.value is not None):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                read = self._read_call(n.value)
                if read and not read[1]:
                    for t in targets:
                        if isinstance(t, ast.Name):
                            bound[t.id] = read[0]
        out = []
        for n in nodes:
            if not (isinstance(n, ast.Compare) and len(n.ops) == 1
                    and isinstance(n.ops[0], (ast.Is, ast.IsNot))):
                continue
            left, right = n.left, n.comparators[0]
            for side, other in ((left, right), (right, left)):
                if not (isinstance(other, ast.Constant) and other.value is None):
                    continue
                if isinstance(side, ast.NamedExpr):
                    side = side.value
                read = self._read_call(side)
                if read and not read[1]:
                    out.append((n.lineno, n.col_offset, read[0]))
                elif isinstance(side, ast.Name) and side.id in bound:
                    out.append((n.lineno, n.col_offset, bound[side.id]))
        return out


def _raw_env_reads(tree: ast.AST) -> list[tuple[int, str]]:
    return _EnvReadScanner(tree).hits()


def _scan(root: Path) -> list[tuple[str, int, str]]:
    out = []
    for f in sorted(root.rglob("*.py")):
        if "_archived" in f.parts:
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        rel = f.relative_to(root).as_posix()
        out += [(rel, line, name) for line, name in _raw_env_reads(tree)]
    return out


_BAD_SNIPPETS = {
    # 既存
    "one_line": ('import os\nx = os.environ.get("DEPPRO_A", "DEPPRO_A-not-set")\n', 2, 'DEPPRO_A'),
    "multi_line": (('import os\nx = os.environ.get(\n    "DEPPRO_A",\n'
                   '    "DEPPRO_A-not-set",\n)\n'), 2, 'DEPPRO_A'),
    "single_quote": ("import os\nx = os.environ.get('DEPPRO_A', 'x')\n", 2, 'DEPPRO_A'),
    "other_default": ("import os\nx = os.environ.get('DEPPRO_OUTPUT_ROOT', 'outputs')\n", 2, 'DEPPRO_OUTPUT_ROOT'),
    "keyword_default": ('import os\nx = os.environ.get("DEPPRO_A", default="x")\n', 2, 'DEPPRO_A'),
    "getenv": ('import os\nx = os.getenv("DEPPRO_A", "x")\n', 2, 'DEPPRO_A'),
    "bare_getenv": ('from os import getenv\nx = getenv("DEPPRO_A", "x")\n', 2, 'DEPPRO_A'),
    "bare_environ": ('from os import environ\nx = environ.get("DEPPRO_A", "x")\n', 2, 'DEPPRO_A'),
    "fstring_name": ('import os\nn = "A"\nx = os.environ.get(f"DEPPRO_{n}", "x")\n', 3, 'DEPPRO_'),
    "subscript_read": ('import os\nx = os.environ["DEPPRO_A"] or "x"\n', 2, 'DEPPRO_A'),
    # 定数・別名
    "import_os_as": ('import os as o\nx = o.environ.get("DEPPRO_A", "x")\n', 2, 'DEPPRO_A'),
    "module_const": ('import os\nNAME = "DEPPRO_A"\nx = os.environ.get(NAME, "x")\n', 3, 'DEPPRO_A'),
    "local_const": (('import os\ndef f():\n    name = "DEPPRO_A"\n'
                    '    return os.environ.get(name, "x")\n'), 4, 'DEPPRO_A'),
    "annotated_const": ('import os\nNAME: str = "DEPPRO_A"\nx = os.getenv(NAME, "x")\n', 3, 'DEPPRO_A'),
    "environ_alias": ('import os\nenv = os.environ\nx = env.get("DEPPRO_A", "x")\n', 3, 'DEPPRO_A'),
    "environ_import_as": ('from os import environ as E\nx = E.get("DEPPRO_A", "x")\n', 2, 'DEPPRO_A'),
    "environ_copy": ('import os\nx = os.environ.copy().get("DEPPRO_A", "x")\n', 2, 'DEPPRO_A'),
    "dict_environ": ('import os\nx = dict(os.environ).get("DEPPRO_A", "x")\n', 2, 'DEPPRO_A'),
    "getenv_alias": ('import os\ng = os.getenv\nx = g("DEPPRO_A", "x")\n', 3, 'DEPPRO_A'),
    "getenv_import_as": ('from os import getenv as g\nx = g("DEPPRO_A", "x")\n', 2, 'DEPPRO_A'),
    "getenv_keywords": ('import os\nx = os.getenv(key="DEPPRO_A", default="x")\n', 2, 'DEPPRO_A'),
    "setdefault_discarded": ('import os\nos.environ.setdefault("DEPPRO_A", "x")\n', 2,
                             'DEPPRO_A'),
    "setdefault_used": ('import os\nx = os.environ.setdefault("DEPPRO_A", "x")\n', 2, 'DEPPRO_A'),
    "pop_used": ('import os\nx = os.environ.pop("DEPPRO_A", "x")\n', 2, 'DEPPRO_A'),
    "alias_subscript": ('import os\nenv = os.environ\nx = env["DEPPRO_A"]\n', 3, 'DEPPRO_A'),
    # 元の別名が関数の中（深い）、別名の別名がモジュール直下（浅い）。
    # 1 回の走査では拾えず、繰り返しが要る配置
    "alias_of_alias": (('import os\ndef init():\n    global x\n    x = os.environ\n'
                       'init()\ny = x\nv = y.get("DEPPRO_A", "d")\n'), 7, 'DEPPRO_A'),
    "getenv_alias_of_alias": ('import os\ng = os.getenv\nh = g\nx = h("DEPPRO_A", "x")\n', 4, 'DEPPRO_A'),
    # 存在判定・None 判定
    "membership": ('import os\nif "DEPPRO_A" in os.environ:\n    pass\n', 2, 'DEPPRO_A'),
    "not_membership": ('import os\nif "DEPPRO_A" not in os.environ:\n    pass\n', 2, 'DEPPRO_A'),
    "is_not_none_direct": ('import os\nif os.environ.get("DEPPRO_A") is not None:\n    pass\n', 2, 'DEPPRO_A'),
    "none_is_reversed": ('import os\nif None is os.getenv("DEPPRO_A"):\n    pass\n', 2, 'DEPPRO_A'),
    "is_none_via_var": (('import os\ndef f():\n    v = os.environ.get("DEPPRO_A")\n'
                        '    if v is None:\n        return None\n    return v\n'), 4, 'DEPPRO_A'),
    "walrus": ('import os\nif (v := os.getenv("DEPPRO_A")) is not None:\n    pass\n', 2, 'DEPPRO_A'),
    "annotated_none_check": ('import os\nfrom typing import Optional\ndef f():\n'
                             '    v: Optional[str] = os.environ.get("DEPPRO_A")\n'
                             '    return v is None\n', 5, 'DEPPRO_A'),
    "const_is_none": (('import os\nNAME = "DEPPRO_A"\nv = os.environ.get(NAME)\n'
                      'ok = v is not None\n'), 4, 'DEPPRO_A'),
}

_OK_SNIPPETS = {
    "no_default": 'import os\nx = os.environ.get("DEPPRO_A")\n',
    "no_default_truthy": 'import os\nv = os.environ.get("DEPPRO_A")\nx = v if v else None\n',
    "helper": 'from deppro.paths import env_value\nx = env_value("DEPPRO_A")\n',
    "helper_is_none": ('from deppro.paths import env_value\nv = env_value("DEPPRO_A")\n'
                       'ok = v is None\n'),
    "other_var": 'import os\nx = os.environ.get("HOME", "x")\n',
    "other_var_is_none": 'import os\nv = os.environ.get("HOME")\nok = v is None\n',
    "other_const": 'import os\nNAME = "HOME"\nx = os.environ.get(NAME, "x")\n',
    "other_membership": 'import os\nok = "HOME" in os.environ\n',
    "plain_dict": 'd = {"DEPPRO_A": 1}\nx = d.get("DEPPRO_A", 0)\nok = "DEPPRO_A" in d\n',
    "subscript_write": 'import os\nos.environ["DEPPRO_A"] = "x"\ndel os.environ["DEPPRO_A"]\n',
    "pop_discarded": 'import os\nos.environ.pop("DEPPRO_A", None)\n',
    "setenv_in_test": 'def t(m):\n    m.setenv("DEPPRO_A", "")\n',
    "none_check_other_scope": ('import os\ndef f():\n    v = os.environ.get("DEPPRO_A")\n'
                               '    return v if v else None\n'
                               'def g():\n    v = None\n    return v is None\n'),
}


@pytest.mark.parametrize("key", sorted(_BAD_SNIPPETS))
def test_guard_detects(tmp_path, key):
    # ガードが空虚でないこと: 取りこぼしやすい書き方を 1 つずつ検出する
    code, line, name = _BAD_SNIPPETS[key]
    (tmp_path / "bad.py").write_text(code)
    assert _scan(tmp_path) == [("bad.py", line, name)]


def test_guard_counts_same_line_reads(tmp_path):
    # 同じ行に 2 つ書いても 2 件と数える（許可リストの件数照合をすり抜けない）
    (tmp_path / "two.py").write_text(
        'import os\nx = (os.environ.get("DEPPRO_A", "a"), os.environ.get("DEPPRO_A", "b"))\n')
    assert _scan(tmp_path) == [("two.py", 2, "DEPPRO_A"), ("two.py", 2, "DEPPRO_A")]


@pytest.mark.parametrize("key", sorted(_OK_SNIPPETS))
def test_guard_ignores(tmp_path, key):
    (tmp_path / "ok.py").write_text(_OK_SNIPPETS[key])
    assert _scan(tmp_path) == []


def test_no_raw_env_reads_in_src():
    src = Path(paths.__file__).resolve().parent
    hits = _scan(src)
    counts: dict[tuple[str, str], int] = {}
    for rel, _, name in hits:
        counts[(rel, name)] = counts.get((rel, name), 0) + 1
    unexpected = [h for h in hits if (h[0], h[2]) not in _ALLOWED_RAW_READERS]
    assert unexpected == []
    # 許可リストが古くなっていないこと（消えた・増えた読み場所を見逃さない）
    assert counts == _ALLOWED_RAW_READERS
