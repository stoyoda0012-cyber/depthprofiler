"""リポジトリ衛生テスト — 「test_ を名乗るならテストを持つ」を機械的に担保する。

なぜ要るか
----------
2026-07 の棚卸しで、`test_*.py` を名乗る 106 ファイルのうち **33 本 (31%)** が
test 関数を 1 つも持たない研究スクリプト（`__main__` で走らせる前提のもの）だった。
pytest から見ると「収集して 0 件」なので、**スイートが緑でも中身が無い**状態に
気付けない。実際、収集エラーが 6 件出ていたことも長く放置されていた。

ここで縛るのは 2 点:

    1. test_*.py という名前のファイルは、必ず 1 つ以上の test を定義する。
    2. 配布対象のモジュールは、pyproject が配布から除外しているパッケージ
       （benchmarks / diagnostics / experiments / tests …）を module 直下で import しない。

2 は 2026-09 の実例から。`deppro/simulate/d2s2d.py` が `deppro.benchmarks.generate_checkered`
を module 直下で import していたため、**インストールした wheel では `import deppro` 自体が
落ちていた**（`deppro/__init__` → `simulate/__init__` → `d2s2d`）。開発 checkout には
benchmarks があるので、そこでは一度も気付けない構造だった。依存の向きは core → benchmarks
ではなく benchmarks → core に固定する。`if TYPE_CHECKING:` 配下だけは許容する。

ベンチマーク駆動やデバッグ用のスクリプトは、名前で役割を宣言すること:

    ベンチマーク  → ``deppro/benchmarks/``   (``benchmark_*.py`` / ``run_*.py``)
    診断スクリプト → ``deppro/diagnostics/``  (``diag_*.py``)
    使用例        → ``deppro/examples/``      (``example_*.py``)
"""

import ast
from pathlib import Path

import pytest


def _repo_root() -> Path:
    """pyproject.toml を持つ最初の親ディレクトリ = リポジトリルート。"""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("リポジトリルート (pyproject.toml) が見つからない")


# 走査から外すディレクトリ。
#   archive/   : 凍結した外部由来コード（復元時にそのまま戻せるよう手を入れない）
#   _archived/ : 各所の凍結置き場（pytest の収集からも外してある）
_SKIP_PARTS = {".git", "__pycache__", ".venv", "venv", "node_modules",
               "archive", "_archived", ".pytest_cache"}


def _iter_test_files():
    root = _repo_root()
    for path in sorted(root.rglob("test_*.py")):
        rel = path.relative_to(root)
        if _SKIP_PARTS & set(rel.parts):
            continue
        yield rel, path


def _count_tests(path: Path) -> int:
    """module 直下の ``def test_*`` と ``class Test*`` のテストメソッドを数える。"""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    n = 0
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test"):
                n += 1
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            n += sum(
                1 for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                and m.name.startswith("test")
            )
    return n


def test_every_test_file_defines_tests():
    """test_*.py を名乗るファイルは必ず test を持つ（スクリプト混入の防止）。"""
    empty = []
    for rel, path in _iter_test_files():
        if _count_tests(path) == 0:
            empty.append(str(rel))

    assert not empty, (
        "test_*.py という名前だが test を 1 つも定義していないファイル:\n  "
        + "\n  ".join(empty)
        + "\n\nスクリプトなら役割の分かる場所・名前へ移すこと:\n"
        "  ベンチマーク  → deppro/benchmarks/  (benchmark_*.py / run_*.py)\n"
        "  診断          → deppro/diagnostics/ (diag_*.py)\n"
        "  使用例        → deppro/examples/    (example_*.py)"
    )


def test_every_test_file_parses():
    """収集エラーの早期検知（構文エラーだけは import 前に潰す）。"""
    broken = []
    for rel, path in _iter_test_files():
        try:
            ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            broken.append(f"{rel}: {exc}")
    assert not broken, "構文エラーのある test ファイル:\n  " + "\n  ".join(broken)


def test_scan_found_test_files():
    """走査自体が空振りしていないことの自己チェック（パス変更の検知）。"""
    files = list(_iter_test_files())
    assert len(files) > 20, f"test_*.py の走査結果が少なすぎる: {len(files)} 件"


@pytest.mark.parametrize("directory", ["benchmarks", "diagnostics"])
def test_script_homes_exist(directory):
    """スクリプトの引き取り先が実在すること（移設先を消すと落ちる）。"""
    target = _repo_root() / "src" / "deppro" / directory
    assert target.is_dir(), f"{target} が無い（スクリプトの置き場）"


# ============================================================================
# 2. 配布対象モジュールは除外パッケージを module 直下で import しない
# ============================================================================

def _packaging_find_rules() -> tuple[list[str], list[str]]:
    """pyproject の ``[tool.setuptools.packages.find]`` から include / exclude パターンを読む。

    tomllib は 3.11+、requires-python は 3.10 なので、無ければ tomli、それも無ければ
    このテーブルだけを最小限に読む（値は文字列のリストという前提）。
    """
    text = (_repo_root() / "pyproject.toml").read_text(encoding="utf-8")
    try:
        import tomllib as _toml  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover — 3.10
        try:
            import tomli as _toml  # type: ignore[import-not-found,no-redef]
        except ModuleNotFoundError:
            _toml = None
    if _toml is not None:
        find = _toml.loads(text)["tool"]["setuptools"]["packages"]["find"]
        return list(find.get("include", ["*"])), list(find.get("exclude", []))
    import re  # pragma: no cover — 3.10 without tomli
    section = re.search(r"\[tool\.setuptools\.packages\.find\](.*?)(?:\n\[|\Z)", text, re.S).group(1)
    def _list(key):
        m = re.search(key + r"\s*=\s*\[(.*?)\]", section, re.S)
        return re.findall(r'"([^"]+)"', m.group(1)) if m else []
    return _list("include") or ["*"], _list("exclude")


def _is_excluded(dotted: str, exclude: list[str]) -> bool:
    """dotted 名かその親パッケージのどれかが exclude パターンに当たれば除外。

    setuptools の find は「パッケージ名が exclude に fnmatch で当たる」判定で、
    ``deppro.tests*`` は ``deppro.tests`` と ``deppro.tests.unit`` の両方に当たる。
    """
    from fnmatch import fnmatchcase
    parts = dotted.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if any(fnmatchcase(name, pat) for pat in exclude):
            return True
    return False


def _iter_shipped_modules():
    """配布対象（include に当たり exclude に当たらない）の .py を (dotted module, path) で返す。"""
    include, exclude = _packaging_find_rules()
    from fnmatch import fnmatchcase
    src = _repo_root() / "src"
    for path in sorted((src / "deppro").rglob("*.py")):
        rel = path.relative_to(src)
        if _SKIP_PARTS & set(rel.parts):
            continue
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        dotted = ".".join(parts)
        if not any(fnmatchcase(parts[0], pat) or fnmatchcase(dotted, pat) for pat in include):
            continue
        if _is_excluded(dotted, exclude):
            continue
        yield dotted, path


def _module_level_imports(tree: ast.AST, module: str, is_package: bool):
    """module 直下（関数・クラスの外）で実行される import を (lineno, dotted, type_checking) で返す。

    ``if`` / ``try`` / ``with`` の中は import 時に実行されるので降りる。関数・クラスの中は
    呼ばれるまで実行されないので降りない。相対 import は絶対名に解決する。
    """
    pkg_parts = module.split(".") if is_package else module.split(".")[:-1]
    out = []

    def resolve(node: ast.ImportFrom) -> str:
        if node.level == 0:
            return node.module or ""
        base = pkg_parts[: len(pkg_parts) - (node.level - 1)]
        return ".".join(base + ([node.module] if node.module else []))

    def walk(body, type_checking):
        for n in body:
            if isinstance(n, ast.ImportFrom):
                out.append((n.lineno, resolve(n), type_checking))
            elif isinstance(n, ast.Import):
                out.extend((n.lineno, a.name, type_checking) for a in n.names)
            elif isinstance(n, ast.If):
                t = n.test
                guard = (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (
                    isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING")
                walk(n.body, type_checking or guard)
                walk(n.orelse, type_checking)
            elif isinstance(n, ast.Try):
                walk(n.body, type_checking)
                for h in n.handlers:
                    walk(h.body, type_checking)
                walk(n.orelse, type_checking)
                walk(n.finalbody, type_checking)
            elif isinstance(n, (ast.With, ast.AsyncWith)):
                walk(n.body, type_checking)

    walk(tree.body, False)
    return out


def _excluded_imports_in(path: Path, module: str, exclude: list[str]):
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    is_package = path.name == "__init__.py"
    return [(lineno, target) for lineno, target, tc in _module_level_imports(tree, module, is_package)
            if not tc and target and _is_excluded(target, exclude)]


def test_packaging_rules_are_read_and_classify_known_modules():
    """pyproject の除外規則が読めて、既知のモジュールを正しく振り分けること（空振り防止）。"""
    include, exclude = _packaging_find_rules()
    assert include and exclude, (include, exclude)
    assert _is_excluded("deppro.tests.unit.test_repo_hygiene", exclude)      # このファイル自身は除外側
    assert _is_excluded("deppro.benchmarks.generate_checkered", exclude)
    assert _is_excluded("deppro.diagnostics.dataset_preflight", exclude)
    assert not _is_excluded("deppro.simulate.d2s2d", exclude)                # 2026-09 の実例は配布側
    assert not _is_excluded("deppro.app.contract.provenance", exclude)
    shipped = dict(_iter_shipped_modules())
    assert len(shipped) > 50, f"配布対象の走査結果が少なすぎる: {len(shipped)}"
    assert "deppro.simulate.d2s2d" in shipped and "deppro.benchmarks.generate_checkered" not in shipped


def test_import_detector_sees_module_level_but_not_type_checking_or_function_bodies():
    """検出器そのものの対照: 拾うべきものを拾い、許容すべきものを拾わないこと。"""
    src = (
        "from typing import TYPE_CHECKING\n"
        "import deppro.benchmarks.a as a\n"                 # 拾う
        "from deppro.diagnostics import b\n"                # 拾う
        "from ..benchmarks import c\n"                      # 相対 → deppro.benchmarks、拾う
        "try:\n    from deppro.experiments import d\nexcept ImportError:\n    d = None\n"   # try でも import は走る → 拾う
        "if TYPE_CHECKING:\n    from deppro.benchmarks import e\n"                          # 許容
        "def f():\n    from deppro.benchmarks import g\n    return g\n"                     # 関数内 → 許容
        "from deppro.simulate import h\n"                   # 配布側 → 対象外
    )
    _, exclude = _packaging_find_rules()
    tree = ast.parse(src)
    hits = [(ln, t) for ln, t, tc in _module_level_imports(tree, "deppro.simulate.x", False)
            if not tc and _is_excluded(t, exclude)]
    assert [t for _, t in hits] == ["deppro.benchmarks.a", "deppro.diagnostics", "deppro.benchmarks", "deppro.experiments"], hits


def test_shipped_modules_do_not_import_excluded_packages_at_module_level():
    """配布対象のモジュールは除外パッケージを module 直下で import しない（wheel で import が通る条件）。"""
    _, exclude = _packaging_find_rules()
    violations = []
    for module, path in _iter_shipped_modules():
        for lineno, target in _excluded_imports_in(path, module, exclude):
            violations.append(f"{path.relative_to(_repo_root())}:{lineno}  imports {target}  (in {module})")
    assert not violations, (
        "配布対象モジュールが配布から除外されたパッケージを module 直下で import している。\n"
        "インストールした wheel ではこの import が ModuleNotFoundError になる。共有部品は配布側へ移すか、"
        "関数内で遅延 import して extra / 未配布であることを名指すエラーにする:\n  " + "\n  ".join(violations))


def test_no_test_files_inside_shipped_packages():
    """配布対象のパッケージに test_*.py を置かない。

    置くと wheel に入り、pytest の testpaths（tests/ 配下）からも外れて誰も走らせない。
    2026-09 の実例: ``deppro/io/test_h5_writer.py``（3 test、収集されず、配布物に同梱）。
    """
    _, exclude = _packaging_find_rules()
    src = _repo_root() / "src"
    offenders = []
    for path in sorted((src / "deppro").rglob("test_*.py")):
        rel = path.relative_to(src)
        if _SKIP_PARTS & set(rel.parts):
            continue
        if not _is_excluded(".".join(rel.parts[:-1]), exclude):
            offenders.append(str(path.relative_to(_repo_root())))
    assert not offenders, "配布対象パッケージ内の test_*.py（tests/ へ移す）: " + ", ".join(offenders)


# ============================================================================
# 3. English-only ratchet for shipped code and public documents
# ============================================================================
#
# Public material (README, CITATION, CHANGELOG, pyproject, docs/, and every
# module that ships in the wheel) is being moved to English, one file per
# commit. The baseline file next to this test lists the files that still
# contain Japanese. The two tests below make the list shrink only:
#
#   * a file outside the baseline must contain no Japanese (new code and new
#     documents are written in English; a translated file is removed from the
#     baseline in the same commit);
#   * a baseline entry must still exist and still contain Japanese, so the
#     list cannot go stale and silently stop protecting a translated file.
#
# Test files are not shipped and are not in scope, and docs/ja/ (Japanese
# translations of the English originals) is excluded by design.

_ENGLISH_ONLY_BASELINE = Path(__file__).with_name("english_only_baseline.txt")
_JAPANESE = __import__("re").compile(r"[぀-ヿ一-鿿]")
_PUBLIC_DOCS = ("README.md", "CITATION.cff", "CHANGELOG.md", "pyproject.toml")


def _iter_public_files():
    """(repo-relative path, path) of every file whose text is public: shipped .py and the documents."""
    root = _repo_root()
    seen = set()
    for _, path in _iter_shipped_modules():
        seen.add(path)
        yield str(path.relative_to(root)), path
    for name in _PUBLIC_DOCS:
        p = root / name
        if p.exists():
            yield name, p
    for p in sorted((root / "docs").rglob("*.md")):
        rel = p.relative_to(root)
        if _SKIP_PARTS & set(rel.parts):
            continue
        if rel.parts[:2] == ("docs", "ja"):
            # docs/ja/ is the designated home of Japanese translations of the
            # English originals; it is public but Japanese by design.
            continue
        yield str(rel), p


def _japanese_lines(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    return sum(1 for line in text.splitlines() if _JAPANESE.search(line))


def _english_only_baseline() -> list[str]:
    lines = _ENGLISH_ONLY_BASELINE.read_text(encoding="utf-8").splitlines()
    entries = [ln.split("#", 1)[0].strip() for ln in lines]
    return [e for e in entries if e]


def test_public_files_outside_the_baseline_contain_no_japanese():
    """New or translated public files are English; the baseline only lists the backlog."""
    baseline = set(_english_only_baseline())
    offenders = []
    for rel, path in _iter_public_files():
        if rel in baseline:
            continue
        n = _japanese_lines(path)
        if n:
            offenders.append(f"{rel}  ({n} lines)")
    assert not offenders, (
        "Japanese text in public files that are not in the English-only baseline. "
        "Write new code and documents in English; a file that was translated must not "
        "regain Japanese. (The baseline is "
        f"{_ENGLISH_ONLY_BASELINE.relative_to(_repo_root())} and is only ever shortened.)\n  "
        + "\n  ".join(offenders)
    )


def test_english_only_baseline_entries_are_still_pending():
    """A translated or deleted file must be removed from the baseline in the same commit (the ratchet)."""
    public = dict(_iter_public_files())
    stale = []
    for rel in _english_only_baseline():
        if rel not in public:
            stale.append(f"{rel}: not a public file any more (deleted, moved, or no longer shipped)")
        elif _japanese_lines(public[rel]) == 0:
            stale.append(f"{rel}: contains no Japanese now")
    assert not stale, (
        "Entries of the English-only baseline that are done; remove them so the list keeps shrinking:\n  "
        + "\n  ".join(stale)
    )


def test_english_only_scan_covers_code_and_documents():
    """The scan itself is not empty (path changes are caught here, not by a silent pass)."""
    files = dict(_iter_public_files())
    assert "README.md" in files and "docs/api_reference.md" in files, sorted(files)[:10]
    assert sum(1 for rel in files if rel.endswith(".py")) > 50
    assert _ENGLISH_ONLY_BASELINE.exists()
