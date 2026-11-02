"""MATLAB DepthProfiler CLI との接続アダプタ。

外部の MATLAB 生成器を呼び、確定した manifest を解析側の契約へ渡すだけの薄い層。
数値も物理も解釈せず、**producer が宣言したものをそのまま持ち回る**。

生成側の契約（`<DEPPRO_MATLAB_SOURCE_ROOT>/DepthProfiler/_cli/README.md` が正本）:

* `_cli/dp <cfg.json> [result.json]` が生成し、**別プロセス**で確認する
* 終了コード 0 = 「生成が成功し、閉じたファイルを読み直して整合が取れた」
* 出力先の ``manifest.json`` が正本。``result.json`` は同一 ``generationId`` の複製
* ``confirmed`` ブロックは**構造的**な確認であって、物理的な正しさの主張ではない

CLI の場所は ``DEPPRO_DEPTHPROFILER_CLI``。**未設定でも
import は通り**、実際に使う段で変数名を含むエラーになる。出力先は既存の
``DEPPRO_SIMULATION_ROOT`` を再利用する。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from deppro.paths import env_path_or_placeholder

__all__ = [
    "MatlabGenerationError",
    "ManifestRejected",
    "UnsupportedLayout",
    "matlab_cli",
    "generate",
    "load_confirmed_manifest",
    "stack_coordinates",
    "grid_coverage",
    "sample_ids",
    "SUPPORTED_LAYOUTS",
    "SCHEMA_MAJOR",
]

SCHEMA_MAJOR = 1
SUPPORTED_LAYOUTS = ("rectangular_grid", "rectangular_grid_partial")
NO_GRID_LAYOUTS = ("flat_samples",)


class MatlabGenerationError(RuntimeError):
    """CLI の起動・実行に失敗した。"""


class ManifestRejected(ValueError):
    """manifest はあるが、受け入れ条件を満たさない。

    途中の JSON、未確認、世代の食い違いなどをここで弾く。
    """


class UnsupportedLayout(NotImplementedError):
    """宣言されたレイアウトをこの層が扱えない。

    **矩形へ黙って reshape しないための例外。** 未知のレイアウトを既知として
    処理するより、明示的に止まるほうがよい。
    """


def matlab_cli() -> Path:
    """MATLAB 生成 CLI (`_cli/dp`) の場所（``DEPPRO_DEPTHPROFILER_CLI``）。

    未設定でも import は通り、ここを呼んだ段で変数名を含むエラーになる。
    既定値を書かないのは、MATLAB 側の置き場が環境ごとに違うため。
    """
    return env_path_or_placeholder("DEPPRO_DEPTHPROFILER_CLI")


@dataclass(frozen=True)
class GenerationResult:
    """1 回の生成の結果。

    Attributes
    ----------
    manifest:
        出力先フォルダの ``manifest.json`` の内容（``confirmed`` 済み）。
    folder:
        生成物の置き場。
    generation_id:
        この生成の識別子。sample_id の前半になる。
    """

    manifest: Dict[str, Any]
    folder: Path
    generation_id: str

    @property
    def main_file(self) -> Path:
        return Path(self.manifest["mainFile"])

    @property
    def layout(self) -> Dict[str, Any]:
        return self.manifest.get("layout", {})


def generate(
    cfg: Dict[str, Any],
    *,
    timeout: Optional[float] = None,
    cli: Optional[Path] = None,
) -> GenerationResult:
    """cfg を MATLAB CLI に渡して生成し、確認済み manifest を返す。

    Parameters
    ----------
    cfg:
        生成設定。キーは生成側 ``_cli/README.md`` の表と 1:1。
        ``outputFolder`` は呼び出し側が決める（この層では既定値を作らない）。
    timeout:
        秒。超えたら :class:`MatlabGenerationError`。
    cli:
        CLI のパス。省略時は ``DEPPRO_DEPTHPROFILER_CLI``。

    Raises
    ------
    MatlabGenerationError
        CLI が見つからない、起動できない、非 0 終了、timeout。
    ManifestRejected
        終了コードは 0 でも manifest が受け入れ条件を満たさない。
    """
    if "outputFolder" not in cfg:
        raise ValueError("cfg に outputFolder が要る（この層では既定値を決めない）")
    exe = Path(cli) if cli is not None else matlab_cli()
    if not exe.exists():
        raise MatlabGenerationError(
            f"MATLAB CLI が見つからない: {exe}. "
            "DEPPRO_DEPTHPROFILER_CLI に _cli/dp のパスを設定する"
        )

    folder = Path(cfg["outputFolder"]).expanduser()
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "cfg.json"
        result_path = Path(tmp) / "result.json"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        # 新しいプロセスグループで起こす。`_cli/dp` は生成段の MATLAB を exec
        # せず子として起こすので、subprocess.run(timeout=...) の既定では直接の
        # 子 (`sh`) しか殺せない。MATLAB は走り続け、「timeout = 失敗」と報告した
        # あとで成功物が現れる。実際に孫が生き残ることを fixture で確認した。
        try:
            proc = subprocess.Popen(
                [str(exe), str(cfg_path), str(result_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            raise MatlabGenerationError(f"CLI を起動できない: {exe} ({exc})") from exc

        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            killed = _kill_process_group(proc)
            proc.communicate()
            raise MatlabGenerationError(
                f"CLI が {timeout} 秒で終わらなかった: {exe}"
                f"（プロセスグループを終了させた: {killed}）"
            ) from exc

        if proc.returncode != 0:
            raise MatlabGenerationError(
                f"CLI が非 0 終了 (returncode={proc.returncode}).\n"
                f"stdout tail:\n{(out or '')[-2000:]}\n"
                f"stderr tail:\n{(err or '')[-2000:]}"
            )

    result = load_confirmed_manifest(folder)
    _check_matches_request(result, cfg, folder)
    return result


def _kill_process_group(proc: "subprocess.Popen") -> str:
    """Kill the CLI and everything it started, not just the直接の子."""
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        proc.kill()
        return "process only (no group)"
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return f"pgid {pgid}"
        except OSError:
            break
        try:
            proc.wait(timeout=5)
            return f"pgid {pgid}"
        except subprocess.TimeoutExpired:
            continue
    return f"pgid {pgid} (may not have exited)"


def _check_matches_request(result: GenerationResult, cfg: Dict[str, Any], folder: Path) -> None:
    """返ってきた manifest が **この cfg の生成物** かを照合する。

    終了コード 0 と `confirmed` の内部整合だけでは、CLI が cfg を無視して古い
    確定済み manifest をそのまま返した場合を見分けられない。`ifExists` の
    扱い次第では実際に起こりうるので、点数と出力先を突き合わせる。
    """
    axes = result.manifest.get("resolved", {}).get("axes", {})
    problems: List[str] = []
    for cfg_key, axis_key in (("gridX", "nx"), ("gridY", "ny"), ("timeSteps", "nt")):
        want = cfg.get(cfg_key)
        if want is None:
            continue
        got = axes.get(axis_key)
        if got is None or int(got) != int(want):
            problems.append(f"cfg.{cfg_key}={want} だが resolved.axes.{axis_key}={got}")
    declared_folder = result.manifest.get("outputFolder")
    if isinstance(declared_folder, str) and declared_folder:
        try:
            same = Path(declared_folder).expanduser().resolve() == folder.resolve()
        except OSError:
            same = str(declared_folder) == str(folder)
        if not same:
            problems.append(
                f"manifest.outputFolder={declared_folder!r} が読み口 {str(folder)!r} と違う"
            )
    if problems:
        raise ManifestRejected(
            "manifest がこの要求の生成物ではない（古い確定済みを再利用した可能性）:\n  "
            + "\n  ".join(problems)
        )


def load_confirmed_manifest(folder: Path | str) -> GenerationResult:
    """出力先の ``manifest.json`` を読み、受け入れ条件を検査して返す。

    終了コード 0 だけでは足りない。ここで見るのは:

    1. manifest が存在し、丸ごと parse できる（途中 JSON を拾わない）
    2. ``schemaVersion`` の major がこの層の想定と一致
    3. ``status == "ok"``
    4. ``confirmed`` が存在し、``confirmed.generationId`` が
       ``generationId`` と一致（**別世代の確認を流用しない**）
    5. ``resolved.files`` が**非空**で、名指しするファイルが実在し記録どおりのサイズ
    6. ``mainFile`` が存在し、実ファイルで、``resolved.files`` の 1 件と同じ
       artifact を指す

    5 と 6 は後から足した。それ以前は ``resolved.files`` が無いと ``[]`` に
    なってループが空回りし、**H5 が 1 つも無い合成 manifest を受理**していた。
    整合が取れているのは manifest の内部だけで、ディスクには何も無い状態を
    「確認済み」と呼んでしまう。

    ``confirmed`` は構造的な確認であって、物理的な正しさでも全 stack の
    正常性でもない。生成側の ``confirmed.scope`` にそう書いてある。
    """
    folder = Path(folder).expanduser()
    path = folder / "manifest.json"
    if not path.is_file():
        derived = folder / "derived_view.json"
        if derived.is_file():
            raise ManifestRejected(
                f"{folder} は derived_view（GUI 検証用のコピー）で、生成データセットでは"
                " ない。実行の入口にしない。値の正本は derived_view.json の parent.folder"
            )
        raise ManifestRejected(f"manifest.json が無い: {folder}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestRejected(f"manifest.json を parse できない: {path} ({exc})") from exc

    if not isinstance(manifest, dict):
        raise ManifestRejected(
            f"manifest.json の最上位が {type(manifest).__name__}（object のはず）: {path}"
        )

    schema = str(manifest.get("schemaVersion", ""))
    major = schema.split(".")[0] if schema else ""
    if major != str(SCHEMA_MAJOR):
        raise ManifestRejected(
            f"schemaVersion {schema!r} はこの層 (major {SCHEMA_MAJOR}) の想定外"
        )

    status = manifest.get("status")
    if status != "ok":
        raise ManifestRejected(f"status が {status!r}（ok ではない）")

    generation_id = manifest.get("generationId")
    if not generation_id:
        raise ManifestRejected("generationId が無い")

    confirmed = manifest.get("confirmed")
    if not isinstance(confirmed, dict):
        raise ManifestRejected(
            "confirmed ブロックが無い。生成プロセスの外からの確認が済んでいない"
        )
    if confirmed.get("status") != "ok":
        # 生成側 1.0.0 の confirmed は at/by/stack/elements しか持たない。
        # 版番号が据え置かれたまま中身が足されたので、版だけでは見分けられない
        # 形から言い当てて理由を具体的にする。
        looks_old = "status" not in confirmed and "generationId" not in confirmed
        hint = ("（confirmed に status も generationId も無い＝生成側 1.1.0 より前の形。"
                "この世代の生成物には /stackxyt も無い）" if looks_old else "")
        raise ManifestRejected(f"confirmed.status が {confirmed.get('status')!r}{hint}")
    if confirmed.get("generationId") != generation_id:
        raise ManifestRejected(
            "confirmed が別世代のもの: "
            f"manifest={generation_id!r} confirmed={confirmed.get('generationId')!r}"
        )

    declared_folder = manifest.get("outputFolder")
    if isinstance(declared_folder, str) and declared_folder:
        try:
            same = Path(declared_folder).expanduser().resolve() == folder.resolve()
        except OSError:
            same = str(declared_folder) == str(folder)
        if not same:
            raise ManifestRejected(
                f"manifest.outputFolder={declared_folder!r} が、読み込んだフォルダ "
                f"{str(folder)!r} と違う。別の出力先の manifest を置き違えている"
            )

    _check_files(manifest, folder)
    return GenerationResult(manifest=manifest, folder=folder, generation_id=generation_id)


def _check_files(manifest: Dict[str, Any], folder: Path) -> None:
    """manifest が名指しするファイルが、記録どおりの大きさで実在するか。

    サイズ一致は cheap check であって改竄検出でも完全性保証でもない
    （生成側の ``unknown.artifactHashes`` にそう書いてある）。ここが見るのは
    「manifest の言うものがディスクに在るか」だけ。

    **空の一覧を成功にしない。** 以前は ``resolved.files`` が無いと ``[]`` に
    なり、for が一度も回らずに素通りしていた。H5 が 1 つも無い合成 manifest が
    受理される。検査の対象が 0 件だったことと、検査に通ったことは別。
    """
    resolved = manifest.get("resolved")
    if not isinstance(resolved, dict):
        raise ManifestRejected("resolved ブロックが無い（または object でない）")
    files = resolved.get("files")
    if not isinstance(files, list) or not files:
        raise ManifestRejected(
            "resolved.files が無いか空。生成物を 1 つも名指ししていない manifest は"
            "「確認済み」とは呼べない"
        )

    problems: List[str] = []
    present: Dict[Path, int] = {}
    for entry in files:
        if not isinstance(entry, dict):
            problems.append(f"resolved.files に object でない要素: {entry!r}")
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            problems.append(f"resolved.files の要素に name が無い: {entry!r}")
            continue
        target = folder / name
        if not target.is_file():
            problems.append(f"{name}: manifest にあるがディスクに無い")
            continue
        size = target.stat().st_size
        expected = entry.get("bytes")
        if expected is not None:
            if not isinstance(expected, (int, float)) or isinstance(expected, bool):
                problems.append(f"{name}: bytes が数値でない ({expected!r})")
                continue
            if size != int(expected):
                problems.append(f"{name}: {size} bytes、manifest は {int(expected)}")
                continue
        try:
            present[target.resolve()] = size
        except OSError:
            present[target] = size

    # mainFile は「在る」だけでは足りない。一覧に載っていない主 H5 は、
    # 検査されていない artifact をこの層が開くことになる。
    main = manifest.get("mainFile")
    if not isinstance(main, str) or not main:
        problems.append("mainFile が無い（または文字列でない）")
    elif not Path(main).expanduser().is_absolute():
        # 相対パスは cwd 基準で解決されるので、同じ manifest が呼び出し元の
        # カレントディレクトリ次第で受理にも拒否にもなる。生成側は常に絶対で
        # 書くので、相対は黙って解決せずここで止める。
        problems.append(
            f"mainFile が相対パス ({main!r})。cwd 依存になるので受け付けない"
            "（生成側は絶対パスで書く）"
        )
    else:
        mp = Path(main).expanduser()
        if not mp.is_file():
            problems.append(f"mainFile がディスクに無い: {main}")
        else:
            try:
                key = mp.resolve()
            except OSError:
                key = mp
            if key not in present:
                problems.append(
                    f"mainFile {main} が resolved.files のどれとも一致しない"
                    "（一覧で検査されていない artifact を開こうとしている）"
                )

    if problems:
        raise ManifestRejected("manifest とディスクが食い違う:\n  " + "\n  ".join(problems))


def stack_coordinates(result: GenerationResult, *, allow_partial: bool = False) -> np.ndarray:
    """各 stack の (x, y, t) を ``(stack, 3)`` で返す。

    ファイル内（h5py が見る形）は ``(3, stack)`` で、MATLAB の報告とは
    転置している。**ここで必ず ``(stack, 3)`` に直す**のがこの層の契約。

    これはこのレイアウトにおける**インデックス対応**であって、物理座標でも
    取得順序でも時刻でもない。それらは別の情報で、記録されていない。

    結果は 3 通りある（:func:`grid_coverage` が事実を返す）:

    1. 格子が成立して完全 → 座標を返す
    2. 格子は成立するが末尾フレームが欠ける → ``allow_partial=True`` を
       明示したときだけ返す。黙って完全なものとして渡さない
    3. 格子が無い / 未知 → :class:`UnsupportedLayout`。**この場合でも
       :func:`sample_ids` は使える**（識別子は格子に依存しない）

    ``stack`` が ``nx*ny`` の倍数である必要は無い。生成器の場面は flat index の
    関数で、フレーム境界という概念を持たない。
    """
    cov = grid_coverage(result)
    ltype = cov["type"]
    if ltype in NO_GRID_LAYOUTS:
        raise UnsupportedLayout(
            f"レイアウト {ltype!r} には格子が無い（実験データなら stack は測定点数で、"
            " dxdydt の格子とは無関係）。座標は作れないが、**sample_ids は使える**"
        )
    if ltype not in SUPPORTED_LAYOUTS:
        raise UnsupportedLayout(
            f"レイアウト {ltype!r} はこの層で扱えない（対応: {SUPPORTED_LAYOUTS}）。"
            " 矩形として reshape しない。sample_ids は使える"
        )
    if not cov["consistent"]:
        raise ManifestRejected(
            f"layout {cov['nx']}x{cov['ny']}x{cov['nt']} が宣言 stack {cov['stack']} と"
            f" 噛み合わない（{cov['note']}）。格子解釈が成り立たないだけで、flat な標本列"
            " としては有効: sample_ids はそのまま使える"
        )
    if cov["partial"] and not allow_partial:
        raise ManifestRejected(
            f"最後のフレームが {cov['last_frame_count']}/{cov['nx']*cov['ny']} 点しか無い"
            " 部分格子。完全な格子として黙って返さない。受け取るなら"
            " stack_coordinates(..., allow_partial=True) と明示する"
        )

    nx, ny, nt, stack = cov["nx"], cov["ny"], cov["nt"], cov["stack"]

    import h5py  # 重いので遅延 import

    with h5py.File(result.main_file, "r") as f:
        if "stackxyt" not in f:
            raise ManifestRejected(
                f"{result.main_file} に /stackxyt が無い。座標を宣言する生成器より"
                " 前の出力（境界は日付ではなく生成器の版。2026-09-07 午前に作られた"
                " 生成物にも入っていないものがある）"
            )
        raw = np.asarray(f["stackxyt"])

    if raw.ndim != 2:
        raise ManifestRejected(f"/stackxyt の次元数が {raw.ndim}（2 のはず）")

    # 向きは **schema で決める**。形から推測すると stack==3 のとき (3,3) が
    # 曖昧になり、転置したものを黙って返してしまう（実際にそうなっていた）。
    # 保存形は (3, stack) と宣言されているので、それに合う軸だけを受け入れる。
    if raw.shape == (3, stack):
        xyt = raw.T
    elif stack == 3 and raw.shape == (3, 3):
        xyt = raw.T          # 宣言どおり行が (x,y,t)。曖昧さは schema で断ち切る
    elif raw.shape == (stack, 3) and stack != 3:
        raise ManifestRejected(
            f"/stackxyt の形が {raw.shape}。保存形は (3, stack) = (3, {stack}) と"
            "宣言されている。転置済みのものを黙って受け入れない"
        )
    else:
        raise ManifestRejected(
            f"/stackxyt の形が {raw.shape}。宣言 stack={stack} に対して "
            f"(3, {stack}) のはず"
        )

    if not np.all(np.isfinite(xyt)):
        raise ManifestRejected("/stackxyt に非有限値がある")
    if not np.array_equal(xyt, np.rint(xyt)):
        raise ManifestRejected("/stackxyt に整数でない値がある（格子インデックス）")

    idx = xyt.astype(np.int64)
    lo = idx.min(axis=0)
    hi = idx.max(axis=0)
    if lo.min() < 1 or hi[0] > nx or hi[1] > ny or hi[2] > nt:
        raise ManifestRejected(
            f"/stackxyt が範囲外: x∈[{lo[0]},{hi[0]}] y∈[{lo[1]},{hi[1]}] "
            f"t∈[{lo[2]},{hi[2]}]、宣言は 1..{nx} / 1..{ny} / 1..{nt}"
        )

    # 宣言式との一致。座標が自分自身と無矛盾なだけでは、行の並びが宣言どおりだ
    # という証拠にならない（行を並べ替えても座標は座標のまま）。
    # The formula is a bijection for ANY stack, so a truncated last frame is
    # checked exactly like a complete one.
    sv = np.arange(1, stack + 1, dtype=np.int64)
    want = np.stack([(sv - 1) % nx + 1,
                     ((sv - 1) % (nx * ny)) // nx + 1,
                     (sv - 1) // (nx * ny) + 1], axis=1)
    if not np.array_equal(idx, want):
        bad = int(np.flatnonzero((idx != want).any(axis=1))[0])
        raise ManifestRejected(
            f"/stackxyt が宣言式 (flattenOrder.stackToXYT) と一致しない。"
            f"最初の不一致 stack={bad + 1}: got {idx[bad].tolist()} want {want[bad].tolist()}"
        )
    return xyt


def grid_coverage(result: GenerationResult) -> Dict[str, Any]:
    """格子解釈が成り立つか、成り立つとして完全かを**事実として**返す。

    推測しない。生成側が宣言した ``layout`` と ``resolved.axes.stack`` を突き
    合わせるだけ。呼び出し側がこれを見ずに座標を受け取れないよう、
    :func:`stack_coordinates` は部分格子で明示の opt-in を要求する。
    """
    layout = result.layout
    ltype = layout.get("type")
    out: Dict[str, Any] = {"type": ltype, "partial": False, "consistent": False,
                           "nx": None, "ny": None, "nt": None,
                           "stack": None, "last_frame_count": None, "note": ""}
    if ltype in NO_GRID_LAYOUTS:
        out["note"] = "格子は宣言されていない"
        return out
    if ltype not in SUPPORTED_LAYOUTS:
        out["note"] = f"未対応のレイアウト {ltype!r}"
        return out

    nx, ny, nt = (_positive_int(layout, k, "layout") for k in ("nx", "ny", "nt"))
    stack = _declared_stack(result)
    ppf = nx * ny
    out.update(nx=nx, ny=ny, nt=nt, stack=stack)

    if ltype == "rectangular_grid":
        out["consistent"] = (ppf * nt == stack)
        out["last_frame_count"] = ppf if out["consistent"] else None
        out["note"] = ("" if out["consistent"] else
                       f"完全格子を名乗るが {nx}*{ny}*{nt} = {ppf*nt} ≠ stack {stack}")
        return out

    # rectangular_grid_partial: nt counts the truncated frame too
    tail = stack - (nt - 1) * ppf
    out["partial"] = True
    out["last_frame_count"] = tail
    out["consistent"] = 0 < tail <= ppf
    out["note"] = ("" if out["consistent"] else
                   f"部分格子を名乗るが末尾フレームの点数が {tail}（1..{ppf} の外）")
    return out


def _positive_int(d: Dict[str, Any], key: str, where: str) -> int:
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v) or int(v) < 1:
        raise ManifestRejected(f"{where}.{key} が正の整数でない: {v!r}")
    return int(v)


def _declared_stack(result: GenerationResult) -> int:
    axes = result.manifest.get("resolved", {}).get("axes", {})
    if not isinstance(axes, dict) or "stack" not in axes:
        raise ManifestRejected("resolved.axes.stack が無い")
    return _positive_int(axes, "stack", "resolved.axes")


def sample_ids(result: GenerationResult, indices: Sequence[int] | None = None) -> List[str]:
    """sample_id を作る。既定は全 stack。

    ``<generationId>#<stackIndex>``、stackIndex は 1-based で**最初に書かれた
    ときの番号**。配列の行番号は位置であって識別子ではない ── 抜き出したり
    並べ替えたりすれば行 7 は別の試料点になる。抽出・並べ替えのあとも、
    元の番号のまま持ち回ること。
    """
    stack = _declared_stack(result)
    if indices is None:
        return [f"{result.generation_id}#{i}" for i in range(1, stack + 1)]

    # 0-based の行番号をそのまま渡す取り違えが起きる。#0 は 1-based では存在
    # しないのに、素通しすると体裁の整った無意味な id ができてしまう。
    # bool と非整数もここで弾く: int() に通してしまうと True -> 1、1.9 -> 1 と
    # 黙って化けて、**別々に指定した点が同じ id になる**。正当な重複抽出と、
    # 型変換で生まれた衝突を区別できなくなる。
    out: List[str] = []
    bad: List[str] = []
    for raw in indices:
        if isinstance(raw, bool):
            bad.append(f"{raw!r} (bool)")
            continue
        if isinstance(raw, (int, np.integer)):
            i = int(raw)
        elif isinstance(raw, (float, np.floating)):
            if not np.isfinite(raw) or float(raw) != int(raw):
                bad.append(f"{raw!r} (整数でない)")
                continue
            i = int(raw)
        else:
            bad.append(f"{raw!r} ({type(raw).__name__})")
            continue
        if i < 1 or i > stack:
            bad.append(f"{i} (範囲外)")
            continue
        out.append(f"{result.generation_id}#{i}")
    if bad:
        raise ValueError(
            f"stackIndex は 1..{stack} の 1-based 整数。受け付けられない値: {bad[:8]}"
            "（配列の行番号や bool を渡していないか確認する）"
        )
    return out
