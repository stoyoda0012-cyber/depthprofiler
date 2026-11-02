# H5 データ形式 — producer 別 dialect

## 0. この文書の読み方

このリポジトリの H5 には**単一の schema が無い**。書き手が 5 系統あり、
group 配置・grid の記録場所・`dxdydt` の有無と意味がそれぞれ違う。

ファイルを触る前に、**どの producer が書いたかを先に確定させること**。
「`/ezdeprof` に全部入っている」「`dxdydt` は表示 grid ではない」といった
一律の前提はどれも成り立たない。

本文は次を明示的に分けて書く:

- **共通保証**（§3）— どの dialect でも成り立つと当てにしてよいこと
- **実装上の慣行**（§4）— その producer の出力に限って依存してよいこと

数値の実例（特定ファイルの `stack` や時間次元）はここに固定しない。
自分のファイルで確かめる手順は §6 にある。

## 1. producer 別 dialect

| ID | producer | トップレベル | `/ezdeprof` の主な中身 | `stack` | `dxdydt` | grid の記録場所 |
|---|---|---|---|---|---|---|
| **D1** benchmark simple | `benchmarks/generate_checkered.py::save_to_h5` | `Dstk`, `Xstk`, `Drlt` | `depth` (1,-1), `thetaobs`, `stepsize`, `gammaeff`, `photonenergy`, `workfunction`, `charge`, `free`/`subst`/`linked`/`linkedconst`, `TransMatObs`, `totaldepth`, `m`, `noise` | 書く `[n_pixels]` | **書かない** | `ez.attrs["nx"]`, `ez.attrs["ny"]` |
| **D2** MATLAB 互換フル | `benchmarks/generate_checkered.py::save_to_matlab_h5` | `Dstk`, `Drlt`, `Dtrn`, `Xstk`, `Xrlt`, ほか | D1 系に加え MATLAB GUI 由来の制御・品質メタ（`loop`, `currentloop`, `currentstack`, `Qini`, `Qopt`, 各種 `*value` フラグ, `prpties` 等） | 書く `[n_pixels]` uint64 | **書く** `[[nx],[ny],[1]]` shape (3,1) uint16 | `dxdydt[0:2]`（＝`config.nx`, `config.ny`）と `ez.attrs["nx"/"ny"]` の両方 |
| **D3** simulate writer | `simulate/writer.py` | `Dstk`, `Xstk`（gzip） | `depth` (-1,1), `thetaobs`, `stepsize` (1,1), `gammaeff`, `photonenergy`, `workfunction`, `elementsymbol`/`orbital`/`compound`, **`element`**（電荷付きラベル `Si(+4)`）, `charge`, `linkedconst`, `free`/`subst`/`linked`, `TransMatObs`, `loop`, `currentloop`, `BindingEnergy` | 書く `[[n_pixels]]` f64 | **書く** `[[nx, ny, nt]]` shape (1,3) f64（D4 と同じ layout）。`finalize(nx=, ny=)` 未指定なら `[[n_pixels, 1, 1]]` | `dxdydt[0:2]`（`finalize` の `nx`/`ny`。`n_pixels` が `nx*ny` の倍数でなければ書く前に `ValueError`） |
| **D4** app result | `app/depth_result_writer.py` | **`/D_est`** group (`data`, `solver`, `timestamp`), **`/Xstk`**。**`Dstk` は無い** | `depth` (-1,1), `stepsize` (1,1), `element`, `dxdydt`, `thetaobs`（任意）, `TransMatObs`（任意） | **書かない** | **書く** `[[n_x, n_y, n_t]]` shape (1,3) f64。`n_t = n_px // (n_x*n_y)` | `dxdydt` が表示 grid そのもの |
| **D5** legacy io writer | `io/h5_writer.py` | MATLAB 準拠の多数 | `SimulationOrExperiment`, `depth`, `totaldepth`, `stepsize`, `m`, `slice`, `thetaobs`, `thetaopt`, `angle`, `elementsymbol`, `orbital`, `compound`, `bindingenergy`, `elementnumber`, `orbitalnumber`, `charge`, `stringcharge`, `element`, `subst`/`free`/`linked` ほか | 書く（`1` または `n_stack`） | **書く**（`[1, 1, 100]` を固定値で。参照ファイルに合わせた **legacy metadata**） | `dxdydt` からは決まらない |
| *(後処理)* metric writer | `benchmark_h5_writer.py` | 既存ファイルを読み書き | `Dopt` 等を**追記** | — | — | — |

## 2. consumer 側の heuristic（`visualize_4d.py` の loader）

| 項目 | 実装 |
|---|---|
| ピクセル数 `n_px` | **トップレベル `Dstk` の shape[0]** から取る（`/ezdeprof/stack` は読まない） |
| `dxdydt` | `f["ezdeprof/dxdydt"]` を**存在チェックなしで**読む |
| `ez.attrs["nx"/"ny"]` | **読まない** |
| grid 解決順 | ① 明示引数 `nx`/`ny` → ② `dxdydt[0]`, `dxdydt[1]` → ③ 親フォルダ名 `{Nx}x{Ny}x{Nt}` の正規表現 → ④ それでも `n_px` を割り切れなければ `ValueError` |
| 時間次元 | `nt = n_px // (nx * ny)`（`dxdydt[2]` は使わない） |
| reshape | MATLAB column-major（Fortran order のピクセルマップ） |
| 存在チェックがある唯一の dataset | `ezdeprof/Col` |

### 現行 loader との相性

| dialect | 相性 |
|---|---|
| **D5** | **現行 loader の想定に最も近い**（`Dstk` と `dxdydt` の両方がある） |
| **D2** | `Dstk` と `dxdydt` の両方があり、`dxdydt[0:2]` が実際の grid と一致する |
| **D3** | `Dstk`、`element`、`dxdydt` がそろい、`dxdydt[0:2]` が `finalize` に渡した grid（既定は 1 行 `n_pixels × 1`）と一致する |
| **D1** | `dxdydt` が無いため `KeyError`。持っている grid 情報（`ez.attrs`）は loader に使われない |
| **D4** | トップレベル `Dstk` が無いため**現行 loader では読めない**（`dxdydt` は持っている） |

この非対応は既知であり、この文書は現状を記述するもので、loader の修正を意味しない。

## 3. 共通保証

どの dialect でも当てにしてよいのは次だけ。

1. `/ezdeprof` group が存在する。
   ただし**主データがそこにあるとは限らない**（D4 は主データを `/D_est` に置く）。
2. 配列は MATLAB レイアウトで保存される
   （`(pixels, elements, depth)` / `(pixels, elements, angles)`）。
3. 深さ軸 `depth` と `stepsize` は `/ezdeprof` にある。
   **shape は dialect 依存**（`(1,-1)` / `(-1,1)` / `(1,1)`）で揃っていない。
4. **index 群（`free` / `subst` / `linked`）が存在する dialect では、それは
   MATLAB 由来の 1-based。** すべての dialect がこれらを持つわけではない
   （D4 は持たない）。

## 4. 実装上の慣行

その producer の出力に限って依存してよいもの。**別 dialect に持ち込まない。**

### 4.1 `stack`

`stack` は「書いた時点の総ピクセル数」。

- **`stack = nx * ny * dt * n_phases` は `benchmarks/generate_checkered.py` の
  `SimulationConfig.n_pixels` 固有の生成則**（`n_phases` は既定 1、2 = 行き＋戻り）。
  D1 / D2 の出力にのみ当てはまり、他 dialect の `stack` をこの式で分解できる
  保証はない。
- D4 は `stack` を書かない。
- runtime package の読み手はいない（loader は `Dstk.shape[0]` を使う）。
  `stack` に依存する新しいコードを書くなら、対象 dialect を明示すること。

### 4.2 `dxdydt`

**意味が producer によって違う。**

- **D4** — 表示 grid そのもの（`[n_x, n_y, n_t]`）。
- **D2** — `[nx, ny, 1]`。前 2 要素は生成時の grid。第 3 要素は 1 固定で、
  時間次元ではない。
- **D5** — `[1, 1, 100]` の固定値。参照ファイルとの互換のために書かれている
  **legacy metadata** で、この値から grid や時間構造を読み取ることはできない。
- **D1 / D3** — 存在しない。

`generate_checkered.py` の空間変調式に出てくる `dx`, `dy`（`eff_dx`/`eff_dy` は
未指定時に `nx`/`ny` へフォールバック）は**生成側の変調レートのパラメータ**で
あって、H5 の `dxdydt` 一般の意味ではない。

### 4.3 `Dopt`

`/ezdeprof/Dopt` を作るのは後処理の `benchmark_h5_writer.py` だけ。
生成直後のファイルにあると期待しない。

## 5. ベンチマークフォルダ命名規則

```
Simulation/DepthProfiler/{Standard,Checkered}/{Nx}x{Ny}x{Nt}_{N}elem[_variant]/
```

- `Standard/` = 6 元素 (OCAlTiSi)、`Checkered/` = 7 元素 (OBCAlTiSi、B 追加)
- **これは表記の慣例ではなく、コードが実際にパースする入力。**
  `visualize_4d._infer_grid_from_path()` が親フォルダ名から `{Nx}x{Ny}` を
  正規表現で読み、`dxdydt` が使えないときの grid 推定に使う。
- 中間階層を挟むフォルダや `Nt=1` の例外があるため、**パス定数の正本は
  `deppro.benchmarks.paths`**（`GT_6ELEM` 等）。個々のパスをスクリプトに
  直書きしない。
- `paths.BENCHMARK_ROOT` は現状ホームディレクトリ相対で、`DEPPRO_*` 環境変数では
  解決されない。

## 6. 自分のファイルで dialect を判定する

```python
import h5py

with h5py.File(path, "r") as f:
    print("top-level:", list(f.keys()))
    print("ezdeprof :", list(f["ezdeprof"].keys()) if "ezdeprof" in f else "(none)")
    print("attrs    :", dict(f["ezdeprof"].attrs) if "ezdeprof" in f else {})
    for key in ("ezdeprof/stack", "ezdeprof/dxdydt"):
        print(key, "=", f[key][()] if key in f else "(absent)")
    for key in ("Dstk", "D_est/data"):
        print(key, "shape =", f[key].shape if key in f else "(absent)")
```

判定の目安:

- `D_est` がある → **D4**
- `Dtrn` / `Xrlt` / `currentstack` がある → **D2**
- `Dstk` があり `dxdydt` が無い → **D1** か **D3**
  （`elementsymbol` があれば D3、`ez.attrs["nx"]` があれば D1）
- `dxdydt == [1, 1, 100]` → **D5** の可能性が高い（固定値なので grid には使えない）
