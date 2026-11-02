# depthprofiler (`deppro`)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23877023.svg)](https://doi.org/10.5281/zenodo.23877023)

## In English

`deppro` reconstructs depth profiles of composition, D(z), in thin films
from angle-resolved XPS and HAXPES intensities. It solves the
transfer-matrix inverse problem I = T·D with classical solvers (SciPy,
the reference implementation), batched solvers on Apple-silicon GPUs
(MLX) and a neural-network inverse solver. Python 3.11 or later.

```bash
git clone https://github.com/stoyoda0012-cyber/depthprofiler.git
cd depthprofiler
pip install -e '.[fit]'      # core plus the physical tables (toyomacro, pinned release)
python -m deppro.examples.freesolver_recommended
```

Optional extras: `[nn]` (torch and dnndenoiser, for HAXPES frame
denoising) and `[mlx]` (Apple GPU solvers). The sibling packages are
pinned to GitHub release tags in `pyproject.toml`. The API is described
in [`docs/api_reference.md`](docs/api_reference.md), the H5 data format
in [`docs/data_format.md`](docs/data_format.md), and how to recompute the
published JVST and SIA results from their public data deposit in
[`docs/REPRODUCTION_JVST_SIA.md`](docs/REPRODUCTION_JVST_SIA.md). Most of
the documentation is in Japanese for now; English versions will follow.

### Known limitations

- `SemiFreeSolverConfig(gradient_mode=True)` is not available in this
  release and raises `NotImplementedError`.
- The MLX solvers are tested against the CPU reference on defined
  synthetic cases. One acceptance test of the semi-free MLX solver
  (`test_mlx_semifree_batch.py::test_one_sided_objective_and_thickness[LLS]`,
  with the default toyomacro tables) does not pass: on that case the MLX
  objective value differs from the reference by more than the test allows. Numerical equivalence of the
  MLX and CPU solvers is not claimed for all settings.
- Two acceptance tests of the Anscombe transform
  (`test_wls_poisson.py::test_anscombe_improves_poisson` and
  `::test_anscombe_similar_for_gaussian`) pass on the Linux CI but fail
  on the author's macOS machine; no improvement is claimed across
  platforms.
- A plain `pytest` therefore reports these acceptance tests as failures
  on macOS. They are kept failing on purpose rather than relaxed: the
  regression suite (what CI runs on every pull request) deselects them,
  and separate acceptance jobs show their state (the MLX one is run by
  hand on macOS). Skipped tests (for example when an optional dependency or
  a user-supplied table set through `DEPPRO_*` is absent) are not counted
  as verified.

### Citation

If you use this software, please cite it via [CITATION.cff](CITATION.cff).
Archived on Zenodo:

- **All versions** (concept DOI; resolves to the latest):
  [10.5281/zenodo.23877023](https://doi.org/10.5281/zenodo.23877023)
- **v0.1.0**:
  [10.5281/zenodo.23877024](https://doi.org/10.5281/zenodo.23877024)

When a result depends on this software, cite the version DOI of the
release you used, not the all-versions DOI: the concept DOI resolves to
whatever release is latest, so it does not say which code produced a
number.

### Use of generative AI

Anthropic Claude, through Claude Code, was used for code, tests,
documentation and repository maintenance; OpenAI ChatGPT and Codex were
used mainly to review changes and designs. The author made the design
decisions, checked the physics and the numbers, decided which changes to
accept, and is responsible for the published content.

MIT License ([`LICENSE`](LICENSE)).

---

角度分解 XPS (AR-XPS / HAXPES) データから、薄膜の**深さ方向組成分布 D(z)** を
再構成する Python パッケージ。転送行列に基づく逆問題（I = T·D）を、古典ソルバ・
NN 逆ソルバ・Apple MLX バッチソルバで解く。

```
AR-XPS / HAXPES スペクトル
    │  (デノイズ)            … dnndenoiser（別リポジトリ）
    │  (Voigt ピークフィット) … toyomacro（別リポジトリ）
    ▼  深さプロファイル逆問題  … deppro（このリポジトリ）
深さ濃度プロファイル D(z)
```

## 構成

| 場所 | 内容 |
|------|------|
| `src/deppro/` | 転送行列・逆ソルバ（LLS / MEM / L1 / L2）、NN 逆ソルバ (`inference/`)、MLX ソルバ (`solvers_mlx/`)、HAXPES の前処理 (`haxpes/`)、シミュレーション (`simulate/`)、例 (`examples/`) |
| `src/deppro/tests/`、`src/deppro/app/tests/` | 回帰スイートと受入ゲート（CI の構成は `.github/workflows/ci.yml`） |
| `docs/` | 環境変数、H5 のデータ形式、ソルバの API、JVST/SIA 論文の再現 |
| `paper/` | ソフトウェア論文の原稿（JOSS 向け。公開後に投稿する予定） |

## インストール

Python 3.11 以上。

```bash
git clone https://github.com/stoyoda0012-cyber/depthprofiler.git
cd depthprofiler
pip install -e .                 # リポジトリルート（src-layout, pyproject.toml がある場所）
pip install -e '.[fit]'          # 化合物・断面積の表と Voigt フィット（toyomacro、固定版を git から取得）
pip install -e '.[nn]'           # DNN デノイザーの橋渡し（torch + dnndenoiser、固定版を git から取得）
pip install -e '.[mlx]'          # Apple GPU のバッチソルバと NN 逆ソルバ（mlx）
```

`[fit]` と `[nn]` は PyPI に無い兄弟パッケージを GitHub のリリースタグで固定して取得する
（`pyproject.toml` に版と commit を記載）。core だけでも `import deppro` と古典ソルバは動くが、
化合物の表を名前で引く経路は `[fit]` か `DEPPRO_COMPOUND_TABLE_PATH` の CSV が要る
（[`docs/environment_variables.md`](docs/environment_variables.md)）。

例のスクリプト（`deppro/examples/*.py`）は配布物に入るが、例が読むデータ（YAML、CSV）は
wheel / sdist に入らないので、同じタグのリポジトリから取得する。テストも配布物には含まれず、
clone した木で引数なしの `pytest` を走らせる（設定の正本は `pyproject.toml`）。

## 使ってみる

3 層膜の合成データを作り、FreeSolver で深さプロファイルを再構成する例:

```bash
pip install -e '.[fit]'
python -m deppro.examples.freesolver_recommended
```

API の入口は `TransferMatrix`（順モデル）と、`FreeSolver` / `SemiFreeSolver` /
`GenericLayerSolver`（逆問題）。引数と戻り値は
[`docs/api_reference.md`](docs/api_reference.md)、H5 のデータ形式は
[`docs/data_format.md`](docs/data_format.md) を参照。外部データの場所は `DEPPRO_*`
環境変数で渡す（一覧は [`docs/environment_variables.md`](docs/environment_variables.md)）。

## JVST / SIA 論文の再現

このパッケージを使った 2 本の論文（JVST A、SIA）について、公開物だけで何を再計算できるか、
何に著者からの提供物が要るか、実際に何を走らせたかを
[`docs/REPRODUCTION_JVST_SIA.md`](docs/REPRODUCTION_JVST_SIA.md) にまとめてある。
寄託物（Zenodo）の保存済みの結果は `python -m deppro.haxpes.deposit_replay` で検算できる
（寄託物を展開した場所を `--sia-dir` / `--jvst-h5` で渡す。手順は同じ文書にある）。

## 関連リポジトリ

| リポジトリ | 役割 |
|------|------|
| **depthprofiler**（このリポジトリ） | 深さ解析（古典 + NN + MLX の逆ソルバ） |
| [**dnndenoiser**](https://github.com/stoyoda0012-cyber/dnndenoiser) | 深層学習による XPS スペクトルのデノイザー（自己教師ありの学習を含む） |
| [**toyomacro**](https://github.com/stoyoda0012-cyber/toyomacro) | Voigt ピークフィットと、化合物・結合エネルギー・断面積の表 |

## 状態

版は 0.1.0。1.0 までは API が変わることがある。

## 既知の制限

- `SemiFreeSolverConfig(gradient_mode=True)` はこの版では使えない（`NotImplementedError`）。
- MLX のソルバーは、決めた合成データの条件で CPU の参照解と比べて検査している。
  SemiFree の MLX には通っていない受入試験が 1 本ある
  （`test_mlx_semifree_batch.py::test_one_sided_objective_and_thickness[LLS]`、
  既定の toyomacro の表のとき。この条件で MLX の目的関数の値が参照から許容幅を超えてずれる）。
  MLX と CPU の数値的な同等性は、すべての設定については主張しない。
- Anscombe 変換の受入試験 2 本（`test_wls_poisson.py::test_anscombe_improves_poisson`、
  `::test_anscombe_similar_for_gaussian`）は、Linux の CI では通るが著者の macOS では通らない。
  プラットフォームを越えた改善は主張しない。
- そのため macOS で引数なしの `pytest` を走らせると、これらの受入試験が失敗として出る。
  閾値を緩めずに失敗のまま残している。PR ごとの CI（回帰スイート）はこれらを外して回し、
  受入試験の状態は別のジョブで見せる（MLX のものは macOS で手動実行）。skip したテスト
  （optional の依存や、`DEPPRO_*` で利用者が渡す表が無いときなど）は、検証済みには数えない。

## 引用

使うときは [CITATION.cff](CITATION.cff) の情報で引用してほしい。Zenodo に保存してある。

- **全版**（concept DOI。常に最新の版を指す）:
  [10.5281/zenodo.23877023](https://doi.org/10.5281/zenodo.23877023)
- **v0.1.0**:
  [10.5281/zenodo.23877024](https://doi.org/10.5281/zenodo.23877024)

結果がこのソフトウェアに依存するときは、全版の DOI ではなく、使った版の version DOI を引用する。
concept DOI はその時点の最新版を指すので、どの版のコードで出した数値かを示せない。

## 生成 AI の利用

コード・テスト・文書の作成とリポジトリの保守に Anthropic Claude（Claude Code）を、
変更と設計のレビューに主に OpenAI ChatGPT と Codex を使った。設計判断、物理モデルと
数値の検証、変更の採否は著者が行い、公開した内容の責任は著者が持つ。
個々のファイルには AI の関与を記していない。

## ライセンス

MIT License（[`LICENSE`](LICENSE)）。toyomacro・dnndenoiser と同じ条件。
