# depthprofiler (`deppro`) — 日本語版 README

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23877023.svg)](https://doi.org/10.5281/zenodo.23877023)

正本はリポジトリ直下の英語版 [`README.md`](../../README.md)。この文書はその日本語版で、
内容が食い違うときは英語版を正とする。`docs/` の他の文書は英語のみ。

`deppro` は、角度分解 XPS（AR-XPS）や HAXPES のピーク強度から、薄膜の**深さ方向の
組成分布 D(z)** を再構成する Python パッケージである。対象は、角度系列のデータ
（1 試料でも、空間マップの全画素でも）をすでに持っていて、推定した組成分布・膜厚・
界面位置が途中の選択——逆解析に入れる構造仮定、正則化の種類と強さ、スペクトルの
前処理（ビニング、背景、断面積、デノイズ）——にどう依存するかを確かめたい表面科学の
研究者である。

**入力**: 任意のピークフィットプログラムが出した強度 I(θ, 元素)、または `haxpes`
サブパッケージが強度に変換する HAXPES のビニング済みフレームスタック。**出力**:
指定した深さグリッド上の D(z)、モデルがそれを持つ場合はフィットした膜厚と界面位置、
順モデルの予測と残差、マップでは画素ごとの配列と評価指標。Python 3.11 以上。

## 何に使うか

以下の 3 つが現在のパッケージが支える用途で、それぞれを担う API を併記する。
順モデル `TransferMatrix` と観測強度はすべてのソルバで共通なので、別の逆解析とは
別のソルバか別の設定のことであって、別のデータではない。

**既知の層構造の膜厚。** 層の並びと各層の化学量論が分かっているときは、
`GenericLayerSolver` が角度系列から各層の厚さ（任意で界面幅と組成比も）をフィットする。
`python -m deppro.cli_depth solve --config multilayer.yaml` は同じことを YAML の記述
（`src/deppro/examples/multilayer.yaml`）から行う。

**構造仮定を変える。** 3 つのソルバは、逆解析に入れる事前構造の量が違う。
`FreeSolver` は各深さの組成を自由にし、正則化と任意の事前分布に頼る。
`SemiFreeSolver` は基板を誤差関数のステップでモデル化してその位置（膜厚）をフィットし、
膜側の組成は電荷均衡または化学量論の拘束のもとで各深さで自由にする。
`GenericLayerSolver` は層の並びを固定する。どのソルバでも `SolverConfig(method=...)` が
正則化なしの最小二乗、L1、L2、最大エントロピーを選び、ソルバの設定が残りの仮定——
界面幅（既定では固定、`optimize_abruptness`）、膜厚の事前分布（`t_prior_center`）、
化学量論モード、ノイズモデル（WLS、Anscombe、Poisson 尤度）——を露出する。同じ強度を
別の仮定で解き直すことが、D(z)・膜厚・界面位置のその仮定への感度を見る方法である。
`src/deppro/examples/freesolver_recommended.py` は合成 3 層膜に対する FreeSolver の例、
各ソルバの設定は `docs/api_reference.md` にある。

**マップ、前処理、評価。** HAXPES の空間マップでは `deppro.haxpes` がフレームスタックから
画素ごとの深さプロファイルまでを扱う（`process_single_pixel`、multiprocessing の
`run_3d_map`）。前処理の選択——角度ビニング、背景、断面積の出典と非双極子補正、角度の
トリミング、測定系の極角オフセット——は `HAXPESSolverConfig` と `HAXPESInstrumentConfig`
にあり、`dnndenoiser` パッケージでフレームをデノイズする橋渡し（`HaxpesDenoiser`）は
任意である。MLX のバッチソルバ（`deppro.solvers_mlx`）はマップの多数の画素を Apple
シリコンの GPU で同時に解く。評価には、`deppro.simulate` が真値つきの合成プロファイルと
強度を生成し、`deppro.depth_profile` に比較指標（`wasserstein_distance_1d`、
`compare_profiles`、`detect_interfaces`）があり、`deppro.tools.gt_vs_recon` が再構成を
真値と並べて表示する。

これまでに行ったこと、まだ行っていないこと: 公開済みの 2 本の研究（後述）は逆解析を
固定し——L1 正則化・界面幅固定の `SemiFreeSolver`——デノイズ法と露光時間を、画素ごとの
プロファイル、膜厚、擬似真値との Wasserstein-1 距離で比較した。固定した界面幅
（2.5–10 Å）に対する結果の感度は JVST A 論文の §III.A にある。同じ実測マップで
3 つのソルバ族や正則化を比較することは、テストスイートとベンチマークの合成データ上で
行ったのみで、公表した研究としては行っていない。`deppro.batch_processor` の
モンテカルロとブートストラップによる誤差推定はこの版では使えない（「既知の制限」参照）。
`deppro.haxpes.bootstrap` には HAXPES パイプライン用のフレーム／ブロックのブートストラップがある。

```
AR-XPS / HAXPES スペクトル
    │  (デノイズ)            … dnndenoiser（別リポジトリ）
    │  (Voigt ピークフィット) … toyomacro（別リポジトリ）
    ▼  深さプロファイル逆問題  … deppro（このリポジトリ）
深さ濃度プロファイル D(z)
```

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
（[`docs/environment_variables.md`](../environment_variables.md)）。

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
[`docs/api_reference.md`](../api_reference.md)、H5 のデータ形式は
[`docs/data_format.md`](../data_format.md) を参照。外部データの場所は `DEPPRO_*`
環境変数で渡す（一覧は [`docs/environment_variables.md`](../environment_variables.md)）。

## JVST / SIA 論文の再現

このパッケージを使った 2 本の論文（JVST A、SIA）について、公開物だけで何を再計算できるか、
何に著者からの提供物が要るか、実際に何を走らせたかを
[`docs/REPRODUCTION_JVST_SIA.md`](../REPRODUCTION_JVST_SIA.md) にまとめてある。
寄託物（Zenodo）の保存済みの結果は `python -m deppro.haxpes.deposit_replay` で検算できる
（寄託物を展開した場所を `--sia-dir` / `--jvst-h5` で渡す。手順は同じ文書にある）。
この検算は寄託されたプロファイルから公表された指標を再計算するもので、デノイザー・
ピークフィット・逆解析を再実行するものではない。

## 構成

| 場所 | 内容 |
|------|------|
| `src/deppro/` | 転送行列・逆ソルバ（LLS / MEM / L1 / L2）、NN 逆ソルバ (`inference/`)、MLX ソルバ (`solvers_mlx/`)、HAXPES の前処理 (`haxpes/`)、シミュレーション (`simulate/`)、例 (`examples/`) |
| `src/deppro/tests/`、`src/deppro/app/tests/` | 回帰スイートと受入ゲート（CI の構成は `.github/workflows/ci.yml`） |
| `docs/` | 環境変数、H5 のデータ形式、ソルバの API、JVST/SIA 論文の再現 |
| `paper/` | ソフトウェア論文の原稿（JOSS 向け。公開後に投稿する予定） |

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
- `deppro.batch_processor` のモンテカルロとブートストラップの経路（`process_batch`、
  `monte_carlo_batch`、`bootstrap_batch`、`parallel_*` 関数、および `cli_depth` の
  `batch` / `monte-carlo` コマンド）は、削除済みのソルバ向けに書かれたもので、移植されるまで
  `NotImplementedError` を送出する。
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

使うときは [CITATION.cff](../../CITATION.cff) の情報で引用してほしい。Zenodo に保存してある。

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

MIT License（[`LICENSE`](../../LICENSE)）。toyomacro・dnndenoiser と同じ条件。
