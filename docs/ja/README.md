# depthprofiler (`deppro`) — 日本語版 README

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23877023.svg)](https://doi.org/10.5281/zenodo.23877023)

正本はリポジトリ直下の英語版 [`README.md`](../../README.md)。この文書はその日本語版で、
内容が食い違うときは英語版を正とする。`docs/` の他の文書は英語のみ。

角度分解 XPS (AR-XPS / HAXPES) の強度から、薄膜の**深さ方向組成分布 D(z)** を
再構成する Python パッケージ。角度分解の深さ系列を、1 か所の測定でも多数の画素や
フレームでも解析し、再構成した分布がそこに至るまでの選択にどれだけ依存するかを
知る必要がある表面科学の研究者に向けて書いている。

- **入力**: 画素ごとの、各元素・各出射角のピーク強度（`n_angles × n_elements` の配列。
  バックグラウンドを引き、断面積で補正したもの）、光子エネルギーと幾何、非弾性平均自由
  行程を求めるための各元素の化合物、どの元素が膜でどの元素が基板かを示す構造の記述。
- **出力**: 画素ごとの深さ格子上の D(z)。構造モデルに応じて、膜厚（基板界面の深さ）、
  界面幅、各層の膜厚。
- **研究での使い方**: 同じ強度を、構造の仮定・正則化・前処理を変えて逆解析し、
  組成分布・膜厚・界面位置がどう変わるかを画素ごとに追う。

```
AR-XPS / HAXPES スペクトル
    │  (デノイズ)            … dnndenoiser（別リポジトリ）
    │  (Voigt ピークフィット) … toyomacro（別リポジトリ）
    ▼  深さプロファイル逆問題  … deppro（このリポジトリ）
深さ濃度プロファイル D(z)
```

## これで何ができるか

三つの古典ソルバは、同じ転送行列（I = T·D の `TransferMatrix`）から予測を作り、同じ強度の
配列を受け取る。そのため構造の仮定を変えても、順モデルの物理は共通のままである。ただし
強度の規格化の仕方、最適化法、既定の事前分布はソルバごとに違うので、二つの再構成の違いは
これらの設定と合わせて読む。

**既知の層構造を当てはめる。** `GenericLayerSolver` は、化学量論を与えた層の積み重ね
（`GenericLayerConfig`、`LayerDefinition`）を受け取り、各層の膜厚を当てはめる。界面幅と、
範囲内での化学量論も当てはめられる。設定は Python で行う。`src/deppro/examples/` の YAML 設定は
`python -m deppro.cli_depth solve --config …` で `SemiFreeSolver` を動かすもので、
このソルバは動かさない。

**同じデータで構造の仮定を変える。** 三つのソルバは異なる事前知識を課す:

| ソルバ | 仮定すること | 返すもの |
|---|---|---|
| `FreeSolver` | 層は仮定しない。各深さで濃度が非負で和が 1（既定では、酸化数を与えたとき電荷中性も） | D(z) |
| `SemiFreeSolver` | 誤差関数の形をした埋もれた基板界面が一つある。膜の組成は各深さで自由 | D(z)、界面の深さ（膜厚）、基板の元素比。界面幅は、当てはめを有効にしない限り固定（既定 5 Å FWHM） |
| `GenericLayerSolver` | 化学量論を与えた層の積み重ね | D(z)、各層の膜厚。任意で界面幅と化学量論 |

`FreeSolver` と `SemiFreeSolver` は正則化を設定（`SolverConfig.method`: 正則化なしの
最小二乗 `LLS`、`L1`、`L2`、最大エントロピー `MEM`）として受け取る。`LLS` には罰則が
無い。それ以外では、罰則が引き寄せる先の事前分布も、既定ではソルバと方法によって変わる。
`FreeSolver` では `L2`/`MEM` で一様、`L1` でほぼ 0 である。`L1` の事前分布は濃度の下限と
等しいので、各深さの総和を 1 とする制約の下では、既定の `L1` 項は一定になり、疎な分布を
優先しない。`SemiFreeSolver` ではほぼ 0 で、罰則は膜厚も初期値へ弱く引き寄せる。罰則だけを変えたいときは、事前分布（`D_prior`）を明示して渡す。
`GenericLayerSolver` は正則化しない。`StructureAnalyzer` は強度からソルバを提案する。

真値がわかっている条件で選択の効果を見るには、合成データの往復計算を使う。分布の
マップを作り、Poisson ノイズを加えて逆解析し、元素ごとの RMSE と Wasserstein 距離 W1 を
出す:

```bash
python -m deppro.simulate.d2s2d --preset 7elem_checkered --nx 3 --ny 3 --dt 1 \
    --noise 0.03 --inverse free
python -m deppro.simulate.d2s2d --preset 7elem_checkered --nx 3 --ny 3 --dt 1 \
    --noise 0.03 --inverse semifree --semifree-init base_fv
```

`SemiFreeSolver` の初期値も仮定の一部である。往復計算は界面幅を当てはめないので、
`--semifree-init` が選ぶ幅はそのまま課される。既定の `gt_fv` は画素ごとの真の膜厚と幅
（測定では得られない情報）を使う。`base_fv` はシーンの公称値（210 Å、20 Å）を使う。
`default` は 30 Å と 5 Å を使い、30 Å の初期値は往復計算の膜厚の下限 50 Å で切られ、
このシーンでは当てはめた膜厚がその下限に留まる。シミュレータは画素ごとの真の組成で
平均自由行程を重み付けするが、逆解析は一様組成の転送行列を使うので、表示される誤差には
その差も含まれる。数値はこの合成シーンとノイズ水準でのもので、ベンチマークではない。

**多数の画素と前処理条件を比べる。** マップやフレーム系列では、画素ごとに独立に逆解析する:

- `deppro.haxpes.pipeline.process_single_pixel` は 1 画素のビン分け済みスペクトルを受け取り、
  バックグラウンド除去、ピーク積分、断面積補正、L1 の `SemiFreeSolver` による当てはめを行う。
  引数で、元素とその役割、深さ格子、正則化の重み、膜厚の初期値、任意の非双極子補正を決める。
  `run_3d_map` はこれを全画素に適用し、D(z) のマップ、膜厚のマップ、画素ごとの残差の
  マップを返す。画素ごとに正則化の重みと膜厚の初期値を渡せるが、元素の役割（JVST/SIA の
  積層の 5 元素 Si/Ti/Al/C/O）、深さ格子、非双極子補正は既定値のままである。他の設定では
  `process_single_pixel` を回す短いループを書く。
- `deppro.haxpes.config.HAXPESSolverConfig` は、`HAXPESPipeline` が 1 組の平均データに
  適用する前処理とソルバの選択（角度のビン分け、バックグラウンドの方法、断面積の表、
  非双極子補正、正則化）をまとめる。`run_depth_profiling` はこのうちソルバの項目だけを読み、
  `run_3d_map` はこれを読まない。
- `deppro.haxpes.bootstrap` はフレームを平均して画素を作り（`build_frame_pixels`、
  `average_pixels`）、フレームのブートストラップ標本を解く（`frame_bootstrap`、
  `block_bootstrap`）。`deppro.haxpes.denoiser.HaxpesDenoiser` は学習済みの `dnndenoiser` の
  ネットワークを逆解析の前にフレームへ適用する（`[nn]` と重みが要る）。
- `python -m deppro.streaming_pipeline` は H5 のマップをチャンクごとに `SemiFreeSolver`
  （または学習済みネットワーク）で解き、画素ごとの D(z) と膜厚を保存する
  （[`docs/data_format.md`](../data_format.md)）。そこでの界面幅は `--abruptness-init` の値に
  固定され、当てはめない。`deppro.solvers_mlx` の MLX のクラスは、画素のバッチを
  Apple silicon の GPU で解く。
- `deppro.depth_profile.wasserstein_distance_1d` と `deppro.app.contract.metrics`
  （`evaluate`、`thickness_half_crossing`）は、再構成を参照分布と比べて採点する。

測定データについて、前処理やソルバの設定を掃引して結果を表にする既製の実行器は無い。
条件のループは、これらの関数を呼ぶ短いスクリプトになる。模擬した積層については、
ソースのチェックアウト（インストールされるパッケージには含まれない）に
`python -m deppro.benchmarks.run_solver_landscape` がある。これはノイズ・膜厚・界面幅・
角度数を変えて 3 つのソルバを解き比べる。正則化は 1 回の実行につき 1 つである。画素どうしは横方向に結合しない。例外は、近傍画素の解から
パスの間に作り直す任意の事前分布だけである（ストリーミングパイプラインの第 2 段
`--mem-loops` と `IterativeMEMSolver.solve_batch_with_neighbor`）。したがってマップは
深さ分布の集まりであり、三次元のトモグラフィー再構成ではない。

### これまでにしたこと、まだしていないこと

- **2 本の論文で実施済み**: [JVST A](https://doi.org/10.1116/6.0005585) では、
  C/Al₂O₃/TiO₂/Si 積層の 1 か所で記録した数千枚の単一フレームを `SemiFreeSolver`
  （L1、界面幅を 5 Å FWHM に固定）で 1 枚ずつ逆解析し、フレーム平均・bin-pool
  リサンプリング・自己教師ありデノイズを二つの露光時間で比べた。
  [SIA](https://doi.org/10.1002/sia.70123) では、同じフレームごとの逆解析を使い、ある露光時間で
  学習したデノイズのネットワークを別の露光時間のデータに適用する試験を、三つの露光時間で、
  フレーム平均の系列と比べて行った。各再構成は、自分のデータセットの擬似真値（そのデータセットの
  全フレームで平均したスペクトルの再構成）と比べ、深さ RMSE・膜厚のずれ・W1 で採点した。
  これらの論文の画素の格子はフレームの集まりで、横方向のマップではない。独立な構造測定を
  真値には使っていない。JVST A で確かめたのは、再構成が同じ試料の RBS 参照と同じ層の順序を
  示し、全膜厚が触針式段差計の値と矛盾しないことだけである。
- **今のコードで可能だが、まだ発表していない**: 横方向に分解したマップでの同じ画素ごとの
  比較、実測データでの構造モデルと正則化の比較。
- **これから検証すべきこと**: 独立な深さ測定（同じ試料の RBS や SIMS など）との定量的な
  比較、2 本の論文の結論が他の試料・装置・ノイズ領域にも当てはまるか。

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
clone した作業ツリーで引数なしの `pytest` を走らせる（設定の正本は `pyproject.toml`）。

## 使ってみる

3 層膜の合成データを作り、FreeSolver で深さプロファイルを再構成する例:

```bash
pip install -e '.[fit]'
python -m deppro.examples.freesolver_recommended
```

`TransferMatrix` とソルバの引数と戻り値は [`docs/api_reference.md`](../api_reference.md)、
H5 のデータ形式は [`docs/data_format.md`](../data_format.md) を参照。外部データの場所は
`DEPPRO_*` 環境変数で渡す（一覧は [`docs/environment_variables.md`](../environment_variables.md)）。

## JVST / SIA 論文の再現

[`docs/REPRODUCTION_JVST_SIA.md`](../REPRODUCTION_JVST_SIA.md) は三つを区別している。
保存済みの結果から指標を再計算すること、同じ入力と重みでデノイザーと逆解析をやり直すこと、
学習をやり直すこと。実施したのは一つ目だけで、それも著者の開発機の上だけである。別の計算機での
再現はまだ無い。`python -m deppro.haxpes.deposit_replay` は、Zenodo の寄託物に保存された
深さ分布から指標を再計算する（寄託物を展開した場所を `--sia-dir` / `--jvst-h5` で渡す）。
デノイザーも逆解析もやり直さない。SIA では、表 1 の値、図 4・図 5 について本文に書かれた深さ RMSE と W1、シード試験の値を
論文と照合する（すべて一致。ほかの補足図や当てはめは再計算していない）。
JVST A では、論文ではなく寄託物の指標ファイルを寄託された配列と照合する。24 s のデータセットで
保存された W1 と深さ RMSE の値は再現しない。公開物から何を再計算できるか、何に著者へ依頼して
受け取る材料が要るかは、同じ文書に図ごとにまとめてある。

## 関連リポジトリ

| リポジトリ | 役割 |
|------|------|
| **depthprofiler**（このリポジトリ） | 深さ解析（古典 + NN + MLX の逆ソルバ） |
| [**dnndenoiser**](https://github.com/stoyoda0012-cyber/dnndenoiser) | 深層学習による XPS スペクトルのデノイザー（自己教師ありの学習を含む） |
| [**toyomacro**](https://github.com/stoyoda0012-cyber/toyomacro) | Voigt ピークフィットと、化合物・結合エネルギー・断面積の表 |

## 状態

版は 0.2.0。1.0 までは API が変わることがある。質問や不具合の報告は
[issue トラッカー](https://github.com/stoyoda0012-cyber/depthprofiler/issues)へ。
対応の目安は [`SUPPORT.md`](../../SUPPORT.md)、変更の提案とテストの走らせ方は
[`CONTRIBUTING.md`](../../CONTRIBUTING.md)（いずれも英語）にある。

## 既知の制限

- `SemiFreeSolverConfig(gradient_mode=True)` はこの版では使えない（`NotImplementedError`）。
- MLX のソルバは、あらかじめ定めた合成データの条件で CPU の参照解と比べて検査している。
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
- 順モデルは電子の直線軌道近似による指数減衰を使い、弾性散乱の項を持たない。
- `BatchProcessor` と `python -m deppro.cli_depth batch` / `monte-carlo` は削除したソルバの
  ためのもので、この版では使えない。多数の画素には上に挙げた関数を使う。

## 構成

| 場所 | 内容 |
|------|------|
| `src/deppro/` | 転送行列・逆ソルバ（LLS / MEM / L1 / L2）、NN 逆ソルバ (`inference/`)、MLX ソルバ (`solvers_mlx/`)、HAXPES の前処理 (`haxpes/`)、シミュレーション (`simulate/`)、例 (`examples/`) |
| `src/deppro/tests/`、`src/deppro/app/tests/` | 回帰スイートと受入ゲート（CI の構成は `.github/workflows/ci.yml`） |
| `docs/` | 環境変数、H5 のデータ形式、ソルバの API、JVST/SIA 論文の再現 |
| `paper/` | JOSS 向けのソフトウェア論文の原稿（未投稿） |

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
