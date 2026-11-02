# 環境変数（外部パス）

外部データ・外部ツールの場所は `DEPPRO_*` 環境変数で渡す。
どの変数も**未設定のままで `import deppro` は通る**。未設定時の扱いは変数ごとに違い、
下表の「未設定時」列のとおり。

**空文字列（`export DEPPRO_X=`）は未設定と同じに扱う。** 値をパスとして使う読み場所は
すべてこの規則に従う。空白だけの値は特別扱いしない。例外は、値をそのまま結果に
記録することが目的の一部の診断スクリプト（`DEPPRO_COMPOUND_TABLE_PATH` の生の値を
書くので、空文字列なら `""` と記録される）。

新しい読み場所は `deppro.paths.env_value()`（値が要るとき）か
`deppro.paths.env_path_or_placeholder()`（`-not-set` 表記が要るとき）を使い、
`os.environ.get(name, default)` / `os.getenv(name, default)` / `os.environ[name]` で
直接読まない。これらは空文字列をそのまま返し、既定値を素通りして `Path("")`
（= カレントディレクトリ）になる。

`tests/unit/test_env_paths.py` は `src/deppro/` を AST で走査し、次の書き方を検出して落とす
（生の値を記録することが目的の読み場所だけを、ファイルごとの出現数つきの許可リスト
`_ALLOWED_RAW_READERS` で除く）。

- 既定値つきの読み取り: `os.environ.get(name, d)`（`default=` も）、`os.getenv(name, d)`
  （`key=` / `default=` のキーワード指定も）
- `os.environ.setdefault(name, d)`（結果を使うかどうかに関係なく。空文字列が入っていると
  既定値が入らないので、既定値を注入する用途にも使えない）
- 値を使う `os.environ.pop(name, ...)`（結果を捨てる文としての呼び出しは、後片付けと
  みなして対象外）
- 添字の読み取り `os.environ[name]`
- 存在判定 `name in os.environ` / `name not in os.environ`
- 既定値なしで読んだ値の None 判定: `os.environ.get(name) is None` と直接比べる形と、
  **同じ関数（またはモジュール）の中で**変数に入れてから `is None` / `is not None` で
  比べる形（`=`、型注釈つきの `=`、`:=`）

`os.environ` には、`environ`（`from os import environ`）、同じファイル内の別名
（`env = os.environ`、`from os import environ as E`、別名の別名も）、`os.environ.copy()`、
`dict(os.environ)` も含む。`os.getenv` には `getenv` と、同じファイル内の別名も含む。
`name` は `"DEPPRO_…"` の文字列リテラル、f-string の先頭、または同じファイル内で
`NAME = "DEPPRO_…"`（型注釈つきも）と代入された名前。

**捕まらない主なもの（網羅ではない）:**

- 変数名を実行時に組み立てる書き方（文字列の連結、`%` や `.format`、`getattr(os, "getenv")`、
  `NAME = f"DEPPRO_{n}"` という定数）
- 別のモジュールで定義した定数や別名、属性に入れた別名（`self.env = os.environ`）
- `{**os.environ}` や `dict(os.environ, A=1)` を経由した読み取り
- 読んだ値を別の関数（クロージャを含む）に渡してから、そちらで None 判定する形
- `== None` での判定、タプルのアンパックで受けた値の None 判定、比較演算子を連ねた形
  （`a is None is b`）
- 関数や lambda の既定引数で読んで None 判定する形、`for` / `with` の対象で受けた値の
  None 判定、モジュールで読んで関数の中で None 判定する形
- 属性に代入してから None 判定する形（`self.v = os.environ.get(...)` のあと `self.v is None`）

これらについては、規則を守る責任は書き手にある。

**誤検出について:** 別名と定数は、スコープも代入の順序も見ずに名前だけで追う。None 判定の
変数は同じ関数（またはモジュール）の中に限って追うが、代入の順序は見ない。そのため、同じ
名前を別のものに使うと、読み取りでなくても検出されることがある（見逃す側ではなく、赤に
なる側に倒れる）。

また、クラス本体と内包表記は、外側の関数（またはモジュール）と同じ範囲として扱う。
そのため、Python では別の変数でも検出されることがある。例えば、クラス本体で読んだ値を
モジュールや別のクラス本体で None 判定する形や、内包表記の対象変数が外側で読んだ変数を
隠す形（`v = os.getenv(...)` のあと `[v is None for v in ...]`）。

既定値に `None` を明示した形（`os.environ.get(name, None)`、`os.getenv(name, None)`、
`default=None`）も、既定値つきの読み取りとして検出される。

誤検出したら、名前を変えるか `deppro.paths` を通す書き方に直す。

許可リストとの照合は、検出した位置（読み取り。None 判定では比較の位置）を行と列で数える。
同じ行に 2 つ書いても、1 つの値を 2 回 None 判定しても、それぞれ別の件になる。

「`-not-set` 表記」は、パスが `<変数名>-not-set/...` という文字列になり、読み込み時の
エラーやログに変数名がそのまま出る方式を指す（`env_path_or_placeholder()`）。

値に `~` を使えるのは、読む側で `expanduser` するものだけ（→ 末尾の「`~` の展開」）。

## データ・ツールの置き場

この文書の表は、公開モジュールが読む場所だけを挙げる。公開版に含めないモジュールだけが読む変数と、
下の変数の非公開の読み場所は `environment_variables_private.md` にある（private リポジトリのみ）。

| 変数 | 指すもの | 読む場所 | 未設定時 |
|---|---|---|---|
| `DEPPRO_FIX_IBW_DIR` | fix モードの IBW フレームのディレクトリ | `deppro.haxpes.config.FIX_IBW_DIR`（`HAXPESPipeline.run_fix_ibw` の例など） | `-not-set` 表記 |
| `DEPPRO_FIX_PXT_DIR` | fix モードの pre-binned PXT ファイルのディレクトリ | `deppro.haxpes.config.FIX_PXT_DIR` | `-not-set` 表記 |
| `DEPPRO_MAP_H5` | フィット済みマップの HDF5 ファイル | `deppro.haxpes.config.MAP_H5` | `-not-set` 表記 |
| `DEPPRO_SIMULATION_ROOT` | MATLAB 生成のベンチマーク H5 のルート（`Standard/`, `Checkered/` の親） | `deppro.paths.simulation_root()` とその呼び出し元、`deppro.paths.legacy_simulation_root()`（`benchmarks/paths.py` の `BENCHMARK_ROOT` / `GT_*`） | `simulation_root()` は `-not-set` 表記。`legacy_simulation_root()` は以前からのホーム相対の置き場（`~/Documents/MATLAB/Simulation/DepthProfiler`）。`Simulation/Checkered` 配下のファイル（`GT_500x500`）は、このルートの 1 つ上の階層を基準にする |
| `DEPPRO_TESTDATA_ROOT` | MATLAB DepthProfiler のテストデータ置き場（`map_*.h5` と MATLAB の結果 `YYMMDD*.h5` を 1 つずつ直下に持つ） | `tools/roundtrip_compare.py` | `-not-set` 表記 |
| `DEPPRO_DEPTHPROFILER_CLI` | MATLAB 生成 CLI（`_cli/dp`） | `deppro.app.matlab_generation` | `-not-set` 表記 |
| `DEPPRO_DATASET_REGISTRY` | ベンチマーク対象データセットの台帳 JSON（`schema: deppro-dataset-registry/1`） | `deppro.app.contract.registry.default_registry()`（`entry` / `benchmark_targets` / `excluded` / `explain` に台帳を渡さなかったとき） | 空の台帳（どのデータセットも「台帳にない」）。設定したパスがファイルでなければ、台帳を使う段で変数名入りの `FileNotFoundError`。台帳を明示的に渡す呼び出し元（`load_registry(path)`）はこの変数を読まない |

## 参照テーブル

| 変数 | 指すもの | 読む場所 | 未設定時 |
|---|---|---|---|
| `DEPPRO_COMPOUND_TABLE_PATH` | 化合物テーブル CSV | `deppro.imfp.CompoundDatabase`、`deppro.producer.materials` | `imfp`: toyomacro 同梱テーブルを使う（toyomacro が無ければ、変数名と install 方法を示す `FileNotFoundError`）。`producer.materials`: `-not-set` 表記 |
| `DEPPRO_BINDING_ENERGY_TABLE_PATH` | 元素の結合エネルギー CSV | `deppro.imfp` | toyomacro 同梱テーブルを使う（toyomacro が無ければ、変数名と install 方法を示す `FileNotFoundError`） |
| `DEPPRO_COMPOUND_BE_TABLE_PATH` | 化合物の結合エネルギー CSV（NIST） | `deppro.imfp`、`deppro.producer.binding` | `imfp`: 既定の表は無い。`CompoundBindingEnergyDatabase` の生成時に、変数名を示す `FileNotFoundError`（同梱する供給源が無く、リポジトリの隣も探さない）。`producer.binding`: 未指定として扱う |
| `DEPPRO_ELEMENT_BE_TABLE_PATH` | 元素の結合エネルギー CSV | `deppro.producer.binding` | 未指定として扱う |
| `DEPPRO_YEH_LINDAU_TABLE_PATH` | Yeh–Lindau の光電離断面積 CSV（`CrossSectionTable_Yeh=Lindau.csv`） | `deppro.haxpes.cross_section_extrap`（`apply_cross_section_correction(..., source='yeh_lindau_extrap')` の Z > 12 の外挿） | 設定すればその CSV を読む。未設定なら **toyomacro の同梱 Yeh–Lindau 表**（同じ CSV から生成）。toyomacro も無ければ、変数名と toyomacro を示す `FileNotFoundError`（リポジトリの隣は探さない）。固定した本番ケースの外挿値は両者で一致する（同テストファイル） |
| `DEPPRO_CROSS_SECTION_DIR` | Scofield (1973) と Yeh–Lindau (1985) の光電離断面積ワークブック（`excel_scofield_1973_pics.xlsx`、`excel_yeh_lindau_1985_pics-4.xlsx`）を置くディレクトリ | `deppro.haxpes.cross_section_loader`（`apply_cross_section_correction` が使う: `source='scofield'` / `'yeh_lindau'` では全元素、既定の `'yeh_lindau_extrap'` でも Z ≤ 12 の元素でワークブックを読む。`deppro.cli_depth` も使う） | 設定すればその Excel ワークブックを読む。未設定なら **toyomacro の同梱 Scofield / Yeh–Lindau 表**（toyomacro が表の完全性と権利レビューを担う: `docs/DATA_SOURCES.md`。内挿は deppro 側のまま）。toyomacro も無ければ、変数名と toyomacro を示す `FileNotFoundError`（リポジトリの隣は探さない）。**射程:** 固定した本番ケース（`tests/unit/test_cross_section_sources.py`）では両者の値は一致する。差が出るのは Scofield の 30 keV 超（toyomacro の表はそこまで）、Yeh–Lindau の一部の非本番セル（ワークブックと toyomacro で転記が異なるものは、一次ソースで照合した toyomacro の表を正とする）、`energy_range` の報告値。しきい値以下の照会は両経路とも NaN |

## 出力先

| 変数 | 指すもの | 読む場所 | 未設定時 |
|---|---|---|---|
| `DEPPRO_OUTPUTS` | 図・生成物の置き場 | `deppro.paths.outputs_dir()` / `ensure_outputs_dir()` | チェックアウトならリポジトリ直下の `outputs/`（ignore 済）。インストール済みパッケージからなら**カレントディレクトリ**の `outputs/`（一度だけ警告） |

## `~` の展開

- **展開する**: `DEPPRO_FIX_IBW_DIR` / `DEPPRO_FIX_PXT_DIR` / `DEPPRO_MAP_H5` / `DEPPRO_SIMULATION_ROOT` / `DEPPRO_TESTDATA_ROOT` /
  `DEPPRO_DEPTHPROFILER_CLI` / `DEPPRO_DATASET_REGISTRY` / `DEPPRO_OUTPUTS`、および `deppro.producer` が読む `DEPPRO_COMPOUND_TABLE_PATH` /
  `DEPPRO_COMPOUND_BE_TABLE_PATH` / `DEPPRO_ELEMENT_BE_TABLE_PATH`、`DEPPRO_YEH_LINDAU_TABLE_PATH`、
  `DEPPRO_CROSS_SECTION_DIR`
- **展開しない**: `deppro.imfp` が読む `DEPPRO_COMPOUND_TABLE_PATH` /
  `DEPPRO_BINDING_ENERGY_TABLE_PATH` / `DEPPRO_COMPOUND_BE_TABLE_PATH`。
  これらには絶対パスを渡す

同じ変数でも読む場所によって扱いが違う（`DEPPRO_COMPOUND_TABLE_PATH` と
`DEPPRO_COMPOUND_BE_TABLE_PATH`）ので、両方に効かせたいときは絶対パスを使う。

新しいコードの出力先は `deppro.paths.outputs_dir()` を使う。新しい外部パスが要るときは
ここに行を足す。
