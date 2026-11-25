# HAXPES-Lab の角度分解データから深さ分布を出す

HAXPES-Lab（SES、Angular56、Fixed モード）で測った `.pxt` のフォルダを渡して、組成の深さ分布 D(z) と膜厚を得るまでの手順です。Claude Code と一緒に進める前提で書いてあります。

確認できている範囲は次のとおりです。

- 試料は C / Al₂O₃ / TiO₂ / Si（JVST A・SIA 論文の試料）。2025-08 測定の HAX-25-005（`20250806_Fixed_Narrow_0001`〜`20250807_Fixed_Narrow_0006.pxt`）で、新しい仮想環境に入れた状態から最後まで通ります。
- ほかの試料・ほかの内殻では、まだ誰も通していません。4 章の手順で設定はできますが、結果は豊田と一緒に見てください。

先に一番大事な注意を書きます。**このソルバーの膜厚は、初期値と深さの刻みに強く依存します。** 上のデータでも、設定によって 202 Å、250 Å、311 Å が出ます（5 章）。出てきた膜厚の数字を、そのまま試料の膜厚として報告しないでください。

## 1. 準備

Python 3.11 以上。

```bash
git clone https://github.com/stoyoda0012-cyber/depthprofiler.git
```

```bash
cd depthprofiler && python -m venv .venv && source .venv/bin/activate
```

```bash
pip install -e '.[fit]'
```

`[fit]` は toyomacro（PXT の読み込みと、化合物・断面積の表）を入れます。これがないと動きません。

動作確認（合成データ、数秒）:

```bash
python -m deppro.examples.freesolver_recommended
```

元素ごとの表が出れば入っています。

インストールで `llvmlite` のビルドが `cmake` がないと言って失敗したら、Python が Intel 版（x86_64）です。Apple シリコンの Mac では arm64 版の Python（miniforge など）で仮想環境を作り直してください。

## 2. データを置く

1 つの測定条件につき 1 つのフォルダに `.pxt` を入れます。フォルダの中の、名前に `Fixed` を含むファイルが全部足し合わされて、1 本の深さ分布になります（`--filter` で変えられます）。試料や条件の違うファイルを同じフォルダに混ぜないでください。

まず中身を見ます。

```bash
python -m deppro.examples.haxpes_lab_depth_profile データのフォルダ --inspect
```

ファイルごと、領域ごとに、領域名・配列の形・レンズモード・Step Time が並びます。見るところは 3 つです。

- 領域名（`Si 1s` など）。4 章の試料ファイルに書く名前と一字一句同じである必要があります。
- 形が `(エネルギー, 12, …)` で、2 番目が角度スライスの数になっていること。
- `(no header)` と出るファイルがないこと。途中で止めた測定のファイルはこうなり、読み飛ばされます（HAX-25-004 の `20250801_Fixed_Narrow_0001.pxt` がこの例で、100 フレームの予定が 43 フレーム分しか入っていません）。

`.ibw` ではなく `.pxt` を使ってください。パッケージの `HAXPESPipeline.run_fix_ibw` は検出器 1001 チャンネルの古い形の IBW 用で、いまの 12 スライスの IBW では止まります。

## 3. 論文と同じ試料で通す

HAX-25-005 の `.pxt` 6 個を 1 つのフォルダに置いて:

```bash
python -m deppro.examples.haxpes_lab_depth_profile データのフォルダ --polar 30
```

新しい環境では次の結果になります（2026-11-25 に確認）。

```text
film thickness 202.4 Å (start 240, bounds 10-500), interface width fixed at 5 Å
rRMSE (intensities normalised to Si) 152.0%

Dependence on the starting thickness (comparison only):
  start [Å]  thickness [Å]   rRMSE
         30           39.5  162.2%
         60          101.9  160.9%
        100          100.0  160.9%
        150          158.1  158.9%
        200          200.0  152.0%
        240          202.4  152.0%   <- written out
        250          250.0  113.8%
        300          262.1  113.8%
        400          400.0  364.7%
```

深さ分布は、表面から C → Al₂O₃ → TiO₂ → Si の順に並びます。同じ数字が出れば、環境とデータの扱いは合っています。

出力フォルダ（既定は `depth_profile_out/`、`--out` で変更）には 4 つのファイルができます。

| ファイル | 中身 |
|---|---|
| `depth_profile.png` | 左が深さ分布、右が角度依存の実測（点）とフィット（線） |
| `depth_profile.csv` | 深さごとの各元素の原子分率 |
| `angle_fit.csv` | 使った角度ごとの、基板で規格化した実測とフィット |
| `summary.json` | 使ったファイル、角度、断面積、設定、膜厚、残差、初期値ごとの結果 |

## 4. 自分の試料に変える

### 4.1 試料ファイルを書く

`src/deppro/examples/haxpes_lab_sample.yaml` をデータのそばにコピーして書き換えます。測った領域 1 つにつき 1 行です。

```yaml
name: HfO2_Si
elements:
  - {region: "Si 1s", symbol: Si, orbital: 1s, compound: Si,   binding_energy: 1839.0, oxidation_state: 0,  role: substrate}
  - {region: "Hf 3d", symbol: Hf, orbital: 3d, compound: HfO2, binding_energy: ここに値, oxidation_state: 4,  role: free, sigma: ここに値}
  - {region: "O 1s",  symbol: O,  orbital: 1s, compound: HfO2, binding_energy: 531.0,  oxidation_state: -2, role: linked}
```

（この HfO₂ の例は書き方を示すだけで、この設定で解いたことはありません。）

| 項目 | 書くもの |
|---|---|
| `region` | SES の領域名。`--inspect` に出たとおり |
| `compound` | その元素が入っている化合物。非弾性平均自由行程の計算に使う。化合物表にある名前だけ使える（下のコマンドで確認） |
| `oxidation_state` | その化合物での形式電荷。基板と単体は 0 |
| `role` | `substrate`＝埋もれた界面より下の基板。`free`＝膜の元素で、深さごとに濃度を解く（陽イオン、炭素など）。`linked`＝電荷のつり合いで決まる元素（ふつう O） |
| `sigma` | 光イオン化断面積。1s 準位で Ti までなら省略できる（下記） |

化合物名の確認:

```bash
python -c "from deppro.imfp import CompoundDatabase as C; print(C().list_compounds())"
```

このソルバーが仮定している構造は「基板の上に膜があり、埋もれた界面は 1 つ、膜の中の組成は深さごとに自由」です。酸化物の膜でしか通していません。金属層（Au など）を含む膜、`linked` の元素がない膜は試していません。

### 4.2 断面積

`sigma` を書かなければ、パッケージの Yeh–Lindau 外挿の値が使われます。これが使えるのは **1s 準位で Ti まで**です。それ以外（Hf 3d、Zr 2p、Au など）は `sigma` を自分で書かないと止まります。

- 値は Trzhaskovskaya–Yarzhemsky の表から、励起エネルギー（Ga Kα 9251.74 eV）に内挿して取ります。表は豊田が持っています。記憶や概算で書かないでください（Claude Code にも書かせないでください）。桁を間違えた前例があります。
- 単位は Mb に揃えます。全元素に同じ倍率を掛けただけでも結果は変わります（全部を 1000 倍したら、同じデータで膜厚が 248 Å から 258 Å に動きました）。
- スピン軌道分裂した準位は、積分した窓に入っている成分の分だけ足します（窓に 3d₅/₂ しかないなら 3d₅/₂ の値）。

ピーク面積は、領域の窓全体を直線バックグラウンドで引いて足したものです。窓の中に別の元素のピークが入っている場合（HfO₂ の O 1s の窓に入る Hf 4s など）は、この手順では分けられません。

### 4.3 角度

```text
放出角 = (スライスの検出器角を逆順にしたもの) + polar
```

`--polar` は、アナライザーのレンズ軸から試料法線までの角度です。論文のデータは 30° です。SES のファイルにはマニピュレーターの角度を書く専用の欄がないので、測定ノートの値を使ってください。ヘッダの `Comments` は手入力で、前の測定から写したままのことがあります（HAX-25 のファイルは C/AlOx/TiOx/Si の測定なのに `Sample Name; Au_Si` のままです）。

polar は結果に効きます。同じデータで 26° と 34° にすると、膜厚が 30〜60 Å 変わりました。

使う角度は既定で 5〜55°（`--angle-min`、`--angle-max`）です。12 スライスのうち両端はこの範囲の外になり、8 枚が残ります。使える枚数が 4 未満だと止まります。

### 4.4 走らせる

```bash
python -m deppro.examples.haxpes_lab_depth_profile データのフォルダ --polar 30 --sample my_sample.yaml --thickness-init 100
```

`--thickness-init` には成膜の設計膜厚（Å）を入れます。膜が厚いときは `--thickness-max`（既定 500）と `--total-depth`（既定 600）も広げます。

## 5. 結果の読み方

### 5.1 膜厚は 1 つの数字として信じない

3 章の表を見てください。初期値 100 → 100.0、200 → 200.0、250 → 250.0 と、解が初期値からほとんど動いていません。深さの刻み（既定は 30 Å）の格子に解が引っかかるためです。スクリプトが初期値を変えた結果を毎回並べるのは、これを見せるためです。

刻みを 10 Å にすると、同じデータでこうなります。

```bash
python -m deppro.examples.haxpes_lab_depth_profile データのフォルダ --polar 30 --n-depth 60
```

```text
film thickness 311.4 Å (start 240, …)      rRMSE 47.8%
  どの初期値（30〜400）からも 311.3〜311.8 Å、rRMSE 47.8〜47.9%
```

初期値によらなくなり、残差も下がります。ただし、これが正しい膜厚だと決まったわけではありません。論文の解析は 30 Å 刻み・初期値 240 Å で行われていて、どちらを標準にするかはまだ決めていません。

当面の手順:

1. 設計膜厚を `--thickness-init` に入れて、既定（30 Å 刻み）で解く。
2. 同じ設定に `--n-depth 60` を足して、もう一度解く。
3. 2 つの膜厚と、初期値の表を並べて見る。食い違うときは、膜厚の数字を外に出す前に豊田に見せる。

化合物表が違うだけでも結果が変わることも分かっています。著者の手元の表（Al₂O₃ の平均自由行程が 2.5% 長い）では、3 章と同じコマンドで 202 Å ではなく 248 Å になります。

### 5.2 図で見ること

- 右の図で、線（フィット）が点（実測）を追っているか。元素ごと全体に上下にずれているなら、断面積か、その領域のピーク面積（窓、バックグラウンド）を疑う。角度に対する傾きが合わないなら、polar か構造の仮定を疑う。
- 左の図で、層の順番が成膜の順番と合っているか。
- rRMSE は基板で規格化した強度の相対残差です。合否の線は決まっていません。同じデータで設定を変えたときの比較に使ってください（上の例では 152% → 114% → 48%）。

### 5.3 走らせる前後の確認

- `--inspect` に `(no header)` のファイルがない。
- 出力の `used (5.0-55.0): […]` の角度が、測定のジオメトリから見ておかしくない。
- `cross sections:` の行で、全元素の値と出どころが意図どおり。
- `WARNING: non-positive peak area` が出ていない。出たら、その領域の窓にピークが入っているか、スペクトルを見る。

強度は、露光時間で割ってあるものとして扱っています。HAX-25-005 では、Step Time が 2000〜8000 ms と 4 倍違う領域でも保存されている値の平均が同じ桁なので、SES が時間あたりの値で保存しているように見えます。確かめたのはこのデータだけです。測定モードが違うデータでは、`--inspect` の Step Time と強度を自分で見比べてください。

## 6. Claude Code と進めるとき

リポジトリの中で Claude Code を開き、最初にこう頼むと早いです。

```text
docs/ja/HAXPES_LAB_QUICKSTART.md と src/deppro/examples/haxpes_lab_depth_profile.py を読んで。
<データのフォルダ> を --inspect で調べて、領域名・形・読めないファイルを報告して。
そのあと、この試料（<層構造と設計膜厚、polar の角度>）用の試料 YAML の案を作って。
断面積は自分で数字を書かずに、必要な準位の一覧だけ出して。
```

Claude Code に守らせること:

- 断面積、束縛エネルギー、平均自由行程の数字を記憶から書かせない。表から引くか、空欄にして聞かせる。
- ファイルのヘッダの `Comments`、フォルダ名、試料名を根拠にさせない。人が書いたもので、間違いが残っています。角度と試料は測定ノートから渡す。
- 残差が大きいときに、角度の範囲や正則化（`--alpha`）を結果が良く見えるまで動かさせない。変えたら、変えた理由と、変える前後の両方の結果を残す。
- 膜厚は必ず初期値の表と `--n-depth 60` の結果を添えて報告させる。

## 7. うまくいかないとき

| 症状 | 見るところ |
|---|---|
| `regions [...] are not in the data` | 試料ファイルの `region` が `--inspect` の領域名と違う。空白も含めて合わせる |
| `No built-in cross section for …` | その準位は組み込みの断面積がない。`sigma` を書く（4.2） |
| `only N angle slices lie in …` | `--polar` が違うか、角度の範囲が狭い |
| `compound [...] is not in the compound table` | `compound` が化合物表にない。4.1 のコマンドで名前を確認 |
| 膜厚が上限・下限に張り付く | `--thickness-max`、`--total-depth` を広げる。初期値の表も見る |
| 層の順番が逆、基板が表面に出る | `role` の割り当てと polar を確認 |

豊田に相談するときは、出力フォルダの `summary.json` と `depth_profile.png`、`--inspect` の出力、試料 YAML を送ってください。これで同じ計算をやり直せます。
