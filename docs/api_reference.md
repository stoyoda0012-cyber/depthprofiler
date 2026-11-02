# deppro - ARXPS Depth Profile Analysis Package

MATLABで開発・検証されたARXPS深さプロファイル解析手法のPython再実装。
脱MATLAB・社会実装を目指すプロジェクト。

## Quick Start

```python
from deppro import (
    create_multilayer_profile,
    TransferMatrix,
    GenericLayerSolver, GenericLayerConfig, LayerDefinition,
    SolverConfig,
)

# 転送行列の計算
tm = TransferMatrix(
    depth=np.arange(0, 200, 1.0),
    step_size=1.0,
    angles=[0, 15, 30, 45, 60],
    photon_energy=1486.6,
)

# GenericLayerSolver（既知構造の膜厚推定）
config = GenericLayerConfig(
    layers=[
        LayerDefinition(name="SiO2", element_indices=[0, 1], stoichiometry={0: 1, 1: 2}),
    ],
    substrate=LayerDefinition(name="Si", element_indices=[2], stoichiometry={2: 1}),
    initial_thicknesses=[30.0],
    initial_widths=[5.0],
    linked_element_index=1,
)
solver = GenericLayerSolver(tm, SolverConfig(method='LLS'), config)
result = solver.solve(I_obs)
print(f"膜厚: {result.layer_thicknesses[0]:.2f} Å")
```

## Solver体系

```
┌─────────────────────────────────────────────────────────────┐
│  GenericLayerSolver                                         │
│    └─ 任意の層数、膜厚を最適化（組成比も可）、化学量論制約 │
│    └─ 用途: 既知構造の膜厚推定                             │
│                                                             │
│  FreeSolver                                                 │
│    └─ 全深さで濃度を自由最適化                             │
│    └─ MEM + Exponential Prior + 束縛条件を指定できる       │
│    └─ 用途: 構造未知の探索的解析                           │
│                                                             │
│  SemiFreeSolver                                             │
│    └─ 基板: erf関数固定、表面: 自由最適化                  │
│    └─ 用途: 中間的なアプローチ                             │
└─────────────────────────────────────────────────────────────┘
```

## Test-inferred capabilities (src/deppro/tests)

> **2026-07-29 の整理**: `test_*.py` を名乗るが test を持たない研究スクリプト 46 本を
> 役割別の置き場へ移した。**pytest が収集するのは本物のテストだけ**になっている。
> 下表の「テスト」と「スクリプト」は別物として読むこと。
>
> | 種別 | 置き場 | 走らせ方 |
> |---|---|---|
> | テスト | `src/deppro/tests/`, `src/deppro/app/tests/` | `pytest`（既定で slow を除外） |
> | ベンチマーク | `src/deppro/benchmarks/{accuracy,noise,optimization,regularization,robustness}/` | `python <file>` |
> | 診断スクリプト | `src/deppro/diagnostics/` | `python <file>` |

テスト群から読み取れる機能一覧（抜粋）:

- 界面検出の自動推定（勾配/50%交差/変曲点の合意推定）
    - tests/unit/test_interface_detection.py
- プロファイル比較メトリクス（Wasserstein-1、重心、モーメント、層間比較）
    - tests/unit/test_metrics.py
- パイプライン統合（構造推定 → ソルバ選択 → 膜厚推定）
    - tests/integration/test_pipeline.py
- FreeSolver（モデルフリー）での逆問題再構成と正則化比較
    - tests/solver/test_free_solver.py
- SemiFreeSolver（半拘束）での層厚推定・正則化比較
    - tests/solver/test_semifree_solver.py
- GenericLayerSolver（既知構造の膜厚推定、化学量論制約）
    - tests/solver/test_generic_layer_solver.py
- ノイズ・ロバスト性評価（Poisson/WLS）
    - tests/benchmark/noise/test_wls_poisson.py
    - tests/benchmark/robustness/test_narrow_angle.py
- 正則化パラメータ選択（L-curve / GCV）
    - tests/benchmark/regularization/test_alpha_selection.py
- MLX バッチソルバの性能回帰
    - tests/benchmark/test_mlx_perf_regression.py
- リポジトリ衛生（`test_` を名乗るならテストを持つ、の機械的担保）
    - tests/unit/test_repo_hygiene.py

スクリプト側（テストではない。旧パスは全て `.../test_<名前>.py`）:

- 均一混合 vs 層構造の識別 → `diagnostics/uniform_detection.py`
- 構造推定の挙動確認 → `diagnostics/structure_analyzer.py`
- 膜厚精度・5層スタック → `diagnostics/thickness_accuracy.py`, `diagnostics/five_layer_free.py`
- GenericLayer の膜構造 → `diagnostics/film_structure_solver.py`
- 失敗率 100 trials → `benchmarks/robustness/failure_rate_100trials.py`
- L1 など正則化比較 → `benchmarks/regularization/l1_regularization.py`
- Numba JIT・ソルバ比較 → `benchmarks/optimization/numba_jit.py`, `benchmarks/accuracy/solver_comparison.py`

注意: ベンチマーク・診断系は探索的な検証スクリプトで、実データ適用時は条件依存。

### ソルバー比較

| ソルバー | 自由度 | 最適化する量 | 用途 |
|---------|--------|-------------|------|
| `GenericLayerSolver` | 低 | 各層の膜厚（`optimize_stoichiometry=True` で組成比も） | 層構造が既知の膜 |
| `FreeSolver` | 高 | 全深さ・全元素の濃度 | 構造が未知の膜の探索 |
| `SemiFreeSolver` | 中 | free 元素の深さごとの濃度と、erf で表した基板界面の位置 | 基板界面を 1 つの erf で表せる膜（多層膜を含む） |

`SemiFreeSolver` は、公表した 2 本の論文（JVST A, doi:10.1116/6.0005585; SIA, doi:10.1002/sia.70123）で
C/Al₂O₃/TiO₂/Si の多層膜の解析に使った（L1 正則化、界面幅は 5 Å FWHM に固定）。

## 界面幅の扱い

角度分解の強度は界面の位置と幅の両方に同じ指数減衰を通して依存し、位置のほうがはるかに強く決まる。
このため、界面幅は推定せず固定して使う（`optimize_abruptness=False` など）。固定した幅は測定値ではなく
正則化の選択である。この扱いの説明と、JVST の解析での幅の変化（2.5–10 Å）に対する感度は
JVST A, doi:10.1116/6.0005585 の §III.A にある。

## 例と機能の説明

### FreeSolver の例（`src/deppro/examples/freesolver_recommended.py`）

合成の 3 層膜（Al₂O₃ 0–15 Å / SiO₂ 15–45 Å / Si、erf 界面 σ = 3 Å、深さ格子 0–98 Å・2 Å 刻み）、
角度 0/30/45/60°、Al Kα、強度に 2% の相対ガウスノイズ（seed 42。加えたあと 1e-6 で下限を切り、角度ごとに規格化）、
`SolverConfig.recommended(alpha=0.0003)`（MEM）、
Exponential Prior、束縛条件。RMSE は全深さ・全元素の濃度の RMSE。

| 化合物の表 | RMSE | χ² | Al³⁺ (0 Å) | Si⁰ (80 Å) | O²⁻ (0 Å) |
|---|---|---|---|---|---|
| 真値 | – | – | 0.400 | 1.000 | 0.600 |
| toyomacro v0.3.1 同梱（既定） | 0.1291 | 1.67e-03 | 0.385 | 0.890 | 0.602 |
| `DEPPRO_COMPOUND_TABLE_PATH` で外部の `CompoundTable.csv` | 0.1210 | 1.38e-03 | 0.387 | 0.912 | 0.602 |

測定: 2026-10-04、Python 3.12.11 / NumPy 2.3.3 / SciPy 1.16.3。結果は化合物の表の供給元で変わる。

以前ここにあった「ほぼ完璧な再構成（RMSE 0.056、2026-01-29）」は撤回した。このリポジトリの最初の版
（2026-07-16）の同じ例でも、同じ CSV の表で RMSE 0.1210 になり、いまの版と同じ値だった。つまり
2026-07-16 以降のコード変更は 0.056 との差の原因ではない。0.056 が書かれた 2026-01-29 時点のコードと
条件は確かめられず、このリポジトリの中には記録されていない。

**この例の設定**:
1. **Exponential Prior** (Livesey 1994): `d_penetration = λ_substrate`
2. **MEM正則化**: α = 0.0003（`SolverConfig.recommended()` の既定値。2% ノイズの合成例で選んだ値で、α を振った比較はいまの版では確かめていない。試料とノイズに応じて調整する）
3. **束縛条件**: `use_sum_constraint=True`, `use_charge_constraint=True`

### 可変組成膜

`LayerDefinition`に`stoichiometry_bounds`を指定することで、組成比を最適化パラメータとして扱える
（酸素欠損の SiOx、窒化した層など）。

**使用例**:
```python
# 酸素欠損SiOxの解析
layer = LayerDefinition(
    name="SiOx",
    element_indices=[0, 1],  # Si, O
    stoichiometry={0: 1, 1: 2.0},  # 初期値: SiO2
    stoichiometry_bounds={1: (1.5, 2.0)},  # O比を1.5-2.0で最適化
)

config = GenericLayerConfig(
    layers=[layer],
    substrate=...,
    optimize_stoichiometry=True,  # 組成最適化を有効化
)
```

**結果の確認**:
```python
result = solver.solve(I_obs)
print(result.layer_stoichiometries)  # 各層の最適化された組成比
print(result.assumptions)            # 使用した仮定（可視化）
print(result.summary())              # サマリー出力
```

**注意**: 組成比を可変にするときも、界面幅は固定して使う（上の「界面幅の扱い」）。

### 基板自動検出

`StructureAnalyzer.detect_substrate()` で角度分解データから基板を自動検出:

**検出アルゴリズム**:
1. **Slope法**: `d(ln I)/d(1/cosθ)` が最も負の元素 = 最も深い（基板）
2. **Ratio法**: `I(60°)/I(0°)` が最も小さい元素 = 最も深い
3. **化合物基板**: 類似した傾きを持つ元素をグループ化

**フォールバック条件**:
- 全元素の角度依存性が類似 → 均一混合の可能性
- 全元素が正の傾き → 基板信号なし（厚い膜の可能性）

## 制約体系

SemiFreeSolverは以下の制約を**代数的に**組み込み：

| 制約 | 数式 | 実装 |
|------|------|------|
| 濃度合計 | Σ D_e = 1 | 代数計算 |
| 電荷保存 | Σ (酸化数 × D_e) = 0 | `free_linked_const` |

## 使用例

### GenericLayerSolver（多層膜）

```python
config = GenericLayerConfig(
    layers=[
        LayerDefinition(name="C", element_indices=[0], stoichiometry={0: 1}),
        LayerDefinition(name="Al2O3", element_indices=[1, 4], stoichiometry={1: 2, 4: 3}),
        LayerDefinition(name="TiO2", element_indices=[2, 4], stoichiometry={2: 1, 4: 2}),
        LayerDefinition(name="SiO2", element_indices=[3, 4], stoichiometry={3: 1, 4: 2}),
    ],
    substrate=LayerDefinition(name="Si", element_indices=[5], stoichiometry={5: 1}),
    initial_thicknesses=[5.0, 15.0, 35.0, 30.0],
    initial_widths=[2.0, 3.0, 4.0, 5.0],
    linked_element_index=4,  # O
)
solver = GenericLayerSolver(tm, SolverConfig(method='LLS'), config)
```

### FreeSolver（`SolverConfig.recommended()` を使う例）

```python
from deppro import (
    FreeSolver, SolverConfig, create_exponential_prior, IMFP
)

# 基板のIMFP計算
imfp_si = IMFP.tpp2m(KE=1383, Nv=4, density=2.33, Aw=28.09, Eg=1.1)

# Exponential Prior (Livesey 1994)
D_prior = create_exponential_prior(
    depth, n_elements=4, substrate_idx=2,
    d_penetration=imfp_si  # 基板のIMFPを使用
)

# MEM + α=0.0003（既定値）+ 束縛条件
config = SolverConfig.recommended()

# ソルバー実行
solver = FreeSolver(tm, config, D_prior, elements=elements)
result = solver.solve(I_obs)
```

### SemiFreeSolver

#### 設定例（多層膜、L1 正則化）

下の値は書き方の例で、論文の解析で使った値ではない。JVST/SIA の解析の設定（α と目的関数の書き方）は
JVST A, doi:10.1116/6.0005585 の §III.A にある。

```python
from deppro import SemiFreeSolver, SemiFreeSolverConfig, SolverConfig

config = SolverConfig(
    method='L1',      # L1正則化（スパース化）
    alpha=1.0,        # 正則化の強さ（例の値）。データとノイズに応じて調整する
    max_iter=100,
)

semi_free_config = SemiFreeSolverConfig(
    thickness_init=70.0,
    abruptness_init=5.0,         # 界面幅（FWHM、Å）。既定では固定（optimize_abruptness=False）
    thickness_bounds=(10.0, 90.0),
    free_indices=[0, 1, 2, 3],   # C, Al, Ti, Si(ox)
    linked_indices=[4],          # O（電荷保存から決定）
    substrate_indices=[5],       # Si
)

solver = SemiFreeSolver(
    tm, config, semi_free_config,
    elements=elements,
    D_prior=None,     # 事前分布なし（下限 min_conc の一様値）。L1 項はほぼ sum|D| になる
    use_numba=True,   # Numba JIT を使う
)
result = solver.solve(I_obs)
```

#### 基本設定（単層膜向け）

```python
from deppro import SemiFreeSolver, SemiFreeSolverConfig

config = SemiFreeSolverConfig(
    thickness_init=30.0,
    abruptness_init=5.0,        # 界面幅は固定して使う（上の「界面幅の扱い」）
    optimize_abruptness=False,
    free_indices=[0],           # 自由最適化
    linked_indices=[1],         # 電荷保存から決定
    substrate_indices=[2],      # 基板（erf関数）
)
solver = SemiFreeSolver(tm, SolverConfig(method='LLS'), config)
```

### SemiFreeSolver + Stoichiometric Mode（NON/ONO解析）

NON（Si₃N₄/SiO₂/Si₃N₄/Si）やONO構造のように、O/Nが独立に変動する系では
`stoichiometric_mode='stoichiometric'` を使用。

```python
from deppro import SemiFreeSolver, SemiFreeSolverConfig

# NON/ONO解析用設定
# 元素インデックス: 0=Si-O, 1=O, 2=N, 3=Si-B
config = SemiFreeSolverConfig(
    free_indices=[1, 2],       # O, N を独立に最適化
    linked_indices=[],          # linkedなし (charge_balanceと異なる)
    substrate_indices=[3],      # Si-B

    stoichiometric_mode='stoichiometric',  # NON/ONO 用
    stoichiometric_rules={
        0: ([1, 2], [0.5, 0.75])  # Si-O = 0.5*D_O + 0.75*D_N (化学量論)
    },
    stoichiometric_constraint_coeffs=[1.5, 1.75],  # (3/2)*O + (7/4)*N = 1 - SiB

    thickness_init=25.0,
    optimize_abruptness=False,
)
solver = SemiFreeSolver(tm, SolverConfig(method='LLS'), config)
result = solver.solve(I_obs)
```

### Stoichiometric Mode 選択ガイド

| 系 | モード | 理由 |
|---|--------|------|
| SiO₂/Si | `charge_balance`（デフォルト） | O/Si比固定 |
| C/Al₂O₃/TiO₂/SiO₂/Si | `charge_balance` | 各層は固定組成 |
| NON（Si₃N₄/SiO₂/Si₃N₄） | `stoichiometric` | O/Nが異なる深さで独立変動 |
| ONO（SiO₂/Si₃N₄/SiO₂） | `stoichiometric` | O/Nが異なる深さで独立変動 |
| SiOx酸素欠損 | `stoichiometric` | O比が深さで変化 |
| SiGe勾配 | `stoichiometry_bounds`推奨 | 組成が連続的に変化 |

**自動検出の限界**:
- `StructureAnalyzer.use_stoichiometric_mode` は参考情報
- 多層酸化物（各層固定組成）で誤検出の可能性あり
- **系の特性を理解した上でユーザーが明示的に選択することを推奨**

**識別のポイント**:
1. **同一元素ペア（O/N）が異なる深さで変動** → `stoichiometric`
2. **異なる化合物が層ごとに存在（各層内は固定）** → `charge_balance`

**モードの選び方**:
1. 化学シフトで区別可能 → 別元素として扱う → GenericLayerSolver
2. 区別不可 + 層間で組成異なる → stoichiometric_mode

### ARXPSPipeline（自動解析）

```python
from deppro import ARXPSPipeline

pipeline = ARXPSPipeline()
result = pipeline.analyze(
    I_obs, tm, elements,
    element_labels=['Si_ox', 'O', 'Si_bulk'],
)

if result.success:
    print(f"膜厚: {result.layer_thicknesses}")
```

### 自動基板検出 + FreeSolver

```python
from deppro import (
    StructureAnalyzer, FreeSolver, SolverConfig,
    create_exponential_prior, IMFP
)

# Step 1: 基板自動検出
analyzer = StructureAnalyzer()
result = analyzer.analyze_with_substrate_detection(I_obs, angles, element_labels)

if result.substrate_detection.has_substrate:
    substrate_idx = result.substrate_detection.substrate_indices[0]
    print(f"検出された基板: {element_labels[substrate_idx]}")

    # Step 2: IMFPと Exponential Prior
    imfp_substrate = IMFP.tpp2m(KE=1383, Nv=4, density=2.33, Aw=28.09, Eg=1.1)
    D_prior = create_exponential_prior(
        depth, n_elements=4, substrate_idx=substrate_idx,
        d_penetration=imfp_substrate
    )

    # Step 3: FreeSolver実行
    solver = FreeSolver(tm, SolverConfig.recommended(), D_prior, elements)
    result = solver.solve(I_obs)
else:
    print(f"⚠️ {result.substrate_detection.fallback_reason}")
    print("substrate_idxを手動指定してください")
```

## Module Structure

```
deppro/
├── __init__.py              # パッケージエクスポート
├── depth_profile.py         # 深さプロファイル生成
├── transfer_matrix.py       # 転送行列、角度分解計算
├── imfp.py                  # TPP-2M IMFP計算
├── inverse_solver.py        # 逆問題ソルバー群
├── structure_analyzer.py    # 自動構造解析
├── pipeline.py              # 自動解析パイプライン
├── benchmark.py             # ベンチマークシステム
├── cli_depth.py             # CLIツール
└── tests/                   # テストスイート
```

## CLI Commands

```bash
# 逆問題を解く
python -m deppro.cli_depth solve --input intensity.csv --config model.yaml

# 設定ファイルテンプレート生成
python -m deppro.cli_depth init-config --model multilayer --output config.yaml

# ベンチマーク
python -m deppro.benchmark --quick
```

## 高角度の注意点

- 放出角が大きいほど基板からの信号は弱くなる。膜が IMFP に比べて厚いと、75° 付近より上では
  ほぼゼロになる（どの角度で弱くなるかは膜厚と IMFP で決まる）
- SemiFreeSolver は既定では基板の信号で規格化するので、基板の信号が弱い角度を含めると不安定になりうる

## 正則化手法

### FreeSolver の設定

```python
# プリセット
config = SolverConfig.recommended()  # MEM + α=0.0003（既定値）+ 束縛条件

# または手動設定
config = SolverConfig(
    method='MEM',
    alpha=0.0003,              # recommended() の既定値。試料とノイズに応じて調整する
    use_sum_constraint=True,   # Σconc = 1
    use_charge_constraint=True # 電荷保存
)
```

**注意**: LLSはαを使わないのでprior項が効かない。Exponential Priorを使うにはMEMが必要。

### 正則化手法比較

| 手法 | 特徴 |
|------|------|
| `LLS` | 正則化なし（α を使わない） |
| `L1` | スパースな解（sum\|D − D_prior\| の最小化） |
| `L2` | 滑らかな解 |
| `MEM` | エントロピー最大化。Exponential Prior と組み合わせる |

α の既定値は `SolverConfig` で 1.0、`SolverConfig.recommended()` で 0.0003。
どちらもすべての試料に合う値ではないので、データとノイズに応じて調整する。

## ノイズモデル

### Gaussian vs Poisson ノイズの比較

⚠️ **Poisson X%とGaussian X%は直接比較できない**

| ノイズ | 定義 | 低強度での挙動 |
|--------|------|----------------|
| Gaussian X% | 全強度で一定σ=X% | 一定 |
| Poisson X% | **最大強度I=1.0でのみ**σ=X% | I=0.1で σ≈3X% |

**RMS等価での比較**（比は強度の分布で変わる。下の値は目安）:
- Poisson 0.5% ≈ Gaussian 2%（RMSベース）
- Poisson 2% ≈ Gaussian 8%（RMSベース）

**推奨**: ベンチマーク結果の比較時はRMS等価ノイズを使用

## 物理的背景

### TPP-2M IMFP計算

```python
from deppro import IMFP

imfp = IMFP.tpp2m(
    KE=1383.1,      # 運動エネルギー (eV)
    Nv=16,          # 価電子数
    density=2.2,    # g/cm³
    Aw=60.08,       # 分子量 (g/mol)
    Eg=9.0,         # バンドギャップ (eV)
)
# → 38.6 Å（SiO2、TPP-2M の式の値）
```

### 参考材料パラメータ

| 化合物 | 密度 | 分子量 | 価電子数 | バンドギャップ |
|--------|------|--------|----------|---------------|
| SiO2 | 2.2 | 60.08 | 16 | 9.0 eV |
| Si | 2.33 | 28.09 | 4 | 1.1 eV |

## Dependencies

- numpy
- scipy
- matplotlib (for visualization)

## 関連ファイル

- `../tests/`: テストスイート
