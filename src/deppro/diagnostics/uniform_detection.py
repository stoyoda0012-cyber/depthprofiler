"""
均一混合検出の閾値検証
======================

問題:
- 完全均一でも、IMFPの違いにより「見かけの角度依存性」が生じる
- これをStructureAnalyzerが「層構造あり」と誤判定するリスク

検証:
1. 完全均一プロファイルでの角度依存性を計算
2. 実際の層構造との比較
3. Go/NoGo閾値の決定

Usage:
    python -m deppro.test_uniform_detection
"""

import numpy as np
from scipy.special import erf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from deppro.depth_profile import ElementConfig
from deppro.transfer_matrix import TransferMatrix
from deppro.structure_analyzer import StructureAnalyzer


def create_uniform_profile(
    composition: dict,
    elements: list,
    depth_max: float = 150.0,
    step_size: float = 1.0,
):
    """
    完全均一プロファイルを生成

    Parameters
    ----------
    composition : dict
        {element_index: fraction} の組成
    elements : list
        ElementConfig のリスト
    """
    depth = np.arange(0, depth_max, step_size)
    n_depth = len(depth)
    n_elements = len(elements)

    D = np.zeros((n_depth, n_elements))
    for idx, frac in composition.items():
        D[:, idx] = frac

    # 正規化
    D = D / D.sum(axis=1, keepdims=True)

    return depth, D, elements


def create_layered_profile(
    thickness: float,
    interface_width: float,
    depth_max: float = 150.0,
    step_size: float = 1.0,
):
    """SiO2/Si層構造プロファイルを生成"""
    depth = np.arange(0, depth_max, step_size)
    n_depth = len(depth)

    sigma = interface_width / (2 * np.sqrt(np.log(2)))
    D = np.zeros((n_depth, 3))

    sio2_profile = 1 - 0.5 * (1 + erf((depth - thickness) / sigma))
    D[:, 0] = (1/3) * sio2_profile  # Si_ox
    D[:, 1] = (2/3) * sio2_profile  # O
    D[:, 2] = 0.5 * (1 + erf((depth - thickness) / sigma))  # Si_bulk

    D = np.clip(D, 1e-6, None)
    D = D / D.sum(axis=1, keepdims=True)

    elements = [
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]

    return depth, D, elements


def compute_angle_resolved_intensity(depth, D, elements, angles):
    """角度分解強度を計算"""
    tm = TransferMatrix(
        depth=depth,
        step_size=depth[1] - depth[0],
        angles=angles,
        photon_energy=1486.6,
    )
    tm.compute(D, elements)
    I = tm.forward(D, normalize=True)
    return I, tm


def analyze_angle_dependence(I, element_labels=None):
    """角度依存性を詳細分析"""
    n_angles, n_elements = I.shape

    results = {}

    # 各元素の変動係数
    cv_per_element = np.std(I, axis=0) / np.maximum(np.mean(I, axis=0), 1e-10)
    results['cv_per_element'] = cv_per_element
    results['mean_cv'] = np.mean(cv_per_element)
    results['max_cv'] = np.max(cv_per_element)

    # 角度間の最大変化率
    max_change = np.max(I, axis=0) - np.min(I, axis=0)
    results['max_change'] = max_change
    results['mean_max_change'] = np.mean(max_change)

    # 低角度 vs 高角度の比率
    low_angle_avg = np.mean(I[:2], axis=0)  # 15°, 30°
    high_angle_avg = np.mean(I[-2:], axis=0)  # 75°, 85°
    ratio = high_angle_avg / np.maximum(low_angle_avg, 1e-10)
    results['angle_ratio'] = ratio
    results['max_angle_ratio'] = np.max(np.abs(ratio - 1))

    return results


def main():
    """メイン"""
    print("=" * 70)
    print("均一混合検出の閾値検証")
    print("=" * 70)

    angles = [15, 30, 45, 60, 75, 85]
    analyzer = StructureAnalyzer()

    # ケース1: 完全均一 SiO2 (Si:O = 1:2)
    print("\n>>> ケース1: 完全均一 SiO2")
    elements_sio2 = [
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
    ]
    depth, D_uniform, elements = create_uniform_profile(
        composition={0: 1/3, 1: 2/3},
        elements=elements_sio2,
    )
    I_uniform, _ = compute_angle_resolved_intensity(depth, D_uniform, elements, angles)

    print(f"組成: Si={1/3:.3f}, O={2/3:.3f} (均一)")
    print(f"\n角度分解強度:")
    print(f"{'角度':<6} {'Si':>10} {'O':>10}")
    for i, ang in enumerate(angles):
        print(f"{ang}°     {I_uniform[i, 0]:>10.4f} {I_uniform[i, 1]:>10.4f}")

    analysis1 = analyze_angle_dependence(I_uniform, ['Si', 'O'])
    print(f"\n統計:")
    print(f"  変動係数 (CV): Si={analysis1['cv_per_element'][0]:.4f}, O={analysis1['cv_per_element'][1]:.4f}")
    print(f"  平均CV: {analysis1['mean_cv']:.4f}")
    print(f"  最大変化: {analysis1['mean_max_change']:.4f}")
    print(f"  高角/低角比率: {analysis1['max_angle_ratio']:.4f}")

    # ケース2: 完全均一 SiO2 + Si基板（3元素）
    print("\n>>> ケース2: 完全均一 3元素系")
    elements_3 = [
        ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
        ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
        ElementConfig('Si', '2p', 'Si', oxidation_state=0),
    ]
    depth, D_uniform3, elements = create_uniform_profile(
        composition={0: 0.2, 1: 0.4, 2: 0.4},
        elements=elements_3,
    )
    I_uniform3, _ = compute_angle_resolved_intensity(depth, D_uniform3, elements, angles)

    print(f"組成: Si_ox=0.2, O=0.4, Si_bulk=0.4 (均一)")
    print(f"\n角度分解強度:")
    print(f"{'角度':<6} {'Si_ox':>10} {'O':>10} {'Si_bulk':>10}")
    for i, ang in enumerate(angles):
        print(f"{ang}°     {I_uniform3[i, 0]:>10.4f} {I_uniform3[i, 1]:>10.4f} {I_uniform3[i, 2]:>10.4f}")

    analysis2 = analyze_angle_dependence(I_uniform3)
    print(f"\n統計:")
    print(f"  各元素CV: {[f'{cv:.4f}' for cv in analysis2['cv_per_element']]}")
    print(f"  平均CV: {analysis2['mean_cv']:.4f}")
    print(f"  高角/低角比率: {analysis2['max_angle_ratio']:.4f}")

    # ケース3: SiO2/Si層構造（比較用）
    print("\n>>> ケース3: SiO2/Si層構造 (30Å)")
    depth, D_layer, elements = create_layered_profile(thickness=30.0, interface_width=5.0)
    I_layer, _ = compute_angle_resolved_intensity(depth, D_layer, elements, angles)

    print(f"\n角度分解強度:")
    print(f"{'角度':<6} {'Si_ox':>10} {'O':>10} {'Si_bulk':>10}")
    for i, ang in enumerate(angles):
        print(f"{ang}°     {I_layer[i, 0]:>10.4f} {I_layer[i, 1]:>10.4f} {I_layer[i, 2]:>10.4f}")

    analysis3 = analyze_angle_dependence(I_layer)
    print(f"\n統計:")
    print(f"  各元素CV: {[f'{cv:.4f}' for cv in analysis3['cv_per_element']]}")
    print(f"  平均CV: {analysis3['mean_cv']:.4f}")
    print(f"  高角/低角比率: {analysis3['max_angle_ratio']:.4f}")

    # ケース4: 薄い層構造（10Å）
    print("\n>>> ケース4: SiO2/Si層構造 (10Å)")
    depth, D_thin, elements = create_layered_profile(thickness=10.0, interface_width=5.0)
    I_thin, _ = compute_angle_resolved_intensity(depth, D_thin, elements, angles)

    analysis4 = analyze_angle_dependence(I_thin)
    print(f"  平均CV: {analysis4['mean_cv']:.4f}")
    print(f"  高角/低角比率: {analysis4['max_angle_ratio']:.4f}")

    # ケース5: 厚い層構造（80Å）
    print("\n>>> ケース5: SiO2/Si層構造 (80Å)")
    depth, D_thick, elements = create_layered_profile(thickness=80.0, interface_width=5.0)
    I_thick, _ = compute_angle_resolved_intensity(depth, D_thick, elements, angles)

    analysis5 = analyze_angle_dependence(I_thick)
    print(f"  平均CV: {analysis5['mean_cv']:.4f}")
    print(f"  高角/低角比率: {analysis5['max_angle_ratio']:.4f}")

    # サマリー
    print("\n" + "=" * 70)
    print("サマリー: 角度依存性の比較")
    print("=" * 70)
    print(f"\n{'ケース':<25} {'平均CV':>10} {'高角/低角比':>12} {'判定':<15}")
    print("-" * 65)

    cases = [
        ("均一 SiO2", analysis1, "均一"),
        ("均一 3元素", analysis2, "均一"),
        ("層構造 10Å", analysis4, "層構造"),
        ("層構造 30Å", analysis3, "層構造"),
        ("層構造 80Å", analysis5, "層構造"),
    ]

    for name, analysis, expected in cases:
        cv = analysis['mean_cv']
        ratio = analysis['max_angle_ratio']
        print(f"{name:<25} {cv:>10.4f} {ratio:>12.4f} {expected:<15}")

    # 閾値の提案
    print("\n" + "=" * 70)
    print("閾値の提案")
    print("=" * 70)

    # 均一ケースのCV最大値
    uniform_max_cv = max(analysis1['mean_cv'], analysis2['mean_cv'])
    # 層構造ケースのCV最小値
    layer_min_cv = min(analysis3['mean_cv'], analysis4['mean_cv'], analysis5['mean_cv'])

    print(f"\n均一ケースの最大CV: {uniform_max_cv:.4f}")
    print(f"層構造ケースの最小CV: {layer_min_cv:.4f}")
    print(f"分離度: {layer_min_cv / uniform_max_cv:.1f}倍")

    # 推奨閾値
    threshold = (uniform_max_cv + layer_min_cv) / 2
    print(f"\n推奨閾値 (中間値): {threshold:.4f}")
    print(f"安全マージン付き閾値: {uniform_max_cv * 1.5:.4f}")

    # StructureAnalyzerでの検証
    print("\n" + "=" * 70)
    print("StructureAnalyzerでの検証")
    print("=" * 70)

    print(f"\n{'ケース':<25} {'構造タイプ':<25} {'推奨ソルバー':<20} {'信頼度':>8}")
    print("-" * 80)

    test_cases = [
        ("均一 SiO2", I_uniform, ['Si', 'O']),
        ("均一 3元素", I_uniform3, ['Si_ox', 'O', 'Si_bulk']),
        ("層構造 10Å", I_thin, ['Si_ox', 'O', 'Si_bulk']),
        ("層構造 30Å", I_layer, ['Si_ox', 'O', 'Si_bulk']),
        ("層構造 80Å", I_thick, ['Si_ox', 'O', 'Si_bulk']),
    ]

    for name, I_obs, labels in test_cases:
        result = analyzer.analyze(I_obs, element_labels=labels, verbose=False)
        print(f"{name:<25} {result.structure_type.value:<25} "
              f"{result.solver_recommendation.value:<20} {result.confidence:>8.2f}")

    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 均一 vs 層構造の角度依存性
    ax1 = axes[0, 0]
    ax1.plot(angles, I_uniform[:, 0], 'o-', label='均一 Si', color='blue', alpha=0.5)
    ax1.plot(angles, I_uniform[:, 1], 's-', label='均一 O', color='red', alpha=0.5)
    ax1.plot(angles, I_layer[:, 0], 'o--', label='層構造 Si_ox', color='blue')
    ax1.plot(angles, I_layer[:, 1], 's--', label='層構造 O', color='red')
    ax1.plot(angles, I_layer[:, 2], '^--', label='層構造 Si_bulk', color='green')
    ax1.set_xlabel('Angle (°)')
    ax1.set_ylabel('Normalized Intensity')
    ax1.set_title('Angle Dependence: Uniform vs Layered')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. CVの比較
    ax2 = axes[0, 1]
    case_names = ['Uniform\n2-elem', 'Uniform\n3-elem', 'Layer\n10Å', 'Layer\n30Å', 'Layer\n80Å']
    cvs = [analysis1['mean_cv'], analysis2['mean_cv'],
           analysis4['mean_cv'], analysis3['mean_cv'], analysis5['mean_cv']]
    colors = ['gray', 'gray', 'blue', 'blue', 'blue']
    bars = ax2.bar(case_names, cvs, color=colors, alpha=0.7)
    ax2.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold={threshold:.3f}')
    ax2.set_ylabel('Mean CV (Coefficient of Variation)')
    ax2.set_title('Angle Dependence Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 膜厚 vs CV
    ax3 = axes[1, 0]
    thicknesses = [0, 0, 10, 30, 80]
    ax3.scatter(thicknesses[:2], cvs[:2], s=100, c='gray', label='Uniform', marker='o')
    ax3.scatter(thicknesses[2:], cvs[2:], s=100, c='blue', label='Layered', marker='s')
    ax3.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
    ax3.set_xlabel('Film Thickness (Å)')
    ax3.set_ylabel('Mean CV')
    ax3.set_title('Thickness vs Angle Dependence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 深さプロファイル比較
    ax4 = axes[1, 1]
    depth_plot = depth[:100]
    ax4.plot(depth_plot, D_uniform3[:100, 0], '-', label='Uniform Si_ox', color='blue', alpha=0.5)
    ax4.plot(depth_plot, D_layer[:100, 0], '--', label='Layer Si_ox (30Å)', color='blue')
    ax4.plot(depth_plot, D_layer[:100, 2], '--', label='Layer Si_bulk (30Å)', color='green')
    ax4.set_xlabel('Depth (Å)')
    ax4.set_ylabel('Concentration')
    ax4.set_title('Depth Profile Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path('/tmp/uniform_detection_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"\n可視化保存: {output_path}")


if __name__ == "__main__":
    main()
