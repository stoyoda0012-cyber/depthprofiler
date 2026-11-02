"""
ARXPS Depth Profile Benchmark System
=====================================

GenericLayerSolver, SemiFreeSolver, FreeSolverの性能を一貫して評価する
ベンチマークシステム。

機能:
- 標準テストケースの定義と実行
- 複数ソルバーの比較
- 精度・ロバスト性・計算時間の評価
- Markdown/HTML/JSONレポート生成
- CI/CD統合用の出力

Usage:
    # フルベンチマーク実行
    python -m deppro.benchmark

    # 特定ソルバーのみ
    python -m deppro.benchmark --solver GenericLayerSolver

    # レポート生成
    python -m deppro.benchmark --output report.md --format markdown
"""

import numpy as np
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from enum import Enum
import warnings

# Internal imports
from .depth_profile import create_multilayer_profile, DepthProfile
from .transfer_matrix import TransferMatrix
from .inverse_solver import (
    GenericLayerSolver,
    GenericLayerConfig,
    LayerDefinition,
    SemiFreeSolver,
    SemiFreeSolverConfig,
    FreeSolver,
    SolverConfig,
)


# =============================================================================
# Test Case Definitions
# =============================================================================

class TestCaseType(Enum):
    """テストケースタイプ"""
    SIO2_SI = "SiO2/Si"
    MULTILAYER_4 = "C/Al2O3/TiO2/SiO2/Si"
    THIN_FILM = "ThinFilm"
    THICK_FILM = "ThickFilm"


@dataclass
class TestCase:
    """ベンチマーク用テストケース"""
    name: str
    case_type: TestCaseType
    layer_names: List[str]
    true_thicknesses: List[float]
    interface_widths: List[float]
    description: str = ""
    tags: List[str] = field(default_factory=list)


# 標準テストケース定義
STANDARD_TEST_CASES = [
    # SiO2/Si 単層系
    TestCase(
        name="SiO2_10A",
        case_type=TestCaseType.SIO2_SI,
        layer_names=["SiO2"],
        true_thicknesses=[10.0],
        interface_widths=[5.0],
        description="薄いSiO2膜（10Å）",
        tags=["thin", "single-layer", "oxide"],
    ),
    TestCase(
        name="SiO2_30A",
        case_type=TestCaseType.SIO2_SI,
        layer_names=["SiO2"],
        true_thicknesses=[30.0],
        interface_widths=[5.0],
        description="標準SiO2膜（30Å）",
        tags=["standard", "single-layer", "oxide"],
    ),
    TestCase(
        name="SiO2_80A",
        case_type=TestCaseType.SIO2_SI,
        layer_names=["SiO2"],
        true_thicknesses=[80.0],
        interface_widths=[5.0],
        description="厚いSiO2膜（80Å）",
        tags=["thick", "single-layer", "oxide"],
    ),
    # 多層膜系
    TestCase(
        name="Multilayer_Standard",
        case_type=TestCaseType.MULTILAYER_4,
        layer_names=["C", "Al2O3", "TiO2", "SiO2"],
        true_thicknesses=[5.0, 15.0, 35.0, 30.0],
        interface_widths=[2.0, 3.0, 4.0, 5.0],
        description="標準4層膜（C/Al2O3/TiO2/SiO2/Si）",
        tags=["multilayer", "standard"],
    ),
    TestCase(
        name="Multilayer_ThinOxide",
        case_type=TestCaseType.MULTILAYER_4,
        layer_names=["C", "Al2O3", "TiO2", "SiO2"],
        true_thicknesses=[3.0, 8.0, 15.0, 10.0],
        interface_widths=[2.0, 2.0, 3.0, 3.0],
        description="薄い4層膜",
        tags=["multilayer", "thin"],
    ),
]


@dataclass
class BenchmarkCondition:
    """ベンチマーク条件"""
    noise_level: float = 0.02
    initial_offset: float = 5.0
    n_trials: int = 30
    angles: List[float] = field(default_factory=lambda: [15, 30, 45, 60, 75, 85])
    seed_base: int = 42


# =============================================================================
# Result Classes
# =============================================================================

@dataclass
class SingleTrialResult:
    """単一試行の結果"""
    estimated_thicknesses: List[float]
    thickness_errors: List[float]
    abs_errors: List[float]
    chi2: float
    elapsed_time: float
    success: bool


@dataclass
class TestCaseResult:
    """テストケースの結果"""
    test_case: TestCase
    solver_name: str
    condition: BenchmarkCondition
    trials: List[SingleTrialResult]

    # 集計統計
    mean_errors: Optional[List[float]] = None
    std_errors: Optional[List[float]] = None
    mean_abs_errors: Optional[List[float]] = None
    success_rate: Optional[float] = None
    mean_time: Optional[float] = None

    def compute_statistics(self):
        """統計量を計算"""
        if not self.trials:
            return

        n_layers = len(self.test_case.true_thicknesses)
        errors = np.array([t.thickness_errors for t in self.trials])
        abs_errors = np.array([t.abs_errors for t in self.trials])
        times = [t.elapsed_time for t in self.trials]
        successes = [t.success for t in self.trials]

        self.mean_errors = np.mean(errors, axis=0).tolist()
        self.std_errors = np.std(errors, axis=0).tolist()
        self.mean_abs_errors = np.mean(abs_errors, axis=0).tolist()
        self.success_rate = sum(successes) / len(successes)
        self.mean_time = np.mean(times)


@dataclass
class BenchmarkReport:
    """ベンチマークレポート"""
    timestamp: str
    version: str
    results: List[TestCaseResult]
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """辞書に変換"""
        return {
            "timestamp": self.timestamp,
            "version": self.version,
            "results": [
                {
                    "test_case": self.results[i].test_case.name,
                    "solver": self.results[i].solver_name,
                    "condition": asdict(self.results[i].condition),
                    "mean_abs_errors": self.results[i].mean_abs_errors,
                    "std_errors": self.results[i].std_errors,
                    "success_rate": self.results[i].success_rate,
                    "mean_time": self.results[i].mean_time,
                }
                for i in range(len(self.results))
            ],
            "summary": self.summary,
        }


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """ベンチマーク実行クラス"""

    def __init__(
        self,
        test_cases: Optional[List[TestCase]] = None,
        conditions: Optional[List[BenchmarkCondition]] = None,
        solvers: Optional[List[str]] = None,
    ):
        self.test_cases = test_cases or STANDARD_TEST_CASES
        self.conditions = conditions or [
            BenchmarkCondition(noise_level=0.0, initial_offset=5.0, n_trials=10),
            BenchmarkCondition(noise_level=0.02, initial_offset=5.0, n_trials=30),
            BenchmarkCondition(noise_level=0.02, initial_offset=20.0, n_trials=30),
            BenchmarkCondition(noise_level=0.05, initial_offset=10.0, n_trials=30),
        ]
        self.solvers = solvers or ["GenericLayerSolver"]
        self.results: List[TestCaseResult] = []

    def run(self, verbose: bool = True) -> BenchmarkReport:
        """ベンチマーク実行"""
        if verbose:
            print("=" * 70)
            print("ARXPS Depth Profile Benchmark")
            print("=" * 70)
            print(f"Test cases: {len(self.test_cases)}")
            print(f"Conditions: {len(self.conditions)}")
            print(f"Solvers: {self.solvers}")
            print()

        self.results = []

        for test_case in self.test_cases:
            for condition in self.conditions:
                for solver_name in self.solvers:
                    if verbose:
                        print(f"Running: {test_case.name} | {solver_name} | "
                              f"noise={condition.noise_level*100:.0f}%, "
                              f"offset=±{condition.initial_offset:.0f}Å")

                    result = self._run_single_benchmark(
                        test_case, condition, solver_name
                    )
                    self.results.append(result)

                    if verbose:
                        self._print_result_summary(result)

        # レポート生成
        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            results=self.results,
        )
        report.summary = self._compute_summary()

        return report

    def _run_single_benchmark(
        self,
        test_case: TestCase,
        condition: BenchmarkCondition,
        solver_name: str,
    ) -> TestCaseResult:
        """単一ベンチマーク実行"""
        trials = []

        for trial_idx in range(condition.n_trials):
            seed = condition.seed_base + trial_idx
            np.random.seed(seed)

            # データ生成
            I_obs, tm, true_thicknesses = self._generate_data(
                test_case, condition, seed
            )

            # 初期値生成
            offsets = condition.initial_offset * (2 * np.random.rand(len(true_thicknesses)) - 1)
            initial_thicknesses = [
                max(1.0, t + o) for t, o in zip(true_thicknesses, offsets)
            ]

            # ソルバー実行
            try:
                result = self._run_solver(
                    solver_name, test_case, tm, I_obs,
                    initial_thicknesses, test_case.interface_widths
                )

                errors = [e - t for e, t in zip(result.layer_thicknesses, true_thicknesses)]
                abs_errors = [abs(e) for e in errors]

                trials.append(SingleTrialResult(
                    estimated_thicknesses=list(result.layer_thicknesses),
                    thickness_errors=errors,
                    abs_errors=abs_errors,
                    chi2=result.chi2,
                    elapsed_time=result.elapsed_time,
                    success=result.success,
                ))
            except Exception as e:
                warnings.warn(f"Trial {trial_idx} failed: {e}")
                trials.append(SingleTrialResult(
                    estimated_thicknesses=[np.nan] * len(true_thicknesses),
                    thickness_errors=[np.nan] * len(true_thicknesses),
                    abs_errors=[np.nan] * len(true_thicknesses),
                    chi2=np.nan,
                    elapsed_time=0,
                    success=False,
                ))

        result = TestCaseResult(
            test_case=test_case,
            solver_name=solver_name,
            condition=condition,
            trials=trials,
        )
        result.compute_statistics()

        return result

    def _generate_data(
        self,
        test_case: TestCase,
        condition: BenchmarkCondition,
        seed: int,
    ) -> Tuple[np.ndarray, TransferMatrix, List[float]]:
        """テストデータ生成"""
        from scipy.special import erf

        if test_case.case_type == TestCaseType.SIO2_SI:
            return self._generate_sio2_si_data(test_case, condition, seed)
        elif test_case.case_type == TestCaseType.MULTILAYER_4:
            return self._generate_multilayer_data(test_case, condition, seed)
        else:
            raise ValueError(f"Unknown test case type: {test_case.case_type}")

    def _generate_sio2_si_data(
        self,
        test_case: TestCase,
        condition: BenchmarkCondition,
        seed: int,
    ) -> Tuple[np.ndarray, TransferMatrix, List[float]]:
        """SiO2/Siデータ生成"""
        from scipy.special import erf

        thickness = test_case.true_thicknesses[0]
        width = test_case.interface_widths[0]

        # 深さグリッド
        depth = np.arange(0, 150, 1.0)
        n_depth = len(depth)

        # プロファイル生成
        sigma = width / (2 * np.sqrt(np.log(2)))
        D = np.zeros((n_depth, 3))  # Si_ox, O, Si_bulk

        # SiO2層
        sio2_profile = 1 - 0.5 * (1 + erf((depth - thickness) / sigma))
        D[:, 0] = (1/3) * sio2_profile  # Si_ox
        D[:, 1] = (2/3) * sio2_profile  # O

        # Si基板
        D[:, 2] = 0.5 * (1 + erf((depth - thickness) / sigma))

        # 正規化
        D = np.clip(D, 1e-6, None)
        D = D / D.sum(axis=1, keepdims=True)

        # 元素情報（簡易版）
        from .depth_profile import ElementConfig
        elements = [
            ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
            ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
            ElementConfig('Si', '2p', 'Si', oxidation_state=0),
        ]

        # 転送行列
        tm = TransferMatrix(
            depth=depth,
            step_size=1.0,
            angles=condition.angles,
            photon_energy=1486.6,
        )
        tm.compute(D, elements)

        # 観測データ
        I_true = tm.forward(D, normalize=True)

        # ノイズ追加
        np.random.seed(seed)
        if condition.noise_level > 0:
            noise = condition.noise_level * np.random.randn(*I_true.shape)
            I_obs = I_true + noise * I_true
            I_obs = np.clip(I_obs, 1e-6, None)
            I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)
        else:
            I_obs = I_true.copy()

        return I_obs, tm, test_case.true_thicknesses

    def _generate_multilayer_data(
        self,
        test_case: TestCase,
        condition: BenchmarkCondition,
        seed: int,
    ) -> Tuple[np.ndarray, TransferMatrix, List[float]]:
        """多層膜データ生成"""
        from scipy.special import erf

        thicknesses = test_case.true_thicknesses
        widths = test_case.interface_widths

        # 深さグリッド
        depth = np.arange(0, 150, 1.0)
        n_depth = len(depth)

        # GenericLayerSolverと一致するモデルでプロファイル生成
        D = np.zeros((n_depth, 6))  # C, Al, Ti, Si_ox, O, Si_bulk

        cumulative = np.cumsum(thicknesses)
        sigmas = np.array(widths) / (2 * np.sqrt(np.log(2)))

        layer_profiles = []

        # 各層のプロファイル
        for i in range(4):
            z_start = 0 if i == 0 else cumulative[i - 1]
            z_end = cumulative[i]
            sigma_start = sigmas[0] if i == 0 else sigmas[i - 1]
            sigma_end = sigmas[i]

            rising = 0.5 * (1 + erf((depth - z_start) / sigma_start)) if i > 0 else np.ones_like(depth)
            falling = 0.5 * (1 + erf((depth - z_end) / sigma_end))
            profile = np.maximum(rising - falling, 0)
            layer_profiles.append(profile)

        # 元素濃度設定
        D[:, 0] = layer_profiles[0]  # C
        D[:, 1] = (2/5) * layer_profiles[1]  # Al (Al2O3: Al:O = 2:3)
        D[:, 2] = (1/3) * layer_profiles[2]  # Ti (TiO2: Ti:O = 1:2)
        D[:, 3] = (1/3) * layer_profiles[3]  # Si_ox (SiO2: Si:O = 1:2)
        D[:, 4] = (3/5) * layer_profiles[1] + (2/3) * layer_profiles[2] + (2/3) * layer_profiles[3]  # O

        # Si基板
        z_sub = cumulative[-1]
        sigma_sub = sigmas[-1]
        D[:, 5] = 0.5 * (1 + erf((depth - z_sub) / sigma_sub))

        # 正規化
        D = np.clip(D, 1e-6, None)
        D = D / D.sum(axis=1, keepdims=True)

        # 元素情報
        from .depth_profile import ElementConfig
        elements = [
            ElementConfig('C', '1s', 'C', oxidation_state=0),
            ElementConfig('Al', '2p', 'Al2O3', oxidation_state=3),
            ElementConfig('Ti', '2p', 'TiO2', oxidation_state=4),
            ElementConfig('Si', '2p', 'SiO2', oxidation_state=4),
            ElementConfig('O', '1s', 'SiO2', oxidation_state=-2),
            ElementConfig('Si', '2p', 'Si', oxidation_state=0),
        ]

        # 転送行列
        tm = TransferMatrix(
            depth=depth,
            step_size=1.0,
            angles=condition.angles,
            photon_energy=1486.6,
        )
        tm.compute(D, elements)

        # 観測データ
        I_true = tm.forward(D, normalize=True)

        # ノイズ追加
        np.random.seed(seed)
        if condition.noise_level > 0:
            noise = condition.noise_level * np.random.randn(*I_true.shape)
            I_obs = I_true + noise * I_true
            I_obs = np.clip(I_obs, 1e-6, None)
            I_obs = I_obs / I_obs.sum(axis=1, keepdims=True)
        else:
            I_obs = I_true.copy()

        return I_obs, tm, test_case.true_thicknesses

    def _run_solver(
        self,
        solver_name: str,
        test_case: TestCase,
        tm: TransferMatrix,
        I_obs: np.ndarray,
        initial_thicknesses: List[float],
        interface_widths: List[float],
    ):
        """ソルバー実行"""
        if solver_name == "GenericLayerSolver":
            return self._run_generic_layer_solver(
                test_case, tm, I_obs, initial_thicknesses, interface_widths
            )
        elif solver_name == "SemiFreeSolver":
            return self._run_semi_free_solver(
                test_case, tm, I_obs, initial_thicknesses, interface_widths
            )
        else:
            raise ValueError(f"Unknown solver: {solver_name}")

    def _run_generic_layer_solver(
        self,
        test_case: TestCase,
        tm: TransferMatrix,
        I_obs: np.ndarray,
        initial_thicknesses: List[float],
        interface_widths: List[float],
    ):
        """GenericLayerSolver実行"""
        if test_case.case_type == TestCaseType.SIO2_SI:
            config = GenericLayerConfig(
                layers=[
                    LayerDefinition(
                        name="SiO2",
                        element_indices=[0, 1],
                        stoichiometry={0: 1, 1: 2},
                    ),
                ],
                substrate=LayerDefinition(
                    name="Si",
                    element_indices=[2],
                    stoichiometry={2: 1},
                ),
                initial_thicknesses=initial_thicknesses,
                initial_widths=interface_widths,
                linked_element_index=1,
            )
        elif test_case.case_type == TestCaseType.MULTILAYER_4:
            config = GenericLayerConfig(
                layers=[
                    LayerDefinition(name="C", element_indices=[0], stoichiometry={0: 1}),
                    LayerDefinition(name="Al2O3", element_indices=[1, 4], stoichiometry={1: 2, 4: 3}),
                    LayerDefinition(name="TiO2", element_indices=[2, 4], stoichiometry={2: 1, 4: 2}),
                    LayerDefinition(name="SiO2", element_indices=[3, 4], stoichiometry={3: 1, 4: 2}),
                ],
                substrate=LayerDefinition(name="Si", element_indices=[5], stoichiometry={5: 1}),
                initial_thicknesses=initial_thicknesses,
                initial_widths=interface_widths,
                linked_element_index=4,
            )
        else:
            raise ValueError(f"Unsupported case type: {test_case.case_type}")

        solver = GenericLayerSolver(tm, SolverConfig(method='LLS'), config)
        return solver.solve(I_obs)

    def _run_semi_free_solver(
        self,
        test_case: TestCase,
        tm: TransferMatrix,
        I_obs: np.ndarray,
        initial_thicknesses: List[float],
        interface_widths: List[float],
    ):
        """SemiFreeSolver実行（SiO2/Siのみ対応）"""
        if test_case.case_type != TestCaseType.SIO2_SI:
            raise ValueError("SemiFreeSolver only supports SiO2/Si test cases")

        config = SemiFreeSolverConfig(
            thickness_init=initial_thicknesses[0],
            abruptness_init=interface_widths[0],
            free_indices=[0],
            linked_indices=[1],
            substrate_indices=[2],
        )

        solver = SemiFreeSolver(tm, SolverConfig(method='LLS'), config)
        result = solver.solve(I_obs)

        # GenericLayerResultと同じ形式に変換
        class CompatibleResult:
            def __init__(self, semi_result):
                self.layer_thicknesses = [semi_result.thickness]
                self.chi2 = semi_result.chi2
                self.elapsed_time = semi_result.elapsed_time
                self.success = semi_result.success

        return CompatibleResult(result)

    def _print_result_summary(self, result: TestCaseResult):
        """結果サマリー表示"""
        print(f"  -> |Error|: {[f'{e:.2f}Å' for e in result.mean_abs_errors]}, "
              f"Success: {result.success_rate*100:.0f}%, "
              f"Time: {result.mean_time*1000:.1f}ms")

    def _compute_summary(self) -> Dict:
        """全体サマリー計算"""
        summary = {
            "total_test_cases": len(self.test_cases),
            "total_conditions": len(self.conditions),
            "total_solvers": len(self.solvers),
            "overall_success_rate": np.mean([r.success_rate for r in self.results]),
        }

        # ソルバー別サマリー
        for solver in self.solvers:
            solver_results = [r for r in self.results if r.solver_name == solver]
            if solver_results:
                all_errors = []
                for r in solver_results:
                    all_errors.extend(r.mean_abs_errors or [])
                summary[f"{solver}_mean_error"] = np.mean(all_errors) if all_errors else None

        return summary


# =============================================================================
# Report Generation
# =============================================================================

class ReportGenerator:
    """レポート生成クラス"""

    @staticmethod
    def to_markdown(report: BenchmarkReport) -> str:
        """Markdownレポート生成"""
        lines = [
            "# ARXPS Depth Profile Benchmark Report",
            "",
            f"**Generated**: {report.timestamp}",
            f"**Version**: {report.version}",
            "",
            "## Summary",
            "",
            f"- Total test cases: {report.summary.get('total_test_cases', 'N/A')}",
            f"- Overall success rate: {report.summary.get('overall_success_rate', 0)*100:.1f}%",
            "",
            "## Detailed Results",
            "",
        ]

        # テストケース別に整理
        test_case_names = list(set(r.test_case.name for r in report.results))

        for tc_name in test_case_names:
            tc_results = [r for r in report.results if r.test_case.name == tc_name]
            if not tc_results:
                continue

            tc = tc_results[0].test_case
            lines.append(f"### {tc_name}")
            lines.append("")
            lines.append(f"**Description**: {tc.description}")
            lines.append(f"**True thicknesses**: {tc.true_thicknesses} Å")
            lines.append(f"**Interface widths**: {tc.interface_widths} Å")
            lines.append("")

            # テーブル作成
            headers = ["Noise", "Offset"] + tc.layer_names + ["Success", "Time"]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

            for r in tc_results:
                row = [
                    f"{r.condition.noise_level*100:.0f}%",
                    f"±{r.condition.initial_offset:.0f}Å",
                ]
                row.extend([f"{e:.2f}" for e in (r.mean_abs_errors or [])])
                row.append(f"{r.success_rate*100:.0f}%")
                row.append(f"{r.mean_time*1000:.1f}ms")
                lines.append("| " + " | ".join(row) + " |")

            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def to_json(report: BenchmarkReport) -> str:
        """JSONレポート生成"""
        return json.dumps(report.to_dict(), indent=2)

    @staticmethod
    def to_ci_summary(report: BenchmarkReport) -> Dict:
        """CI/CD用サマリー生成"""
        return {
            "status": "pass" if report.summary.get("overall_success_rate", 0) > 0.95 else "fail",
            "success_rate": report.summary.get("overall_success_rate", 0),
            "timestamp": report.timestamp,
            "errors": {
                r.test_case.name: r.mean_abs_errors
                for r in report.results
            },
        }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """メイン実行"""
    import argparse

    parser = argparse.ArgumentParser(description="ARXPS Benchmark System")
    parser.add_argument("--solver", type=str, default="GenericLayerSolver",
                        help="Solver to benchmark")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path")
    parser.add_argument("--format", type=str, default="markdown",
                        choices=["markdown", "json", "ci"],
                        help="Output format")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (fewer trials)")

    args = parser.parse_args()

    # 条件設定
    if args.quick:
        conditions = [
            BenchmarkCondition(noise_level=0.02, initial_offset=5.0, n_trials=5),
        ]
    else:
        conditions = None  # デフォルト

    # ベンチマーク実行
    runner = BenchmarkRunner(
        solvers=[args.solver],
        conditions=conditions,
    )
    report = runner.run(verbose=True)

    # レポート生成
    if args.format == "markdown":
        output = ReportGenerator.to_markdown(report)
    elif args.format == "json":
        output = ReportGenerator.to_json(report)
    else:
        output = json.dumps(ReportGenerator.to_ci_summary(report), indent=2)

    # 出力
    if args.output:
        Path(args.output).write_text(output)
        print(f"\nReport saved to: {args.output}")
    else:
        print("\n" + "=" * 70)
        print("BENCHMARK REPORT")
        print("=" * 70)
        print(output)


if __name__ == "__main__":
    main()
