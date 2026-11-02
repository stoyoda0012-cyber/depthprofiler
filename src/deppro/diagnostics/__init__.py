"""deppro 本体（ソルバ・構造解析）の診断スクリプト置き場。

ここに入るのは **テストではない**もの — 手で走らせて図や数値を眺めるための
調査用スクリプト。pytest は収集しない（`test_*.py` を名乗らないため）。

    python -m deppro.diagnostics.structure_analyzer
    python src/deppro/diagnostics/uniform_detection.py

2026-07-29 に `tests/{solver,integration,unit}/` から移設した。移設前は
`test_*.py` を名乗っていたので pytest が収集していたが、test 関数はゼロで、
「収集 0 件」を誰も見ていなかった。同じ理由でベンチマークは `deppro/benchmarks/` にある。

移設元（参照が残っている文書のため）。旧パスは全て `test_<新名>.py`:

    tests/integration/ → five_layer_free (旧 test_5layer_free), film_structure_solver,
                         structure_analyzer, thickness_accuracy
    tests/unit/        → uniform_detection

ほかに公開版に含めない診断スクリプトも同時に移した（一覧は 2026-07-29 の移設
commit にある）。その一部は `test_` を名乗る**引数つきヘルパー**しか持たず、
pytest が fixture 注入に失敗して **26 件の setup error** を出し続けていた。
ヘルパーは `run_*` に改名済み。

図の出力先は `deppro.paths.ensure_outputs_dir()` で解決する。`DEPPRO_OUTPUTS`
が設定されていればそこ、未設定ならリポジトリ直下の `outputs/`。
かつてここには `savefig('/Users/.../src/outputs/...')` という**個人の絶対
パス**が直書きされていた（2026-08-03 に解消）。
"""
