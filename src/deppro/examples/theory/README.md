# Verification Scripts for Depth Resolution Theory

Cumpson (1995) の理論予測を検証するためのスクリプト群。

## ファイル一覧

| ファイル | 内容 |
|---------|------|
| `theory_comparison.py` | **Cumpson理論 vs MCシミュレーション比較**（公表済みの理論との比較） |
| `xps_noise_summary.py` | ガウス vs ポアソンノイズ比較（Strohmeier 式の誤差伝播） |

どちらも `python <file>` で実行したときだけ計算と作図を行う（import しても何も書かない）。

## 実行方法

```bash
cd <repo>

# Cumpson理論との比較（推奨）。deppro が import できる環境なら `python -m` でもよい
python src/deppro/examples/theory/theory_comparison.py
python -m deppro.examples.theory.theory_comparison

# ノイズ影響サマリー
python src/deppro/examples/theory/xps_noise_summary.py
python -m deppro.examples.theory.xps_noise_summary
```

## 出力先

すべてのプロットは `deppro.paths.ensure_outputs_dir()` が返す場所に保存されます。

- `DEPPRO_OUTPUTS` が設定されていればそこ
- 未設定で、チェックアウトから使っていればリポジトリ直下の `outputs/`（`.gitignore` 済み。無ければ作られる）
- 未設定で、インストール済みの deppro からなら、カレントディレクトリの `outputs/`（一度だけ警告が出る）

```bash
DEPPRO_OUTPUTS=~/Desktop/figs python -m deppro.examples.theory.theory_comparison
```

## 結果

数値は各スクリプトを実行して得る（この README には結果を転記しない）。

### 参考文献

P.J. Cumpson, "Angle-resolved XPS and AES: Depth-resolution limits...",
J. Electron Spectrosc. Relat. Phenom. 73 (1995) 25-52
