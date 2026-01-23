# 引き継ぎ手順書: `adjointGradientDescent` の調査・改善（性能/警告/テスト）

このドキュメントは、`adjointGradientDescent` 周辺で発生していた **SVD 警告**・**テストの watch 待機**・**性能の“怪しさ”**を調査し、現時点までに実施した改善と、今後の掘り下げ方針を引き継ぐための手順書です。

> 注意: `docs/` は Typedoc の生成物です（HTML/JS/CSS）。手書きドキュメントはそこへ追加せず、本ファイルのようにリポジトリ直下へ置く方針にしています。

---

## 現状の結論（わかったこと）

- **SVD 警告の正体**: `ml-matrix` の SVD が「列数 > 行数」の行列で動くと `autoTranspose` を推奨する警告を `stderr` に出す。  
  これは `adjointGradientDescent` の **本体計算ではなく診断ログ用の SVD** 呼び出しが原因だった。

- **`adjointGradientDescent` が遅く見える要因（テスト観測）**:
  - ログ出力が無効でも、`adjointGradientDescent.ts` 内で **SVD/行列統計などの“診断計算”を先に実施**していた箇所があり、これが無駄コストになりやすかった。
  - residual/cost 判定のために、ユーザー関数（cost/residual）が **同一点で二重に評価**される箇所があり、性能だけでなく「副作用があると結果が揺れる」リスクもあった。

- **テスト出力の `stdout | test/stress-constrained...` は失敗ではない**:
  - ストレステストが `console.log` / Logger の INFO を出しているだけ。テストは PASS でも表示される。

---

## 実施済み変更（このリポジトリで反映済み）

### 1) テストが watch 待機に入らないように変更

- `package.json`
  - `test`: `vitest run`（1回実行して終了）
  - `test:watch`: `vitest`（従来の watch）

確認コマンド:

```bash
npm test
```

### 2) SVD 警告（`ml-matrix`）を出さない

- `src/core/adjointGradientDescent.ts`
  - 診断用 SVD 呼び出しに `autoTranspose: true` を付けるラッパを追加し、警告を抑止。
  - 特異値は転置しても同一なので、**診断値の意味は基本的に不変**。

確認コマンド（`stderr` に警告が出ないこと）:

```bash
npm test -- test/adjointGradientDescent.test.ts
```

### 3) ログが無効なとき「診断計算自体」をやらない

- `src/core/logger.ts`
  - `Logger.isEnabled(level)` を追加（そのレベルが出力されるかを問い合わせ可能に）

- `src/core/adjointGradientDescent.ts`
  - `solveAdjointEquation` / `logAdjointDiagnostics` で、SVD/行列走査などの **高コスト診断をログ出力時のみ計算**するように gating。

狙い:
- `logLevel` 未指定（= 実質ログ無効）でも、診断計算が走ってしまうケースを潰す。

### 4) residual/cost の「二重評価」を削減し、モードを固定化

- `src/core/adjointGradientDescent.ts`
  - `evaluateCostOrResidual` を導入し、**型判定のための余計な呼び出し**を避ける。
  - 初回評価で `isResidualMode` を確定し、以降はそのモードで処理する。

リスク/前提:
- ユーザーの `costFunction` は「常に number」か「常に Float64Array」を返す想定（途中で返り型が変わるのは仕様外）。

### 5) backtracking line search に不要な “trial 勾配計算” をさせない

- `adjointGradientDescent` は `backtrackingLineSearch` を使用しており、これは **trial 点で勾配を要求しない**（現在点のみ）。
- そのため、trial 点用の複雑な gradient wrapper は不要なので簡略化し、関連の未使用コードも削除。

注意:
- 将来 `adjointGradientDescent` が `strongWolfeLineSearch` を使う場合は、trial 点の勾配が必要になり、再度 wrapper が必要になる可能性がある。

---

## 検証手順（最低限の再現コマンド）

### 1) 全テスト（watch なし）

```bash
npm test
```

期待:
- `Waiting for file changes...` が出ない
- 失敗なし（PASS）
- `Computing SVD on a matrix with more columns than rows...` が `stderr` に出ない

### 2) 対象テストのみ

```bash
npm test -- test/adjointGradientDescent.test.ts
```

### 3) watch が必要なら（開発用）

```bash
npm run test:watch
```

---

## 今後やるべきこと（優先度/重要度/難易度/リスク）

以下は「本当に性能が効くところ」を狙って、ベストプラクティスに沿って進めるための推奨順です。

### A. 有限差分（finite diff）周りのメモリ/評価回数削減（最優先）

- **重要度**: 高（実利用で支配的になりがち）
- **難易度**: 中〜高（API/実装の整理が必要）
- **リスク**: 中（数値誤差や再利用バッファのバグ）

狙い:
- `finiteDiffConstraintPartialX/P` や residual/cost の評価回数は、状態/制約次元が大きくなると爆発する。
- 現状は `Float64Array` / `Matrix` の生成が多く、GC/割当が増えやすい。

候補:
- 使い回せるバッファの導入（in-place、ワーク領域のキャッシュ）
- 中央差分で共通化できる箇所の削減（同一点評価の再利用）
- `stepSize` を次元ごとに持てる（スケーリング問題の軽減）

### B. ラインサーチ中の state 更新（`updateStates`）の高速化

- **重要度**: 中〜高（バックトラック回数が増えると効く）
- **難易度**: 高（線形代数の設計が必要）
- **リスク**: 中〜高（解の意味が変わりうる）

現状:
- backtracking の trial ごとに `updateStates` が走り、内部で least-squares solve が発生し得る。

候補:
- 「感度行列」 \(S = -(\partial c/\partial x)^+ (\partial c/\partial p)\) を1回作って trial では `dx = S * Δp` にする（solve 回数削減）
- ただし疑似逆や正規方程式の安定性、正則化の扱いが難しくなるので設計が必要。

### C. ログ/診断の設計整理（共通化）

- **重要度**: 中
- **難易度**: 低〜中
- **リスク**: 低

候補:
- SVD 診断用ユーティリティを一箇所に寄せる（現状は `adjointGradientDescent.ts` 側ローカル）
- `docs/` 生成物に混ぜない運用を README に追記するか、`DEV_NOTES.md` を導入する

---

## 追加メモ（調査時の観測ポイント）

- `vitest` の表示する「(xx tests) N ms」はテスト関数実行部分の目安で、トランスパイル/収集/準備の時間は別枠で出る。
- 「遅い」原因を切り分ける際は:
  - まず **ログ/診断計算の有無**（今回ここが当たりだった）
  - 次に **関数評価回数（finite diff）**
  - 最後に **線形ソルバ（solveLeastSquares / Cholesky）** の回数とサイズ

