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

## 現状わかっている課題一覧（引き継ぎ用）

### 1) 状態更新（`x`）が線形近似のみで、制約多様体からドリフトし得る

- **症状**: `checkConstraintViolation` が WARN を出す／制約違反が累積する可能性。
- **原因候補**: `p` 更新後に `x` を `dx`（線形 LS 解）で更新するのみで、非線形制約に対する復元を行っていない。
- **影響**: 収束性低下、誤った勾配、ラインサーチ評価の不整合（実際は制約外で cost を評価してしまう等）。
- **優先度**: 高
- **対応案**: Restoration Step（`p` 固定で `c(p,x)=0` を満たすように `x` を数回ニュートン補正）を導入。既に `src/core/constrainedUtils.ts` に `projectStatesToConstraints` があるため、まずはそれを流用するのが最短。

### 2) 有限差分（finite diff）の評価回数/割り当てが支配的になりやすい

- **症状**: 次元が上がると極端に遅くなる（特に `finiteDiffConstraintPartialX/P`）。
- **原因候補**: central difference により 1次元あたり2回の関数評価、かつ `Float64Array`/`Matrix` の生成が多い。
- **影響**: 実問題でスケールしない、GC によるスパイク。
- **優先度**: 高
- **対応案**: 評価回数の再利用（同一点評価のキャッシュ）、ワークバッファ再利用（in-place）、必要なら次元別 `stepSize` を導入（スケーリング問題対策）。

### 3) ラインサーチ中の trial 評価が「制約付きの正しい点」になっていない可能性

- **症状**: trial 点の cost 評価が制約から外れた状態で行われる/行われ得る。
- **原因候補**: 現状は trial `p` に対して線形近似 `x` 更新のみ（復元なし）。制約が強非線形だと trial ごとにズレが増える。
- **影響**: ラインサーチ判定（Armijo）が本来の制約多様体上の目的関数と一致しない。
- **優先度**: 中〜高
- **対応案**: trial 評価でも必要に応じて復元（ただしコスト増）。まずは「制約違反が閾値超なら復元」の条件付きが現実的。

### 4) `costFunction` / `residualFunction` の返り型が途中で変わると破綻する（仕様上の前提）

- **症状**: 初回評価で `isResidualMode` を確定して以降固定しているため、途中で返り型が変わると不整合。
- **原因候補**: ユーザー関数が条件分岐で number/Float64Array を返してしまう等（仕様外）。
- **影響**: 例外や誤った計算。
- **優先度**: 中
- **対応案**: README/型で明示、もしくはデバッグ時に「初回以降の返り型変化」を検知して例外化。

### 5) 大規模問題では GC（割り当て）負荷がボトルネックになりやすい

- **症状**: 反復中に `Float64Array` / `Matrix` が大量生成され、速度が落ちる・スパイクする。
- **原因候補**: `add/subtract/scale` 等が新規配列を返す設計、`float64ArrayToMatrix` 変換の往復。
- **影響**: 大規模ケースでの実行時間・安定性悪化。
- **優先度**: 中
- **対応案**: クリティカルパスのみワーク領域を持つ（in-place API or 内部バッファ）、変換回数削減。

### 6) 関数引数が多く、保守性/変更容易性に課題

- **症状**: `performAdjointGradientDescentIteration` / `handleStepSizeAndUpdate` などが多引数になりがち。
- **原因候補**: 状態/設定/キャッシュ/ログを個別引数で受け渡し。
- **影響**: 変更時に破壊的変更やバグ混入が起きやすい。
- **優先度**: 中
- **対応案**: Context パターン（`OptimizationContext`）の導入で整理（ただし影響範囲が広いので計画的に）。

### 7) `ml-matrix` の `SingularValueDecomposition` の型定義が弱く、`@ts-ignore` が残りがち

- **症状**: TS 型安全性が落ちる。
- **原因候補**: upstream の型定義不足。
- **影響**: IDE 支援低下、将来の破壊的変更の検知が遅れる。
- **優先度**: 低〜中
- **対応案**: ローカルで型拡張（`*.d.ts`）を追加し、`@ts-ignore` を排除。

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

