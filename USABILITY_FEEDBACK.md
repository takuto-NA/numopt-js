# 初見として使いにくかった点 - 正直な感想

実際にコードを書いて使ってみた経験から、改善すべき点をまとめました。

## 1. APIの一貫性の問題

### 1.1 関数の引数順序が直感的でない

**問題点:**
- `finiteDiffGradient(costFunction, parameters, options)` の順序
- 他の関数（`gradientDescent`, `levenbergMarquardt`）は `parameters` が最初
- 引数の順序が統一されていない

**実際の体験:**
```typescript
// 最初、こう書いてしまった（間違い）
finiteDiffGradient(params, rosenbrockFunction, { h: 1e-6 })

// 正しくは
finiteDiffGradient(rosenbrockFunction, params, { stepSize: 1e-6 })
```

**改善案:**
- すべての関数で `parameters` を最初の引数にする
- または、オプションオブジェクトで統一する

### 1.2 オーバーロードされた関数シグネチャが分かりにくい

**問題点:**
```typescript
levenbergMarquardt(
  initialParameters: Float64Array,
  residualFunction: ResidualFn,
  jacobianFunctionOrOptions?: JacobianFn | LevenbergMarquardtOptions,
  options?: LevenbergMarquardtOptions
)
```

- `jacobianFunctionOrOptions` という名前が分かりにくい
- TypeScriptの型推論が効きにくい
- 関数を渡すかオプションを渡すかで動作が変わる

**実際の体験:**
- 最初、どの引数をどう渡せばいいか迷った
- IDEの補完がうまく機能しない

**改善案:**
- 別々の関数にする（`levenbergMarquardt` と `levenbergMarquardtWithJacobian`）
- または、オプションオブジェクト内に `jacobian` プロパティを追加

## 2. 戻り値の型が一貫していない

**問題点:**
- `gaussNewton` は `OptimizationResult` を返す（`finalResidualNorm` がない）
- `levenbergMarquardt` は `LevenbergMarquardtResult` を返す（`finalResidualNorm` がある）
- 同じような問題（非線形最小二乗）なのに、返す型が違う

**実際の体験:**
```typescript
const result = gaussNewton(...);
console.log(result.finalResidualNorm); // undefined エラー！
```

**改善案:**
- すべての非線形最小二乗アルゴリズムで `finalResidualNorm` を含む型を返す
- または、`finalCost` から計算できることを明示

## 3. オプション名の一貫性がない

**問題点:**
- `tolerance` (gradientDescent)
- `tolGradient`, `tolStep`, `tolResidual` (levenbergMarquardt)
- `tolGradient` (gaussNewton)

**実際の体験:**
- どのアルゴリズムでどのオプション名を使えばいいか覚えられない
- ドキュメントを見返す必要がある

**改善案:**
- 統一した命名規則を採用
- または、すべてのアルゴリズムで共通のオプション名を使用

## 4. エラーメッセージが不十分

**問題点:**
```typescript
throw new Error('Jacobian function must be provided or useNumericJacobian must be true');
```

- 何が間違っていたのか具体的でない
- どう修正すればいいか分からない

**実際の体験:**
- エラーが出ても、原因を特定するのに時間がかかった
- デバッグ情報が少ない

**改善案:**
- より具体的なエラーメッセージ
- 使用例を含むエラーメッセージ
- デバッグモードでの詳細情報

## 5. ドキュメントの不足

**問題点:**
- READMEには基本的な使い方はあるが、詳細が不足
- エッジケースの説明がない
- トラブルシューティングの情報がない
- デフォルト値がコード内にしか書かれていない

**実際の体験:**
- オプションのデフォルト値を調べるのにコードを読む必要があった
- エラー時の対処法が分からなかった

**改善案:**
- APIリファレンスの充実
- デフォルト値の明記
- よくある問題と解決策のセクション
- より多くの例（エラーハンドリング、高度な使い方）

## 6. 型定義の複雑さ

**問題点:**
- `ml-matrix` の `Matrix` 型に依存しているが、それが明示的でない
- `JacobianFn` が `Matrix` を返すが、なぜ `Matrix` なのか分からない
- `Float64Array` を多用しているが、その理由が分からない

**実際の体験:**
- `Matrix` 型の使い方が分からず、ドキュメントを探す必要があった
- `Float64Array` と通常の配列の違いが分からなかった

**改善案:**
- なぜ `Matrix` を使うのか、なぜ `Float64Array` を使うのかの説明
- 型定義のコメントの充実
- 変換関数の提供（通常の配列から `Float64Array` へ）

## 7. 例の不足

**問題点:**
- 基本的な例はあるが、以下が不足：
  - エラーハンドリングの例
  - 高度な使い方の例
  - パフォーマンス最適化の例
  - 複雑な問題の例

**実際の体験:**
- 新しい問題に適用する際、例を参考にしづらかった
- エラー時の対処法が分からなかった

**改善案:**
- より多様な例の追加
- 実用的な問題の例（画像処理、機械学習など）
- エラーハンドリングの例

## 8. デバッグの難しさ

**問題点:**
- `verbose` オプションがあるが、何が出力されるか分からない
- エラー時の診断情報が少ない
- 収束しない場合の原因が分かりにくい

**実際の体験:**
- `verbose: true` にしても、出力が多すぎて見づらい
- 収束しない場合、何が問題なのか分からなかった

**改善案:**
- デバッグレベルの設定（`debug: 'minimal' | 'normal' | 'verbose'`）
- より構造化されたログ出力
- 収束しない場合の診断情報の提供

## 9. オプションのデフォルト値が分かりにくい

**問題点:**
- デフォルト値がコード内にしか書かれていない
- ドキュメントに明記されていない
- IDEの補完でデフォルト値が表示されない

**実際の体験:**
- どのオプションを設定すべきか分からなかった
- デフォルト値が適切かどうか判断できなかった

**改善案:**
- ドキュメントにデフォルト値を明記
- JSDocコメントにデフォルト値を追加
- 型定義にデフォルト値を含める

## 10. 学習曲線が急

**問題点:**
- 基本的な使い方は簡単だが、高度な使い方が難しい
- 概念（Jacobian、残差、収束条件など）の理解が必要
- ドキュメントが専門的すぎる

**実際の体験:**
- 最初は簡単だったが、複雑な問題に適用する際に苦労した
- 数学的な背景知識が必要だと感じた

**改善案:**
- 段階的なチュートリアル
- 概念の説明を追加
- 初心者向けガイドの追加

## まとめ

全体的には、**基本的な使い方は簡単**ですが、**高度な使い方やエラー時の対処が難しい**と感じました。

特に改善してほしい点：
1. **APIの一貫性** - 引数の順序、オプション名の統一
2. **戻り値の型の一貫性** - 同じような問題で同じ型を返す
3. **エラーメッセージの改善** - より具体的で有用な情報
4. **ドキュメントの充実** - デフォルト値、エッジケース、トラブルシューティング

これらが改善されれば、**もっと使いやすいライブラリ**になると思います。

