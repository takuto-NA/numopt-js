# Adjoint 準ニュートン法（numopt-js 実装方針）

この文書は、numopt-js に Adjoint BFGS（および後続の Adjoint L-BFGS）を足すときの規範である。新しい更新式を発明する文書ではない。既存の縮約空間 Adjoint と、既存の無制約 BFGS をどう接続するかを決める。

`docs/` には置かない。そこは TypeDoc の生成先であり `.gitignore` されている。`npm run docs` で消える。

主張は (一次資料の事実) と (このリポジトリへの推奨) を分けて書く。

## この文書の結論（先に読む）

- 「Adjoint BFGS」は特殊な BFGS 公式ではない。陰的状態 \(x(p)\) を制約 \(c(p,x)=0\) で消した縮約目的 \(\tilde f(p)=f(p,x(p))\) に、既存の BFGS（Nocedal & Wright Algorithm 6.1）を適用する。勾配は既存の随伴式で取る。
- 最初に実装するのは **dense `adjointBfgs`**。このライブラリの Adjoint の存在理由は「少ない \(p\)・多い \(x\)」であり、Hessian は \(p\) 次元にだけ置けばよい。`adjointLbfgs` は \(p\) が大きくなったあとの第二弾。
- 公開 API は `adjointGradientDescent` と同じ入力（\(p_0,x_0,f,c\) と偏微分オプション）を取る専用入口にする。公開の `bfgs()` に問題を投げない。更新式 `updateInverseHessianApproximation` と Strong Wolfe は再利用する。
- 直線探索は **Strong Wolfe**（既存 `bfgs` / `lbfgs` と同じ）。試行点と受理点の \((\tilde f,\nabla\tilde f)\) は、射影後の実現可能点で毎回作り直す。凍結した \(\partial c/\partial x\) の接線予測だけで勾配を作らない。
- 残差最小二乗 \(f=\tfrac12\|r\|^2\) は、すでに Constrained GN/LM（有効ヤコビアン = 縮約 \(J^\top J\)）がある。Adjoint BFGS の主対象はスカラーコスト \(f(p,x)\)。残差を受けてもよいが、小残差問題では GN/LM を先に勧める。

## 現状（リポジトリ）

事実（コード）:

- 縮約空間 Adjoint は `src/core/adjointGradientDescent.ts`。探索変数は \(p\) だけ。\(x\) は接線予測 `updateStates` のあと Newton 射影 `projectStatesToConstraints` で \(c(p,x)=0\) に戻す。
- 縮約勾配は \((\partial c/\partial x)^\top\lambda=(\partial f/\partial x)^\top\) を解き、\(\mathrm{d}f/\mathrm{d}p=\partial f/\partial p-\lambda^\top\partial c/\partial p\)。共有実装は `solveAdjointEquation`（`src/core/constrainedUtils.ts`）。
- Adjoint は正方の陰的状態系（`constraintCount === stateCount`）を要求する。Constrained GN/LM は矩形 \(\partial c/\partial x\) を許し、縮約残差ヤコビアンは `src/core/effectiveJacobian.ts`。
- 無制約 `bfgs` / `lbfgs` は Strong Wolfe（`src/core/lineSearch.ts`）。\(s^\top y\) が弱いと逆 Hessian / 履歴を捨てる。Adjoint GD の直線探索は Armijo 後退だけ。
- README の Out of Scope: 自動微分、不等式、疎行列、並列。参考文献に Nocedal & Wright 第2版 (2006) がある。
- 典型例は `examples/adjoint-reduced-space.ts`（\(p\) が1、陰的状態が20）。論文ベンチの Circle 仮説は「Adjoint GD の反復は Constrained GN/LM より多い」（`scripts/paper-benchmark/hypotheses.ts`）。

仮定（まだコードにない）:

- 利用者が欲しいのは、同じ縮約空間のまま準ニュートン化すること。全空間 SQP や PDE の連続随伴ではない。

## 用語

| 用語 | このリポジトリでの意味 |
|------|------------------------|
| 陰的状態 | \(c(p,x)=0\) が局所的に一意な \(x(p)\) を定める。Adjoint の契約。 |
| 縮約目的 \(\tilde f(p)\) | \(f(p,x(p))\)。\(x\) は探索変数ではない。 |
| 縮約勾配 \(\nabla\tilde f(p)\) | 随伴で計算する \(\mathrm{d}f/\mathrm{d}p\)。 |
| 縮約 Hessian \(\nabla^2\tilde f(p)\) | \(p\) 空間の二階微分。BFGS が近似する対象。全空間の \(Z^\top\nabla^2\mathcal L\,Z\) と同値だが、実装は \(p\) だけで持つ。 |
| 随伴 (discrete adjoint) | 離散化した \(c\) のヤコビアン転置を解く。このライブラリのやり方。連続随伴 PDE は対象外。 |
| 実現可能性の維持 | 受理した反復で \(\|c(p,x)\|\) を `constraintTolerance` 以下に保つ。ROL の reduced-space と同じ思想。 |

混同しないこと: Nocedal & Wright 第18章の **reduced-Hessian SQP** は、全変数 \((p,x)\) 上で制約の零空間に BFGS を置く。メリット関数で制約を扱い、反復は多様体から外れうる。numopt-js の Adjoint 列はそれではない。

## 一次資料が言っていること

### 1. Adjoint BFGS は「縮約目的への標準 BFGS」である

- Giles & Pierce は、設計変数 \(\alpha\) と陰的状態 \(U\)（\(N(U,\alpha)=0\)）の設計問題を、随伴で勾配を取ったうえで「BFGS のような準ニュートン」で進める、と書いている。新しい BFGS 式は出していない。[Giles2000, §2.6]
- dolfin-adjoint は縮約汎関数 \(\tilde J(m):=J(u(m),m)\) を作り、既定で SciPy の L-BFGS-B に渡す。PDE 制約は各反復で満たす、と明記している。[dolfin-adjoint-opt]
- ROL は縮約問題 \(\min_z J(S(z),z)\)（\(u=S(z)\) が \(c(u,z)=0\)）の勾配を、状態求解 → 随伴 \(c_u^*\lambda=-\nabla_u J\) → \(\nabla J=\nabla_z\mathcal L\) で計算し、その縮約空間に L-BFGS などの secant を載せる。[ROL-slides]
- したがって実装すべき更新は、既存 `bfgs.ts` と同じ逆 Hessian 更新（Nocedal (6.17)）である。[NW2006, Alg. 6.1, (6.17)]

### 2. 縮約勾配は陰関数定理と随伴の同じ式である

陰関数定理: \(\partial c/\partial x\) が正則なら局所的に \(x(p)\) が一意で、

\[
\frac{\mathrm{d}x}{\mathrm{d}p}=-\Bigl(\frac{\partial c}{\partial x}\Bigr)^{-1}\frac{\partial c}{\partial p}.
\]

連鎖律から

\[
\nabla\tilde f=\frac{\partial f}{\partial p}-\Bigl(\frac{\partial c}{\partial p}\Bigr)^\top\lambda,\qquad
\Bigl(\frac{\partial c}{\partial x}\Bigr)^\top\lambda=\Bigl(\frac{\partial f}{\partial x}\Bigr)^\top.
\]

これは現在の `computeAdjointGradient` / `solveAdjointEquation` そのもの。ROL は随伴右辺にマイナスを置く記法だが、Lagrange 関数の停留条件としては同値。[ROL-slides][NW2006, Ch. 12 IFT][Giles2000, linear-algebra adjoint]

有効ヤコビアン `J_eff = r_p - r_x C_x^+ C_p`（`effectiveJacobian.ts`）は、同じ IFT を残差に適用した列である。スカラー \(f=\tfrac12\|r\|^2\) なら \(\nabla\tilde f = J_{\mathrm{eff}}^\top r\)。

### 3. BFGS には Wolfe（できれば Strong Wolfe）が要る

- 曲率条件 \(s^\top y>0\) が無いとセカント方程式を満たす正定値 \(B_{k+1}\) は存在しない。[NW2006, (6.7)]
- Wolfe / Strong Wolfe を課せば \(s^\top y>0\) は保証される。[NW2006, (6.8); Exercise 6.2]
- 直線探索はまず \(\alpha=1\) を試す。準ニュートンが Newton 方向に近づけば単位ステップが受理され、超一次収束の前提になる。[NW2006, §3.5, §6.1 Implementation]
- 典型値は \(c_1=10^{-4}\)、\(c_2=0.9\)。不正確な直線探索のほうが関数評価は少ない、とある。[NW2006, §6.1]
- このリポジトリの `strongWolfeLineSearch` と `BfgsOptions` は、すでにその既定値と「初回 \(\alpha=1\)」を実装している。

Strong Wolfe の対価: 各試行で勾配が要る。縮約空間では勾配 1 回 = 実現可能復元 + 偏微分 + 随伴 1 回。Giles も、準ニュートンは最急降下より精度の高い状態・随伴を要求すると注意している。[Giles2000, §2.6]

Armijo だけのまま BFGS を回す選択もあるが、そのときは \(s^\top y\le 0\) を別途処理する必要がある（更新スキップ、または Powell 減衰）。Powell 減衰はもともと **SQP の Lagrange Hessian**（負曲率がありうる）向けである。[Powell1978][NW2006, Ch. 18]

### 4. 縮約勾配上の \(s,y\) の意味

受理ステップのあと

\[
s=\Delta p,\qquad y=\nabla\tilde f(p_+\,)-\nabla\tilde f(p)
\]

と置けば、BFGS は \(\nabla^2\tilde f\) のセカント近似になる。\(\nabla\tilde f\) は射影後の点で計算する。接線予測だけの \(x\) で取った勾配は \(\tilde f\) の勾配ではない。

全空間 SQP で近似するのは \(Z^\top\nabla^2\mathcal L\,Z\) である。[ByrdNocedal1990][NW2006, §18.3] 多様体上に留まれば、IFT の基底 \(Z=[I;\,\mathrm{d}x/\mathrm{d}p]\) に対してこれは \(\nabla^2\tilde f\) と一致する。numopt-js は \(Z\) を作らず、\(p\) 空間の \((s,y)\) だけを使う。

曲率が弱いとき:

- 現行 `bfgs.ts` は閾値以下で \(H\) を単位行列に戻す。
- SciPy `_minimize_bfgs` は Wolfe 直線探索のあと (6.17) を適用する。\(y^\top s=0\) のとき \(\rho\) を大きな値に置き、更新を捨てない（数値上の応急処置）。[SciPy-BFGS]
- Powell 減衰 \(\bar y=\theta y+(1-\theta)Bs\) は、Lagrange Hessian の更新を生き延びさせる技法。[Powell1978]

(推奨) 第1弾は現行 `bfgs.ts` に合わせ、弱い \(s^\top y\) では更新を捨てて \(H\leftarrow I\)。Powell 減衰は全空間 SQP をやるまで入れない。

### 5. dense BFGS を先に出す理由

- L-BFGS は「大きい \(n\) で dense \(H\) が置けない」ための制限メモリ法である。[LiuNocedal1989]
- Adjoint の売りは「\(p\) が小さく \(x\) が大きい」（README、`adjoint-reduced-space`）。メモリ律速になるのは \(x\) 側の制約ヤコビアンであり、\(p\times p\) の \(H\) ではない。
- dolfin-adjoint が L-BFGS-B を既定にするのは、制御変数 \(m\) が関数空間で大きいからである。[dolfin-adjoint-opt]
- 既存の無制約側も、小さい \(n\) 向け dense `bfgs` と大きい \(n\) 向け `lbfgs` を分けている。

### 6. 公開 `bfgs()` をラップしてはいけない理由

dolfin-adjoint が SciPy に \(\tilde J\) を渡せるのは、テープ再生が「\(m\) だけの関数」を本当に提供するからである。[dolfin-adjoint-opt]

このリポジトリには AD が無い。`bfgs(cost, grad)` は \(x\) を知らない。ラップで擬似的に \(\tilde f\) を作ることはできるが、次が欠ける。

- 初期の正方性検査と射影失敗（`adjointGradientDescent` は射影不能なら throw / 停止する）
- `finalStates` / `finalConstraintNorm`
- `regularization` 付きの \(\partial c/\partial x\) 求解
- 試行点が射影不能なときの扱い（`+∞` コスト。Strong Wolfe に偽の勾配を渡してはならない）

Adjoint GD が `gradientDescent()` を呼んでいないのと同じ理由で、Adjoint BFGS も公開 `bfgs()` を呼ばない。共有するのは更新式と直線探索だけである。

凍結ヤコビアンで試行コストだけを近似する、現在の Adjoint GD ラッパは Armijo 用の近道である。BFGS の \(y\) をそれで作るとセカントが壊れる。

### 7. 残差問題では Constrained GN/LM が先

\(f=\tfrac12\|r\|^2\) の Hessian は \(J^\top J+\sum r_i\nabla^2 r_i\)。Gauss–Newton は第一項目だけを使う。小残差・ほぼ線形ならこれで十分で、二次微分は不要。[NW2006, §10.3]

大残差では第二項が無視できず、準ニュートンや hybrid が議論される。[NW2006, §10.3 Methods for Large-Residual Problems]

このリポジトリはすでに縮約 \(J_{\mathrm{eff}}\) 上の Constrained GN/LM を持っている。論文ベンチも Circle で Adjoint GD より GN/LM の反復が少ないことを仮説にしている。したがって

- スカラー \(f(p,x)\) → Adjoint BFGS
- 残差 \(r(p,x)\) → Constrained GN/LM
- 残差を Adjoint BFGS に渡すのは「構造を捨てた準ニュートン」であり、API 互換のための副次経路

### 8. なぜ Adjoint 列は縮約空間のままか

ROL の対比 [ROL-slides]:

- 縮約空間: 各反復で実現可能。\(S(z)\) が無い \(z\) では目的が定義されない。微分のたびに状態・随伴を解く。
- 全空間: \((u,z)\) を同時に動かす。\(S(z)\) が未定義でも問題は置ける。微分に非線形求解は不要。

numopt-js の Adjoint は前者を選んでいる。全空間の等式は Constrained GN/LM（残差）と、将来やるなら SQP の話である。Adjoint BFGS で全空間に乗り換えない。

Giles の one-shot（状態・随伴を収束させずに設計変数を動かす）は、このライブラリの「受理点で \(\|c\|\) を見る」契約と衝突する。第1弾ではやらない。[Giles2000, §2.6]

## numopt-js への写像

| 論点 | 選択 | 理由 | 根拠 |
|------|------|------|------|
| 算法の中身 | 標準 BFGS (6.17) を \(\tilde f(p)\) に適用 | 新しい公式は資料に無い | [NW2006][Giles2000][dolfin-adjoint-opt] |
| 公開入口 | `adjointBfgs`（専用） | 状態復元・結果型・正方契約を Adjoint GD と揃える | 現行 `adjointGradientDescent` |
| 共有 | 逆 Hessian 更新、Strong Wolfe、`constrainedUtils` | 二重実装しない | 現行 `bfgs.ts` / `lineSearch.ts` |
| 直線探索 | Strong Wolfe、初回 \(\alpha=1\)、\(c_1=10^{-4}\)、\(c_2=0.9\) | 曲率と超一次の前提。既存 BFGS と同じ | [NW2006, §6.1] |
| 試行点の勾配 | 射影 + 再偏微分 + 随伴 | \(y\) を本物の \(\nabla\tilde f\) 差分にする | [NW2006, (6.7)] + 現行射影 |
| 射影失敗 | コスト \(+\infty\)、その点の勾配は Wolfe / \(y\) に使わない | \(\tilde f\) が未定義 | [ROL-slides] の \(S(z)\) |
| 弱い \(s^\top y\) | 更新スキップ、\(H\leftarrow I\) | 現行 `bfgs.ts`。Powell は SQP 用 | [NW2006][Powell1978] |
| 初期 \(H_0\) | 単位行列（現行 `bfgs`） | 第1弾は既存に合わせる。(6.20) スケールは後で | [NW2006, (6.20)] は任意改善 |
| 第1弾の密度 | dense BFGS | \(p\) が小さい | README / Liu-Nocedal の前提の裏 |
| 残差 | 受けてもよいが文書では GN/LM を先に | 構造化 Hessian | [NW2006, §10.3] |
| 正方制約 | Adjoint GD と同じ検査 | IFT で \(x(p)\) が一意 | 現行 assert |
| 不等式・AD・疎・信頼領域 | やらない | README Out of Scope | README |

## 実装方針（規範）

実装者が再議論しなくてよい規則。

1. 公開関数名は `adjointBfgs`。入力は `adjointGradientDescent` と同じ（初期 \(p,x\)、コストまたは残差、制約、偏微分オプション、`regularization`、`constraintTolerance`）。BFGS 側は `useLineSearch`（既定 true）、`lineSearchOptions`（Strong Wolfe）、固定 `stepSize`。
2. 結果型は `AdjointGradientDescentResult` を流用するか、同じフィールド（`finalParameters`, `finalStates`, `finalConstraintNorm`, `finalGradientNorm`, `usedLineSearch`）を持つ。
3. 初期化は Adjoint GD と同じ: 正方検査、初期射影。射影不能なら throw。
4. 反復の探索方向は \(p=-H\nabla\tilde f\)。降下方向でなければ \(H\leftarrow I\)、\(p=-\nabla\tilde f\)（現行 `bfgs.ts`）。
5. 直線探索の `cost` / `grad` クロージャは、試行 \(p\) ごとに (接線予測) → 射影 → 失敗なら \(+\infty\) / 勾配なし → 成功ならその点で随伴勾配、とする。Adjoint GD の「開始点勾配を失敗時に使い回す」は Strong Wolfe では禁止。
6. 受理ステップのあと、新しい点で偏微分と随伴を取り直して \(y\) を作る。予測子の \(x\) で \(y\) を作らない。
7. \(s^\top y\le\varepsilon\)（現行 `MINIMUM_CURVATURE_THRESHOLD`）なら \(H\leftarrow I\)。それ以外は (6.17)。
8. 収束判定は Adjoint GD と同じ: \(\|\nabla\tilde f\|\) とステップノルム、かつ \(\|c\|\le\) 許容。
9. \(\partial c/\partial x\) の正則化は既存 `regularization` を随伴と射影にそのまま渡す。
10. 残差入力時のコストは現行どおり \(f=\tfrac12\|r\|^2\)、偏微分は \(r^\top\partial r/\partial(\cdot)\)。新しい最小二乗 Hessian はここで作らない。
11. `index.ts` と結果フォーマッタに載せる。README のアルゴリズム表に「スカラーコスト + 等式」の行へ BFGS を足す。
12. 第1弾でやらない: `adjointLbfgs`、Powell 減衰、信頼領域、one-shot、全空間 SQP、縮約 Hessian-vector（ROL の2回随伴）、不等式。

## 推奨 API とモジュール境界

```
adjointBfgs                    公開。ループと状態を所有する
  ├ constrainedUtils           随伴・予測・射影（既存）
  ├ adjoint 偏微分ヘルパ       adjointGradientDescent から抽出してよい
  ├ strongWolfeLineSearch      既存
  └ updateInverseHessianApproximation  bfgs.ts から共有
```

抽出してよいもの: 目的の種類判定、`computeAdjointGradient`、実現可能復元。Adjoint GD と BFGS でコピーしない。

抽出してはいけないもの: Adjoint GD の Armijo 用「凍結ヤコビアン勾配ラッパ」を BFGS に使い回すこと。

内部に `createReducedObjective(p, x, …)` を置き、\(\tilde f\) と \(\nabla\tilde f\) だけを返す形にするとテストしやすい。公開の `bfgs()` には渡さない。

## 検証方針

終わったときにできること（二次等式、非線形円、陰的鎖、残差互換）を、単体ではなくシナリオテストで固定する。詳細なスライス順は実装計画に置く。CI は `npm test` だけが総合相当を回す。`benchmark:paper` のフルスイートは CI に無いので、受け入れの代替にしない。

単体・契約（`test/adjointBfgs.test.ts`）:

- 既存 Adjoint GD の単純問題 \(f=p^2+x^2,\ c=p+x-1\)。解 \((0.5,0.5)\)。解析偏微分と差分の両方。
- 非正方制約は throw。初期が射影不能なら throw。
- `regularization` を付けても単純問題が解ける。
- 直線探索中の射影失敗で偽収束しない（`converged === false`、開始点のまま）。
- \(s^\top y\) が弱い経路で \(H\) がリセットされても例外にならない。

総合（`test/adjointBfgs.scenarios.test.ts`、公開 `adjointBfgs` 経由）:

- 非線形円 \(f=(p-1)^2+(x-1)^2,\ c=p^2+x^2-2\)。実現可能に \((1,1)\)。同じ開始で反復数 < Adjoint GD。
- 陰的鎖（\(p\) が1、状態が多数）。解析解 \(p=1/n\)。決定変数は1。同じ開始で反復数 < Adjoint GD。
- 残差 LS（`test/fixtures/constrainedLeastSquares.ts`）。収束する。Constrained GN より少なくなくてよい。
- `src/index.ts` からの import で少なくとも一つの旗艦問題が解ける。

例（CI 外、実装後に実行）:

- `examples/constrained-compare.ts` と `examples/adjoint-reduced-space.ts` に BFGS を足す。決定変数は1のまま、鎖では反復が GD より少ないことを示す。

ベンチ（配線は第1弾に含める。フルスイート実行は検証であり CI 必須ではない）:

- `benchmark:paper` に Adjoint BFGS を Circle / Chain へ足す。成功判定は現行どおりパラメータ誤差と \(\|c\|\)。
- 仮説: Circle / Chain で Adjoint BFGS の反復中央値が Adjoint GD 未満。残差 Circle では Constrained GN/LM より少なくなくてよい。
- `test/paperBenchmark.test.ts` の fixture 行を更新する。これがハーネス単体であり、総合の代替ではない。

## 読まなくてよいもの / 将来

第1弾の対象外:

- 連続随伴・境界条件の随伴（Jameson / Giles の PDE 本体）
- one-shot / 不正確随伴 [Giles2000]
- 全空間 SQP、メリット関数、二次補正 [NW2006, Ch. 18][ByrdNocedal1990]
- Powell 減衰 [Powell1978]
- 信頼領域 L-BFGS（ROL Lin-More）
- 縮約 Hessian-vector の Newton–CG（ROL の数値レシピ）
- hybrid Gauss–Newton–BFGS（大残差専用）
- 不等式、自動微分、疎行列（README）

将来の分岐:

- \(p\) が大きい → `adjointLbfgs`（two-loop、[LiuNocedal1989]、既存 `lbfgs.ts`）
- 大残差の構造化更新 → 既存 Constrained LM を先に強化する
- \(S(p)\) が頻繁に死ぬ問題 → そのときはじめて全空間を検討する [ROL-slides]

## 読む順番（実装者）

必須:

1. `src/core/adjointGradientDescent.ts` と `src/core/bfgs.ts`（このリポジトリの現状）
2. Nocedal & Wright 第2版、第3章（Wolfe / Algorithm 3.5）と第6章（(6.7), (6.17), Algorithm 6.1, 初回 \(\alpha=1\)）[NW2006]
3. この文書の「実装方針」節

次:

4. Giles & Pierce §2.6（随伴勾配のあと BFGS、精度の要求）[Giles2000]
5. dolfin-adjoint 「reduced functional」ページ（縮約してから汎用ソルバ）[dolfin-adjoint-opt]
6. ROL スライドの reduced vs full space [ROL-slides]

残差や「なぜ SQP ではないか」が必要になったら:

7. Nocedal 第10章 §10.3 [NW2006]
8. Nocedal 第18章 §18.3（やらないほうの reduced Hessian）[NW2006]
9. Liu & Nocedal 1989（L-BFGS を足すとき）[LiuNocedal1989]
10. Powell 1978（減衰を検討するとき）[Powell1978]

## 文献

1. [NW2006] Jorge Nocedal, Stephen J. Wright. *Numerical Optimization*, 2nd ed. Springer, 2006. DOI: [10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5). 特に Ch. 3, 6, 10.3, 12 (IFT), 18.3。README 既存引用。
2. [Giles2000] Michael B. Giles, Niles A. Pierce. “An introduction to the adjoint approach to design.” *Flow, Turbulence and Combustion* 65 (2000), 393–415. DOI: [10.1023/A:1011430410075](https://doi.org/10.1023/A:1011430410075). 著者 PDF: [https://people.maths.ox.ac.uk/~gilesm/files/ftc00.pdf](https://people.maths.ox.ac.uk/~gilesm/files/ftc00.pdf)
3. [Jameson1988] Antony Jameson. “Aerodynamic design via control theory.” *Journal of Scientific Computing* 3 (1988), 233–260. DOI: [10.1007/BF01061285](https://doi.org/10.1007/BF01061285). NASA PDF: [https://ntrs.nasa.gov/citations/19890004037](https://ntrs.nasa.gov/citations/19890004037)
4. [LiuNocedal1989] Dong C. Liu, Jorge Nocedal. “On the limited memory BFGS method for large scale optimization.” *Mathematical Programming* 45 (1989), 503–528. DOI: [10.1007/BF01589116](https://doi.org/10.1007/BF01589116)
5. [ByrdNocedal1990] Richard H. Byrd, Jorge Nocedal. “An analysis of reduced Hessian methods for constrained optimization.” *Mathematical Programming* 49 (1990), 285–323. DOI: [10.1007/BF01588794](https://doi.org/10.1007/BF01588794)
6. [Powell1978] M. J. D. Powell. “Algorithms for nonlinear constraints that use Lagrangian functions.” *Mathematical Programming* 14 (1978), 224–248. DOI: [10.1007/BF01588967](https://doi.org/10.1007/BF01588967). 減衰 BFGSの出典。同時代の計算手順は LNM 630（README の Moré LM と同じ巻）にもある。
7. [ROL-slides] Sandia National Laboratories. Rapid Optimization Library 概要スライド. [https://trilinos.github.io/pdfs/ROL.pdf](https://trilinos.github.io/pdfs/ROL.pdf). 公式 docs: [https://trilinos.github.io/docs/rol/index.html](https://trilinos.github.io/docs/rol/index.html)
8. [dolfin-adjoint-opt] dolfin-adjoint. “PDE-constrained optimisation.” [https://www.dolfin-adjoint.org/en/latest/documentation/optimisation.html](https://www.dolfin-adjoint.org/en/latest/documentation/optimisation.html). 実装: [pyadjoint/optimization/optimization.py](https://github.com/dolfin-adjoint/pyadjoint/blob/master/pyadjoint/optimization/optimization.py)
9. [SciPy-BFGS] SciPy `scipy.optimize._minimize_bfgs`. 参照実装（逆 Hessian (6.17)、Wolfe 直線探索）. [https://github.com/scipy/scipy/blob/v1.11.4/scipy/optimize/_optimize.py](https://github.com/scipy/scipy/blob/v1.11.4/scipy/optimize/_optimize.py)
10. [Heinkenschloss2008] Matthias Heinkenschloss. “PDE Constrained Optimization.” SIAM Optimization 2008 tutorial. [https://archive.siam.org/meetings/op08/Heinkenschloss.pdf](https://archive.siam.org/meetings/op08/Heinkenschloss.pdf)
