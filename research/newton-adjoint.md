# Newton–Adjoint with the section-12 approximation

This document records the mathematical contract, safeguards, and evaluation protocol for `adjointNewton`. The implementation follows the simplified Hessian in [section 12 of the requested article](https://qiita.com/naturemon/items/df0242e8a2e5814503f5), rather than silently replacing it with BFGS, Gauss–Newton, or an exact reduced Hessian.

## Model and missing terms

For a square, locally nonsingular state system c(p,x)=0, define x(p), the reduced cost f̃(p)=f(p,x(p)), and S=dx/dp. The solver evaluates

\[
c_x^T\lambda=f_x,\qquad g=f_p-c_p^T\lambda,\qquad
S=-c_x^{-1}c_p,\qquad \widehat H=f_{pp}+S^Tf_{xx}S.
\]

Gradients are column vectors; f_pp and f_xx are partial Hessians with the other input held fixed. Residual input means f=1/2 ||r||², including all curvature of this scalar f within each block. It does not mean replacing the partial Hessians with residual normal equations.

The exact reduced Hessian is

\[
\nabla^2\tilde f=f_{pp}+f_{px}S+S^Tf_{xp}+S^Tf_{xx}S
+\sum_i f_{x_i}\nabla_p^2x_i.
\]

The implementation omits both mixed terms and the final state-curvature term. This is a substantive approximation, even with exact analytical partial Hessians. It can be indefinite or singular even when the reduced objective has a well-defined minimum.

The article's discussion of quadratic convergence is conditional. For a smooth problem with unit steps and a positive definite limiting model, the local error satisfies

\[
e_{k+1}=[I-\widehat H_*^{-1}\nabla^2\tilde f(p_*)]e_k+O(\|e_k\|^2).
\]

A persistent Hessian error generally leaves a linear term. Affine states and a separable objective eliminate the omitted terms; vanishing omitted terms near the solution can also give Newton-like local behavior. Fixed finite-difference, projection, and regularization errors eventually limit accuracy. A line search and positive definiteness alone do not establish quadratic or unconditional global convergence.

## Explicit counterexamples

Mixed partials: let c=x-p and f=1/2(p-x)²+1/2(p-1)². On the feasible set, f̃=1/2(p-1)² and the exact Hessian is 1, while f_pp=2, f_xx=1, S=1, and Ĥ=3. Unit Newton-adjoint steps satisfy e_next=(2/3)e. This is linear convergence, despite a convex quadratic cost and a linear constraint. BFGS and GD can solve this particular reduced quadratic in a single step.

State curvature: let c=x-p² and f=1/2(p-1)²+x. Then f̃=3/2 p²-p+1/2, with minimizer p=1/3 and exact Hessian 3. The simplified model is 1 because f_xx=0. Full steps overshoot; Armijo backtracking accepts a half step in the tested neighborhood, giving an error factor of -1/2. The missing term is f_x x_pp=2.

These are regression cases, not failures to hide or tune away. Damping cannot recreate the omitted terms.

## Numerical implementation

- Common reduced-space initialization checks square dimensions and restores the initial states. Unprojectable initial states throw. Trial states use tangent prediction plus Newton correction; failed restorations have infinite trial cost.
- The adjoint and feasibility routines are shared with the existing solvers. The original approximate-version commit (`63aafde`) included an automatic shift even when `regularization` was zero, which could require an extra correction on an exact quadratic. The full-Newton branch corrects this for every solver: try the unshifted system first and regularize on failure. Sensitivities use the requested regularization and otherwise solve the unshifted system.
- S is solved with a matrix right-hand side so the dense factorization is reused across parameter columns. No inverse is formed. Sensitivities require O(stateCount × parameterCount) storage; dense factorization remains cubic in the state dimension.
- Analytical `d2fdp2` and `d2fdx2` must have the expected dimensions, finite entries, and numerical symmetry. Inputs are cloned before any matrix operation that mutates its receiver.
- Without analytical blocks, the parameter Hessian uses central second differences, and state curvature is evaluated directly along columns of S at fixed p. This is O(parameterCount²) objective samples instead of O(stateCount²); no dense f_xx is allocated. It still requires a dense state Jacobian and sensitivities. Coordinate-relative step scaling avoids using one large coordinate to set perturbations for all others.
- Numerical partial evaluations are intentionally off the constraint manifold: restoring them would differentiate the reduced objective and introduce terms absent from the requested approximation. The objective must be defined in those neighborhoods, as with existing numerical partial derivatives. Non-finite curvature samples produce an actionable error.
- A Cholesky solve first tries the requested minimum Hessian damping (zero by default). If necessary, a Gershgorin lower bound supplies a positive-definite shift, followed by bounded retries. A finite descent direction is verified; otherwise the direction falls back to -g. These safeguards do not add missing physical curvature.
- Armijo backtracking starts at a unit step and evaluates cost only, avoiding unnecessary trial adjoints. Cached feasible states and cost are reused at acceptance. The shared backtracker returns a step even after exhausting its budget, so this solver explicitly rechecks sufficient decrease before accepting it.
- Convergence requires both a small finite reduced-gradient norm and feasible states. Tiny steps with a large gradient report `stepStagnation` and `converged=false`; repeated unchanged costs report `costStagnation`. Search exhaustion and iteration limits have separate stop reasons. Newton restoration accuracy is tightened with the gradient tolerance to avoid feasibility error overwhelming the predicted decrease.
- Shared linear solves clone the coefficient matrix before diagonal shifts. This prevents cumulative shifts across retries and mutation of analytical Jacobians retained by callers. Non-finite constraint residuals also fail restoration instead of passing an ordered comparison with NaN.

## Validation and performance protocol

Run:

```sh
npm run typecheck
npm test -- --maxWorkers=2 --minWorkers=1
npm run lint:headers
npm run build
npm run example:adjoint-newton
npm run benchmark:adjoint-newton
```

Tests independently check the mathematical approximation on a coupled two-parameter problem; analytical, numerical, and mixed block inputs; objective-sample scaling with state count; immutable user matrices; damping and descent fallback; residual input; feasibility failures; exhausted line search; invalid options; and strict stationarity versus stagnation. The counterexamples check the expected linear contraction and slower convergence than BFGS.

The comparison includes an anisotropic affine quadratic, a 20-state chain, a nonlinear circle, and both counterexamples. All scalar methods receive the same first derivatives in analytical mode; Newton-adjoint additionally receives exact partial Hessians. Numerical mode supplies no derivatives. GN/LM are also compared in numerical mode on residual problems.

One warmup run is discarded, then seven runs are retained. The report contains median/IQR elapsed time, objective/residual calls, constraint calls, first/second partial callback counts, parameter error, feasibility, solver gradient norms, and stop reasons. Success depends on parameter error and feasibility, independently of `converged`. Gradient norms are retained because older solvers can stop on a small step at different stationarity tolerances. Timing is secondary for these small dense problems and is not a CI assertion. Callback counts do not account for arbitrary internal work in user derivatives.

Reports are written to gitignored `benchmark-results/adjoint-newton.md` and `.json`. Expected qualitative findings: the affine quadratic and circle benefit relative to BFGS; the chain can match BFGS iteration counts while avoiding its repeated trial adjoints; the two omitted-curvature problems are substantially worse. Constrained GN/LM remain attractive for the residual examples. No blanket speedup claim or replacement of existing defaults follows from this suite.

## Background

- [Requested article, sections 11–12](https://qiita.com/naturemon/items/df0242e8a2e5814503f5): full reduced second derivative and the requested simplified model.
- [Giles and Pierce, “An Introduction to the Adjoint Approach to Design”, sections 2.6–2.7](https://people.maths.ox.ac.uk/~gilesm/files/ftc00.pdf): derivative accuracy versus per-iteration work, and direct sensitivities for least squares.
- [Sandia Rapid Optimization Library overview](https://trilinos.github.io/pdfs/ROL.pdf): reduced-space state/adjoint computations and exact Hessian-vector alternatives. Those exact second-order adjoints are outside this implementation.
