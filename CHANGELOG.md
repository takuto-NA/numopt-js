# Changelog

## [Unreleased]

- Keep GN/LM normal equations unchanged across damping retries, including unconstrained LM. Attempt undamped GN before ridge fallback.
- Return zero from exhausted Wolfe/Armijo searches instead of an unverified step. Adjoint BFGS now distinguishes verified gradient convergence from step stagnation and exposes `stopReason` plus a separate relative `stepTolerance`.
- Add multistart four-bar and slider-crank validation, near-toggle sensitivity probes, and inconsistent/infeasible-state checks (`benchmark:mechanical-character:robustness`).
- Expose `computeEffectiveJacobian` for BFGS/custom reduced objectives, including consistent redundant constraints. Reuse a single LU/QR solve across parameter columns and validate partial-derivative dimensions and finite results.
- Use relative curvature checks in BFGS/L-BFGS so small positive secant pairs remain usable; skip invalid dense updates without discarding the previous model. Add objective-scaling regressions and synthetic mechanical-character fitting examples and benchmarks.
- Add `adjointLbfgs` with shared two-loop recursion, bounded correction history, restored-state adjoint gradients, verified decrease and explicit stagnation/failure reasons. Compare time, peak RSS and convergence with dense BFGS and approximate/full Newton–Adjoint.
- Add process-isolated Newton–Adjoint scaling experiments with common accuracy targets, warm timings, cold OS peak RSS, crossover refinements, and reproducible plots. Include analytical/action/numerical-curvature paths and heterogeneous starts; solver algorithms are unchanged.
- Add `adjointNewtonExact` with all terms of the reduced Hessian, analytical joint Hessian/Hessian-vector callbacks, and numerical projected Lagrangian curvature. Compare accuracy, convergence traces, and cost against approximate Newton and BFGS.
- Try unshifted state/adjoint solves first when `regularization=0`; tighten Newton feasibility accuracy and report cost stagnation without false convergence.
- Add `adjointNewton` with the partial-Hessian approximation `f_pp + S^T f_xx S`, numerical directional curvature, damped directions, and verified Armijo search. Include analytical/numerical comparisons and omitted-curvature counterexamples; quadratic convergence is not guaranteed.
- Keep shared regularized solves from mutating caller-owned Jacobians, and reject non-finite feasibility residuals.
- File-header lint (`npm run lint:headers`) and CODING_RULES cleanup.
- Typed `format*` / `print*` helpers are deprecated; use `formatResult` / `printResult`.
- LM / Constrained LM treat `tolerance` as a fallback when `tolGradient` / `tolStep` / `tolResidual` are omitted.
- Packed-artifact smoke now executes the browser bundle once.
- Shared constrained normal-equation helpers in `constrainedNormalEquations.ts`.

## [0.5.0] - 2026-09-17

- Add reduced-space `adjointBfgs` (standard dense BFGS on \(\tilde f(p)\); gradient is the adjoint).
- Keep adjoint states feasible on nonlinear constraints during line search and accepted steps.
- Add paper-grade public solver benchmark (`npm run benchmark:paper`); success is parameter error, not `result.converged`.
- Docs and CI: TypeDoc GitHub Pages, pack smoke, purge stale demos/scripts.

## [0.4.0] - 2026-02-04

- CMA-ES (vanilla + IPOP) and related examples.

## [0.3.1] - 2026-01-20

- Use official `ml-matrix` types and typed-array-safe copies.

## [0.3.0] - 2026-01-16

- Dense BFGS and L-BFGS.

## [0.2.0] - 2026-12-08

- Earlier public solver set and npm release.
