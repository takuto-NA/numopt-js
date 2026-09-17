# Changelog

## [Unreleased]

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
