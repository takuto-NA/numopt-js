# Full reduced Newton–Adjoint

The full implementation is `adjointNewtonExact`. `adjointNewton` retains the article's section-12 partial-Hessian approximation, and `adjointBfgs` remains the quasi-Newton alternative. This branch starts from the approximate implementation at commit `63aafde`; the original approximate branch remains available.

## Exact Hessian without a second-state-sensitivity tensor

Take column gradients, c(p,x)=0, and a square nonsingular C_x. Define

\[
C_x^T\lambda=f_x,\quad S=-C_x^{-1}C_p,\quad
L(p,x,\lambda)=f(p,x)-\lambda^Tc(p,x),\quad Z=[I;S].
\]

The full reduced Hessian is

\[
H=Z^TL_{zz}Z=L_{pp}+L_{px}S+S^TL_{xp}+S^TL_{xx}S.
\]

To see where the state second derivative went, for any parameter directions u,v differentiate the constraint twice:

\[
C_x x_{pp}[u,v]=-c_{zz}[Zu,Zv].
\]

Consequently f_x^T x_pp[u,v] = -lambda^T c_zz[Zu,Zv]. Therefore

\[
H=f_{pp}+f_{px}S+S^Tf_{xp}+S^Tf_{xx}S
+\sum_i f_{x_i}\nabla^2_p x_i.
\]

All terms are present. Contracting constraint curvature with lambda avoids constructing or solving for the entire stateCount × parameterCount × parameterCount tensor. This is the finite-dimensional projected Lagrangian formulation of the reduced Hessian; [Sandia's ROL presentation, pages 43–45](https://trilinos.github.io/pdfs/ROL.pdf) gives the corresponding full reduced-Hessian and Hessian-vector formulas (with the opposite multiplier sign convention).

## Derivative interfaces

- `lagrangianHessian(p,x,lambda)` returns L_zz, with joint coordinates ordered `[p,x]`, lambda held fixed, and the sign L=f-lambda^T c. It must be finite, symmetric, and dimensionally correct. Supplying f_zz alone omits constraint/state curvature.
- `lagrangianHessianVector(p,x,lambda,direction)` is an alternative action callback. Each tangent basis column is applied once; the resulting products are projected with Z^T. This avoids the joint dense Hessian. The projected operator must be symmetric. At most one analytical curvature callback is allowed.
- With neither callback, central second differences evaluate L(p+dp,x+S dp,lambda). Lambda and S are fixed for one model. No feasibility restoration is performed inside these derivative samples; the Lagrangian correction requires the off-manifold constraint values. Restoring samples would differentiate a different expression. `hessianStepSize` controls coordinate-relative perturbations.
- First derivatives and residual input follow the existing adjoint contracts. For residuals, f=1/2 ||r||². “Exact” refers to the inclusion of every mathematical Hessian term, not exact arithmetic or exact finite differences.

`runAdjointNewton` owns one shared safeguarded iteration for both Newton models. This keeps initialization, damping, line search, acceptance and termination identical while changing only the Hessian construction. The adjoint variable is returned by the shared reduced-gradient evaluator and reused by the full model.

## Precision and safeguards

An unconditional 1e-8 shift in the existing square state/adjoint solve biased otherwise nonsingular problems. `regularization=0` now first tries an unshifted system; failed singular solves still retry with regularization. This correction applies to all three methods in the comparison. A positive explicit regularization still changes the computed sensitivities/gradient, and can limit Newton accuracy.

Both Newton variants restore states to min(user constraint tolerance, max(100 eps max(1,||x_initial||), gradient tolerance²)). The user tolerance is an upper bound. A loose fixed inner residual can dominate the O(||gradient||²) predicted decrease, especially with a nonzero adjoint. The roundoff scale avoids asking ordinary state solves for accuracy far below representable state values. Difficult scaling can still require rescaling or tuning.

An exact Hessian can be indefinite away from a minimum. Both variants first try zero damping, then a positive-definite shift if necessary, and verify descent. Armijo search starts at a unit step and rejects failed restorations and inadequate decrease. Exact curvature is not replaced by damping in the benchmark; damped-step counts are reported. Small steps or repeated unchanged costs with an insufficiently small gradient stop with converged=false.

Local quadratic convergence requires smoothness, a nonsingular state Jacobian, a positive definite reduced Hessian at the solution, sufficiently accurate derivatives/state/adjoint solves, and eventual unit steps with vanishing damping. These conditions do not give global optimality, nor do they imply less wall time than BFGS. Numerical differences, cancellation in f-lambda^T c, noisy callbacks, ill conditioning, and limited restoration iterations can reduce the observed order or cause stagnation.

## Verification

The independent matrix test uses two nonlinear implicit states, one with p-x constraint curvature and another with x-x curvature. Its reference Hessian is derived from explicit state functions and their first/second sensitivities, rather than from the projected Lagrangian formula. Analytical joint matrices, action callbacks and numerical Lagrangian differences must all match it.

The former omitted-curvature examples have full reduced quadratic objectives. Their analytical complete Newton solves finish in one undamped step. The nonlinear convergence example has c=x-p² and

\[
f=\tfrac14(p-1)^4+\tfrac13(p-1)^3+\tfrac12(p-3)^2+x.
\]

At p*=1 the adjoint is 1, so the state-curvature contribution remains nonzero. With e=p-1, the reduced gradient is e³+e²+3e, the exact Hessian is 3e²+2e+3, and the exact unit-step recurrence is

\[
e_{next}=\frac{e^2+2e^3}{3+2e+3e^2},\qquad e_{next}/e^2\to1/3.
\]

Tests check this quadratic regime directly and require zero damping. The approximate model tends to 1 instead of 3 and needs backtracking; BFGS learns curvature from gradient differences. Additional tests exercise indefinite exact curvature, malformed analytical operators, residual input, unbiased square solves, and singular regularization fallback.

## Comparison protocol and computational cost

Run `npm run benchmark:adjoint-newton-exact`. The report includes the original five problems, the nonlinear convergence example, an 80-state chain, and mixed quadratics with 4 and 16 parameters. Analytical and numerical modes are separate; analytical action callbacks are additionally compared where available. All solvers receive identical first-derivative information. Newton variants receive the appropriate exact second derivatives for their respective models.

One warmup is discarded and seven runs are retained. An independent closed-form reduced gradient and known parameter solution define the common target: gradient norm and parameter error at most 1e-6. Primary iteration counts, evaluation counts and elapsed time are captured at the first callback meeting that target. The solver is allowed to finish so its stricter final state, gradient and termination diagnostics can also be checked. Counts exclude the independent reference gradient; time includes the same monitoring for every solver. Every final state must satisfy the common 1e-12 feasibility bound. The internal solver tolerance is 1e-9. A target success and a solver convergence flag are deliberately reported separately.

The generated Markdown/JSON retains complete traces, final reported and true gradients, solver flags, damping, objective/constraint evaluations, first/second derivative callback counts, median time and IQR. Timing is machine dependent and secondary for these small problems. Different curvature callbacks carry different amounts of information and internal work; a callback count is not a floating-point-operation count.

Let m be the parameter count and n the state count. Both Newton methods solve for S using one dense C_x factorization with m right-hand sides: O(n³+n²m), and O(nm) sensitivity storage. The reduced Hessian requires O(m²) storage and a dense O(m³) direction solve. Analytical joint curvature stores (m+n)² entries and costs O((m+n)²m) to project; action callbacks avoid that joint allocation. Numerical full curvature uses O(m²) objective **and constraint** samples; the approximate numerical model uses O(m²) objective samples for its two blocks. BFGS avoids these explicit sensitivity/curvature constructions but can require more iterations and trial adjoints under Strong Wolfe. Thus expensive constraints or analytical Hessians can outweigh a reduced iteration count. Sparse and matrix-free Newton-CG solvers remain outside this dense implementation.

The report is saved to `benchmark-results/adjoint-newton-exact.md` and `.json` (gitignored). Favor conclusions supported by true-gradient accuracy and measured evaluation counts; do not assume that the complete method always beats the approximate method or BFGS.
