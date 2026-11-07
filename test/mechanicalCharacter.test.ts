/** Checks mechanism derivatives, assembly branches, quadrature, and public-solver fitting against geometry. */

import { Matrix, solve } from 'ml-matrix';
import { describe, expect, it } from 'vitest';
import { adjointBfgs, constrainedLevenbergMarquardt, finiteDiffConstraintPartialP, finiteDiffConstraintPartialX, finiteDiffResidualPartialP,
  finiteDiffResidualPartialX, vectorNorm } from '../src/index';
import { addRedundantConstraints, assembleStates, createFourBarProblem, createReducedEvaluator, createWorkCounters, implicitGradient,
  oracleStates, PARAMETER_COUNT } from '../examples/mechanical-character/four-bar';
import { CONSTRAINT_TOLERANCE, GRADIENT_TOLERANCE, MAXIMUM_DESIGN_ITERATIONS,
  METHODS, runMechanismSolver, verifyMechanismOutcome } from '../examples/mechanical-character/solvers';

const TEST_SAMPLE_COUNT = 16;
const DERIVATIVE_TOLERANCE = 1e-7;
const MATRIX_TOLERANCE = 1e-8;

function maximumDifference(left: Matrix, right: Matrix): number {
  return Math.max(...left.clone().sub(right).to1DArray().map(Math.abs));
}

describe('mechanical character continuous optimization', () => {
  it('assembles the same branch as independent circle intersections over a full cycle', () => {
    const problem = createFourBarProblem(TEST_SAMPLE_COUNT);
    for (const parameters of [problem.initialParameters, problem.targetParameters,
      new Float64Array([4, -4, 4, -4, 2.5, 0.3])]) {
      const numerical = assembleStates(parameters, problem.phases);
      const exact = oracleStates(parameters, problem.phases);
      expect(vectorNorm(problem.constraint(parameters, numerical))).toBeLessThan(1e-9);
      expect(vectorNorm(Float64Array.from(numerical, (value, index) => value - exact[index]))).toBeLessThan(1e-9);
    }
  });

  it('matches all analytical partials to finite differences with the other argument held fixed', () => {
    const problem = createFourBarProblem(TEST_SAMPLE_COUNT);
    const parameters = problem.initialParameters;
    const states = oracleStates(parameters, problem.phases);
    const pairs = [
      [problem.derivatives.dcdp(parameters, states), finiteDiffConstraintPartialP(parameters, states, problem.constraint)],
      [problem.derivatives.dcdx(parameters, states), finiteDiffConstraintPartialX(parameters, states, problem.constraint)],
      [problem.derivatives.drdp(parameters, states), finiteDiffResidualPartialP(parameters, states, problem.residual)],
      [problem.derivatives.drdx(parameters, states), finiteDiffResidualPartialX(parameters, states, problem.residual)]
    ];
    for (const [analytical, numerical] of pairs) expect(maximumDifference(analytical, numerical)).toBeLessThan(MATRIX_TOLERANCE);
  });

  it('matches paper sensitivities, adjoints, and finite differences of fully reassembled costs', () => {
    const problem = createFourBarProblem(TEST_SAMPLE_COUNT, true);
    const parameters = problem.initialParameters;
    const states = oracleStates(parameters, problem.phases);
    const reference = implicitGradient(problem, parameters, states, 'analytic');
    const forwardGradients = ['paper', 'public-ift', 'redundant-ift'].map(mode =>
      createReducedEvaluator(problem, mode as 'paper' | 'public-ift' | 'redundant-ift').gradient(parameters));
    const numerical = createReducedEvaluator(problem, 'finite-difference').gradient(parameters);
    const multiplier = solve(problem.derivatives.dcdx(parameters, states).transpose(),
      Matrix.columnVector(problem.costPartials.dfdx(parameters, states)));
    const adjoint = Matrix.columnVector(problem.costPartials.dfdp(parameters, states))
      .sub(problem.derivatives.dcdp(parameters, states).transpose().mmul(multiplier)).to1DArray();
    for (const candidate of [...forwardGradients, numerical, adjoint]) {
      expect(vectorNorm(Float64Array.from(reference, (value, index) => value - candidate[index])))
        .toBeLessThan(DERIVATIVE_TOLERANCE);
    }
  });

  it('reuses one assembly for IFT while full central differences reassemble 2 * parameterCount times', () => {
    const problem = createFourBarProblem(TEST_SAMPLE_COUNT);
    for (const mode of ['paper', 'finite-difference'] as const) {
      const counters = createWorkCounters();
      const evaluator = createReducedEvaluator(problem, mode, counters);
      evaluator.gradient(problem.initialParameters);
      expect(counters.assemblies).toBe(TEST_SAMPLE_COUNT * (mode === 'paper' ? 1 : 2 * PARAMETER_COUNT));
    }
  });

  it('normalizes the objective as a phase integral rather than an unscaled sample sum', () => {
    const sparse = createFourBarProblem(TEST_SAMPLE_COUNT);
    const dense = createFourBarProblem(TEST_SAMPLE_COUNT * 2);
    const evaluate = (problem: ReturnType<typeof createFourBarProblem>) =>
      problem.cost(problem.initialParameters, oracleStates(problem.initialParameters, problem.phases));
    expect(evaluate(sparse)).toBeCloseTo(evaluate(dense), 9);
  });

  it('handles consistent redundant constraints through LM, while adjoint BFGS explicitly rejects them', () => {
    const problem = createFourBarProblem(TEST_SAMPLE_COUNT);
    const parameters = problem.initialParameters;
    const states = assembleStates(parameters, problem.phases);
    const redundantConstraint = (trial: Float64Array, angles: Float64Array): Float64Array => {
      const base = problem.constraint(trial, angles);
      return new Float64Array([...base, ...base.map(value => 2 * value)]);
    };
    const stack = (matrix: Matrix): Matrix => new Matrix([...matrix.to2DArray(), ...matrix.clone().mul(2).to2DArray()]);
    const options = {
      ...problem.derivatives,
      dcdp: (trial: Float64Array, angles: Float64Array) => stack(problem.derivatives.dcdp(trial, angles)),
      dcdx: (trial: Float64Array, angles: Float64Array) => stack(problem.derivatives.dcdx(trial, angles)),
      constraintTolerance: CONSTRAINT_TOLERANCE, tolGradient: GRADIENT_TOLERANCE,
      tolStep: 1e-12, tolResidual: 1e-12, maxIterations: MAXIMUM_DESIGN_ITERATIONS
    };
    const result = constrainedLevenbergMarquardt(parameters, states, problem.residual, redundantConstraint, options);
    expect(verifyMechanismOutcome(problem, { result, states: result.finalStates, counters: createWorkCounters(), history: [] }, false).passed)
      .toBe(true);
    expect(() => adjointBfgs(parameters, states, problem.cost, redundantConstraint, { ...options, ...problem.costPartials }))
      .toThrow('requires a square implicit-state system');
  });

  it.each(METHODS)('%s returns a feasible trajectory with independently verified cost reduction', method => {
    const problem = createFourBarProblem(TEST_SAMPLE_COUNT);
    const outcome = runMechanismSolver(problem, method);
    const verification = verifyMechanismOutcome(problem, outcome, false);
    expect(verification.finalCost).toBeLessThan(verification.initialCost / 100);
    expect(verification.constraintNorm).toBeLessThan(1e-9);
    expect(verification.stateCurveError).toBeLessThan(1e-8);
    expect(verification.passed).toBe(true);
    if (method === 'BFGS + redundant IFT') {
      expect(vectorNorm(addRedundantConstraints(problem.constraint(outcome.result.finalParameters, outcome.states))))
        .toBeLessThan(CONSTRAINT_TOLERANCE);
    }
  });

  it.each(METHODS)('%s reaches a feasible stationary fit with a nonzero residual', method => {
    const problem = createFourBarProblem(TEST_SAMPLE_COUNT, true);
    const outcome = runMechanismSolver(problem, method);
    expect(verifyMechanismOutcome(problem, outcome, true).passed).toBe(true);
  });
});
