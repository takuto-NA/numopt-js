/** Numerically assembled costs and phase-block gradients implementing Eqs. (2)-(4), with controlled omissions. */

import { Matrix, solve } from 'ml-matrix';
import { bfgs, computeEffectiveJacobian, finiteDiffGradient, gaussNewton, vectorNorm } from '../../src/index';
import { type PaperMechanism } from './cases';

export const DIFFERENCE_STEP = 1e-5;
export const STATE_TOLERANCE = 1e-13;
export const CLOSURE_TOLERANCE = 1e-10;
export const STATE_DIMENSION = 2;
const MAXIMUM_ASSEMBLY_ITERATIONS = 40;
const MAXIMUM_DESIGN_ITERATIONS = 800;
const DESIGN_GRADIENT_TOLERANCE = 1e-10;
const HALF = 0.5;
export const GRADIENT_METHODS = ['paper-ift', 'full-fd'] as const;
export type PaperGradient = typeof GRADIENT_METHODS[number] | 'analytic-ift' | 'frozen-state' | 'omit-direct';
export type PaperCounters = { assemblies: number; constraints: number };

export function assemblePaperStates(model: PaperMechanism, parameters: Float64Array, phases = model.phases,
  counters: PaperCounters = { assemblies: 0, constraints: 0 }, tolerance: number = STATE_TOLERANCE, requireClosure = true): Float64Array {
  const states = new Float64Array(STATE_DIMENSION * phases.length);
  let previous: Float64Array = new Float64Array([1, 2]);
  phases.forEach((phase, index) => {
    counters.assemblies++;
    const constraint = (angles: Float64Array) => { counters.constraints++; return model.constraint(parameters, angles, phase); };
    const result = gaussNewton(previous, constraint, { jacobian: angles => model.dcdx(parameters, angles),
      maxIterations: MAXIMUM_ASSEMBLY_ITERATIONS, tolerance });
    previous = result.finalParameters;
    const closure = vectorNorm(constraint(previous));
    if (!Number.isFinite(closure) || (requireClosure && closure > Math.max(CLOSURE_TOLERANCE, tolerance))) {
      throw new Error(`Assembly failed: ${model.name} at ${phase}`);
    }
    states.set(previous, STATE_DIMENSION * index);
  });
  return states;
}

export function paperCurve(model: PaperMechanism, parameters: Float64Array, states: Float64Array, phases = model.phases): number[][] {
  return phases.map((phase, index) => Array.from(model.marker(parameters,
    states.subarray(STATE_DIMENSION * index, STATE_DIMENSION * (index + 1)), phase)));
}

export function paperCost(model: PaperMechanism, parameters: Float64Array, states: Float64Array, phases = model.phases): number {
  const target = model.targetAt(phases);
  const curve = paperCurve(model, parameters, states, phases);
  return HALF * curve.reduce((sum, point, index) => sum + point.reduce((subtotal, value, coordinate) =>
    subtotal + (value - target[index][coordinate]) ** 2, 0), 0) / phases.length;
}

/** Analytical reference solves C_s S=-C_p directly; the paper path calls the public residual Jacobian instead. */
export function paperGradientAt(model: PaperMechanism, parameters: Float64Array, states: Float64Array,
  method: Exclude<PaperGradient, 'full-fd'>, counters: PaperCounters = { assemblies: 0, constraints: 0 }, redundant = false): Float64Array {
  const target = model.targetAt(model.phases);
  const gradient = new Float64Array(parameters.length);
  const scale = 1 / Math.sqrt(model.phases.length);
  const stack = (matrix: Matrix) => redundant ? new Matrix([...matrix.to2DArray(), ...matrix.clone().mul(2).to2DArray()]) : matrix;
  model.phases.forEach((phase, index) => {
    const angles = states.subarray(STATE_DIMENSION * index, STATE_DIMENSION * (index + 1));
    const residual = (trial: Float64Array, fixed: Float64Array) => model.marker(trial, fixed, phase)
      .map((value, coordinate) => scale * (value - target[index][coordinate]));
    const direct = model.dmdp(parameters, angles, phase);
    let jacobian: Matrix;
    if (method === 'analytic-ift') {
      const sensitivity = solve(model.dcdx(parameters, angles), model.dcdp(parameters, angles, phase)).mul(-1);
      jacobian = direct.add(model.dmdx(parameters, angles).mmul(sensitivity)).mul(scale);
    } else if (method === 'frozen-state') jacobian = direct.mul(scale);
    else jacobian = computeEffectiveJacobian(parameters, angles, residual, (trial, fixed) => {
      counters.constraints++;
      const values = model.constraint(trial, fixed, phase);
      return redundant ? new Float64Array([...values, ...values.map(value => 2 * value)]) : values;
    }, { drdp: (trial, fixed) => method === 'omit-direct' ? Matrix.zeros(STATE_DIMENSION, parameters.length)
      : model.dmdp(trial, fixed, phase).mul(scale), drdx: (trial, fixed) => model.dmdx(trial, fixed).mul(scale),
      dcdx: (trial, fixed) => stack(model.dcdx(trial, fixed)), stepSizeP: DIFFERENCE_STEP });
    const contribution = jacobian.transpose().mmul(Matrix.columnVector(residual(parameters, angles))).to1DArray();
    contribution.forEach((value, column) => { gradient[column] += value; });
  });
  return gradient;
}

export function createPaperEvaluator(model: PaperMechanism, method: PaperGradient = 'paper-ift', tolerance = STATE_TOLERANCE) {
  const counters: PaperCounters = { assemblies: 0, constraints: 0 };
  // Targets are fixed input. Their independent geometric generation is excluded from optimizer work.
  const targets = model.targetAt(model.phases);
  const activeModel = { ...model, targetAt: (phases: number[]) => phases === model.phases ? targets : model.targetAt(phases) };
  let cachedParameters: Float64Array | undefined;
  let cachedStates: Float64Array;
  const statesAt = (parameters: Float64Array): Float64Array => {
    if (!cachedParameters || !parameters.every((value, index) => value === cachedParameters![index])) {
      cachedStates = assemblePaperStates(activeModel, parameters, activeModel.phases, counters, tolerance);
      cachedParameters = parameters.slice();
    }
    return cachedStates;
  };
  const cost = (parameters: Float64Array) => paperCost(activeModel, parameters, statesAt(parameters));
  const gradient = (parameters: Float64Array) => method === 'full-fd'
    ? finiteDiffGradient(parameters, cost, { stepSize: DIFFERENCE_STEP })
    : paperGradientAt(activeModel, parameters, statesAt(parameters), method, counters);
  return { cost, gradient, statesAt, counters };
}

export function fitPaperMechanism(model: PaperMechanism, method: PaperGradient = 'paper-ift', initial = model.initialParameters) {
  const evaluator = createPaperEvaluator(model, method);
  const history: Array<{ iteration: number; cost: number; parameters: number[] }> = [];
  const result = bfgs(initial, evaluator.cost, evaluator.gradient, { maxIterations: MAXIMUM_DESIGN_ITERATIONS,
    tolerance: DESIGN_GRADIENT_TOLERANCE,
    onIteration: (iteration, cost, parameters) => { history.push({ iteration, cost, parameters: Array.from(parameters) }); } });
  return { result, states: evaluator.statesAt(result.finalParameters), history, counters: evaluator.counters };
}
