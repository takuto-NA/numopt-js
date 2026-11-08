/** Independently finite-difference the normalized physical design box to diagnose tanh saturation. */

import { vectorNorm } from '../../src/index';
import { type PaperMechanism } from './cases';

const DIFFERENCE_STEP = 1e-5;
const SECOND_SAMPLE_OFFSET = 2;
const THREE_POINT_BASE_WEIGHT = -3;
const THREE_POINT_FIRST_WEIGHT = 4;
const THREE_POINT_SECOND_WEIGHT = -1;
const OPEN_BOUND_MARGIN = 8 * Number.EPSILON;
export const BOUNDARY_GRADIENT_TOLERANCE = 1e-6;

/** q_i=tanh(p_i) for bounded dimensions; unbounded marker coordinates are left unchanged. */
export function checkBoundaryStationarity(model: PaperMechanism, parameters: Float64Array,
  exactCost: (parameters: Float64Array) => number) {
  const bounded = new Set(model.boundedParameterIndices);
  const coordinates = parameters.map((value, index) => bounded.has(index) ? Math.tanh(value) : value);
  const expand = (trial: Float64Array) => trial.map((value, index) => bounded.has(index)
    ? Math.atanh(Math.max(-1 + OPEN_BOUND_MARGIN, Math.min(1 - OPEN_BOUND_MARGIN, value))) : value);
  const costAt = (trial: Float64Array) => exactCost(expand(trial));
  const baseCost = costAt(coordinates);
  const gradient = coordinates.map((value, index) => {
    const trialAt = (displacement: number) => { const trial = coordinates.slice(); trial[index] += displacement; return costAt(trial); };
    const nearUpper = bounded.has(index) && value > 1 - SECOND_SAMPLE_OFFSET * DIFFERENCE_STEP;
    const nearLower = bounded.has(index) && value < -1 + SECOND_SAMPLE_OFFSET * DIFFERENCE_STEP;
    if (!nearUpper && !nearLower) return (trialAt(DIFFERENCE_STEP) - trialAt(-DIFFERENCE_STEP)) / (SECOND_SAMPLE_OFFSET * DIFFERENCE_STEP);
    const direction = nearUpper ? -1 : 1;
    return direction * (THREE_POINT_BASE_WEIGHT * baseCost + THREE_POINT_FIRST_WEIGHT * trialAt(direction * DIFFERENCE_STEP)
      + THREE_POINT_SECOND_WEIGHT * trialAt(direction * SECOND_SAMPLE_OFFSET * DIFFERENCE_STEP)) / (SECOND_SAMPLE_OFFSET * DIFFERENCE_STEP);
  });
  const projectedResidual = coordinates.map((value, index) => bounded.has(index)
    ? value - Math.max(-1, Math.min(1, value - gradient[index])) : gradient[index]);
  const projectedGradientNorm = vectorNorm(projectedResidual);
  return { projectedGradientNorm, gradient: Array.from(gradient), passed: Number.isFinite(projectedGradientNorm)
    && projectedGradientNorm <= BOUNDARY_GRADIENT_TOLERANCE,
    interpretation: 'Finite-difference first-order check in the closed normalized physical box; not a global or unconstrained optimum certificate.' };
}
