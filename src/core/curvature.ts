/** Scale-independent secant-pair checks shared by BFGS and L-BFGS updates. */

import { dotProduct } from '../utils/matrix.js';

export const MINIMUM_CURVATURE_COSINE = 1e-10;

function stableNorm(vector: Float64Array): number {
  let norm = 0;
  for (const value of vector) norm = Math.hypot(norm, value);
  return norm;
}

export function hasUsableCurvature(
  step: Float64Array, gradientChange: Float64Array, minimumCosine = MINIMUM_CURVATURE_COSINE
): boolean {
  const curvature = dotProduct(step, gradientChange);
  if (!(curvature > 0) || !Number.isFinite(curvature) || !Number.isFinite(1 / curvature)) return false;
  const stepNorm = stableNorm(step);
  const changeNorm = stableNorm(gradientChange);
  if (!(stepNorm > 0) || !(changeNorm > 0) || !Number.isFinite(stepNorm) || !Number.isFinite(changeNorm)) return false;
  let cosine = 0;
  for (let index = 0; index < step.length; index++) {
    cosine += (step[index] / stepNorm) * (gradientChange[index] / changeNorm);
  }
  return cosine > minimumCosine;
}
