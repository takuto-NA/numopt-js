/** Multistart geometry checks, near-toggle derivatives, and expected rejection of inconsistent states. */

import { describe, expect, it } from 'vitest';
import { evaluateFourBarInitializations, evaluateMechanismSafety, evaluateSliderInitializations } from '../examples/mechanical-character/robustness';

describe('mechanical-character robustness', () => {
  it('fits reachable and distorted four-bar curves from three initial designs', () => {
    const rows = evaluateFourBarInitializations();
    expect(rows.filter(row => !row.passed)).toEqual([]);
    expect(rows).toHaveLength(18);
  });

  it('fits a slider-crank through implicit BFGS and LM from three initial designs', () => {
    const rows = evaluateSliderInitializations();
    expect(rows.filter(row => !row.passed)).toEqual([]);
    expect(rows).toHaveLength(6);
  });

  it('checks near-toggle sensitivity and distinguishes inconsistent constraints from redundancy', () => {
    const checks = evaluateMechanismSafety();
    expect(checks.filter(check => !check.passed)).toEqual([]);
    expect(checks).toHaveLength(8);
  });
});
