/** Malformed external WASM results must fail at the public LM seam. */
import { Matrix } from 'ml-matrix';
import {
  createKiboLinearSolver, LinearSolverConfigurationError, levenbergMarquardt,
  type PreparedKiboLinearAlgebra
} from '../src/index';

function externalWorkspace(output: Float64Array): PreparedKiboLinearAlgebra {
  return {
    rows: 1, cols: 1, allocatedNumericBytes: 64,
    module: { _kibo_abi_version: () => 1 },
    solveLlt: () => ({ status: 0, output }),
    dispose() {}
  };
}

function runLinearProblem(linearSolver: ReturnType<typeof createKiboLinearSolver>) {
  return levenbergMarquardt(
    new Float64Array([0]),
    parameters => new Float64Array([parameters[0] - 2]),
    { jacobian: () => new Matrix([[1]]), maxIterations: 2, linearSolver }
  );
}

describe('kibo integration failures', () => {
  it('rejects an incompatible ABI before optimization starts', () => {
    const workspace = externalWorkspace(new Float64Array([1]));
    workspace.module._kibo_abi_version = () => 2;
    expect(() => createKiboLinearSolver(workspace)).toThrow(LinearSolverConfigurationError);
  });

  it('rejects a WASM capacity error without using LU', () => {
    const workspace = externalWorkspace(new Float64Array([1]));
    workspace.solveLlt = () => ({ status: 4 });
    const solver = createKiboLinearSolver(workspace);
    try {
      expect(() => runLinearProblem(solver)).toThrow(LinearSolverConfigurationError);
      expect(solver.statistics.fallbackAttempts).toBe(0);
    } finally { solver.dispose(); }
  });
  it.each([new Float64Array([NaN]), new Float64Array([Infinity]), new Float64Array([])])(
    'rejects malformed successful WASM output without reporting convergence', output => {
      const solver = createKiboLinearSolver(externalWorkspace(output));
      try {
        expect(() => runLinearProblem(solver)).toThrow(LinearSolverConfigurationError);
        expect(solver.statistics.fallbackAttempts).toBe(0);
      } finally { solver.dispose(); }
    }
  );

  it('rejects a solver prepared for a different parameter count', () => {
    const workspace = externalWorkspace(new Float64Array([1]));
    const solver = createKiboLinearSolver({ ...workspace, rows: 2, cols: 2 });
    try {
      expect(() => runLinearProblem(solver)).toThrow(LinearSolverConfigurationError);
    } finally { solver.dispose(); }
  });

  it('propagates a failed WASM call instead of retrying the optimizer', () => {
    const workspace = externalWorkspace(new Float64Array([1]));
    workspace.solveLlt = () => { throw new Error('WASM instance failed'); };
    const solver = createKiboLinearSolver(workspace);
    try {
      expect(() => runLinearProblem(solver)).toThrow(LinearSolverConfigurationError);
    } finally { solver.dispose(); }
  });

  it('rejects a missing external result without retrying the optimizer', () => {
    const workspace = externalWorkspace(new Float64Array([1]));
    Object.defineProperty(workspace, 'solveLlt', { value: () => undefined });
    const solver = createKiboLinearSolver(workspace);
    try {
      expect(() => runLinearProblem(solver)).toThrow(LinearSolverConfigurationError);
    } finally { solver.dispose(); }
  });
});
