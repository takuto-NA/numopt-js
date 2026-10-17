/** Generate ignored diagnostic copies of the existing LM modules and kibo facade. */
import fs from 'node:fs';
import path from 'node:path';
import ts from 'typescript';
import { projectRoot, assetDirectory, sha256 } from './kibo-assets.mjs';

export const PROBE_NAME = '__numoptLmPhaseProbe';
const PROBE = `globalThis.${PROBE_NAME}`;
const DAMPING_VARIABLES = ['parameterCount', 'identity', 'dampedHessian', 'negativeJtr'];

function dampingEdit(source, node) {
  const matches = [];
  function visit(candidate) {
    if (ts.isBlock(candidate)) {
      const statements = [...candidate.statements];
      const variableName = statement => ts.isVariableStatement(statement) &&
        statement.declarationList.declarations.length === 1
        ? statement.declarationList.declarations[0].name.getText(source) : undefined;
      for (let index = 0; index <= statements.length - DAMPING_VARIABLES.length; index++) {
        if (!DAMPING_VARIABLES.every((name, offset) => variableName(statements[index + offset]) === name)) continue;
        const start = statements[index].getStart(source);
        const end = statements[index + DAMPING_VARIABLES.length - 1].getEnd();
        matches.push({ start, end, text: `const {dampedHessian, negativeJtr} = ${PROBE}.measure("damping", () => {
          ${source.text.slice(start, end)}
          return {dampedHessian, negativeJtr};
        });` });
      }
    }
    ts.forEachChild(candidate, visit);
  }
  visit(node);
  if (matches.length !== 1) throw new Error('Expected one unchanged LM damping group');
  return matches[0];
}

export function instrumentSource(text, configuration) {
  const source = ts.createSourceFile(configuration.file, text, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS);
  const edits = [], counts = {};
  const count = name => { counts[name] = (counts[name] ?? 0) + 1; };
  function wrapExpression(node, phase) {
    edits.push({ start: node.getStart(source), end: node.getStart(source), text: `${PROBE}.measure("${phase}", () => (` });
    edits.push({ start: node.getEnd(), end: node.getEnd(), text: '))' });
    count(phase);
  }
  function visit(node) {
    const name = node.name?.getText(source);
    const phase = ts.isFunctionDeclaration(node) ? configuration.functions?.[name]
      : ts.isMethodDeclaration(node) ? configuration.methods?.[name] : undefined;
    if (phase && node.body) {
      edits.push({ start: node.body.getStart(source) + 1, end: node.body.getStart(source) + 1,
        text: `${PROBE}.enter("${phase}"); try {` });
      edits.push({ start: node.body.getEnd() - 1, end: node.body.getEnd() - 1,
        text: `} finally { ${PROBE}.exit("${phase}"); }` });
      count(phase);
    }
    if (configuration.damping && ts.isFunctionDeclaration(node) && name === 'tryLevenbergMarquardtStep') {
      edits.push(dampingEdit(source, node)); count('damping');
    }
    if (ts.isNewExpression(node) && configuration.cholesky && node.expression.getText(source) === 'CholeskyDecomposition')
      wrapExpression(node, 'mlFactorization');
    if (ts.isCallExpression(node) && configuration.cholesky && node.expression.getText(source) === 'cholesky.solve')
      wrapExpression(node, 'mlTriangularSolve');
    if (ts.isCallExpression(node) && configuration.wasm &&
        ts.isPropertyAccessExpression(node.expression) && node.expression.name.text === '_kibo_llt')
      wrapExpression(node, 'wasmExport');
    ts.forEachChild(node, visit);
  }
  visit(source);
  const expected = [...Object.values(configuration.functions ?? {}), ...Object.values(configuration.methods ?? {}),
    ...(configuration.damping ? ['damping'] : []), ...(configuration.cholesky ? ['mlFactorization', 'mlTriangularSolve'] : []),
    ...(configuration.wasm ? ['wasmExport'] : [])];
  for (const phase of expected)
    if (counts[phase] !== 1) throw new Error(`Expected one ${phase} anchor in ${configuration.file}`);
  let output = text;
  for (const edit of edits.sort((left, right) => right.start - left.start))
    output = output.slice(0, edit.start) + edit.text + output.slice(edit.end);
  return { output: output.replace(/^\/\/# sourceMappingURL=.*$/gm, ''), counts };
}

export function prepareProfileModules() {
  const directory = path.join(projectRoot, '.cache/lm-profile');
  fs.mkdirSync(directory, { recursive: true });
  fs.cpSync(path.join(projectRoot, 'dist'), path.join(directory, 'dist'), { recursive: true });
  fs.copyFileSync(path.join(assetDirectory, 'facade.mjs'), path.join(directory, 'facade.mjs'));
  const configurations = [
    { file: 'dist/core/levenbergMarquardt.js', functions: {
      levenbergMarquardt: 'lm', tryLevenbergMarquardtStep: 'step'
    }, damping: true },
    { file: 'dist/core/jacobianComputation.js', functions: { computeJacobianMatrix: 'jacobian' } },
    { file: 'dist/core/constrainedNormalEquations.js', functions: { computeNormalEquationsMatrices: 'normalEquations' } },
    { file: 'dist/core/lmLinearSolver.js', methods: { solve: 'linearSolver' }, cholesky: true },
    { file: 'dist/core/kiboLinearSolver.js', methods: {
      solve: 'linearSolver', copyInputs: 'matrixConversion', callWasm: 'facade', fallback: 'luFallback'
    } },
    { file: 'facade.mjs', methods: { '#copy': 'copyIn', copyOutput: 'copyOut' }, wasm: true }
  ];
  const transformations = configurations.map(configuration => {
    const file = path.join(directory, configuration.file);
    const originalHash = sha256(file);
    const { output, counts } = instrumentSource(fs.readFileSync(file, 'utf8'), configuration);
    fs.writeFileSync(file, output);
    return { file: configuration.file, originalHash, instrumentedHash: sha256(file), counts };
  });
  return { directory, transformations };
}
