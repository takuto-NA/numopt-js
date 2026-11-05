/** Audits isolated comparison trials and writes tables relative to adjoint L-BFGS. */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { compareTiming } from './scaling/statistics.mjs';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const OUTPUT = path.join(ROOT, 'benchmark-results');
const NAMES = ['curved', 'heterogeneous', 'conditioned', 'conditioned-bfgs-refined'];
const METHODS = ['bfgs', 'lbfgs-5', 'lbfgs', 'lbfgs-20', 'approximate', 'full-dense', 'full-hvp'];
const reports = NAMES.map(name => ({ name, ...JSON.parse(fs.readFileSync(
  path.join(OUTPUT, `adjoint-lbfgs-${name}.json`), 'utf8')) }));

let timingTrials = 0;
let memoryTrials = 0;
let failedTimingTrials = 0;
const incompleteRows = [];
function verifyAccuracy(trial, protocol, description) {
  if (trial.status !== 'success' || !Number.isFinite(trial.gradientRms) || !Number.isFinite(trial.errorRms) ||
    !Number.isFinite(trial.constraintRms) || trial.gradientRms > protocol.gradientRmsTarget ||
    trial.errorRms > protocol.parameterRmsTarget || trial.constraintRms > protocol.constraintRmsTarget) {
    throw new Error(`Invalid accuracy target: ${description}`);
  }
}
for (const report of reports) {
  const expectedRows = report.protocol.sizes.length * report.protocol.methods.length * report.protocol.families.length;
  if (report.rows.length !== expectedRows) throw new Error(`${report.name}: missing rows`);
  for (const row of report.rows) {
    if (row.status !== 'success') incompleteRows.push({ report: report.name, parameters: row.parameters,
      states: row.states, method: row.method, timing: row.timing });
    const successfulTimes = (row.timing.trials ?? []).filter(trial => trial.status === 'success');
    for (const trial of [...successfulTimes, ...row.memories.map(memory => memory.trial)]) {
      verifyAccuracy(trial, report.protocol, `${report.name} ${row.method} ${row.parameters}x${row.states}`);
    }
    timingTrials += successfulTimes.length;
    failedTimingTrials += (row.timing.trials ?? []).filter(trial => trial.status !== 'success').length;
    memoryTrials += row.memories.length;
  }
}

const refinedPath = path.join(OUTPUT, 'adjoint-lbfgs-refined.json');
const refinement = fs.existsSync(refinedPath) ? JSON.parse(fs.readFileSync(refinedPath, 'utf8')) : undefined;
if (refinement) {
  const expectedRows = refinement.protocol.sizes.length * refinement.protocol.methods.length * refinement.protocol.families.length;
  if (refinement.rows.length !== expectedRows) throw new Error('Missing crossover refinement rows');
  for (const row of refinement.rows) {
    if (row.status !== 'success') throw new Error('Incomplete crossover refinement');
    for (const trial of row.timing.trials) verifyAccuracy(trial, refinement.protocol, `refined ${row.method} ${row.parameters}`);
    timingTrials += row.timing.trials.length;
  }
}

const lines = ['# Adjoint L-BFGS 比較結果', '',
  `測定コード: ${reports[0].environment.commit}; ${reports[0].environment.node}; ${reports[0].environment.cpu}。`,
  `精度を監査した成功試行: 時間 ${timingTrials}、独立したコールドプロセスのピーク RSS ${memoryTrials}。未達の時間試行 ${failedTimingTrials}。ウォームアップは試行数に含めない。`,
  '時間とメモリは各3試行の中央値 [IQR]。時間は2回ウォームアップ後、メモリは別の新規プロセス。',
  '共通基準: 真の勾配 RMS≤1e−6、解誤差 RMS≤1e−6、実制約残差 RMS≤1e−12。すべて解析一次・二次微分。',
  'lbfgs の既定履歴数は10。n は状態数、m は設計変数数。ピーク RSS はランタイムや入力も含む全プロセスの値。',
  '時間比は同じ問題の L-BFGS(q=10) に対する値。quartile separation は記述的な比較であり有意差検定ではない。', ''];
function appendTable(rows, includeMemory = true) {
  lines.push('| m | n | Method | 状態 | 反復 | 時間 ms [IQR] | ピーク RSS MiB [IQR] | 時間 / L-BFGS | 四分位範囲 |',
    '|---:|---:|---|---|---:|---:|---:|---:|---|');
  const sorted = [...rows].sort((left, right) => left.states - right.states || left.parameters - right.parameters ||
    METHODS.indexOf(left.method) - METHODS.indexOf(right.method));
  for (const row of sorted) {
    const reference = rows.find(other => other.parameters === row.parameters && other.states === row.states && other.method === 'lbfgs');
    const comparison = compareTiming(row.timeMs, reference?.timeMs);
    const memory = includeMemory && row.peakRssMiB;
    lines.push(`| ${row.parameters} | ${row.states} | ${row.method} | ${row.status} | ${row.iterations?.median ?? row.timing.trials?.[0]?.iterations ?? 'n/a'} | ` +
      `${row.timeMs ? `${row.timeMs.median.toFixed(2)} [${row.timeMs.iqr.toFixed(2)}]` : '精度未達'} | ` +
      `${memory ? `${memory.median.toFixed(1)} [${memory.iqr.toFixed(1)}]` : '未計測'} | ` +
      `${comparison.ratio?.toFixed(2) ?? 'n/a'} | ${comparison.classification} |`);
  }
  lines.push('');
}
for (const report of reports) {
  lines.push(`## ${report.name}`, '');
  if (report.protocol.solverToleranceFactor !== undefined) lines.push(
    `追試のソルバ終了許容誤差: ${report.protocol.solverToleranceFactor} × 1e−6 × sqrt(m)。共通の目標精度は同じ。`, '');
  appendTable(report.rows);
}
for (const row of incompleteRows) lines.push(
  `未達の記録: ${row.report} ${row.method}, m=${row.parameters}, n=${row.states}; ` +
  `最初の試行の勾配 RMS=${row.timing.trials?.[0]?.gradientRms ?? 'n/a'}。この終了時間は目標達成までの時間として比較しない。`, '');
if (refinement) {
  lines.push('## 交差点の追加測定', '', '8回ウォームアップ後7回測定。時間のみ測り、メモリは再測定しない。', '');
  appendTable(refinement.rows, false);
}
lines.push('L-BFGS は受理済み勾配と復元済み状態を再利用するため、その効果も時間に含む。既存 BFGS の密行列更新は現実装では O(m³)。',
  '状態系は密行列で表現した良条件な結合系。疎な PDE や任意の目的関数への次元閾値の一般化はできない。');
fs.writeFileSync(path.join(OUTPUT, 'adjoint-lbfgs-summary.md'), `${lines.join('\n')}\n`);
fs.writeFileSync(path.join(OUTPUT, 'adjoint-lbfgs-summary.json'), JSON.stringify({
  audit: { timingTrials, memoryTrials, failedTimingTrials, incompleteRows }, reports, refinement
}, null, 2));
console.log(JSON.stringify({ timingTrials, memoryTrials, failedTimingTrials, report: path.join(OUTPUT, 'adjoint-lbfgs-summary.md') }));
