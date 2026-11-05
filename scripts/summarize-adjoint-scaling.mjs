/**
 * Combine the primary sweep, independent memory samples and timing refinements without discarding provenance.
 */

import fs from 'node:fs';
import path from 'node:path';
import { compareTiming } from './scaling/statistics.mjs';

const DIRECTORY = path.resolve('benchmark-results');
const PRIMARY_REPORTS = ['adjoint-scaling-full', 'adjoint-scaling-extended',
  'adjoint-scaling-refine-easy', 'adjoint-scaling-refine-curved'];
const METHOD_LABELS = { bfgs: 'BFGS', approximate: '近似版', 'full-dense': '完全版・密行列', 'full-hvp': '完全版・HVP',
  'approximate-fd': '近似版・曲率を差分', 'full-fd': '完全版・曲率を差分' };
const METHODS = ['bfgs', 'approximate', 'full-dense', 'full-hvp'];

function readReport(name) { return JSON.parse(fs.readFileSync(path.join(DIRECTORY, `${name}.json`), 'utf8')); }
function rowKey(row) { return [row.family, row.parameters, row.states, row.method].join('/'); }
const combined = new Map();
const sourceReports = [];
for (const name of PRIMARY_REPORTS) {
  const report = readReport(name);
  sourceReports.push({ name, environment: report.environment, protocol: report.protocol });
  for (const row of report.rows) {
    if (row.status !== 'success') continue;
    const previous = combined.get(rowKey(row));
    combined.set(rowKey(row), { ...row,
      peakRssMiB: row.peakRssMiB ?? previous?.peakRssMiB,
      timingReport: name, memoryReport: row.peakRssMiB ? name : previous?.memoryReport,
      timeRepeats: report.protocol.repeats, warmups: report.protocol.warmups,
      memoryRepeats: row.peakRssMiB ? report.protocol.memoryRepeats : previous?.memoryRepeats });
  }
}
const rows = [...combined.values()];
const reference = row => rows.find(other => other.family === row.family && other.parameters === row.parameters &&
  other.states === row.states && other.method === 'bfgs');
const time = row => row ? `${row.timeMs.median.toFixed(2)} [${row.timeMs.iqr.toFixed(2)}]` : '未測定';
const memory = row => row?.peakRssMiB ? `${row.peakRssMiB.median.toFixed(1)} [${row.peakRssMiB.iqr.toFixed(1)}]` : '未測定';
const environment = sourceReports[0].environment;
const primary = readReport(PRIMARY_REPORTS[0]);
const timedOut = primary.rows.filter(row => row.timing.status === 'timeout');
const lines = ['# Newton–Adjoint 大規模比較', '',
  `環境: ${environment.node} / ${environment.platform}-${environment.architecture} / ${environment.cpu}。`,
  'ソルバ実装は c5a98c7 のまま。設計変数 m と状態変数 n を別々に拡大した密行列実装の測定。',
  '共通精度: 真の縮約勾配 RMS ≤ 1e-6、パラメータ誤差 RMS ≤ 1e-6、実際の制約残差 RMS ≤ 1e-12。',
  '時間は中央値 [IQR] ms。通常は2回ウォームアップ後3回、境界の再測定は8回ウォームアップ後7回。',
  'ピークRSSは3つの独立したコールドプロセスの中央値 [IQR] MiB。Node・入力行列・JIT・未回収の一時領域も含む。',
  '時間とメモリは別のプロセスで測定。同期計算のピークはOSの maxRSS を用い、イベントループのサンプリングで代用しない。',
  'easy は3方式とも1反復で解ける制御問題。curved は非線形状態と非ゼロの随伴変数を持つ制御問題。',
  timedOut.length ? `初回に${timedOut.length}条件がプロセス全体の${primary.protocol.timeoutMs / 1000}秒上限に達した。長い時間枠での再実測を別ファイルに保存し採用。` :
    '主測定にプロセスの時間切れはない。各測定条件の時間枠は元JSONに保存。', '',
  '## 非線形問題: 設計変数を増加、状態数は128に固定', '',
  '| m | 手法 | 反復数 | 時間 ms [IQR] | RSS MiB [IQR] | 時間/BFGS | 四分位範囲の比較 | 時間の標本数 |',
  '|---:|---|---:|---:|---:|---:|---|---:|'];
for (const row of rows.filter(row => row.family === 'curved' && row.states === 128)
  .sort((left, right) => left.parameters - right.parameters || METHODS.indexOf(left.method) - METHODS.indexOf(right.method))) {
  const comparison = compareTiming(row.timeMs, reference(row)?.timeMs);
  lines.push(`| ${row.parameters} | ${METHOD_LABELS[row.method]} | ${row.iterations.median} | ${time(row)} | ${memory(row)} | ` +
    `${comparison.ratio.toFixed(2)} | ${comparison.classification} | ${row.timeRepeats} |`);
}

lines.push('', '## 大きい状態数・設計変数数での実測', '',
  '| 問題 | m | n | 手法 | 反復数 | 時間 ms [IQR] | RSS MiB [IQR] |', '|---|---:|---:|---|---:|---:|---:|');
for (const [parameters, states] of [[16, 512], [16, 1024], [256, 512], [512, 512]]) {
  for (const method of METHODS) {
    const row = rows.find(row => row.family === 'curved' && row.parameters === parameters && row.states === states && row.method === method);
    if (row) lines.push(`| curved | ${parameters} | ${states} | ${METHOD_LABELS[method]} | ${row.iterations.median} | ${time(row)} | ${memory(row)} |`);
  }
}

lines.push('', '## 完全版の時間がBFGSを上回る領域', '',
  '下表は中央値の比。1より大きいとBFGSより遅い。四分位範囲が重なる場合は境界を断定しない。', '',
  '| 問題 | n | m | 密行列版/BFGS | HVP版/BFGS | 密行列版の範囲比較 | HVP版の範囲比較 |',
  '|---|---:|---:|---:|---:|---|---|');
for (const family of ['easy', 'curved']) {
  for (const states of [128, 256, 512]) {
    const sizes = [...new Set(rows.filter(row => row.family === family && row.states === states).map(row => row.parameters))].sort((a, b) => a - b);
    for (const parameters of sizes) {
      const baseline = rows.find(row => row.family === family && row.parameters === parameters && row.states === states && row.method === 'bfgs');
      const dense = rows.find(row => row.family === family && row.parameters === parameters && row.states === states && row.method === 'full-dense');
      const hvp = rows.find(row => row.family === family && row.parameters === parameters && row.states === states && row.method === 'full-hvp');
      const first = compareTiming(dense?.timeMs, baseline?.timeMs);
      const second = compareTiming(hvp?.timeMs, baseline?.timeMs);
      lines.push(`| ${family} | ${states} | ${parameters} | ${first.ratio?.toFixed(2) ?? '未測定'} | ` +
        `${second.ratio?.toFixed(2) ?? '未測定'} | ${first.classification} | ${second.classification} |`);
    }
  }
}

for (const [name, title] of [['adjoint-scaling-fd', '曲率を数値差分で計算する場合'],
  ['adjoint-scaling-heterogeneous', '初期値を設計変数ごとに変えた非線形問題']]) {
  const report = readReport(name);
  lines.push('', `## ${title}`, '',
    '一次偏微分は解析式。-fd のみ二階の曲率を差分で計算する。数値勾配まで含む比較ではない。', '',
    '| m | n | 手法 | 状態 | 反復数 | 時間 ms [IQR] | RSS MiB [IQR] |', '|---:|---:|---|---|---:|---:|---:|');
  for (const row of report.rows) lines.push(`| ${row.parameters} | ${row.states} | ${METHOD_LABELS[row.method]} | ${row.status} | ` +
    `${row.iterations?.median ?? '未測定'} | ${row.timeMs ? time(row) : '未測定'} | ${memory(row)} |`);
}
lines.push('', '## 解釈上の制限', '',
  '- 逆転点は目的関数、反復数、m/n、ヘッセ行列の渡し方、実装、実行環境に依存する。普遍的な設計変数数は得られない。',
  '- HVP は結合ヘッセ行列の保存を省くが、縮約ヘッセ行列は密行列として構築・分解する。matrix-free Newton-CG ではない。',
  '- 現BFGSの逆ヘッセ更新は一般の密行列積を使い O(m³)。標準的な O(m²) のrank-two更新に最適化したBFGSとの境界とは異なる。',
  '- RSSはプロセス全体のOS高水位。使用中の配列だけのサイズではない。反復回数・割当量・GCでも変わる。小差は過剰解釈しない。',
  '- 制御問題の式は研究メモに記載。大規模な疎PDE実問題へそのまま外挿しない。', '',
  '## 元データ', '', ...[...PRIMARY_REPORTS, 'adjoint-scaling-fd', 'adjoint-scaling-heterogeneous']
    .map(name => `- [${name}.json](${name}.json) / [測定表](${name}.md)`),
  '', '時間の精密化では元の3回のRSS測定を保持している。統合JSONに時間・メモリそれぞれの出典を保存。');
fs.writeFileSync(path.join(DIRECTORY, 'adjoint-scaling-summary.md'), `${lines.join('\n')}\n`);
fs.writeFileSync(path.join(DIRECTORY, 'adjoint-scaling-summary.json'), JSON.stringify({ sourceReports, rows }, null, 2));
console.log(path.join(DIRECTORY, 'adjoint-scaling-summary.md'));
