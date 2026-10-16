/** Exercise the shipped browser bundle and real WASM through HTTP in Chromium. */
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { chromium } from 'playwright';
import { projectRoot, collectNumoptProvenance, verifyKiboAssets, toolchains } from '../../scripts/kibo-assets.mjs';

const TIMEOUT_MILLISECONDS = 30_000;
const manifest = verifyKiboAssets();
const playwrightPackage = JSON.parse(fs.readFileSync(path.join(projectRoot, 'node_modules/playwright/package.json'), 'utf8'));
if (playwrightPackage.version !== toolchains.playwrightVersion) throw new Error('Playwright version mismatch');
const contentTypes = { '.mjs': 'text/javascript', '.js': 'text/javascript', '.html': 'text/html', '.wasm': 'application/wasm' };
const server = http.createServer((request, response) => {
  const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
  const file = path.resolve(projectRoot, `.${pathname}`);
  const relative = path.relative(projectRoot, file);
  const allowed = ['dist/', '.cache/kibo/', 'integration/kibo/'].some(prefix => relative.replaceAll('\\', '/').startsWith(prefix));
  if (!allowed || relative.startsWith('..') || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
    response.writeHead(404); response.end(); return;
  }
  response.writeHead(200, { 'Content-Type': contentTypes[path.extname(file)] ?? 'application/octet-stream' });
  fs.createReadStream(file).pipe(response);
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  page.setDefaultTimeout(TIMEOUT_MILLISECONDS);
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(`http://127.0.0.1:${server.address().port}/integration/kibo/browser.html`);
  await page.waitForFunction(() => window.kiboReport !== undefined);
  const result = await page.evaluate(() => window.kiboReport);
  const report = { browser: browser.version(), playwright: playwrightPackage.version,
    driverRuntime: process.version, numopt: collectNumoptProvenance(), manifest, errors, ...result };
  fs.mkdirSync('benchmark-results/kibo', { recursive: true });
  fs.writeFileSync('benchmark-results/kibo/chromium.json', `${JSON.stringify(report, null, 2)}\n`);
  if (!result.passed || errors.length) throw new Error(JSON.stringify(report));
  console.log(JSON.stringify({ browser: report.browser, passed: result.passed, cases: result.cases.length,
    configurationFailures: result.configurationFailures.length, numericFallbacks: result.numericFallbacks.length,
    resources: result.resources }));
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
