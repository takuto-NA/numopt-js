/**
 * Fail if docs/ is dirty after TypeDoc regeneration.
 * Uses porcelain status so untracked pages are caught (unlike git diff alone).
 */

import { execSync } from 'node:child_process';

const status = execSync('git status --porcelain -- docs', {
  encoding: 'utf8'
}).trim();

if (status.length > 0) {
  console.error('docs/ is out of date or has uncommitted changes:\n' + status);
  process.exit(1);
}

console.log('docs/ is clean');
