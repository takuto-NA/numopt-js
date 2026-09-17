import fs from 'node:fs';
import path from 'node:path';

const SOURCE_ROOT = path.resolve(process.cwd(), 'src');
const BLOCK_COMMENT_START = '/**';
const FORBIDDEN_HEADER_PHRASES = ['For first-time readers', 'Phase', 'MVP'];

function collectTypeScriptFiles(directory) {
  const entries = fs.readdirSync(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...collectTypeScriptFiles(fullPath));
      continue;
    }
    if (entry.isFile() && entry.name.endsWith('.ts')) {
      files.push(fullPath);
    }
  }
  return files;
}

function extractLeadingBlockComment(source) {
  const trimmed = source.trimStart();
  if (!trimmed.startsWith(BLOCK_COMMENT_START)) {
    return null;
  }
  const commentEnd = trimmed.indexOf('*/');
  if (commentEnd === -1) {
    return null;
  }
  return trimmed.slice(0, commentEnd + 2);
}

function findForbiddenPhrases(comment) {
  return FORBIDDEN_HEADER_PHRASES.filter((phrase) => comment.includes(phrase));
}

const files = collectTypeScriptFiles(SOURCE_ROOT);
const failures = [];

for (const filePath of files) {
  const source = fs.readFileSync(filePath, 'utf8');
  const comment = extractLeadingBlockComment(source);
  const relativePath = path.relative(process.cwd(), filePath).replaceAll('\\', '/');
  if (comment === null) {
    failures.push(`${relativePath}: file must start with a /** responsibility header */`);
    continue;
  }
  const forbidden = findForbiddenPhrases(comment);
  if (forbidden.length > 0) {
    failures.push(`${relativePath}: header contains forbidden phrase(s): ${forbidden.join(', ')}`);
  }
}

if (failures.length > 0) {
  console.error(`Header check failed (${failures.length}):`);
  for (const failure of failures) {
    console.error(`  ${failure}`);
  }
  process.exit(1);
}

console.log(`Header check passed (${files.length} files)`);
