import { webcrypto } from 'node:crypto';
import { defineConfig } from 'vitest/config';

type GlobalWithCrypto = typeof globalThis & { crypto?: typeof webcrypto };
const globalScope = globalThis as GlobalWithCrypto;

if (!globalScope.crypto) {
  globalScope.crypto = webcrypto;
}

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    setupFiles: ['test/setup-vitest.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'test/',
        'dist/',
        '**/*.d.ts',
        '**/*.config.*'
      ]
    }
  }
});

