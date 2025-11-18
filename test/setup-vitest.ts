import { webcrypto } from 'node:crypto';

type GlobalWithCrypto = typeof globalThis & { crypto?: typeof webcrypto };
const globalScope = globalThis as GlobalWithCrypto;

// Node 16 does not expose `crypto` on the global scope by default.
if (!globalScope.crypto) {
  globalScope.crypto = webcrypto;
}
