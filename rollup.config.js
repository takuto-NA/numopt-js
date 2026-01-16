import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import typescript from '@rollup/plugin-typescript';

const tsPlugin = typescript({
  tsconfig: './tsconfig.json',
  declaration: false,
  declarationMap: false,
});

/**
 * We ship:
 * - Browser ESM bundle (dependency-bundled): dist/index.browser.js
 * - Node CJS bundle (ml-matrix external): dist/index.cjs
 *
 * Note: package.json uses "type": "module", so CommonJS output must be .cjs.
 */
export default [
  // Browser bundle (includes dependencies)
  {
    input: 'src/index.ts',
    output: {
      file: 'dist/index.browser.js',
      format: 'es',
      sourcemap: true,
    },
    plugins: [
      resolve({
        browser: true,
        preferBuiltins: false,
      }),
      commonjs(),
      tsPlugin,
    ],
    external: [], // ml-matrixもバンドルに含める
  },

  // Node CJS entry (keeps dependencies external)
  {
    input: 'src/index.ts',
    output: {
      file: 'dist/index.cjs',
      format: 'cjs',
      sourcemap: true,
      exports: 'named',
    },
    plugins: [
      resolve({
        browser: false,
        preferBuiltins: true,
      }),
      commonjs(),
      tsPlugin,
    ],
    external: ['ml-matrix'],
  },
];

