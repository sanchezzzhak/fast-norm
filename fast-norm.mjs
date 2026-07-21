import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let nativeBinding = null;

if (process.platform === 'linux' && process.arch === 'x64') {
	nativeBinding = require(join(__dirname, 'fast-norm.linux-x64-gnu.node'));
} else if (process.platform === 'win32' && process.arch === 'x64') {
	nativeBinding = require(join(__dirname, 'fast-norm.win32-x64-msvc.node'));
} else {
	throw new Error(`Platform ${process.platform} not supported`);
}

export const normalizeRgb = nativeBinding.normalizeRgb;
export const normalizeRgbAsync = nativeBinding.normalizeRgbAsync;

export const normalizeRgba = nativeBinding.normalizeRgba;
export const normalizeRgbaAsync = nativeBinding.normalizeRgbaAsync;

export const normalizeBgr = nativeBinding.normalizeBgr;
export const normalizeBgrAsync = nativeBinding.normalizeBgrAsync;

export default {
	normalizeRgb,
	normalizeRgbAsync,
	normalizeRgba,
	normalizeRgbaAsync,
	normalizeBgr,
	normalizeBgrAsync
};
