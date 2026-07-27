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

export const denormalizeRgb = nativeBinding.denormalizeRgb;
export const denormalizeRgbAsync  = nativeBinding.denormalizeRgbAsync;


export const processYolo11Seg = nativeBinding.processYolo11Seg;
export const processYolo11SegAsync = nativeBinding.processYolo11SegAsync;
export const processYolo11Det = nativeBinding.processYolo11Det;
export const processYolo11DetAsync = nativeBinding.processYolo11DetAsync;


export default {
	// input image buffer
	normalizeRgb,
	normalizeRgbAsync,
	normalizeRgba,
	normalizeRgbaAsync,
	normalizeBgr,
	normalizeBgrAsync,
	// output image buffer
	denormalizeRgb,
	denormalizeRgbAsync,
  // output seg/box
	processYolo11Seg,
	processYolo11SegAsync,
	processYolo11Det,
	processYolo11DetAsync
};
