const { join } = require('node:path');
let nativeBinding = null;

if (process.platform === 'linux' && process.arch === 'x64') {
	nativeBinding = require(join(__dirname, 'fast-norm.linux-x64-gnu.node'));
} else if (process.platform === 'win32' && process.arch === 'x64') {
	nativeBinding = require(join(__dirname, 'fast-norm.win32-x64-msvc.node'));
} else {
	throw new Error(`Platform ${process.platform} not supported`);
}

module.exports = {
	// input image buffer
	normalizeRgb: nativeBinding.normalizeRgb,
	normalizeRgbAsync: nativeBinding.normalizeRgbAsync,
	normalizeRgba: nativeBinding.normalizeRgba,
	normalizeRgbaAsync: nativeBinding.normalizeRgbaAsync,
	normalizeBgr: nativeBinding.normalizeBgr,
	normalizeBgrAsync: nativeBinding.normalizeBgrAsync,
	// output image buffer
	denormalizeRgb: nativeBinding.denormalizeRgb,
	denormalizeRgbAsync: nativeBinding.denormalizeRgbAsync,
	// output seg/box
	processYolo11Seg: nativeBinding.processYolo11Seg,
	processYolo11SegAsync: nativeBinding.processYolo11SegAsync,
	processYolo11Det: nativeBinding.processYolo11Det,
	processYolo11DetAsync: nativeBinding.processYolo11DetAsync,
};
