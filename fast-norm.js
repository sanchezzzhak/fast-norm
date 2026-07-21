const { join } = require('node:path');
const {denormalizeRgbAsync, denormalizeRgb} = require("./fast-norm");
let nativeBinding = null;

if (process.platform === 'linux' && process.arch === 'x64') {
	nativeBinding = require(join(__dirname, 'fast-norm.linux-x64-gnu.node'));
} else if (process.platform === 'win32' && process.arch === 'x64') {
	nativeBinding = require(join(__dirname, 'fast-norm.win32-x64-msvc.node'));
} else {
	throw new Error(`Platform ${process.platform} not supported`);
}

module.exports = {
	normalizeRgb: nativeBinding.normalizeRgb,
	normalizeRgbAsync: nativeBinding.normalizeRgbAsync,

	normalizeRgba: nativeBinding.normalizeRgba,
	normalizeRgbaAsync: nativeBinding.normalizeRgbaAsync,

	normalizeBgr: nativeBinding.normalizeBgr,
	normalizeBgrAsync: nativeBinding.normalizeBgrAsync,

	denormalizeRgb: nativeBinding.denormalizeRgb,
	denormalizeRgbAsync: nativeBinding.denormalizeRgbAsync,
};
