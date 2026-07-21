const {
	normalizeBgr, normalizeBgrAsync,
	normalizeRgb, normalizeRgbAsync,
	normalizeRgba, normalizeRgbaAsync
} = require('../fast-norm');
const { Suite } = require('bench-node');

const width = 640;
const height = 640;
const size = width * height;

const mean = new Float32Array([0.48145466, 0.4578275, 0.40821073]);
const std = new Float32Array([0.26862954, 0.26130258, 0.27577711]);

// For RGBA, add the 4th channel to mean and std (for example, transparency without changes)
const meanRgba = new Float32Array([0.48145466, 0.4578275, 0.40821073, 0.0]);
const stdRgba = new Float32Array([0.26862954, 0.26130258, 0.27577711, 1.0]);

// Input Buffers
const rawRgbBuffer = Buffer.alloc(width * height * 3, 200);
const rawRgbaBuffer = Buffer.alloc(width * height * 4, 255);

// Optimization Factors for JS
const rFactor = 1 / (255 * std[0]);
const gFactor = 1 / (255 * std[1]);
const bFactor = 1 / (255 * std[2]);
const rOffset = -mean[0] / std[0];
const gOffset = -mean[1] / std[1];
const bOffset = -mean[2] / std[2];

// Common output buffers
const sharedOutputBuffer = new Float32Array(3 * size);
const sharedOutputBufferRgba = new Float32Array(4 * size);

// JS function for 3-channel images (RGB/BGR)
function jsNormalize3Ch(buffer, size, outBuf, rf, gf, bf, ro, go, bo) {
	const gOffsetIdx = size;
	const bOffsetIdx = size * 2;
	let cIdx = 0;
	let len = buffer.length;
	for (let i = 0; i < len; i += 3) {
		outBuf[cIdx] = buffer[i] * rf + ro;
		outBuf[cIdx + gOffsetIdx] = buffer[i + 1] * gf + go;
		outBuf[cIdx + bOffsetIdx] = buffer[i + 2] * bf + bo;
		cIdx++;
	}
}

// JS function for 4-channel images (RGBA)
function jsNormalize4Ch(buffer, size, outBuf, rf, gf, bf, ro, go, bo) {
	const gOffsetIdx = size;
	const bOffsetIdx = size * 2;
	const aOffsetIdx = size * 3;
	let cIdx = 0;
	let len = buffer.length;
	for (let i = 0; i < len; i += 4) {
		outBuf[cIdx] = buffer[i] * rf + ro;
		outBuf[cIdx + gOffsetIdx] = buffer[i + 1] * gf + go;
		outBuf[cIdx + bOffsetIdx] = buffer[i + 2] * bf + bo;
		outBuf[cIdx + aOffsetIdx] = buffer[i + 3] / 255.0;
		cIdx++;
	}
}

const dd = new Suite();

// --- BGR methods ---
dd.add('native js (3 channels)', () => {
	jsNormalize3Ch(rawRgbBuffer, size, sharedOutputBuffer, rFactor, gFactor, bFactor, rOffset, gOffset, bOffset);
});

dd.add('rust normalizeBgr sync', () => {
	normalizeBgr(rawRgbBuffer, sharedOutputBuffer, width, height, mean, std, true);
});

dd.add('rust normalizeBgrAsync async', async () => {
	await normalizeBgrAsync(rawRgbBuffer, sharedOutputBuffer, width, height, mean, std, true);
});

// --- RGB methods ---
dd.add('rust normalizeRgb sync', () => {
	normalizeRgb(rawRgbBuffer, sharedOutputBuffer, width, height, mean, std, true);
});

dd.add('rust normalizeRgbAsync async', async () => {
	await normalizeRgbAsync(rawRgbBuffer, sharedOutputBuffer, width, height, mean, std, true);
});

// --- RGBA methods ---
dd.add('native js (4 channels)', () => {
	jsNormalize4Ch(rawRgbaBuffer, size, sharedOutputBufferRgba, rFactor, gFactor, bFactor, rOffset, gOffset, bOffset);
});

dd.add('rust normalizeRgba sync', () => {
	normalizeRgba(rawRgbaBuffer, sharedOutputBufferRgba, width, height, meanRgba, stdRgba, true);
});

dd.add('rust normalizeRgbaAsync async', async () => {
	await normalizeRgbaAsync(rawRgbaBuffer, sharedOutputBufferRgba, width, height, meanRgba, stdRgba, true);
});

dd.run()
