const { denormalizeRgb, denormalizeRgbAsync } = require('../fast-norm');
const { Suite } = require('bench-node');

// We simulate the output of a 4x upscaler (the original image was 640x640)
const outWidth = 640 * 4;   // 2560
const outHeight = 640 * 4;  // 2560
const outTotalPixels = outWidth * outHeight;

// Simulate raw data from ONNX Runtime (Float32Array in CHW format)
// Fill with random values from 0.0 to 1.0
const outputData = new Float32Array(3 * outTotalPixels);
for (let i = 0; i < outputData.length; i++) {
	outputData[i] = Math.random();
}

// Allocating buffers for results (Interleaved RGB u8)
const sharedJsBuffer = Buffer.alloc(outTotalPixels * 3);
const sharedRustBuffer = Buffer.alloc(outTotalPixels * 3);

// Settings for denormalization
const mean = new Float32Array([0.0, 0.0, 0.0]);
const std  = new Float32Array([1.0, 1.0, 1.0]);

// Old JS implementation for comparison
function jsDenormalize(srcData, dstBuffer, totalPixels) {
	for (let i = 0; i < totalPixels; i++) {
		let r = Math.min(Math.max(srcData[i], 0.0), 1.0) * 255;
		let g = Math.min(Math.max(srcData[i + totalPixels], 0.0), 1.0) * 255;
		let b = Math.min(Math.max(srcData[i + 2 * totalPixels], 0.0), 1.0) * 255;

		const dstIdx = i * 3;
		dstBuffer[dstIdx] = Math.round(r);
		dstBuffer[dstIdx + 1] = Math.round(g);
		dstBuffer[dstIdx + 2] = Math.round(b);
	}
}

const suite = new Suite();

suite.add('native js denormalize', () => {
	jsDenormalize(outputData, sharedJsBuffer, outTotalPixels);
});

suite.add('rust denormalizeRgb sync', () => {
	denormalizeRgb(outputData, sharedRustBuffer, outWidth, outHeight, mean, std, true);
});

suite.add('rust denormalizeRgbAsync async', async () => {
	await denormalizeRgbAsync(outputData, sharedRustBuffer, outWidth, outHeight, mean, std, true);
});

suite.run();
