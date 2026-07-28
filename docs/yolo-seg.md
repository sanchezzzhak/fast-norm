# segmentation dd-cars parts
```js
const ort = require('onnxruntime-node');
const sharp = require('sharp');
const fs = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');
const {normalizeRgbAsync, processYolo11SegAsync} = require("fast-norm");

const MEAN_ARR = new Float32Array([0.0, 0.0, 0.0]);
const STD_ARR = new Float32Array([1.0, 1.0, 1.0]);

const IMG_SIZE = 896;         // model image size fixed
const NUM_CLASSES = 33;       // count classes detects
const CONF_THRESHOLD = 0.25;
const MAX_DETECTIONS = 300;  
const MAX_POLYGON_POINTS = 100;

const CLASS_NAMES = {
	0: 'back-bumper',
	1: 'back-windshield',
	2: 'front-bumper',
  // ... etc
}

async function main() {
	const modelPath = 'model.onnx'
	const imagePath = 'test.png'
  
	const session = await ort.InferenceSession.create(modelPath, { executionProviders: ['cpu'] });

	const image = sharp(imagePath);
	const metadata = await image.metadata();
	const origWidth = metadata.width;
	const origHeight = metadata.height;
	const scale = Math.min(IMG_SIZE / origWidth, IMG_SIZE / origHeight);
	const newWidth = Math.round(origWidth * scale);
	const newHeight = Math.round(origHeight * scale);
	const padX = (IMG_SIZE - newWidth) / 2;
	const padY = (IMG_SIZE - newHeight) / 2;

	const paddedBuffer = await image
		.resize(newWidth, newHeight)
		.removeAlpha()
		.extend({
			top: Math.floor(padY), bottom: Math.ceil(padY),
			left: Math.floor(padX), right: Math.ceil(padX),
			background: { r: 255, g: 255, b: 255 },
		})
		.raw()
		.toBuffer();

	const float32Data = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
	await normalizeRgbAsync(paddedBuffer, float32Data, IMG_SIZE, IMG_SIZE, MEAN_ARR, STD_ARR, true);
	
	const maskH = outputs.output1.dims[2];
	const maskW = outputs.output1.dims[3];

	const memBefore = process.memoryUsage().heapUsed / 1024 / 1024;
	const startTime = performance.now();
	
	const result = await processYolo11SegAsync(
		outputs.output0.data,
		outputs.output1.data,
		maskH,
		maskW,
		origWidth,
		origHeight,
		scale,
		padX,
		padY,
		IMG_SIZE,
		CONF_THRESHOLD,
		NUM_CLASSES,
		MAX_DETECTIONS,
		MAX_POLYGON_POINTS
	)

	const endTime = performance.now();
	const memAfter = process.memoryUsage().heapUsed / 1024 / 1024;
	const avgTimeMs = totalTimeMs / 1;
	const totalTimeMs = endTime - startTime;
	const memDeltaMb = memAfter - memBefore;
	
	console.log('=== Performance test results ===============');
	console.log(`Total time:              ${totalTimeMs.toFixed(2)} ms`);
	console.log(`Average time/frame:      ${avgTimeMs.toFixed(2)} ms`);
	console.log(`FPS (frames per second): ${((1000 / avgTimeMs)).toFixed(1)}`);
	console.log(`Memory before:           ${memBefore.toFixed(2)} mb`);
	console.log(`Memory after:            ${memAfter.toFixed(2)} mb`);
	console.log(`Memory gain (Δ):         ${memDeltaMb.toFixed(2)} mb`);
	console.log('===========================================');
}

main().catch(console.error)
```