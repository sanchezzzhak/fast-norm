const {
	processYolo11SegAsync,
	processYolo11Seg
} = require('../fast-norm');

const { Suite } = require('bench-node');
const fs = require('fs');
const path = require('path');

const IMG_SIZE = 896;
const NUM_CLASSES = 33;
const NUM_MASK_COEFFS = 32;
const CONF_THRESHOLD = 0.25;
const MAX_DETECTIONS = 300;
const MAX_POLYGON_POINTS = 100;
const maskH = 224,
	maskW= 224,
	origWidth = 400,
	origHeight = 300,
	scale = 2.24,
	padX = 0,
	padY = 112;

const data0 = new Float32Array(JSON.parse(fs.readFileSync(path.join(__dirname , 'fixtures/data0.json'))))
const data1 = new Float32Array(JSON.parse(fs.readFileSync(path.join(__dirname , 'fixtures/data1.json'))))

const outputs = {
	output0: { data: data0 },
	output1: { data: data1, dims: [1, 32, maskH, maskW] }
};
function postprocessDetectionsEnd2End(output0, origWidth, origHeight, scale, padX, padY) {
	const data = output0.data;
	const detections = [];

	for (let i = 0; i < MAX_DETECTIONS; i++) {
		const offset = i * 38;

		const x1 = data[offset + 0];
		const y1 = data[offset + 1];
		const x2 = data[offset + 2];
		const y2 = data[offset + 3];
		const confidence = data[offset + 4];
		const classId = Math.round(data[offset + 5]);
		if (confidence < CONF_THRESHOLD || classId < 0 || classId >= NUM_CLASSES) {
			continue;
		}
		const origX1 = Math.max(0, (x1 - padX) / scale);
		const origY1 = Math.max(0, (y1 - padY) / scale);
		const origX2 = Math.min(origWidth, (x2 - padX) / scale);
		const origY2 = Math.min(origHeight, (y2 - padY) / scale);

		const maskCoeffs = new Float32Array(NUM_MASK_COEFFS);
		for (let m = 0; m < NUM_MASK_COEFFS; m++) {
			maskCoeffs[m] = data[offset + 6 + m];
		}
		const className = '';
		detections.push({
			classId,
			className,
			confidence,
			bbox_xyxy: [origX1, origY1, origX2, origY2],
			maskCoeffs
		});
	}

	return detections;
}

function processMasks(detections, output1, origWidth, origHeight, scale, padX, padY) {
	const output1Data = output1.data;
	const maskH = output1.dims[2];
	const maskW = output1.dims[3];
	const maskArea = maskH * maskW;

	for (const det of detections) {
		const finalMask = new Float32Array(maskArea);

		for (let m = 0; m < NUM_MASK_COEFFS; m++) {
			const coeff = det.maskCoeffs[m];
			const protoOffset = m * maskArea;
			for (let i = 0; i < maskArea; i++) {
				finalMask[i] += coeff * output1Data[protoOffset + i];
			}
		}

		const binaryMask = new Uint8Array(maskArea);
		for (let i = 0; i < maskArea; i++) {
			const sigmoidVal = 1.0 / (1.0 + Math.exp(-finalMask[i]));
			binaryMask[i] = sigmoidVal > 0.5 ? 1 : 0;
		}

		const [x1, y1, x2, y2] = det.bbox_xyxy;
		const mx1 = Math.max(0, Math.floor((x1 * scale + padX) / (IMG_SIZE / maskW)));
		const my1 = Math.max(0, Math.floor((y1 * scale + padY) / (IMG_SIZE / maskH)));
		const mx2 = Math.min(maskW, Math.ceil((x2 * scale + padX) / (IMG_SIZE / maskW)));
		const my2 = Math.min(maskH, Math.ceil((y2 * scale + padY) / (IMG_SIZE / maskH)));

		const cropW = mx2 - mx1;
		const cropH = my2 - my1;

		if (cropW <= 0 || cropH <= 0) {
			det.mask_polygon_xy = [];
			det.mask_polygon_xyn = [];
			continue;
		}

		const croppedMask = new Uint8Array(cropW * cropH);
		for (let y = 0; y < cropH; y++) {
			for (let x = 0; x < cropW; x++) {
				croppedMask[y * cropW + x] = binaryMask[(my1 + y) * maskW + (mx1 + x)];
			}
		}

		const contour = traceContour(croppedMask, cropW, cropH);
		const scaleX = (x2 - x1) / cropW;
		const scaleY = (y2 - y1) / cropH;

		det.mask_polygon_xy = contour.map(([px, py]) => [
			parseFloat((x1 + px * scaleX).toFixed(2)),
			parseFloat((y1 + py * scaleY).toFixed(2))
		]);

		det.mask_polygon_xyn = det.mask_polygon_xy.map(([px, py]) => [
			parseFloat((px / origWidth).toFixed(6)),
			parseFloat((py / origHeight).toFixed(6))
		]);

		delete det.maskCoeffs;
	}
}

function traceContour(binaryMask, w, h) {
	let startX = -1, startY = -1;
	outer: for (let y = 0; y < h; y++) {
		for (let x = 0; x < w; x++) {
			if (binaryMask[y * w + x] === 1) {
				startX = x; startY = y;
				break outer;
			}
		}
	}
	if (startX === -1) return [];

	const dirs = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];
	let contour = [];
	let cx = startX, cy = startY;
	let dir = 0;
	let maxSteps = w * h;
	let steps = 0;

	do {
		contour.push([cx, cy]);
		let found = false;
		let startDir = (dir + 5) % 8;

		for (let i = 0; i < 8; i++) {
			let d = (startDir + i) % 8;
			let nx = cx + dirs[d][0];
			let ny = cy + dirs[d][1];

			if (nx >= 0 && nx < w && ny >= 0 && ny < h && binaryMask[ny * w + nx] === 1) {
				dir = d;
				cx = nx;
				cy = ny;
				found = true;
				break;
			}
		}
		if (!found) break;
		steps++;
	} while ((cx !== startX || cy !== startY) && steps < maxSteps);

	if (contour.length > 100) {
		const step = Math.floor(contour.length / 100);
		contour = contour.filter((_, i) => i % step === 0);
	}

	return contour;
}

function runNativeJsPipeline() {
	const detections = postprocessDetectionsEnd2End(outputs.output0, origWidth, origHeight, scale, padX, padY);
	processMasks(detections, outputs.output1, origWidth, origHeight, scale, padX, padY);
}

const dd = new Suite();

dd.add('native js (end2end find counter)', () => {
	runNativeJsPipeline();
})

dd.add('rust processYolo11Seg sync', () => {
	processYolo11Seg(
		data0,
		data1,
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
	);
});

dd.add('rust processYolo11SegAsync async', async () => {
	await processYolo11SegAsync(
		data0,
		data1,
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
	);
});

dd.run()
