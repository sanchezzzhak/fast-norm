const { denormalizeRgb } = require('../fast-norm');
const assert = require('assert');

function runValidationTest() {
	// 1. Setup mock dimensions
	const width = 100;
	const height = 100;
	const totalPixels = width * height;

	console.log(`[Config] Image resolution: ${width}x${height} (${totalPixels} pixels)`);
	console.log(`[Config] Total tensor elements to validate: ${3 * totalPixels}`);

	// 2. Generate smooth gradient float data (CHW format) to test edge cases
	const outputData = new Float32Array(3 * totalPixels);
	for (let i = 0; i < outputData.length; i++) {
		outputData[i] = i / outputData.length;
	}

	// 3. Inject extreme values to test clamping behavior
	outputData[0] = -15.42;                    // Deep sub-zero value (should clamp to 0)
	outputData[totalPixels] = 0.0;             // Exact boundary zero
	outputData[totalPixels * 2] = 1.0;         // Exact boundary one
	outputData[outputData.length - 1] = 99.88; // Extreme value above 1.0 (should clamp to 255)

	// 4. Allocate independent target buffers for verification
	const jsBuffer = Buffer.alloc(totalPixels * 3);
	const rustBuffer = Buffer.alloc(totalPixels * 3);

	const mean = new Float32Array([0.0, 0.0, 0.0]);
	const std  = new Float32Array([1.0, 1.0, 1.0]);

	// 5. Reference JS implementation (Using strict Math.round)
	console.log('\n[Process] Computing baseline array using Javascript loop...');
	for (let i = 0; i < totalPixels; i++) {
		let r = Math.min(Math.max(outputData[i], 0.0), 1.0) * 255;
		let g = Math.min(Math.max(outputData[i + totalPixels], 0.0), 1.0) * 255;
		let b = Math.min(Math.max(outputData[i + 2 * totalPixels], 0.0), 1.0) * 255;

		const dstIdx = i * 3;
		jsBuffer[dstIdx] = Math.round(r);
		jsBuffer[dstIdx + 1] = Math.round(g);
		jsBuffer[dstIdx + 2] = Math.round(b);
	}

	// 6. Compute using optimized Rust Addon
	console.log('[Process] Computing optimized array using Rust Native Addon...');
	denormalizeRgb(outputData, rustBuffer, width, height, mean, std, true);

	// 7. Perform bitwise verification loop
	console.log('\n[Verify] Running deep byte-by-byte comparison...');
	let mismatches = 0;
	const maxLoggedErrors = 5;

	for (let i = 0; i < jsBuffer.length; i++) {
		if (jsBuffer[i] !== rustBuffer[i]) {
			mismatches++;
			if (mismatches <= maxLoggedErrors) {
				console.error(`  ❌ Mismatch at byte index [${i}] -> JS: ${jsBuffer[i]} | Rust: ${rustBuffer[i]} (Diff: ${Math.abs(jsBuffer[i] - rustBuffer[i])})`);
			}
		}
	}

	// 8. Assert and print summary report
	try {
		// Assert that the buffers are identical down to the last byte
		assert.deepStrictEqual(rustBuffer, jsBuffer);

		console.log('✅ TEST PASSED: SUCCESS!');
		console.log(' Rust native buffer matches JS baseline 100% bit-for-bit.');
		console.log(' Fast instruction rounding (+0.5 as u8) is safe and mathematically valid.');
		process.exit(0);
	} catch (error) {
		console.error('❌ TEST FAILED: MISMATCH DETECTED!');
		console.error(` Total byte discrepancies found: ${mismatches} out of ${jsBuffer.length} bytes.`);
		console.error(` Error rate: ${((mismatches / jsBuffer.length) * 100).toFixed(4)}%`);
		console.error(' Note: If the diff is exactly 1, it represents acceptable rounding precision tolerance in graphics pipelines.');
		process.exit(1);
	}
}

// Execute the test suite
runValidationTest();
