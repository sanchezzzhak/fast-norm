const test = require('node:test');
const assert = require('node:assert');
const { performance } = require('perf_hooks');
const addon = require('./../fast-norm');

const WIDTH = 640;
const HEIGHT = 640;
const SIZE = WIDTH * HEIGHT;
const MEAN = new Float32Array([0.0, 0.0, 0.0]);
const STD = new Float32Array([1.0, 1.0, 1.0]);
const ITERATIONS = 100;

// Auxiliary buffer generators
function createDummyRgbImage(size) {
  const buffer = Buffer.alloc(size * 3);
  for (let i = 0; i < size * 3; i++) buffer[i] = Math.floor(Math.random() * 256);
  return buffer;
}

function createDummyRgbaImage(size) {
  const buffer = Buffer.alloc(size * 4);
  for (let i = 0; i < size * 4; i++) {
    buffer[i] = (i % 4 === 3) ? 255 : Math.floor(Math.random() * 256);
  }
  return buffer;
}

test('Correctness test suite fast-norm', async (t) => {

  await t.test('End-to-end test: RGB normalizeRgb -> denormalizeRgb', () => {
    const originalImage = createDummyRgbImage(SIZE);
    const tensorBuffer = new Float32Array(SIZE * 3);
    const restoredImage = Buffer.alloc(SIZE * 3);

    addon.normalizeRgb(originalImage, tensorBuffer, WIDTH, HEIGHT, MEAN, STD, true);
    addon.denormalizeRgb(tensorBuffer, restoredImage, WIDTH, HEIGHT, MEAN, STD, true);

    assert.deepStrictEqual(originalImage,
      restoredImage,
      'The restored image must be identical to the original'
    );
  });

  await t.test(
    'normalizeRgba: Correctly ignoring the Alpha channel', () => {
    const rgbImage = createDummyRgbImage(SIZE);
    const rgbaImage = Buffer.alloc(SIZE * 4);

    // Создаем rgba из rgb
    for(let i = 0; i < SIZE; i++) {
      rgbaImage[i*4] = rgbImage[i*3];
      rgbaImage[i*4+1] = rgbImage[i*3+1];
      rgbaImage[i*4+2] = rgbImage[i*3+2];
      rgbaImage[i*4+3] = 128;
    }

    const tensorRgb = new Float32Array(SIZE * 3);
    const tensorRgba = new Float32Array(SIZE * 3);
    addon.normalizeRgb(rgbImage, tensorRgb, WIDTH, HEIGHT, MEAN, STD, true);
    addon.normalizeRgba(rgbaImage, tensorRgba, WIDTH, HEIGHT, MEAN, STD, true);

    assert.deepStrictEqual(tensorRgba,
      tensorRgb,
      'The output RGB and RGBA tensors must match, ignoring the alpha channel'
    );
  });

  await t.test('normalizeBgr: Checking the correct channel rearrangement in Planar', () => {
    const rgbImage = createDummyRgbImage(SIZE);
    const tensorRgb = new Float32Array(SIZE * 3);
    const tensorBgr = new Float32Array(SIZE * 3);

    addon.normalizeRgb(rgbImage, tensorRgb, WIDTH, HEIGHT, MEAN, STD, true);
    addon.normalizeBgr(rgbImage, tensorBgr, WIDTH, HEIGHT, MEAN, STD, true);

    assert.strictEqual(tensorBgr[0], tensorRgb[SIZE * 2],
      'Blue channel at the first BGR position should match the RGB end'
    );
    assert.strictEqual(tensorBgr[SIZE], tensorRgb[SIZE],
      'Green channel should remain in place'
    );
    assert.strictEqual(tensorBgr[SIZE * 2], tensorRgb[0],
      'Red channel at the end of BGR must match the beginning of RGB'
    );
  });
});

test('Addon performance benchmarks', { skip: false }, async () => {
  console.log(`Rolling up the load: ${ITERATIONS} кадров (${WIDTH}x${HEIGHT})...`);

  const rgbImage = createDummyRgbImage(SIZE);
  const rgbaImage = createDummyRgbaImage(SIZE);
  const tensorBuffer = new Float32Array(SIZE * 3);
  const restoredImage = Buffer.alloc(SIZE * 3);

  // Функция-хелпер для замера
  const measure = async (name, fnAsync = false, action) => {
    const start = performance.now();
    if (fnAsync) {
      const promises = [];
      for (let i = 0; i < ITERATIONS; i++) promises.push(action());
      await Promise.all(promises);
    } else {
      for (let i = 0; i < ITERATIONS; i++) action();
    }
    const end = performance.now();
    console.log(`[${name}]: Total ${(end - start).toFixed(2)} ms | per frame: ${((end - start) / ITERATIONS).toFixed(3)} ms`);
  };

  console.log('\n--- Synchronous Core (Single Frames) ---');
  await measure('normalizeRgb (Sync)', false,
    () => addon.normalizeRgb(rgbImage, tensorBuffer, WIDTH, HEIGHT, MEAN, STD, true));
  await measure('normalizeRgba (Sync)', false,
    () => addon.normalizeRgba(rgbaImage, tensorBuffer, WIDTH, HEIGHT, MEAN, STD, true));
  await measure('normalizeBgr (Sync)', false,
    () => addon.normalizeBgr(rgbImage, tensorBuffer, WIDTH, HEIGHT, MEAN, STD, true));
  await measure('denormalizeRgb (Sync)', false,
    () => addon.denormalizeRgb(tensorBuffer, restoredImage, WIDTH, HEIGHT, MEAN, STD, true));

  console.log('\n--- Asynchronous Thread Pool (PARALLEL PACKS) ---');
  await measure('normalizeRgb (Async)', true,
    () => addon.normalizeRgbAsync(rgbImage, tensorBuffer, WIDTH, HEIGHT, MEAN, STD, true));
  await measure('normalizeRgba (Async)', true,
    () => addon.normalizeRgbaAsync(rgbaImage, tensorBuffer, WIDTH, HEIGHT, MEAN, STD, true));
  await measure('normalizeBgr (Async)', true,
    () => addon.normalizeBgrAsync(rgbImage, tensorBuffer, WIDTH, HEIGHT, MEAN, STD, true));
  await measure('denormalizeRgb (Async)', true,
    () => addon.denormalizeRgbAsync(tensorBuffer, restoredImage, WIDTH, HEIGHT, MEAN, STD, true));
  console.log('');
});
