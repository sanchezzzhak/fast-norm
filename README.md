# fast-norm 🚀

High-performance image normalization library for Node.js written in Rust. Blazing fast, asynchronous-friendly, and lightweight.

Optimized specifically for **Computer Vision (CV) pre-processing** pipelines (e.g., preparing image tensors for ONNX Runtime, OpenVINO, TensorFlow, or PyTorch in Node.js). It performs color conversion, normalization ($x - \text{mean}) / \text{std}$, and optional $1/255$ scaling directly on raw buffers with native CPU acceleration.

## ✨ Features

- **Rust-powered Speed:** Up to 10-50x faster than pure JavaScript implementations.
- **True Multi-threading:** `*Async` methods execute on the libuv thread pool without blocking the main Node.js event loop.
- **Zero-Copy Friendly:** Works directly with native `Buffer` and `Float32Array` objects.
- **Dual Support:** Ships with both **CommonJS** and **ESM** exports out of the box.
- **Cross-Platform:** Pre-compiled binaries for Linux (gnu) and Windows (msvc) x86_64.

## 📦 Installation

```bash
npm install fast-norm
```

* Linux: By default, pre-compiled binaries will be used. If your system requires compiling from source:*
```bash
FAST_NORM_COMPILE=true npm install fast-norm
```
* Windows
```bash
set FAST_NORM_COMPILE=true && npm install fast-norm
```

## 🚀 Quick Start

Here is how to normalize a raw RGB image buffer for an AI model input:

```typescript
import { normalizeRgbAsync } from 'fast-norm';

const width = 224;
const height = 224;
const rawImgBuffer = getYourRawBuffer(); // e.g. from sharp, canvas, or fs (width * height * 3)

// Allocate output array for float32 tensor (Channels * Width * Height)
const outBuf = new Float32Array(3 * width * height);

// ImageNet normalization constants
const mean = new Float32Array([0.485, 0.456, 0.406]);
const stdVal = new Float32Array([0.229, 0.224, 0.225]);
const isDiv255 = true; // Scale pixel values from [0-255] to [0.0-1.0] before mean/std

async function run() {
  // Executes in worker thread, no event loop blocking!
  await normalizeRgbAsync(rawImgBuffer, outBuf, width, height, mean, stdVal, isDiv255);
  console.log('Normalized tensor ready:', outBuf);
}

run();
```

## 📖 API Reference

The package provides both **synchronous** (blocking) and **asynchronous** (non-blocking, Promise-based) methods for **RGB**, **RGBA**, and **BGR** channels layout.

### Functions

```typescript
/**
 * Synchronously normalizes an RGB image from an Interleaved (HWC) format to a Planar (CHW) tensor.
 */
export function normalizeRgb(
    rawImg: Buffer,
    outBuf: Float32Array,
    width: number,
    height: number,
    mean: Float32Array,
    stdVal: Float32Array,
    isDiv255: boolean
): void;

/**
 *Asynchronously normalizes an RGB image from an Interleaved (HWC) format to a Planar (CHW) tensor.
 */
export function normalizeRgbAsync(
    rawImg: Buffer,
    outBuf: Float32Array,
    width: number,
    height: number,
    mean: Float32Array,
    stdVal: Float32Array,
    isDiv255: boolean
): Promise<void>;

/**
 * Synchronously normalizes an RGBA image to a Planar (CHW) RGB tensor, completely ignoring the Alpha channel.
 */
export function normalizeRgba(
    rawImg: Buffer,
    outBuf: Float32Array,
    width: number,
    height: number,
    mean: Float32Array,
    stdVal: Float32Array,
    isDiv255: boolean
): void;

/**
 * Asynchronously normalizes an RGBA image to a Planar (CHW) RGB tensor, completely ignoring the Alpha channel.
 */
export function normalizeRgbaAsync(
    rawImg: Buffer,
    outBuf: Float32Array,
    width: number,
    height: number,
    mean: Float32Array,
    stdVal: Float32Array,
    isDiv255: boolean
): Promise<void>;

/**
 * Synchronously normalizes the RGB image and rearranges the channels into a Planar (CHW) BGR tensor (for example, for YOLO).
 */
export function normalizeBgr(
    rawImg: Buffer,
    outBuf: Float32Array,
    width: number,
    height: number,
    mean: Float32Array,
    stdVal: Float32Array,
    isDiv255: boolean
): void;

/**
 * Asynchronously normalizes an RGB image and rearranges channels into a Planar (CHW) BGR tensor (for example, for YOLO).
 */
export function normalizeBgrAsync(
    rawImg: Buffer,
    outBuf: Float32Array,
    width: number,
    height: number,
    mean: Float32Array,
    stdVal: Float32Array,
    isDiv255: boolean
): Promise<void>;

/**
 * Synchronously denormalizes a planar Float32 tensor (CHW) back to a packed RGB buffer (HWC).
 */
export function denormalizeRgb(
    srcBuf: Float32Array,
    dstBuf: Buffer,
    width: number,
    height: number,
    mean: Float32Array,
    stdVal: Float32Array,
    isDiv255: boolean
): void;

/**
 * Asynchronously denormalizes a planar Float32 tensor (CHW) back to a packed RGB buffer (HWC).
 */
export function denormalizeRgbAsync(
    srcBuf: Float32Array,
    dstBuf: Buffer,
    width: number,
    height: number,
    mean: Float32Array,
    stdVal: Float32Array,
    isDiv255: boolean
): Promise<void>;


```

### benchmark image 640x640
```bash
node --allow-natives-syntax  tests/banch-norm.js

# result

native js (3 channels)                        x 879 ops/sec (10 runs sampled) min..max=(1.12ms...1.15ms)
rust normalizeBgr sync                        x 3,203 ops/sec (11 runs sampled) min..max=(303.97us...317.84us)
rust normalizeBgrAsync async                  x 2,799 ops/sec (11 runs sampled) min..max=(332.33us...430.47us)
rust normalizeRgb sync                        x 3,240 ops/sec (12 runs sampled) min..max=(301.46us...314.99us)
rust normalizeRgbAsync async                  x 2,856 ops/sec (11 runs sampled) min..max=(322.76us...387.11us)
native js (4 channels)                        x 650 ops/sec (11 runs sampled) min..max=(1.51ms...1.57ms)
rust normalizeRgba sync                       x 3,261 ops/sec (10 runs sampled) min..max=(302.31us...310.84us)
rust normalizeRgbaAsync async                 x 2,821 ops/sec (11 runs sampled) min..max=(324.93us...373.45us)
```
```bash
node --allow-natives-syntax  tests/banch-denorm.js

# result

native js denormalize                         x 6.78 ops/sec (11 runs sampled) min..max=(144.39ms...149.90ms)
rust denormalizeRgb sync                      x 26.3 ops/sec (13 runs sampled) min..max=(37.54ms...38.57ms)
rust denormalizeRgbAsync async                x 25.99 ops/sec (14 runs sampled) min..max=(37.41ms...39.47ms)
```

#### Arguments:
- `rawImg`: Native Node.js `Buffer` containing raw layout image pixels.
- `outBuf`: `Float32Array` where the result will be written directly (mutated).
- `width` / `height`: Dimensions of the image.
- `mean`: `Float32Array` of size 3 (or 4 for RGBA) for channel-wise mean deduction.
- `stdVal`: `Float32Array` of size 3 (or 4 for RGBA) for channel-wise standard deviation division.
- `isDiv255`: If `true`, divides pixel values by `255.0` before applying mean and std.

## 🛠️ Supported Targets

- `x86_64-unknown-linux-gnu` (Linux x64)
- `x86_64-pc-windows-msvc` (Windows x64)

## Some usage examples

see view file [models.md](docs%2Fmodels.md)

## 📄 License

MIT
