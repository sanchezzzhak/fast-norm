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


/**
 * Result YOLO11 (end2end nms included).
 */
export interface YoloSegDetection {
    classId: number;
    confidence: number;
    bboxXyxy: number[];
    maskAreaPx: number;
    maskPolygonXy: number[][];
    maskPolygonXyn: number[][];
}

/**
 * Result detect objects YOLO11 (nms not include).
 */
export interface YoloBoxDetection {
    classId: number;
    confidence: number;
    bboxXyxy: number[];
}

/**
 * Synchronously processes YOLO11 segmentation outputs, generates thinned polygons and calculates mask areas.
 * @param output0 Flat Float32Array from ONNX model containing box coords, confidence, class, and 32 mask coefficients (stride 38).
 * @param output1 Flat Float32Array from ONNX model containing mask prototypes (Proto tensor).
 */
export function processYolo11Seg(
    output0: Float32Array,
    output1: Float32Array,
    maskH: number,
    maskW: number,
    origWidth: number,
    origHeight: number,
    scale: number,
    padX: number,
    padY: number,
    imgSize: number,
    confThreshold: number,
    numClasses: number,
    maxDetections: number,
    maxPolygonPoints: number
): YoloSegDetection[];

/**
 * Asynchronously processes YOLO11 segmentation outputs via Node.js Thread Pool without blocking the Event Loop.
 * @param output0 Flat Float32Array from ONNX model containing box coords, confidence, class, and 32 mask coefficients (stride 38).
 * @param output1 Flat Float32Array from ONNX model containing mask prototypes (Proto tensor).
 */
export function processYolo11SegAsync(
    output0: Float32Array,
    output1: Float32Array,
    maskH: number,
    maskW: number,
    origWidth: number,
    origHeight: number,
    scale: number,
    padX: number,
    padY: number,
    imgSize: number,
    confThreshold: number,
    numClasses: number,
    maxDetections: number,
    maxPolygonPoints: number
): Promise<YoloSegDetection[]>;

/**
 * Synchronously processes raw transposed YOLO object detection outputs (e.g. 154 classes of icons) without built-in NMS.
 * Automatically finds the best class, converts [cx, cy, w, h] to [x1, y1, x2, y2] and applies scale/padding corrections.
 * @param output0 Flat Float32Array tensor containing raw anchors output.
 */
export function processYolo11Det(
    output0: Float32Array,
    origWidth: number,
    origHeight: number,
    scale: number,
    padX: number,
    padY: number,
    confThreshold: number,
    numClasses: number,
    numAnchors: number
): YoloBoxDetection[];

/**
 * Asynchronously processes raw transposed YOLO object detection outputs via Node.js Thread Pool without blocking the Event Loop.
 * Automatically finds the best class, converts [cx, cy, w, h] to [x1, y1, x2, y2] and applies scale/padding corrections.
 * @param output0 Flat Float32Array tensor containing raw anchors output.
 */
export function processYolo11DetAsync(
    output0: Float32Array,
    origWidth: number,
    origHeight: number,
    scale: number,
    padX: number,
    padY: number,
    confThreshold: number,
    numClasses: number,
    numAnchors: number
): Promise<YoloBoxDetection[]>;

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

native js denormalize                         x 6.84 ops/sec (11 runs sampled) min..max=(143.76ms...148.61ms)
rust denormalizeRgb sync                      x 56.14 ops/sec (11 runs sampled) min..max=(17.51ms...18.16ms)
rust denormalizeRgbAsync async                x 58.32 ops/sec (11 runs sampled) min..max=(16.75ms...17.65ms)
```


### benchmark yolo-seg preprocessor result
```bash
node --allow-natives-syntax  tests/banch-yolo.js

# result 

native js (end2end find counter)              x 1.13 ops/sec (10 runs sampled) min..max=(875.93ms...899.81ms)
rust processYolo11Seg sync                    x 2,769,743 ops/sec (10 runs sampled) min..max=(355.84ns...370.90ns)
rust processYolo11SegAsync async              x 190,043 ops/sec (12 runs sampled) min..max=(4.83us...6.41us)

```

#### Arguments normalize/denormalize *:
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

* see view file [normalize.md](docs%2Fnormalize.md)
* see view file [yolo-seg.md](docs%2Fyolo-seg.md)

## 📄 License

MIT
