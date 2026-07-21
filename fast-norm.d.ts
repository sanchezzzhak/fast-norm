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
