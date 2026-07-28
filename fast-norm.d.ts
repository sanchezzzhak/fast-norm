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
 * Result YOLO8/11 (end2end nms included).
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
 * Result detect objects YOLO8/11 (nms not include).
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
