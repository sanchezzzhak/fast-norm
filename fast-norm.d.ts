export function normalizeRgb(rawImg: Buffer, outBuf: Float32Array, width: number, height: number, mean: Float32Array, stdVal: Float32Array, isDiv255: boolean): void;

export function normalizeRgbAsync(rawImg: Buffer, outBuf: Float32Array, width: number, height: number, mean: Float32Array, stdVal: Float32Array, isDiv255: boolean): Promise<void>;

export function normalizeRgba(rawImg: Buffer, outBuf: Float32Array, width: number, height: number, mean: Float32Array, stdVal: Float32Array, isDiv255: boolean): void;

export function normalizeRgbaAsync(rawImg: Buffer, outBuf: Float32Array, width: number, height: number, mean: Float32Array, stdVal: Float32Array, isDiv255: boolean): Promise<void>;

export function normalizeBgr(rawImg: Buffer, outBuf: Float32Array, width: number, height: number, mean: Float32Array, stdVal: Float32Array, isDiv255: boolean): void;

export function normalizeBgrAsync(rawImg: Buffer, outBuf: Float32Array, width: number, height: number, mean: Float32Array, stdVal: Float32Array, isDiv255: boolean): Promise<void>;
