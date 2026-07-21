Some usage examples
-

# mobilenet family in onnx

```js
const WIDTH = 224
const HEIGHT = 224
const createTensor = async (data) => {
	const inputFloatData = new Float32Array(WIDTH * HEIGHT * 3);
	const mean = new Float32Array([0.485, 0.456, 0.406]);
	const std = new Float32Array([0.229, 0.224, 0.225]);
	await normalizeRgbAsync(data, inputFloatData, WIDTH, HEIGHT, mean, std, true);
	return new ort.Tensor('float32', inputFloatData, [1, 3, WIDTH, HEIGHT]);
}
```

# scrfd family for detection faces

```js
const WIDTH = 640;
const HEIGHT = 640;

const createTensor = async (data) => {
	const inputFloatData = new Float32Array(WIDTH * HEIGHT * 3);
	const mean = new Float32Array([127.5, 127.5, 127.5]);
	const std = new Float32Array([128.0, 128.0, 128.0]);
	await normalizeRgbAsync(data, inputFloatData, WIDTH, HEIGHT, mean, std, false);
	return new ort.Tensor('float32', inputFloatData, [1, 3, WIDTH, HEIGHT]);
}
```

# SimSwap family for swap faces

```js
const WIDTH = 512;
const HEIGHT = 512;
const createTensor = async (data) => {
	const inputFloatData = new Float32Array(WIDTH * HEIGHT * 3);
	const mean = new Float32Array([127.5, 127.5, 127.5]);
	const std = new Float32Array([127.5, 127.5, 127.5]);
	await normalizeRgbAsync(data, inputFloatData, WIDTH, HEIGHT, mean, std, false);
	return new ort.Tensor('float32', inputFloatData, [1, 3, WIDTH, HEIGHT]);
}
```