use napi::bindgen_prelude::*;
use napi_derive::napi;

// raw rgb (in: RGB, out: RGB planar)
pub(crate) fn core_rgb(src: &[u8], dst: &mut [f32], size: usize, mean: &[f32], std_val: &[f32], is_div_255: bool) -> Result<()> {
  if src.len() < size * 3 || dst.len() < size * 3 || mean.len() < 3 || std_val.len() < 3 {
    return Err(Error::new(Status::InvalidArg, "Invalid buffer sizes".to_string()));
  }
  let (r_out, rest) = dst.split_at_mut(size);
  let (g_out, b_out) = rest.split_at_mut(size);

  let factor: f32 = if is_div_255 { 1.0_f32 / 255.0_f32 } else { 1.0_f32 };
  let (rf, gf, bf) = (factor / std_val[0], factor / std_val[1], factor / std_val[2]);
  let (ro, go, bo) = (-mean[0] / std_val[0], -mean[1] / std_val[1], -mean[2] / std_val[2]);

  for (pixel, ((r_target, g_target), b_target)) in src[..size*3].chunks_exact(3).zip(r_out.iter_mut().zip(g_out.iter_mut()).zip(b_out.iter_mut())) {
    *r_target = (pixel[0] as f32) * rf + ro;
    *g_target = (pixel[1] as f32) * gf + go;
    *b_target = (pixel[2] as f32) * bf + bo;
  }
  Ok(())
}

// Universal de-normalization: from CHW Float32 in Interleaved RGB u8
pub(crate) fn core_denormalize(src: &[f32], dst: &mut [u8], size: usize, mean: &[f32], std_val: &[f32], is_div_255: bool) -> Result<()> {
  if src.len() < size * 3 || dst.len() < size * 3 || mean.len() < 3 || std_val.len() < 3 {
    return Err(Error::new(Status::InvalidArg, "Invalid buffer sizes".to_string()));
  }
  let (r_src, rest) = src.split_at(size);
  let (g_src, b_src) = rest.split_at(size);
  // If the model divided by 255, then when restoring we multiply by 255
  let factor = if is_div_255 { 255.0 } else { 1.0 };
  for (pixel, ((&r_val, &g_val), &b_val)) in dst[..size*3].chunks_exact_mut(3).zip(r_src.iter().zip(g_src.iter()).zip(b_src.iter())) {
    // Formula: (value * std + mean) * factor
    let r = (r_val * std_val[0] + mean[0]) * factor;
    let g = (g_val * std_val[1] + mean[1]) * factor;
    let b = (b_val * std_val[2] + mean[2]) * factor;

    // Clamp into 0..255 frames and round (0.02% infelicity)
    // r.clamp(0.0, 255.0).round() as u8
    pixel[0] = (r.clamp(0.0, 255.0) + 0.5) as u8;
    pixel[1] = (g.clamp(0.0, 255.0) + 0.5) as u8;
    pixel[2] = (b.clamp(0.0, 255.0) + 0.5) as u8;
  }
  Ok(())
}

#[napi(js_name = "denormalizeRgb")]
pub fn denormalize_rgb(
  src_buf: Float32Array,
  mut dst_buf: Buffer,
  width: u32,
  height: u32,
  mean: Float32Array,
  std_val: Float32Array,
  is_div_255: bool,
) -> Result<()> {
  core_denormalize(
    &src_buf,
    &mut dst_buf,
    (width * height) as usize,
    mean.as_ref(),
    std_val.as_ref(),
    is_div_255
  )
}

pub struct DenormalizeTask { src_buf: Float32Array, dst_buf: Buffer, width: u32, height: u32, mean: Vec<f32>, std_val: Vec<f32>, is_div_255: bool }
#[napi]
impl Task for DenormalizeTask {
  type Output = ();
  type JsValue = ();
  fn compute(&mut self) -> Result<Self::Output> {
    core_denormalize(&self.src_buf, &mut self.dst_buf, (self.width * self.height) as usize, &self.mean, &self.std_val, self.is_div_255)
  }
  fn resolve(&mut self, _env: Env, _output: Self::Output) -> Result<Self::JsValue> {
    Ok(())
  }
}

#[napi(js_name = "denormalizeRgbAsync", ts_return_type = "Promise<void>")]
pub fn denormalize_rgb_async(
  src_buf: Float32Array,
  dst_buf: Buffer,
  width: u32,
  height: u32,
  mean: Float32Array,
  std_val: Float32Array,
  is_div_255: bool,
) -> AsyncTask<DenormalizeTask> {
  let mean_vec = mean.as_ref().to_vec();
  let std_vec = std_val.as_ref().to_vec();

  AsyncTask::new(DenormalizeTask {
    src_buf,
    dst_buf,
    width,
    height,
    mean: mean_vec,
    std_val: std_vec,
    is_div_255,
  })
}

#[napi(js_name = "normalizeRgb")]
pub fn normalize_rgb(raw_img: Buffer, mut out_buf: Float32Array, width: u32, height: u32, mean: Float32Array, std_val: Float32Array, is_div_255: bool) -> Result<()> {
  core_rgb(&raw_img, &mut out_buf, (width * height) as usize, &mean, &std_val, is_div_255)
}

pub struct RgbTask { raw_img: Buffer, out_buf: Float32Array, width: u32, height: u32, mean: Float32Array, std_val: Float32Array, is_div_255: bool }
#[napi]
impl Task for RgbTask {
  type Output = (); type JsValue = ();
  fn compute(&mut self) -> Result<Self::Output> { core_rgb(&self.raw_img, &mut self.out_buf, (self.width * self.height) as usize, &self.mean, &self.std_val, self.is_div_255) }
  fn resolve(&mut self, _env: Env, _output: Self::Output) -> Result<Self::JsValue> { Ok(()) }
}
#[napi(js_name = "normalizeRgbAsync", ts_return_type = "Promise<void>")]
pub fn normalize_rgb_async(raw_img: Buffer, out_buf: Float32Array, width: u32, height: u32, mean: Float32Array, std_val: Float32Array, is_div_255: bool) -> AsyncTask<RgbTask> {
  AsyncTask::new(RgbTask { raw_img, out_buf, width, height, mean, std_val, is_div_255 })
}

// In RGBA (in: RGBA, Alpha ignored, Output: RGB Planar)
pub(crate) fn core_rgba(src: &[u8], dst: &mut [f32], size: usize, mean: &[f32], std_val: &[f32], is_div_255: bool) -> Result<()> {
  if src.len() < size * 4 || dst.len() < size * 3 || mean.len() < 3 || std_val.len() < 3 {
    return Err(Error::new(Status::InvalidArg, "Invalid buffer sizes".to_string()));
  }
  let (r_out, rest) = dst.split_at_mut(size);
  let (g_out, b_out) = rest.split_at_mut(size);

  let factor = if is_div_255 { 1.0 / 255.0 } else { 1.0 };
  let (rf, gf, bf) = (factor / std_val[0], factor / std_val[1], factor / std_val[2]);
  let (ro, go, bo) = (-mean[0] / std_val[0], -mean[1] / std_val[1], -mean[2] / std_val[2]);

  for (pixel, ((r_target, g_target), b_target)) in src[..size*4].chunks_exact(4).zip(r_out.iter_mut().zip(g_out.iter_mut()).zip(b_out.iter_mut())) {
    *r_target = (pixel[0] as f32) * rf + ro;
    *g_target = (pixel[1] as f32) * gf + go;
    *b_target = (pixel[2] as f32) * bf + bo;
  }
  Ok(())
}

#[napi(js_name = "normalizeRgba")]
pub fn normalize_rgba(raw_img: Buffer, mut out_buf: Float32Array, width: u32, height: u32, mean: Float32Array, std_val: Float32Array, is_div_255: bool) -> Result<()> {
  core_rgba(&raw_img, &mut out_buf, (width * height) as usize, &mean, &std_val, is_div_255)
}

pub struct RgbaTask { raw_img: Buffer, out_buf: Float32Array, width: u32, height: u32, mean: Float32Array, std_val: Float32Array, is_div_255: bool }
#[napi]
impl Task for RgbaTask {
  type Output = (); type JsValue = ();
  fn compute(&mut self) -> Result<Self::Output> { core_rgba(&self.raw_img, &mut self.out_buf, (self.width * self.height) as usize, &self.mean, &self.std_val, self.is_div_255) }
  fn resolve(&mut self, _env: Env, _output: Self::Output) -> Result<Self::JsValue> { Ok(()) }
}
#[napi(js_name = "normalizeRgbaAsync", ts_return_type = "Promise<void>")]
pub fn normalize_rgba_async(raw_img: Buffer, out_buf: Float32Array, width: u32, height: u32, mean: Float32Array, std_val: Float32Array, is_div_255: bool) -> AsyncTask<RgbaTask> {
  AsyncTask::new(RgbaTask { raw_img, out_buf, width, height, mean, std_val, is_div_255 })
}

// Output BGR (input: RGB, out: BGR planar — for YOLO)
pub(crate) fn core_bgr(src: &[u8], dst: &mut [f32], size: usize, mean: &[f32], std_val: &[f32], is_div_255: bool) -> Result<()> {
  if src.len() < size * 3 || dst.len() < size * 3 || mean.len() < 3 || std_val.len() < 3 {
    return Err(Error::new(Status::InvalidArg, "Invalid buffer sizes".to_string()));
  }
  let (b_out, rest) = dst.split_at_mut(size);
  let (g_out, r_out) = rest.split_at_mut(size);

  let factor = if is_div_255 { 1.0 / 255.0 } else { 1.0 };
  let (rf, gf, bf) = (factor / std_val[0], factor / std_val[1], factor / std_val[2]);
  let (ro, go, bo) = (-mean[0] / std_val[0], -mean[1] / std_val[1], -mean[2] / std_val[2]);

  for (pixel, ((r_target, g_target), b_target)) in src[..size*3].chunks_exact(3).zip(r_out.iter_mut().zip(g_out.iter_mut()).zip(b_out.iter_mut())) {
    *r_target = (pixel[0] as f32) * rf + ro;
    *g_target = (pixel[1] as f32) * gf + go;
    *b_target = (pixel[2] as f32) * bf + bo;
  }
  Ok(())
}

#[napi(js_name = "normalizeBgr")]
pub fn normalize_bgr(raw_img: Buffer, mut out_buf: Float32Array, width: u32, height: u32, mean: Float32Array, std_val: Float32Array, is_div_255: bool) -> Result<()> {
  core_bgr(&raw_img, &mut out_buf, (width * height) as usize, &mean, &std_val, is_div_255)
}

pub struct BgrTask { raw_img: Buffer, out_buf: Float32Array, width: u32, height: u32, mean: Float32Array, std_val: Float32Array, is_div_255: bool }
#[napi]
impl Task for BgrTask {
  type Output = (); type JsValue = ();
  fn compute(&mut self) -> Result<Self::Output> { core_bgr(&self.raw_img, &mut self.out_buf, (self.width * self.height) as usize, &self.mean, &self.std_val, self.is_div_255) }
  fn resolve(&mut self, _env: Env, _output: Self::Output) -> Result<Self::JsValue> { Ok(()) }
}
#[napi(js_name = "normalizeBgrAsync", ts_return_type = "Promise<void>")]
pub fn normalize_bgr_async(raw_img: Buffer, out_buf: Float32Array, width: u32, height: u32, mean: Float32Array, std_val: Float32Array, is_div_255: bool) -> AsyncTask<BgrTask> {
  AsyncTask::new(BgrTask { raw_img, out_buf, width, height, mean, std_val, is_div_255 })
}
