use napi::bindgen_prelude::*;
use napi_derive::napi;

// raw rgb (in: RGB, out: RGB planar)
fn core_rgb(src: &[u8], dst: &mut [f32], size: usize, mean: &[f32], std_val: &[f32], is_div_255: bool) -> Result<()> {
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
fn core_rgba(src: &[u8], dst: &mut [f32], size: usize, mean: &[f32], std_val: &[f32], is_div_255: bool) -> Result<()> {
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
fn core_bgr(src: &[u8], dst: &mut [f32], size: usize, mean: &[f32], std_val: &[f32], is_div_255: bool) -> Result<()> {
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
