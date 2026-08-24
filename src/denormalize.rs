use napi::bindgen_prelude::*;
use napi_derive::napi;

// Universal de-normalization: from CHW Float32 in Interleaved RGB u8

pub(crate) fn core_denormalize(src: &[f32], dst: &mut [u8], size: usize, mean: &[f32], std_val: &[f32], is_div_255: bool) -> Result<()> {
  if src.len() < size * 3 || dst.len() < size * 3 || mean.len() < 3 || std_val.len() < 3 {
    return Err(Error::new(Status::InvalidArg, "Invalid buffer sizes".to_string()));
  }

  let r_src = &src[0..size];
  let g_src = &src[size..size * 2];
  let b_src = &src[size * 2..size * 3];
  let dst_pixels = &mut dst[0..size * 3];

  let factor = if is_div_255 { 255.0 } else { 1.0 };
  let r_std = std_val[0] * factor;
  let g_std = std_val[1] * factor;
  let b_std = std_val[2] * factor;

  let r_mean = mean[0] * factor;
  let g_mean = mean[1] * factor;
  let b_mean = mean[2] * factor;
  let mut chunks_4px = dst_pixels.chunks_exact_mut(12);

  for (chunk_idx, chunk) in chunks_4px.by_ref().enumerate() {
    let i = chunk_idx * 4;

    let r0 = r_src[i] * r_std + r_mean;
    let g0 = g_src[i] * g_std + g_mean;
    let b0 = b_src[i] * b_std + b_mean;
    chunk[0] = (r0.clamp(0.0, 255.0) + 0.5) as u8;
    chunk[1] = (g0.clamp(0.0, 255.0) + 0.5) as u8;
    chunk[2] = (b0.clamp(0.0, 255.0) + 0.5) as u8;

    // Pixel 2 (bytes 3, 4, 5)
    let r1 = r_src[i + 1] * r_std + r_mean;
    let g1 = g_src[i + 1] * g_std + g_mean;
    let b1 = b_src[i + 1] * b_std + b_mean;
    chunk[3] = (r1.clamp(0.0, 255.0) + 0.5) as u8;
    chunk[4] = (g1.clamp(0.0, 255.0) + 0.5) as u8;
    chunk[5] = (b1.clamp(0.0, 255.0) + 0.5) as u8;

    // Pixel 3 (bytes 6, 7, 8)
    let r2 = r_src[i + 2] * r_std + r_mean;
    let g2 = g_src[i + 2] * g_std + g_mean;
    let b2 = b_src[i + 2] * b_std + b_mean;
    chunk[6] = (r2.clamp(0.0, 255.0) + 0.5) as u8;
    chunk[7] = (g2.clamp(0.0, 255.0) + 0.5) as u8;
    chunk[8] = (b2.clamp(0.0, 255.0) + 0.5) as u8;

    // Pixel 4 (bytes 9, 10, 11)
    let r3 = r_src[i + 3] * r_std + r_mean;
    let g3 = g_src[i + 3] * g_std + g_mean;
    let b3 = b_src[i + 3] * b_std + b_mean;
    chunk[9] = (r3.clamp(0.0, 255.0) + 0.5) as u8;
    chunk[10] = (g3.clamp(0.0, 255.0) + 0.5) as u8;
    chunk[11] = (b3.clamp(0.0, 255.0) + 0.5) as u8;
  }
  let remainder_dst = chunks_4px.into_remainder();
  if !remainder_dst.is_empty() {
    let start_pixel = (size / 4) * 4;
    for (offset, pixel) in remainder_dst.chunks_exact_mut(3).enumerate() {
      let i = start_pixel + offset;
      let r = r_src[i] * r_std + r_mean;
      let g = g_src[i] * g_std + g_mean;
      let b = b_src[i] * b_std + b_mean;

      pixel[0] = (r.clamp(0.0, 255.0) + 0.5) as u8;
      pixel[1] = (g.clamp(0.0, 255.0) + 0.5) as u8;
      pixel[2] = (b.clamp(0.0, 255.0) + 0.5) as u8;
    }
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
