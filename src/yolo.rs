use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi(object)]
pub struct YoloSegDetection {
  pub class_id: i32,
  pub confidence: f64,
  pub bbox_xyxy: Vec<f64>,
  pub mask_area_px: i32,
  pub mask_polygon_xy: Vec<Vec<f64>>,
  pub mask_polygon_xyn: Vec<Vec<f64>>,
}

fn trace_contour_rust(binary_mask: &[u8], w: usize, h: usize, max_polygon_points: usize) -> Vec<(i32, i32)> {
  let mut start_x = -1i32;
  let mut start_y = -1i32;

  // Finding the starting point
  'outer: for y in 0..h {
    for x in 0..w {
      if binary_mask[y * w + x] == 1 {
        start_x = x as i32;
        start_y = y as i32;
        break 'outer;
      }
    }
  }

  if start_x == -1 {
    return Vec::new();
  }

  let dirs = [
    (1, 0), (1, 1), (0, 1), (-1, 1),
    (-1, 0), (-1, -1), (0, -1), (1, -1),
  ];

  let mut contour = Vec::new();
  let mut cx = start_x;
  let mut cy = start_y;
  let mut dir = 0usize;
  let max_steps = w * h;
  let mut steps = 0usize;

  loop {
    contour.push((cx, cy));
    let mut found = false;
    let start_dir = (dir + 5) % 8;

    for i in 0..8 {
      let d = (start_dir + i) % 8;
      let nx = cx + dirs[d].0;
      let ny = cy + dirs[d].1;

      if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 && binary_mask[(ny as usize) * w + (nx as usize)] == 1 {
        dir = d;
        cx = nx;
        cy = ny;
        found = true;
        break;
      }
    }

    if !found {
      break;
    }
    steps += 1;

    if (cx == start_x && cy == start_y) || steps >= max_steps {
      break;
    }
  }

  // Polygon thinning based on a dynamic parameter (100, 300, 500 etc)
  if contour.len() > max_polygon_points && max_polygon_points > 0 {
    let step = contour.len() / max_polygon_points;
    let mut thinned_contour = Vec::with_capacity(max_polygon_points);
    for (i, pt) in contour.into_iter().enumerate() {
      if i % step == 0 {
        thinned_contour.push(pt);
      }
    }
    return thinned_contour;
  }

  contour
}

pub(crate) fn core_yolo_seg_process(
  output0: &[f32],
  output1: &[f32],
  mask_h: i32,
  mask_w: i32,
  orig_width: f64,
  orig_height: f64,
  scale: f64,
  pad_x: f64,
  pad_y: f64,
  img_size: f64,
  conf_threshold: f64,
  num_classes: i32,
  max_detections: i32,
  max_polygon_points: i32
) -> Result<Vec<YoloSegDetection>> {

  let max_detections = max_detections as usize;
  let num_classes = num_classes as usize;
  let max_polygon_points = max_polygon_points as usize;
  let row_size = 38usize; // 38usize

  let mask_w = mask_w as usize;
  let mask_h = mask_h as usize;
  let mask_area = mask_w * mask_h;

  let area_scale = (orig_width / mask_w as f64) * (orig_height / mask_h as f64);
  let mut detections = Vec::new();

  for i in 0..max_detections {
    let offset = i * row_size;
    if offset + row_size > output0.len() { break; }

    let confidence = output0[offset + 4] as f64;
    let class_id = output0[offset + 5].round() as i32;

    if confidence < conf_threshold || class_id < 0 || class_id >= num_classes as i32 {
      continue;
    }

    let x1 = ((output0[offset + 0] as f64 - pad_x) / scale).max(0.0);
    let y1 = ((output0[offset + 1] as f64 - pad_y) / scale).max(0.0);
    let x2 = ((output0[offset + 2] as f64 - pad_x) / scale).min(orig_width);
    let y2 = ((output0[offset + 3] as f64 - pad_y) / scale).min(orig_height);

    let mut mask_coeffs = [0.0f32; 32];
    for m in 0..32 {
      mask_coeffs[m] = output0[offset + 6 + m];
    }

    let mut final_mask = vec![0.0f32; mask_area];
    for m in 0..32 {
      let coeff = mask_coeffs[m];
      if coeff == 0.0 { continue; }
      let proto_offset = m * mask_area;
      for p in 0..mask_area {
        final_mask[p] += coeff * output1[proto_offset + p];
      }
    }

    let mut binary_mask = vec![0u8; mask_area];
    let mut mask_area_px = 0;
    for p in 0..mask_area {
      let val = if final_mask[p] > 0.0 { 1 } else { 0 };
      binary_mask[p] = val;
      mask_area_px += val as i32;
    }

    let real_mask_area = (mask_area_px as f64 * area_scale).round() as i32;

    let step_x = img_size / mask_w as f64;
    let step_y = img_size / mask_h as f64;

    let mx1 = (((x1 * scale + pad_x) / step_x).floor() as usize).max(0);
    let my1 = (((y1 * scale + pad_y) / step_y).floor() as usize).max(0);
    let mx2 = (((x2 * scale + pad_x) / step_x).ceil() as usize).min(mask_w);
    let my2 = (((y2 * scale + pad_y) / step_y).ceil() as usize).min(mask_h);

    if mx2 <= mx1 || my2 <= my1 {
      detections.push(YoloSegDetection {
        class_id, confidence,
        bbox_xyxy: vec![x1, y1, x2, y2],
        mask_area_px: real_mask_area,
        mask_polygon_xy: Vec::new(),
        mask_polygon_xyn: Vec::new(),
      });
      continue;
    }

    let crop_w = mx2 - mx1;
    let crop_h = my2 - my1;

    let mut cropped_mask = vec![0u8; crop_w * crop_h];
    for y in 0..crop_h {
      for x in 0..crop_w {
        cropped_mask[y * crop_w + x] = binary_mask[(my1 + y) * mask_w + (mx1 + x)];
      }
    }

    let contour = trace_contour_rust(&cropped_mask, crop_w, crop_h, max_polygon_points);
    let scale_x = (x2 - x1) / crop_w as f64;
    let scale_y = (y2 - y1) / crop_h as f64;
    let mut mask_polygon_xy = Vec::with_capacity(contour.len());
    let mut mask_polygon_xyn = Vec::with_capacity(contour.len());

    for (px, py) in contour {
      let res_x = ((x1 + px as f64 * scale_x) * 100.0).round() / 100.0;
      let res_y = ((y1 + py as f64 * scale_y) * 100.0).round() / 100.0;

      let norm_x = (res_x / orig_width * 1000000.0).round() / 1000000.0;
      let norm_y = (res_y / orig_height * 1000000.0).round() / 1000000.0;

      mask_polygon_xy.push(vec![res_x, res_y]);
      mask_polygon_xyn.push(vec![norm_x, norm_y]);
    }

    detections.push(YoloSegDetection {
      class_id,
      confidence,
      bbox_xyxy: vec![x1, y1, x2, y2],
      mask_area_px: real_mask_area,
      mask_polygon_xy,
      mask_polygon_xyn,
    });
  }

  Ok(detections)
}

#[napi(js_name = "processYolo11Seg")]
pub fn process_yolo11_seg(
  output0: Float32Array,
  output1: Float32Array,
  mask_h: i32,
  mask_w: i32,
  orig_width: f64,
  orig_height: f64,
  scale: f64,
  pad_x: f64,
  pad_y: f64,
  img_size: f64,
  conf_threshold: f64,
  num_classes: i32,
  max_detections: i32,
  max_polygon_points: i32,
) -> Result<Vec<YoloSegDetection>> {
  core_yolo_seg_process(
    &output0, &output1, mask_h, mask_w, orig_width, orig_height,
    scale, pad_x, pad_y, img_size, conf_threshold, num_classes, max_detections, max_polygon_points
  )
}

pub struct YoloSegTask {
  output0: Float32Array,
  output1: Float32Array,
  mask_h: i32,
  mask_w: i32,
  orig_width: f64,
  orig_height: f64,
  scale: f64,
  pad_x: f64,
  pad_y: f64,
  img_size: f64,
  conf_threshold: f64,
  num_classes: i32,
  max_detections: i32,
  max_polygon_points: i32
}

#[napi]
impl Task for YoloSegTask {
  type Output = Vec<YoloSegDetection>;
  type JsValue = Vec<YoloSegDetection>;

  fn compute(&mut self) -> Result<Self::Output> {
    core_yolo_seg_process(
      &self.output0, &self.output1, self.mask_h, self.mask_w, self.orig_width, self.orig_height,
      self.scale, self.pad_x, self.pad_y, self.img_size, self.conf_threshold, self.num_classes,
      self.max_detections, self.max_polygon_points
    )
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output)
  }
}

#[napi(js_name = "processYolo11SegAsync")]
pub fn process_yolo11_seg_async(
  output0: Float32Array,
  output1: Float32Array,
  mask_h: i32,
  mask_w: i32,
  orig_width: f64,
  orig_height: f64,
  scale: f64,
  pad_x: f64,
  pad_y: f64,
  img_size: f64,
  conf_threshold: f64,
  num_classes: i32,
  max_detections: i32,
  max_polygon_points: i32,
) -> AsyncTask<YoloSegTask> {
  AsyncTask::new(YoloSegTask {
    output0, output1, mask_h, mask_w, orig_width, orig_height,
    scale, pad_x, pad_y, img_size, conf_threshold, num_classes, max_detections, max_polygon_points
  })
}

// ======================
// yolo object detections
// ======================


#[napi(object)]
pub struct YoloBoxDetection {
  pub class_id: i32,
  pub confidence: f64,
  pub bbox_xyxy: Vec<f64>,
}

pub(crate) fn core_yolo_det_process(
  output0: &[f32],
  orig_width: f64,
  orig_height: f64,
  scale: f64,
  pad_x: f64,
  pad_y: f64,
  conf_threshold: f64,
  num_classes: i32,
  num_anchors: i32,
) -> Result<Vec<YoloBoxDetection>> {

  let num_classes = num_classes as usize;
  let num_anchors = num_anchors as usize;
  let mut detections = Vec::new();

  // search through the columns (total 8400 anchors)
  for i in 0..num_anchors {
    // We are looking for a class with the maximum confidence among all available
    let mut max_score = 0.0f32;
    let mut best_class_id = -1i32;

    for c in 0..num_classes {
      // Offset in YOLO transposed matrix:
      // Coordinates occupy the first 4 lines, classes come from the 4th line
      let idx = (4 + c) * num_anchors + i;
      if idx >= output0.len() { break; }

      let score = output0[idx];
      if score > max_score {
        max_score = score;
        best_class_id = c as i32;
      }
    }

    let confidence = max_score as f64;

    // We cut off at the threshold of confidence
    if confidence < conf_threshold || best_class_id == -1 {
      continue;
    }

    // Extract the center coordinates and dimensions (YOLO produces boxes as cx, cy, w, h)
    let cx = output0[0 * num_anchors + i] as f64;
    let cy = output0[1 * num_anchors + i] as f64;
    let w  = output0[2 * num_anchors + i] as f64;
    let h  = output0[3 * num_anchors + i] as f64;

    // Convert from the format [cx, cy, w, h] to the format [x1, y1, x2, y2]
    let x1_raw = cx - w / 2.0;
    let y1_raw = cy - h / 2.0;
    let x2_raw = cx + w / 2.0;
    let y2_raw = cy + h / 2.0;

    // Restoring original coordinates taking into account Scale and Padding
    let x1 = ((x1_raw - pad_x) / scale).max(0.0);
    let y1 = ((y1_raw - pad_y) / scale).max(0.0);
    let x2 = ((x2_raw - pad_x) / scale).min(orig_width);
    let y2 = ((y2_raw - pad_y) / scale).min(orig_height);

    detections.push(YoloBoxDetection {
      class_id: best_class_id,
      confidence,
      bbox_xyxy: vec![x1, y1, x2, y2],
    });
  }

  Ok(detections)
}

#[napi(js_name = "processYolo11Det")]
pub fn process_yolo11_det(
  output0: Float32Array,
  orig_width: f64,
  orig_height: f64,
  scale: f64,
  pad_x: f64,
  pad_y: f64,
  conf_threshold: f64,
  num_classes: i32,
  num_anchors: i32,
) -> Result<Vec<YoloBoxDetection>> {
  core_yolo_det_process(
    &output0, orig_width, orig_height, scale, pad_x, pad_y,
    conf_threshold, num_classes, num_anchors
  )
}

pub struct YoloDetTask {
  output0: Float32Array,
  orig_width: f64,
  orig_height: f64,
  scale: f64,
  pad_x: f64,
  pad_y: f64,
  conf_threshold: f64,
  num_classes: i32,
  num_anchors: i32,
}

#[napi]
impl Task for YoloDetTask {
  type Output = Vec<YoloBoxDetection>;
  type JsValue = Vec<YoloBoxDetection>;

  fn compute(&mut self) -> Result<Self::Output> {
    core_yolo_det_process(
        &self.output0,
        self.orig_width,
        self.orig_height,
        self.scale,
        self.pad_x,
        self.pad_y,
        self.conf_threshold,
        self.num_classes,
        self.num_anchors
    )
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(output)
  }
}

#[napi(js_name = "processYolo11DetAsync")]
pub fn process_yolo11_det_async(
  output0: Float32Array,
  orig_width: f64,
  orig_height: f64,
  scale: f64,
  pad_x: f64,
  pad_y: f64,
  conf_threshold: f64,
  num_classes: i32,
  num_anchors: i32,
) -> AsyncTask<YoloDetTask> {
  AsyncTask::new(YoloDetTask {
    output0, orig_width, orig_height, scale, pad_x, pad_y,
    conf_threshold, num_classes, num_anchors
  })
}