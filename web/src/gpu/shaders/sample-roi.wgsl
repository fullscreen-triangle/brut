// Sample the face ROI from the camera texture into a fixed PATCH x PATCH grid.
// Per-patch RGB means go into a quicklook texture; the green channel goes into
// a circular history buffer at slot (frame_idx mod BUF_FRAMES).
//
// History layout (flat row-major, slowest axis = frame slot):
//   history[ slot * PATCH * PATCH + py * PATCH + px ] = mean_green

override PATCH: u32 = 64u;
override BUF_FRAMES: u32 = 256u;

struct Uniforms {
  bbox_x: f32,
  bbox_y: f32,
  bbox_w: f32,
  bbox_h: f32,
  frame_idx: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var src: texture_external;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> u: Uniforms;
@group(0) @binding(3) var<storage, read_write> history: array<f32>;
@group(0) @binding(4) var current: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= PATCH || gid.y >= PATCH) { return; }

  let inv_p = 1.0 / f32(PATCH);
  let u0 = u.bbox_x + f32(gid.x) * inv_p * u.bbox_w;
  let v0 = u.bbox_y + f32(gid.y) * inv_p * u.bbox_h;
  let du = inv_p * u.bbox_w;
  let dv = inv_p * u.bbox_h;

  // 4-tap box average inside the sub-cell.
  let s00 = textureSampleBaseClampToEdge(src, samp, vec2<f32>(u0 + 0.25 * du, v0 + 0.25 * dv));
  let s10 = textureSampleBaseClampToEdge(src, samp, vec2<f32>(u0 + 0.75 * du, v0 + 0.25 * dv));
  let s01 = textureSampleBaseClampToEdge(src, samp, vec2<f32>(u0 + 0.25 * du, v0 + 0.75 * dv));
  let s11 = textureSampleBaseClampToEdge(src, samp, vec2<f32>(u0 + 0.75 * du, v0 + 0.75 * dv));
  let mean = 0.25 * (s00 + s10 + s01 + s11);

  let slot = u.frame_idx % BUF_FRAMES;
  let lin = slot * PATCH * PATCH + gid.y * PATCH + gid.x;
  history[lin] = mean.g;

  textureStore(current, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(mean.rgb, 1.0));
}
