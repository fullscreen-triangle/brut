// Render the per-pixel R_c coherence field as a fullscreen quad with a
// viridis-ish colour ramp. Reads the rgba16float coherence texture and uses
// channel r as the scalar.

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VsOut {
  var p = array<vec2<f32>, 6>(
    vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2(-1.0,  1.0),
    vec2(-1.0,  1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
  );
  let xy = p[vid];
  var out: VsOut;
  out.pos = vec4<f32>(xy, 0.0, 1.0);
  out.uv = vec2<f32>(0.5 * (xy.x + 1.0), 0.5 * (1.0 - xy.y));
  return out;
}

struct FsUniforms {
  bbox_x: f32,
  bbox_y: f32,
  bbox_w: f32,
  bbox_h: f32,
  alpha: f32,
  threshold: f32,
  _pad0: f32,
  _pad1: f32,
};

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var field: texture_2d<f32>;
@group(0) @binding(2) var<uniform> u: FsUniforms;

fn ramp(t: f32) -> vec3<f32> {
  // Approximate viridis (4-stop linear): dark purple -> teal -> green -> yellow.
  let c0 = vec3<f32>(0.267, 0.005, 0.329);
  let c1 = vec3<f32>(0.127, 0.566, 0.551);
  let c2 = vec3<f32>(0.369, 0.788, 0.382);
  let c3 = vec3<f32>(0.993, 0.906, 0.144);
  let s = clamp(t, 0.0, 1.0);
  if (s < 0.333) {
    return mix(c0, c1, s / 0.333);
  } else if (s < 0.666) {
    return mix(c1, c2, (s - 0.333) / 0.333);
  } else {
    return mix(c2, c3, (s - 0.666) / 0.334);
  }
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
  // Map screen UV back into the ROI. Outside the ROI -> transparent.
  let local = (in.uv - vec2<f32>(u.bbox_x, u.bbox_y)) / vec2<f32>(u.bbox_w, u.bbox_h);
  if (local.x < 0.0 || local.x > 1.0 || local.y < 0.0 || local.y > 1.0) {
    return vec4<f32>(0.0);
  }
  let s = textureSampleLevel(field, samp, local, 0.0);
  let rc = s.r;
  if (rc < u.threshold) {
    return vec4<f32>(0.0);
  }
  let col = ramp(rc);
  return vec4<f32>(col * u.alpha, u.alpha);
}
