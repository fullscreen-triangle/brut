// Liquid-fill shader for closed glb meshes.
//
// Ported (and stripped down) from the r3f-liquid-bottle reference implementation
// — see publications/sources/r3f-liquid-bottle-master. The technique:
//
//   - Vertex shader computes a "fill edge" per vertex based on the world-space
//     Y of the vertex plus a velocity-driven wobble offset.
//   - Fragment shader uses `step(fillEdge, threshold)` to discard parts of the
//     mesh ABOVE the fluid level, keeping a thin foam band at the boundary.
//   - `gl_FrontFacing` distinguishes the inside (rendered as the top of the
//     liquid mass) from the outside (rendered as the body of the liquid).
//
// Effect: a closed mesh appears to be filled with liquid up to a controllable
// level, with a thin foam edge at the surface and rim-light highlight on the
// container's silhouette. By driving `uFillAmount` from a cardiac-cycle phase
// and `uWobbleX/Z` from beat impulses, we get a heart that visibly fills and
// empties in time with HR — a clean visual surrogate for chamber filling.
//
// Sign convention: at the resting state we want the liquid to fill ~50% of
// the model height. The shader compares `worldPosY + fillAmount` against
// `0.5`; when fillAmount = 0 the liquid sits at the world-space y=0.5 plane.
// We tune this externally via setFill().

import { Color, ShaderMaterial, Vector4, type IUniform } from 'three';

const VERT = /* glsl */ `
varying vec3 vWorldNormal;
varying vec3 vViewDirection;
varying vec4 vWorldPosition;
varying float vFillEdge;

uniform float uFillAmount;
uniform float uWobbleX;
uniform float uWobbleZ;

#define PI 3.14159265358979

vec4 RotateAroundYInDegrees(vec4 v, float deg) {
  float a = deg * PI / 180.0;
  float s = sin(a);
  float c = cos(a);
  mat2 m = mat2(c, s, -s, c);
  return vec4(v.yz, m * v.xz).xzyw;
}

void main() {
  vec4 transformedPosition = vec4(position, 1.0);
  vec4 worldPosition = modelMatrix * transformedPosition;
  vWorldPosition = worldPosition;

  // Twirl-rotate the local position to derive two orthogonal wobble axes,
  // then add scaled offsets to the world-space Y of the vertex. This is the
  // velocity-driven wobble of the r3f reference — we keep the effect visual
  // rather than physical.
  vec3 worldPosX = RotateAroundYInDegrees(vec4(position, 0.0), 360.0).xyz;
  vec3 worldPosZ = vec3(worldPosX.y, worldPosX.z, worldPosX.x);
  vec3 worldPosAdjusted =
      worldPosition.xyz
      + (worldPosX * uWobbleX)
      + (worldPosZ * uWobbleZ);

  vFillEdge = worldPosAdjusted.y + uFillAmount;

  vWorldNormal = normalize(modelViewMatrix * vec4(normal, 0.0)).xyz;
  vViewDirection = normalize(worldPosition.xyz - cameraPosition);
  gl_Position = projectionMatrix * modelViewMatrix * transformedPosition;
}
`;

const FRAG = /* glsl */ `
uniform vec4 uTopColor;
uniform vec4 uRimColor;
uniform vec4 uFoamColor;
uniform vec4 uTint;
uniform float uRim;
uniform float uRimPower;
uniform float uOpacity;

varying vec3 vWorldNormal;
varying vec3 vViewDirection;
varying vec4 vWorldPosition;
varying float vFillEdge;

void main() {
  vec4 body = uTint;

  // Rim-light at silhouette grazing angles.
  float facing = clamp(dot(vWorldNormal, vViewDirection), 0.0, 1.0);
  float rimFactor = 1.0 - pow(facing, uRimPower);
  vec4 rimResult = vec4(smoothstep(0.5, 1.0, rimFactor)) * uRimColor * uRimColor.w;

  // Foam band at the liquid surface.
  vec4 foam = vec4(step(vFillEdge, 0.5) - step(vFillEdge, 0.5 - uRim));
  vec4 foamColoured = foam * (uFoamColor * 0.95);

  // Body of the liquid (below foam).
  vec4 result = step(vFillEdge, 0.5) - foam;
  vec4 resultColoured = result * body;

  vec4 finalResult = resultColoured + foamColoured;
  finalResult.rgb += rimResult.rgb;

  // Top of the liquid (visible from inside the mesh on backfaces).
  vec4 topColor = uTopColor * (foam + result);

  gl_FragColor = gl_FrontFacing ? finalResult : topColor;
  gl_FragColor.a *= uOpacity;
}
`;

export interface LiquidUniforms extends Record<string, IUniform<unknown>> {
  uFillAmount: IUniform<number>;
  uWobbleX: IUniform<number>;
  uWobbleZ: IUniform<number>;
  uTint: IUniform<Vector4>;
  uTopColor: IUniform<Vector4>;
  uRimColor: IUniform<Vector4>;
  uFoamColor: IUniform<Vector4>;
  uRim: IUniform<number>;
  uRimPower: IUniform<number>;
  uOpacity: IUniform<number>;
}

export interface LiquidMaterialOptions {
  /** Initial fill amount; range tuning depends on the target geometry's bounds. */
  fillAmount?: number;
  /** Body colour as an RGBA Vector4; alpha multiplies the body opacity. */
  tint?: Color;
  tintAlpha?: number;
  /** Foam edge colour. */
  foamColor?: Color;
  /** Top-surface colour (the liquid surface seen from above through the container). */
  topColor?: Color;
  /** Rim highlight colour and strength. */
  rimColor?: Color;
  rim?: number;
  rimPower?: number;
  /** Global opacity multiplier. */
  opacity?: number;
}

export interface LiquidMaterial {
  material: ShaderMaterial;
  uniforms: LiquidUniforms;
}

export function createLiquidMaterial(opts: LiquidMaterialOptions = {}): LiquidMaterial {
  const tint = opts.tint ?? new Color(0.78, 0.05, 0.08); // arterial red
  const foam = opts.foamColor ?? new Color(0.96, 0.78, 0.84);
  const top = opts.topColor ?? new Color(0.85, 0.10, 0.15);
  const rimC = opts.rimColor ?? new Color(1.0, 0.70, 0.78);
  const tintAlpha = opts.tintAlpha ?? 0.85;

  const uniforms: LiquidUniforms = {
    uFillAmount: { value: opts.fillAmount ?? 0 },
    uWobbleX: { value: 0 },
    uWobbleZ: { value: 0 },
    uTint: { value: new Vector4(tint.r, tint.g, tint.b, tintAlpha) },
    uTopColor: { value: new Vector4(top.r, top.g, top.b, 1.0) },
    uRimColor: { value: new Vector4(rimC.r, rimC.g, rimC.b, 1.0) },
    uFoamColor: { value: new Vector4(foam.r, foam.g, foam.b, 1.0) },
    uRim: { value: 0.04 },
    uRimPower: { value: 4.0 },
    uOpacity: { value: opts.opacity ?? 1.0 },
  };

  const material = new ShaderMaterial({
    name: 'liquid-fill',
    uniforms,
    vertexShader: VERT,
    fragmentShader: FRAG,
    transparent: true,
    depthWrite: false,
  });

  return { material, uniforms };
}

/**
 * Drive the liquid fill from a cardiac-cycle phase and a beat impulse.
 *
 *   - phase ∈ [0, 2π] cycles once per heartbeat
 *   - On each phase wrap (systole onset) inject a wobble impulse
 *   - Fill amount oscillates between min/max sampling the phase as 0.5*(1-cos(phase))
 */
export class LiquidPulse {
  private phaseLast = 0;
  private wobbleX = 0;
  private wobbleZ = 0;

  constructor(
    private mat: LiquidMaterial,
    private opts: {
      fillMin: number;        // fill amount at end-systole (low blood)
      fillMax: number;        // fill amount at end-diastole (high blood)
      maxWobble?: number;     // amplitude of wobble injected per beat
      recovery?: number;      // 1/τ (s⁻¹) for wobble decay
    },
  ) {}

  /**
   * Update the shader uniforms.  `phaseRad` advances 2π per heartbeat;
   * `dt` is the frame interval in seconds.
   */
  tick(phaseRad: number, dt: number): void {
    const { fillMin, fillMax } = this.opts;
    const maxW = this.opts.maxWobble ?? 0.06;
    const recovery = this.opts.recovery ?? 1.4;

    // Detect a beat onset (phase wrap from ~2π to ~0) and inject impulses.
    if (phaseRad < this.phaseLast - Math.PI) {
      this.wobbleX += (Math.random() - 0.5) * 2 * maxW;
      this.wobbleZ += (Math.random() - 0.5) * 2 * maxW;
    }
    this.phaseLast = phaseRad;

    // Decay wobble exponentially.
    const decay = Math.exp(-recovery * dt);
    this.wobbleX *= decay;
    this.wobbleZ *= decay;

    // Asymmetric cardiac pulse: rapid drop early in cycle (ejection),
    // slow rise across diastole (filling). Use a shifted-cos:
    //   f(phase) = 0.5 - 0.5*cos(phase)   -> rises 0..1..0 across [0..2π]
    // Map to (fillMin..fillMax).
    const t = 0.5 - 0.5 * Math.cos(phaseRad);
    const fill = fillMin + (fillMax - fillMin) * t;

    this.mat.uniforms.uFillAmount.value = fill;
    this.mat.uniforms.uWobbleX.value = this.wobbleX;
    this.mat.uniforms.uWobbleZ.value = this.wobbleZ;
  }

  reset(): void {
    this.phaseLast = 0;
    this.wobbleX = 0;
    this.wobbleZ = 0;
  }
}
