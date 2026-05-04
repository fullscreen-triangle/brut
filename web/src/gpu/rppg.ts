// rPPG GPU pipeline.
//
// Per frame:
//   1. Build an external texture from the <video> element.
//   2. Compute pass A (sample-roi): downsample face ROI to PATCH x PATCH means;
//      append the green channel into a circular history buffer.
//   3. Compute pass B (coherence-field): from the recent history, estimate a
//      per-patch R_c via detrended autocorrelation in the cardiac band.
//      Output: rgba16float (R_c, amp, period_s, snr) per patch.
//   4. Render pass (heatmap): composite the R_c field into the overlay canvas
//       within the face bbox.
//
// We also produce a single global BVP sample per frame (mean over all patches)
// for the CPU-side HR/HRV/regime stack.

import sampleRoiSrc from './shaders/sample-roi.wgsl?raw';
import coherenceSrc from './shaders/coherence-field.wgsl?raw';
import heatmapSrc from './shaders/heatmap.wgsl?raw';

import type { GpuContext } from './device';
import type { FaceROI } from '../camera/landmarks';
import { log } from '../util/log';

export const PATCH = 64;
export const BUF_FRAMES = 256;

export interface RppgPipeline {
  tick(video: HTMLVideoElement, roi: FaceROI, sampleRateHz: number, timestampMs: number): Promise<RppgFrameResult>;
  destroy(): void;
}

export interface RppgFrameResult {
  globalBvp: number;       // mean of all patch BVP residuals this frame
  rcMean: number;          // spatial mean of R_c over face patches
  rcStd: number;           // spatial std of R_c (heterogeneity proxy)
  snr: number;             // mean SNR over patches
  samplesFilled: number;   // history fill count, 0..BUF_FRAMES
}

export async function createRppgPipeline(gpu: GpuContext): Promise<RppgPipeline> {
  const { device } = gpu;

  const sampleRoiModule = device.createShaderModule({ label: 'sample-roi', code: sampleRoiSrc });
  const coherenceModule = device.createShaderModule({ label: 'coherence-field', code: coherenceSrc });
  const heatmapModule = device.createShaderModule({ label: 'heatmap', code: heatmapSrc });

  const constants = { PATCH, BUF_FRAMES };

  const sampleRoiPipeline = await device.createComputePipelineAsync({
    label: 'sample-roi',
    layout: 'auto',
    compute: { module: sampleRoiModule, entryPoint: 'main', constants },
  });

  const coherencePipeline = await device.createComputePipelineAsync({
    label: 'coherence-field',
    layout: 'auto',
    compute: { module: coherenceModule, entryPoint: 'main', constants },
  });

  const heatmapPipeline = await device.createRenderPipelineAsync({
    label: 'heatmap',
    layout: 'auto',
    vertex: { module: heatmapModule, entryPoint: 'vs' },
    fragment: {
      module: heatmapModule,
      entryPoint: 'fs',
      targets: [{
        format: gpu.format,
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
        },
      }],
    },
    primitive: { topology: 'triangle-list' },
  });

  // Resources -----------------------------------------------------------
  const historyBuf = device.createBuffer({
    label: 'history',
    size: BUF_FRAMES * PATCH * PATCH * 4, // f32
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const currentTex = device.createTexture({
    label: 'current-mean',
    size: { width: PATCH, height: PATCH },
    format: 'rgba32float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });

  const coherenceTex = device.createTexture({
    label: 'coherence-field',
    size: { width: PATCH, height: PATCH },
    format: 'rgba16float',
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC,
  });

  const sampler = device.createSampler({
    label: 'roi-sampler',
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });

  const sampleUbo = device.createBuffer({
    label: 'sample-ubo',
    size: 32, // 4 floats + 4 u32 = 32 bytes
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const coherenceUbo = device.createBuffer({
    label: 'coherence-ubo',
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const heatmapUbo = device.createBuffer({
    label: 'heatmap-ubo',
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Readback for per-frame statistics. We stage a small RGBA16Float -> RGBA32Float
  // copy via a download buffer of 64*64*4*4 = 16 KiB. With a single in-flight readback
  // the worst-case latency is one frame, which is fine for v1.
  const COHERENCE_BYTES = PATCH * PATCH * 4 * 2; // rgba16f = 8 bytes/pixel
  const readbackBuf = device.createBuffer({
    label: 'coherence-readback',
    size: COHERENCE_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  let readbackInFlight = false;
  let frameIdx = 0;
  let samplesFilled = 0;
  let lastResult: RppgFrameResult = {
    globalBvp: 0,
    rcMean: 0,
    rcStd: 0,
    snr: 0,
    samplesFilled: 0,
  };

  // Bind group helper for sample-roi (rebuilt each frame because external texture changes).
  function buildSampleRoiBg(externalTex: GPUExternalTexture): GPUBindGroup {
    return device.createBindGroup({
      label: 'sample-roi-bg',
      layout: sampleRoiPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: externalTex },
        { binding: 1, resource: sampler },
        { binding: 2, resource: { buffer: sampleUbo } },
        { binding: 3, resource: { buffer: historyBuf } },
        { binding: 4, resource: currentTex.createView() },
      ],
    });
  }

  const coherenceBg = device.createBindGroup({
    label: 'coherence-bg',
    layout: coherencePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: coherenceUbo } },
      { binding: 1, resource: { buffer: historyBuf } },
      { binding: 2, resource: coherenceTex.createView() },
    ],
  });

  const heatmapBg = device.createBindGroup({
    label: 'heatmap-bg',
    layout: heatmapPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: sampler },
      { binding: 1, resource: coherenceTex.createView() },
      { binding: 2, resource: { buffer: heatmapUbo } },
    ],
  });

  log(`rPPG pipeline ready: PATCH=${PATCH} BUF_FRAMES=${BUF_FRAMES}`);

  async function tick(
    video: HTMLVideoElement,
    roi: FaceROI,
    sampleRateHz: number,
    _timestampMs: number,
  ): Promise<RppgFrameResult> {
    const externalTex = device.importExternalTexture({ source: video });

    const sampleU = new ArrayBuffer(32);
    const sampleU_f = new Float32Array(sampleU);
    const sampleU_u = new Uint32Array(sampleU);
    sampleU_f[0] = roi.bbox.x;
    sampleU_f[1] = roi.bbox.y;
    sampleU_f[2] = roi.bbox.w;
    sampleU_f[3] = roi.bbox.h;
    sampleU_u[4] = frameIdx >>> 0;
    device.queue.writeBuffer(sampleUbo, 0, sampleU);

    const coherenceU = new ArrayBuffer(32);
    const cohU_f = new Float32Array(coherenceU);
    const cohU_u = new Uint32Array(coherenceU);
    cohU_f[0] = sampleRateHz;
    cohU_u[1] = frameIdx >>> 0;
    cohU_u[2] = Math.min(samplesFilled + 1, BUF_FRAMES);
    cohU_u[3] = 0;
    cohU_f[4] = 0.7;
    cohU_f[5] = 3.0;
    device.queue.writeBuffer(coherenceUbo, 0, coherenceU);

    const heatU = new ArrayBuffer(32);
    const heatU_f = new Float32Array(heatU);
    heatU_f[0] = roi.bbox.x;
    heatU_f[1] = roi.bbox.y;
    heatU_f[2] = roi.bbox.w;
    heatU_f[3] = roi.bbox.h;
    heatU_f[4] = 0.55;  // alpha
    heatU_f[5] = 0.05;  // threshold
    device.queue.writeBuffer(heatmapUbo, 0, heatU);

    const encoder = device.createCommandEncoder({ label: 'rppg-frame' });

    // Pass A: sample ROI + write history slot.
    {
      const pass = encoder.beginComputePass({ label: 'sample-roi-pass' });
      pass.setPipeline(sampleRoiPipeline);
      pass.setBindGroup(0, buildSampleRoiBg(externalTex));
      pass.dispatchWorkgroups(PATCH / 8, PATCH / 8, 1);
      pass.end();
    }

    // Pass B: per-pixel coherence.
    {
      const pass = encoder.beginComputePass({ label: 'coherence-pass' });
      pass.setPipeline(coherencePipeline);
      pass.setBindGroup(0, coherenceBg);
      pass.dispatchWorkgroups(PATCH / 8, PATCH / 8, 1);
      pass.end();
    }

    // Pass C: render heatmap onto the surface canvas (overlay).
    {
      const view = gpu.ctx.getCurrentTexture().createView();
      const pass = encoder.beginRenderPass({
        label: 'heatmap-pass',
        colorAttachments: [{
          view,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      pass.setPipeline(heatmapPipeline);
      pass.setBindGroup(0, heatmapBg);
      pass.draw(6, 1, 0, 0);
      pass.end();
    }

    // Pass D: stage coherence texture for CPU readback (skipped if previous still in flight).
    if (!readbackInFlight) {
      encoder.copyTextureToBuffer(
        { texture: coherenceTex },
        { buffer: readbackBuf, bytesPerRow: PATCH * 8, rowsPerImage: PATCH },
        { width: PATCH, height: PATCH },
      );
    }

    device.queue.submit([encoder.finish()]);

    if (!readbackInFlight) {
      readbackInFlight = true;
      // Fire-and-forget map; we update lastResult when it resolves.
      readbackBuf.mapAsync(GPUMapMode.READ).then(() => {
        try {
          const arr = new Uint16Array(readbackBuf.getMappedRange()).slice();
          const rcs = new Float32Array(PATCH * PATCH);
          const snrs = new Float32Array(PATCH * PATCH);
          let bvpAcc = 0;
          let bvpN = 0;
          for (let i = 0; i < PATCH * PATCH; i++) {
            const base = i * 4;
            const rc = float16ToFloat32(arr[base + 0]);
            const amp = float16ToFloat32(arr[base + 1]);
            const snr = float16ToFloat32(arr[base + 3]);
            rcs[i] = rc;
            snrs[i] = snr;
            if (rc > 0.05) {
              bvpAcc += amp;
              bvpN += 1;
            }
          }
          const rcMean = mean(rcs);
          const rcStd = std(rcs, rcMean);
          const snrMean = mean(snrs);
          lastResult = {
            globalBvp: bvpN > 0 ? bvpAcc / bvpN : 0,
            rcMean,
            rcStd,
            snr: snrMean,
            samplesFilled,
          };
        } finally {
          readbackBuf.unmap();
          readbackInFlight = false;
        }
      }).catch((err) => {
        log(`readback failed: ${err}`);
        readbackInFlight = false;
      });
    }

    frameIdx = (frameIdx + 1) >>> 0;
    if (samplesFilled < BUF_FRAMES) samplesFilled += 1;

    return { ...lastResult, samplesFilled };
  }

  function destroy(): void {
    historyBuf.destroy();
    currentTex.destroy();
    coherenceTex.destroy();
    sampleUbo.destroy();
    coherenceUbo.destroy();
    heatmapUbo.destroy();
    readbackBuf.destroy();
  }

  return { tick, destroy };
}

function mean(a: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / Math.max(1, a.length);
}

function std(a: ArrayLike<number>, m: number): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - m;
    s += d * d;
  }
  return Math.sqrt(s / Math.max(1, a.length));
}

// IEEE 754 binary16 -> f32 (no native API in the DOM).
function float16ToFloat32(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) {
    return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  } else if (e === 0x1f) {
    return f ? NaN : (s ? -Infinity : Infinity);
  }
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}
