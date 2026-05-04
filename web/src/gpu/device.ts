import { log } from '../util/log';

export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  ctx: GPUCanvasContext;
  format: GPUTextureFormat;
}

export async function initGpu(canvas: HTMLCanvasElement): Promise<GpuContext> {
  if (!('gpu' in navigator)) {
    throw new Error('WebGPU not available. Use Chrome/Edge ≥ 113 on Windows/macOS, or enable the flag.');
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No GPU adapter');

  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 268_435_456),
    },
  });

  device.lost.then((info) => {
    log(`gpu device lost: ${info.reason} ${info.message}`);
  });

  const ctx = canvas.getContext('webgpu');
  if (!ctx) throw new Error('Could not acquire webgpu canvas context');

  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({
    device,
    format,
    alphaMode: 'premultiplied',
  });

  log(`gpu ready: ${adapter.info?.vendor ?? 'unknown'} ${adapter.info?.architecture ?? ''} fmt=${format}`);

  return { adapter, device, canvas, ctx, format };
}

export function resizeCanvasToDisplay(canvas: HTMLCanvasElement): boolean {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.floor(canvas.clientWidth * dpr);
  const h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = Math.max(1, w);
    canvas.height = Math.max(1, h);
    return true;
  }
  return false;
}
