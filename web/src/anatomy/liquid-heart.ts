// Liquid-filled beating heart for the landing scene.
//
// Loads `beating_heart.glb`, replaces its materials with the liquid shader
// from `liquid-shader.ts`, and drives the fill amount + wobble from a
// default heart-rate phase counter. Used by the landing screen as the
// only visual on the page until the user clicks "begin".

import {
  AmbientLight,
  Box3,
  Clock,
  Color,
  DirectionalLight,
  Group,
  Mesh,
  PerspectiveCamera,
  Scene,
  Vector3,
  WebGLRenderer,
} from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { createLiquidMaterial, LiquidPulse, type LiquidMaterial } from './liquid-shader';
import { log } from '../util/log';

const HEART_GLB = '/glb/beating_heart.glb';
const DEFAULT_BPM = 60;

export interface LiquidHeartHandle {
  setBpm(bpm: number): void;
  destroy(): void;
}

export async function mountLiquidHeart(canvas: HTMLCanvasElement): Promise<LiquidHeartHandle> {
  const renderer = new WebGLRenderer({
    canvas,
    alpha: true,
    antialias: true,
    premultipliedAlpha: true,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(new Color(0x000000), 0);

  const scene = new Scene();
  scene.background = null;

  scene.add(new AmbientLight(0xffffff, 0.5));
  const key = new DirectionalLight(0xffeae0, 1.6);
  key.position.set(1.5, 2.5, 3);
  scene.add(key);
  const fill = new DirectionalLight(0x9fb6ff, 0.6);
  fill.position.set(-2, -0.5, -2);
  scene.add(fill);

  const camera = new PerspectiveCamera(28, 1, 0.01, 100);

  const container = new Group();
  scene.add(container);

  // Liquid material shared across all heart meshes.
  const liquid: LiquidMaterial = createLiquidMaterial({
    tint: new Color(0.78, 0.05, 0.08),
    tintAlpha: 0.88,
    foamColor: new Color(1.0, 0.85, 0.88),
    topColor: new Color(0.85, 0.10, 0.15),
    rimColor: new Color(1.0, 0.55, 0.55),
    rim: 0.14,
    rimPower: 3.5,
  });

  const loader = new GLTFLoader();
  const gltf = await loader.loadAsync(HEART_GLB).catch((err) => {
    log(`liquid-heart load failed: ${err}`);
    throw err;
  });
  container.add(gltf.scene);

  // Replace all materials with the liquid material.
  gltf.scene.traverse((obj) => {
    const mesh = obj as Mesh;
    if (!(mesh as { isMesh?: boolean }).isMesh) return;
    const old = mesh.material;
    if (Array.isArray(old)) for (const m of old) (m as { dispose?: () => void }).dispose?.();
    else if (old && typeof (old as { dispose?: () => void }).dispose === 'function') {
      (old as { dispose: () => void }).dispose();
    }
    mesh.material = liquid.material;
    // Render the inside of the mesh too (for the top-of-liquid effect).
    mesh.renderOrder = 0;
  });

  // Frame the heart.
  const box = new Box3().setFromObject(gltf.scene);
  const size = box.getSize(new Vector3());
  const centre = box.getCenter(new Vector3());
  gltf.scene.position.sub(centre);
  const maxDim = Math.max(size.x, size.y, size.z);
  const fovRad = (camera.fov * Math.PI) / 180;
  const distance = (maxDim / (2 * Math.tan(fovRad / 2))) * 1.65;
  camera.position.set(0, size.y * 0.05, distance);
  camera.lookAt(0, 0, 0);
  camera.near = Math.max(0.001, distance / 100);
  camera.far = distance * 100;
  camera.updateProjectionMatrix();
  log(`liquid heart bounds: y∈[${box.min.y.toFixed(2)}, ${box.max.y.toFixed(2)}]`);

  // Tune fill range to the heart's vertical extent.  Since the geometry is
  // centred at origin after the position offset above, the y-range is
  // [-size.y/2, +size.y/2] and the shader compares (worldPosY + uFillAmount)
  // to 0.5. We want fill spanning roughly the lower 35-65% of the model.
  const halfH = size.y / 2;
  const fillMid = 0.5 + halfH * 0.05;        // sits slightly above centre
  const fillSwing = halfH * 0.45;            // 45% of half-height swing
  const pulse = new LiquidPulse(liquid, {
    fillMin: fillMid - fillSwing,
    fillMax: fillMid + fillSwing,
    maxWobble: 0.05,
    recovery: 1.5,
  });

  // Render loop.
  let bpm = DEFAULT_BPM;
  let phase = 0;
  let running = true;
  const clock = new Clock();

  function fitToCanvasSize(): void {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (w <= 0 || h <= 0) return;
    const dpr = renderer.getPixelRatio();
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
  }
  fitToCanvasSize();
  const ro = new ResizeObserver(fitToCanvasSize);
  ro.observe(canvas);

  function tick(): void {
    if (!running) return;
    const dt = clock.getDelta();
    const hz = bpm / 60;
    phase = (phase + 2 * Math.PI * hz * dt) % (2 * Math.PI);

    pulse.tick(phase, dt);

    renderer.render(scene, camera);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  return {
    setBpm(b: number): void { bpm = Math.max(20, Math.min(220, b)); },
    destroy(): void {
      running = false;
      ro.disconnect();
      renderer.dispose();
      liquid.material.dispose();
      gltf.scene.traverse((obj) => {
        const mesh = obj as Mesh;
        if ((mesh as { isMesh?: boolean }).isMesh && mesh.geometry?.dispose) mesh.geometry.dispose();
      });
    },
  };
}
