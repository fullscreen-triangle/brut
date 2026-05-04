// Generic three.js glb mount: transparent canvas, scene, ambient + key light,
// animation mixer driven by a tempo signal. The same engine drives both the
// lungs and the heart corner widgets.

import {
  AmbientLight,
  AnimationMixer,
  Box3,
  Clock,
  Color,
  DirectionalLight,
  Group,
  PerspectiveCamera,
  Scene,
  Vector3,
  WebGLRenderer,
  type AnimationAction,
  type AnimationClip,
} from 'three';
import { GLTFLoader, type GLTF } from 'three/addons/loaders/GLTFLoader.js';
import { log } from '../util/log';

export interface GlbWidgetOptions {
  url: string;
  canvas: HTMLCanvasElement;
  // Camera framing fudge factor; >1 zooms out.
  framePadding?: number;
  // Optional initial rotation around Y in radians (turns the model to face camera).
  initialYaw?: number;
  // Optional rotation speed (rad/s) so the model gently rotates if the user wants.
  ambientYawRate?: number;
}

export interface GlbWidget {
  setTempoHz(hz: number): void;     // 0 pauses; positive scales animation playback rate
  setTempoFromBpm(bpm: number, baseBpm?: number): void;
  setVisible(visible: boolean): void;
  destroy(): void;
}

/**
 * Mount a glb into `canvas`, framing the model and playing all embedded
 * animation clips through an AnimationMixer. The mixer's timeScale is updated
 * by setTempoHz / setTempoFromBpm so that one period of the animation matches
 * the requested oscillator frequency.
 */
export async function mountGlb(opts: GlbWidgetOptions): Promise<GlbWidget> {
  const { url, canvas } = opts;

  const renderer = new WebGLRenderer({
    canvas,
    alpha: true,
    antialias: true,
    premultipliedAlpha: true,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(new Color(0x000000), 0); // fully transparent

  const scene = new Scene();
  scene.background = null;

  const ambient = new AmbientLight(0xffffff, 0.45);
  scene.add(ambient);

  const key = new DirectionalLight(0xc8d8ff, 1.6);
  key.position.set(2, 3, 4);
  scene.add(key);

  const fill = new DirectionalLight(0xffd8c0, 0.5);
  fill.position.set(-3, -1, -2);
  scene.add(fill);

  const camera = new PerspectiveCamera(35, 1, 0.01, 1000);
  camera.position.set(0, 0, 5);

  // Container for the model so we can rotate without touching the loaded gltf
  const container = new Group();
  if (opts.initialYaw) container.rotation.y = opts.initialYaw;
  scene.add(container);

  const loader = new GLTFLoader();
  let gltf: GLTF;
  try {
    gltf = await loader.loadAsync(url);
  } catch (err) {
    log(`glb load failed (${url}): ${err}`);
    throw err;
  }

  container.add(gltf.scene);

  // Frame the camera to fit the model.
  fitCameraToObject(camera, gltf.scene, opts.framePadding ?? 1.4);

  // Animation mixer — all clips play on loop with a shared timeScale.
  const mixer = new AnimationMixer(gltf.scene);
  const actions: AnimationAction[] = [];
  const clips: AnimationClip[] = gltf.animations ?? [];
  for (const clip of clips) {
    const action = mixer.clipAction(clip);
    action.play();
    actions.push(action);
  }
  if (clips.length === 0) {
    log(`glb has no animation tracks: ${url}`);
  } else {
    log(`glb loaded: ${url} (${clips.length} clip(s), longest ${clips
      .map((c) => c.duration.toFixed(2))
      .join('/')}s)`);
  }

  let tempoHz = 0;          // 0 = paused; >0 plays at scaled speed
  let visible = true;
  let running = true;

  // For tempo locking we need a baseline animation period to scale against.
  const baselineDuration =
    clips.length > 0 ? Math.max(...clips.map((c) => c.duration)) : 1.0;

  function setTempoHz(hz: number): void {
    tempoHz = Math.max(0, hz);
    // We want the animation to complete one full clip per (1/hz) seconds.
    // mixer.timeScale = 1 means real-time playback; if baselineDuration is 1s,
    // then 1 Hz tempo => timeScale 1.0; 2 Hz => 2.0; etc.
    mixer.timeScale = tempoHz === 0 ? 0 : tempoHz * baselineDuration;
  }

  function setTempoFromBpm(bpm: number, baseBpm?: number): void {
    if (bpm <= 0) {
      setTempoHz(0);
      return;
    }
    if (baseBpm && baseBpm > 0) {
      setTempoHz((bpm / baseBpm) * (1 / baselineDuration));
    } else {
      setTempoHz(bpm / 60);
    }
  }

  function setVisible(v: boolean): void {
    visible = v;
    canvas.style.opacity = v ? '1' : '0';
  }

  function destroy(): void {
    running = false;
    mixer.stopAllAction();
    actions.length = 0;
    renderer.dispose();
    gltf.scene.traverse((obj) => {
      const mesh = obj as { geometry?: { dispose: () => void }; material?: unknown };
      if (mesh.geometry?.dispose) mesh.geometry.dispose();
      const mat = mesh.material;
      if (Array.isArray(mat)) {
        for (const m of mat) (m as { dispose?: () => void }).dispose?.();
      } else if (mat && typeof (mat as { dispose?: () => void }).dispose === 'function') {
        (mat as { dispose: () => void }).dispose();
      }
    });
  }

  // Resize handling.
  function fitToCanvasSize(): void {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (w === 0 || h === 0) return;
    if (canvas.width !== w * renderer.getPixelRatio() || canvas.height !== h * renderer.getPixelRatio()) {
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
  }
  fitToCanvasSize();

  const ro = new ResizeObserver(fitToCanvasSize);
  ro.observe(canvas);

  // Render loop.
  const clock = new Clock();
  function tick(): void {
    if (!running) return;
    const dt = clock.getDelta();
    if (visible && tempoHz > 0) {
      mixer.update(dt);
    } else if (visible && opts.ambientYawRate) {
      // No tempo and no animation — let the model gently rotate so it's
      // recognisably "alive" before we have a tempo to lock to.
      container.rotation.y += (opts.ambientYawRate ?? 0) * dt;
    }
    if (visible) renderer.render(scene, camera);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  return { setTempoHz, setTempoFromBpm, setVisible, destroy };
}

/**
 * Position the camera so the bounding sphere of `object` fills the frustum
 * with `padding`× extra room on each side. Centres the object at the origin.
 */
function fitCameraToObject(
  camera: PerspectiveCamera,
  object: import('three').Object3D,
  padding: number,
): void {
  const box = new Box3().setFromObject(object);
  const size = box.getSize(new Vector3());
  const center = box.getCenter(new Vector3());

  // Recentre the object so it sits at world origin.
  object.position.sub(center);

  const maxDim = Math.max(size.x, size.y, size.z);
  if (maxDim === 0) return;

  const fovRad = (camera.fov * Math.PI) / 180;
  const distance = (maxDim / (2 * Math.tan(fovRad / 2))) * padding;
  camera.position.set(0, size.y * 0.05, distance);
  camera.lookAt(0, 0, 0);
  camera.near = Math.max(0.001, distance / 100);
  camera.far = distance * 100;
  camera.updateProjectionMatrix();
}
