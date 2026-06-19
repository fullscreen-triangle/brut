// BrutScript sandbox — GLB model viewer panel.
//
// Parses a cardiac/anatomical GLB from the /glb/ asset catalogue and renders
// it with the same StrainMaterial the observatory uses, driven by live trace
// signal values. The parser exposes mesh names, animation clips, bounding box,
// and per-node metadata so the sandbox can display a structured asset report
// alongside the 3D view.

import {
  AmbientLight,
  AnimationMixer,
  Box3,
  Clock,
  Color,
  DirectionalLight,
  Mesh,
  PerspectiveCamera,
  Scene,
  Vector3,
  WebGLRenderer,
  Group,
} from 'three';
import { GLTFLoader, type GLTF } from 'three/addons/loaders/GLTFLoader.js';
import {
  createStrainMaterial,
  updateStrainUniforms,
  type StrainMaterial,
} from '../anatomy/strain-shader';

// ─── GLB parse report ─────────────────────────────────────────────────────────

export interface GlbMeshInfo {
  name: string;
  vertexCount: number;
  faceCount: number;
  hasMorph: boolean;
  hasAnimation: boolean;
}

export interface GlbParseReport {
  url: string;
  loadedAt: number;          // session ms
  meshes: GlbMeshInfo[];
  animationClips: Array<{ name: string; durationS: number; trackCount: number }>;
  boundingBox: { min: [number, number, number]; max: [number, number, number] };
  totalVertices: number;
  totalFaces: number;
}

// ─── Available cardiac GLB catalogue ─────────────────────────────────────────

export interface GlbEntry {
  id: string;
  label: string;
  url: string;
  description: string;
  tags: string[];
}

export const CARDIAC_GLB_CATALOGUE: GlbEntry[] = [
  {
    id: 'heart_animated',
    label: 'Heart — Animated Anatomical',
    url: '/glb/heart__animated_anatomical_3d_model.glb',
    description: 'Full anatomical heart with embedded cardiac cycle animation. Strain shader maps EOS-derived Ees/Ea/EF to surface compression colour.',
    tags: ['heart', 'animated', 'strain'],
  },
  {
    id: 'beating_heart',
    label: 'Beating Heart',
    url: '/glb/beating_heart.glb',
    description: 'Stylised beating heart. Animation driven at live HR from rPPG.',
    tags: ['heart', 'animated'],
  },
  {
    id: 'anatomical_codominance',
    label: 'Heart — Codominance Anatomy',
    url: '/glb/anatomical_heart__codominance.glb',
    description: 'High-fidelity codominant coronary anatomy. Static; illuminated by strain shader so surface colour reflects fitted cardiac state.',
    tags: ['heart', 'anatomy', 'coronary'],
  },
  {
    id: 'echo_a2c',
    label: 'Echo — Apical 2-Chamber',
    url: '/glb/cardiac_anatomy_apical_2_chamber_echo_plane.glb',
    description: 'A2C echocardiographic plane geometry. Strain colour illustrates ventricular long-axis deformation.',
    tags: ['echo', 'a2c', 'plane'],
  },
  {
    id: 'echo_a5c',
    label: 'Echo — Apical 5-Chamber',
    url: '/glb/cardiac_anatomy_apical_5_chamber_echo_plane.glb',
    description: 'A5C plane including LVOT. Useful for assessing aortic outflow contribution to CO.',
    tags: ['echo', 'a5c', 'plane'],
  },
  {
    id: 'echo_psax',
    label: 'Echo — PSAX Aortic Valve',
    url: '/glb/cardiac_anatomy_psax_aortic_valve_echo_plane.glb',
    description: 'Parasternal short-axis at aortic valve level. Strain colour encodes Ea (afterload).',
    tags: ['echo', 'psax', 'aortic'],
  },
  {
    id: 'external_view',
    label: 'Heart — External View',
    url: '/glb/cardiac_anatomy_external_view.glb',
    description: 'Full external cardiac anatomy. Pericardial surface; suitable for visualising regional wall motion.',
    tags: ['heart', 'anatomy', 'external'],
  },
  {
    id: 'heart_with_flow',
    label: 'Heart — Blood Flow',
    url: '/glb/heart_with_blood_flow.glb',
    description: 'Heart with embedded blood flow geometry. Useful for visualising CO and SV concepts.',
    tags: ['heart', 'flow', 'blood'],
  },
  {
    id: 'thorax',
    label: 'Heart in Thorax',
    url: '/glb/human_anatomy_heart_in_thorax.glb',
    description: 'Heart within thoracic context. Shows spatial relationship to lungs and chest wall.',
    tags: ['heart', 'thorax', 'context'],
  },
  {
    id: 'circulatory',
    label: 'Circulatory System',
    url: '/glb/circulatory_01.glb',
    description: 'Full circulatory system. Strain colour codes CO and MAP from live PCHR decomposition.',
    tags: ['circulatory', 'vasculature'],
  },
];

// ─── GLB parser ──────────────────────────────────────────────────────────────

export async function parseGlb(url: string, sessionStartMs: number): Promise<GlbParseReport> {
  const loader = new GLTFLoader();
  const gltf = await loader.loadAsync(url);

  const meshes: GlbMeshInfo[] = [];
  let totalVerts = 0, totalFaces = 0;

  gltf.scene.traverse(obj => {
    const mesh = obj as Mesh;
    if (!(mesh as unknown as { isMesh: boolean }).isMesh) return;
    const geo = mesh.geometry;
    const vCount = geo.attributes.position?.count ?? 0;
    const fCount = geo.index ? geo.index.count / 3 : vCount / 3;
    totalVerts += vCount;
    totalFaces += Math.round(fCount);
    meshes.push({
      name: mesh.name || '(unnamed)',
      vertexCount: vCount,
      faceCount: Math.round(fCount),
      hasMorph: geo.morphAttributes && Object.keys(geo.morphAttributes).length > 0,
      hasAnimation: gltf.animations.some(a => a.tracks.some(t => t.name.startsWith(mesh.name ?? ''))),
    });
  });

  const box = new Box3().setFromObject(gltf.scene);
  const mn = box.min, mx = box.max;

  const clips = gltf.animations.map(a => ({
    name: a.name || '(unnamed)',
    durationS: a.duration,
    trackCount: a.tracks.length,
  }));

  // Dispose after parsing — caller will reload for rendering if needed
  gltf.scene.traverse(obj => {
    const mesh = obj as { geometry?: { dispose: () => void }; material?: unknown };
    mesh.geometry?.dispose();
    const mat = mesh.material;
    if (Array.isArray(mat)) for (const m of mat) (m as { dispose?: () => void }).dispose?.();
    else if (mat) (mat as { dispose?: () => void }).dispose?.();
  });

  return {
    url,
    loadedAt: performance.now() - sessionStartMs,
    meshes,
    animationClips: clips,
    boundingBox: {
      min: [mn.x, mn.y, mn.z],
      max: [mx.x, mx.y, mx.z],
    },
    totalVertices: totalVerts,
    totalFaces: totalFaces,
  };
}

// ─── GLB viewer handle ────────────────────────────────────────────────────────

export interface GlbViewerHandle {
  updateCardiacState(state: {
    HR: number; Ees: number; Ea: number; EDV: number; ESV: number; EF: number; Rc: number;
  }): void;
  setTempoHz(hz: number): void;
  destroy(): void;
}

// ─── GLB renderer (viewer) ────────────────────────────────────────────────────

export async function mountGlbViewer(
  canvas: HTMLCanvasElement,
  url: string,
): Promise<GlbViewerHandle> {
  const renderer = new WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(new Color(0), 0);

  const scene = new Scene();
  scene.add(new AmbientLight(0xffffff, 0.5));
  const key = new DirectionalLight(0xc8d8ff, 1.8);
  key.position.set(2, 3, 4);
  scene.add(key);
  const fill = new DirectionalLight(0xffd8c0, 0.4);
  fill.position.set(-3, -1, -2);
  scene.add(fill);

  const camera = new PerspectiveCamera(35, 1, 0.01, 1000);
  const container = new Group();
  scene.add(container);
  container.rotation.y = Math.PI * 0.1;

  const strain: StrainMaterial = createStrainMaterial(new Color(0.95, 0.92, 1.0));

  const loader = new GLTFLoader();
  let gltf: GLTF;
  try {
    gltf = await loader.loadAsync(url);
  } catch {
    renderer.dispose();
    throw new Error(`Failed to load ${url}`);
  }

  container.add(gltf.scene);

  // Apply strain material to all meshes
  gltf.scene.traverse(obj => {
    const mesh = obj as Mesh;
    if (!(mesh as unknown as { isMesh: boolean }).isMesh) return;
    const old = mesh.material;
    if (Array.isArray(old)) for (const m of old) (m as { dispose?: () => void }).dispose?.();
    else if (old) (old as { dispose?: () => void }).dispose?.();
    mesh.material = strain.material;
  });

  // Frame camera
  const box = new Box3().setFromObject(gltf.scene);
  const size = box.getSize(new Vector3());
  const center = box.getCenter(new Vector3());
  gltf.scene.position.sub(center);
  const maxDim = Math.max(size.x, size.y, size.z);
  const fovRad = (camera.fov * Math.PI) / 180;
  const dist = maxDim / (2 * Math.tan(fovRad / 2)) * 1.5;
  camera.position.set(0, size.y * 0.05, dist);
  camera.lookAt(0, 0, 0);
  camera.near = dist / 100;
  camera.far = dist * 100;
  camera.updateProjectionMatrix();

  // Animation
  const mixer = new AnimationMixer(gltf.scene);
  const baseDuration = gltf.animations.length > 0
    ? Math.max(...gltf.animations.map(a => a.duration))
    : 1.0;
  for (const clip of gltf.animations) mixer.clipAction(clip).play();

  let tempoHz = 1.0;
  mixer.timeScale = tempoHz * baseDuration;

  const clock = new Clock();
  let phaseRad = 0;
  let running = true;

  // Resize
  function resize(): void {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();
  const ro = new ResizeObserver(resize);
  ro.observe(canvas);

  function tick(): void {
    if (!running) return;
    const dt = clock.getDelta();
    mixer.update(dt);
    phaseRad = (phaseRad + 2 * Math.PI * tempoHz * dt) % (2 * Math.PI);
    strain.uniforms.uPhase.value = phaseRad;
    strain.uniforms.uTime.value += dt;

    renderer.render(scene, camera);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  return {
    updateCardiacState(s) {
      updateStrainUniforms(strain, s);
    },
    setTempoHz(hz) {
      tempoHz = Math.max(0, hz);
      mixer.timeScale = tempoHz * baseDuration;
    },
    destroy() {
      running = false;
      ro.disconnect();
      mixer.stopAllAction();
      renderer.dispose();
      gltf.scene.traverse(obj => {
        const m = obj as { geometry?: { dispose: () => void }; material?: unknown };
        m.geometry?.dispose();
        const mat = m.material;
        if (Array.isArray(mat)) for (const x of mat) (x as { dispose?: () => void }).dispose?.();
        else if (mat) (mat as { dispose?: () => void }).dispose?.();
      });
    },
  };
}

// ─── Render a parse report as an HTML table ───────────────────────────────────

export function renderParseReport(report: GlbParseReport, container: HTMLElement): void {
  container.innerHTML = '';

  const hdr = document.createElement('div');
  hdr.className = 'glb-report-header';
  hdr.innerHTML = `
    <div class="glb-report-url">${report.url.split('/').pop()}</div>
    <div class="glb-report-meta">
      <span>${report.totalVertices.toLocaleString()} vertices</span>
      <span>${report.totalFaces.toLocaleString()} faces</span>
      <span>${report.meshes.length} meshes</span>
      <span>${report.animationClips.length} clips</span>
    </div>`;
  container.appendChild(hdr);

  if (report.animationClips.length > 0) {
    const clipH = document.createElement('div');
    clipH.className = 'glb-section-title';
    clipH.textContent = 'Animation Clips';
    container.appendChild(clipH);
    const clipT = document.createElement('table');
    clipT.className = 'glb-table';
    clipT.innerHTML = `<thead><tr><th>name</th><th>duration</th><th>tracks</th></tr></thead>
      <tbody>${report.animationClips.map(c =>
        `<tr><td>${c.name}</td><td>${c.durationS.toFixed(3)}s</td><td>${c.trackCount}</td></tr>`
      ).join('')}</tbody>`;
    container.appendChild(clipT);
  }

  const meshH = document.createElement('div');
  meshH.className = 'glb-section-title';
  meshH.textContent = 'Mesh Inventory';
  container.appendChild(meshH);

  const meshT = document.createElement('table');
  meshT.className = 'glb-table';
  meshT.innerHTML = `<thead><tr><th>mesh</th><th>vertices</th><th>faces</th><th>morph</th><th>anim</th></tr></thead>
    <tbody>${report.meshes.map(m => `
      <tr>
        <td class="mesh-name">${m.name}</td>
        <td>${m.vertexCount.toLocaleString()}</td>
        <td>${m.faceCount.toLocaleString()}</td>
        <td>${m.hasMorph ? '✓' : '—'}</td>
        <td>${m.hasAnimation ? '✓' : '—'}</td>
      </tr>`).join('')}
    </tbody>`;
  container.appendChild(meshT);

  const [bmin, bmax] = [report.boundingBox.min, report.boundingBox.max];
  const bboxH = document.createElement('div');
  bboxH.className = 'glb-section-title';
  bboxH.textContent = 'Bounding Box';
  container.appendChild(bboxH);
  const bbox = document.createElement('div');
  bbox.className = 'glb-bbox';
  bbox.innerHTML = `
    <span>min [${bmin.map(v => v.toFixed(3)).join(', ')}]</span>
    <span>max [${bmax.map(v => v.toFixed(3)).join(', ')}]</span>
    <span>size [${bmin.map((v, i) => (bmax[i] - v).toFixed(3)).join(', ')}]</span>`;
  container.appendChild(bbox);
}
