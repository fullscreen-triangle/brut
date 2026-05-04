// Stealth-mode toggle. Self-view (camera, overlay, coherence heatmap) is
// hidden via CSS; rPPG pipeline keeps running because the <video> element
// is still rendering camera frames into its internal buffer — it just isn't
// visible. MediaPipe and WebGPU still see the same pixels.
//
// Persists across reloads via localStorage.

const KEY = 'brut.stealth';
const btn = document.getElementById('stealth') as HTMLButtonElement | null;

export function initStealthToggle(): void {
  // Default behaviour: stealth ON unless the user has explicitly toggled it OFF
  // in a prior session. Looking at oneself on camera is a real source of
  // friction, and the framework's measurements don't depend on the self-view.
  const stored = localStorage.getItem(KEY);
  const initiallyStealth = stored !== '0';

  if (initiallyStealth) {
    document.body.classList.add('stealth');
    if (btn) btn.textContent = 'reveal';
  } else if (btn) {
    btn.textContent = 'stealth';
  }

  btn?.addEventListener('click', () => {
    const on = !document.body.classList.contains('stealth');
    document.body.classList.toggle('stealth', on);
    btn.textContent = on ? 'reveal' : 'stealth';
    try {
      localStorage.setItem(KEY, on ? '1' : '0');
    } catch {
      /* private mode etc */
    }
  });
}

export function isStealth(): boolean {
  return document.body.classList.contains('stealth');
}
