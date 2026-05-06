// Landing screen: minimal black background, a single liquid-filled beating
// heart in the centre, and a subtle "begin" affordance. On click we fade
// the landing out and reveal the observatory, which is constructed lazily
// to avoid eating cycles before the user wants it.

import { mountLiquidHeart, type LiquidHeartHandle } from '../anatomy/liquid-heart';
import { setStatus } from '../util/log';

export interface LandingHandle {
  beginPromise: Promise<void>;
  destroy(): void;
}

export function mountLanding(): LandingHandle {
  const root = document.getElementById('landing');
  const canvas = document.getElementById('landing-canvas') as HTMLCanvasElement | null;
  const beginBtn = document.getElementById('landing-begin') as HTMLButtonElement | null;
  if (!root || !canvas || !beginBtn) {
    return {
      beginPromise: Promise.resolve(),
      destroy(): void {},
    };
  }

  let heart: LiquidHeartHandle | null = null;
  let resolved = false;

  const beginPromise = new Promise<void>((resolve) => {
    void mountLiquidHeart(canvas)
      .then((h) => { heart = h; setStatus('idle — click begin'); })
      .catch((err) => {
        setStatus(`landing error: ${err instanceof Error ? err.message : err}`);
      });

    beginBtn.addEventListener('click', () => {
      if (resolved) return;
      resolved = true;
      // Fade landing, dismount heart after the transition finishes.
      root.classList.add('dismissing');
      setStatus('entering observatory');
      setTimeout(() => {
        heart?.destroy();
        root.style.display = 'none';
        resolve();
      }, 450);
    });

    document.addEventListener('keydown', (ev) => {
      if (resolved) return;
      if (ev.key === 'Enter' || ev.key === ' ') {
        beginBtn.click();
      }
    });
  });

  return {
    beginPromise,
    destroy(): void { heart?.destroy(); },
  };
}
