// Side-panel + drawer invocation. Click handlers wired here; chart mounts
// happen in the panel modules that import this.

const heartCorner = document.getElementById('heart-corner')!;
const lungsCorner = document.getElementById('lungs-corner')!;
const heartPanel = document.getElementById('heart-panel')!;
const lungsPanel = document.getElementById('lungs-panel')!;
const drawer = document.getElementById('drawer')!;
const drawerHandle = document.getElementById('drawer-handle')!;

export type PanelId = 'heart-panel' | 'lungs-panel' | 'drawer';

const listeners: Partial<Record<PanelId, ((open: boolean) => void)[]>> = {};

export function onPanelToggle(id: PanelId, fn: (open: boolean) => void): void {
  if (!listeners[id]) listeners[id] = [];
  listeners[id]!.push(fn);
}

function emit(id: PanelId, open: boolean): void {
  for (const fn of listeners[id] ?? []) fn(open);
}

function setOpen(el: HTMLElement, open: boolean): void {
  el.classList.toggle('open', open);
  el.setAttribute('aria-hidden', open ? 'false' : 'true');
}

export function openPanel(id: PanelId): void {
  const el = document.getElementById(id);
  if (!el) return;
  setOpen(el, true);
  if (id === 'drawer') drawerHandle.classList.add('open');
  emit(id, true);
}

export function closePanel(id: PanelId): void {
  const el = document.getElementById(id);
  if (!el) return;
  setOpen(el, false);
  if (id === 'drawer') drawerHandle.classList.remove('open');
  emit(id, false);
}

export function togglePanel(id: PanelId): void {
  const el = document.getElementById(id);
  if (!el) return;
  if (el.classList.contains('open')) closePanel(id);
  else openPanel(id);
}

// Wire the click handlers.
heartCorner.addEventListener('click', () => togglePanel('heart-panel'));
lungsCorner.addEventListener('click', () => togglePanel('lungs-panel'));
drawerHandle.addEventListener('click', () => togglePanel('drawer'));

document.querySelectorAll('[data-close]').forEach((btn) => {
  btn.addEventListener('click', (ev) => {
    ev.stopPropagation();
    const id = (btn as HTMLElement).dataset.close as PanelId;
    closePanel(id);
  });
});

// ESC closes any open panel.
document.addEventListener('keydown', (ev) => {
  if (ev.key !== 'Escape') return;
  if (heartPanel.classList.contains('open')) closePanel('heart-panel');
  if (lungsPanel.classList.contains('open')) closePanel('lungs-panel');
  if (drawer.classList.contains('open')) closePanel('drawer');
});
