const logEl = document.getElementById('log');
const statusEl = document.getElementById('status');

const MAX_LINES = 80;
const lines: string[] = [];

export function log(msg: string): void {
  const ts = new Date().toISOString().slice(11, 23);
  const line = `${ts}  ${msg}`;
  lines.push(line);
  if (lines.length > MAX_LINES) lines.shift();
  if (logEl) logEl.textContent = lines.join('\n');
  // eslint-disable-next-line no-console
  console.log(line);
}

export function setStatus(msg: string): void {
  if (statusEl) statusEl.textContent = msg;
}
