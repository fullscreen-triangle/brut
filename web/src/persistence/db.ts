// IndexedDB-backed persistence for ObservatoryRecord.
//
// Schema v1:
//   db: brut-observatory
//   stores:
//     records (autoincrement key, indexed on `t`)
//
// API is intentionally minimal: putRecord, listSince, listAll, exportAll,
// clearAll, count. The dashboard hydrates from listSince(now - 7d) at
// startup and writes via putRecord on every push.
//
// All operations are best-effort: any IDB failure is logged and the rest
// of the app keeps running. Persistence is a convenience; the framework's
// O(1) memory claim is intact.

import type { ObservatoryRecord } from '../charts/dashboard';
import { log } from '../util/log';

const DB_NAME = 'brut-observatory';
const DB_VERSION = 1;
const STORE = 'records';

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: 'id', autoIncrement: true });
        store.createIndex('t', 't', { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
    req.onblocked = () => log('idb open blocked (other tab may be holding the db)');
  });
  return dbPromise;
}

/** Wall-clock anchor mapped against performance.now() so persisted timestamps survive page reloads. */
let wallAnchorEpoch = Date.now();
let perfAnchor = performance.now();

function toWall(t: number): number {
  return wallAnchorEpoch + (t - perfAnchor);
}

function fromWall(wt: number): number {
  return perfAnchor + (wt - wallAnchorEpoch);
}

/** Reset the wall/perf anchors. Call at session start. */
export function anchorNow(): void {
  wallAnchorEpoch = Date.now();
  perfAnchor = performance.now();
}

export interface PersistedRecord extends ObservatoryRecord {
  id?: number;
  // Wall-clock timestamp (ms epoch) so records survive across reloads.
  wallT: number;
}

export async function putRecord(rec: ObservatoryRecord): Promise<void> {
  try {
    const db = await openDb();
    const tx = db.transaction(STORE, 'readwrite');
    const store = tx.objectStore(STORE);
    store.add({ ...rec, wallT: toWall(rec.t) });
    await txComplete(tx);
  } catch (err) {
    log(`idb put failed: ${(err as Error).message}`);
  }
}

export async function listSince(wallMsAgo: number): Promise<ObservatoryRecord[]> {
  try {
    const db = await openDb();
    const cutoff = Date.now() - wallMsAgo;
    const tx = db.transaction(STORE, 'readonly');
    const idx = tx.objectStore(STORE).index('t');
    const range = IDBKeyRange.lowerBound(0); // all; we filter by wallT in code
    const out: ObservatoryRecord[] = [];
    return new Promise<ObservatoryRecord[]>((resolve, reject) => {
      const req = idx.openCursor(range);
      req.onsuccess = () => {
        const cursor = req.result;
        if (cursor) {
          const v = cursor.value as PersistedRecord;
          if (v.wallT >= cutoff) {
            // Map wall-clock back into our perf-time domain so charts behave.
            out.push({ ...v, t: fromWall(v.wallT) });
          }
          cursor.continue();
        } else {
          resolve(out);
        }
      };
      req.onerror = () => reject(req.error);
    });
  } catch (err) {
    log(`idb list failed: ${(err as Error).message}`);
    return [];
  }
}

export async function exportAll(): Promise<PersistedRecord[]> {
  try {
    const db = await openDb();
    const tx = db.transaction(STORE, 'readonly');
    const store = tx.objectStore(STORE);
    return new Promise<PersistedRecord[]>((resolve, reject) => {
      const req = store.getAll();
      req.onsuccess = () => resolve(req.result as PersistedRecord[]);
      req.onerror = () => reject(req.error);
    });
  } catch (err) {
    log(`idb export failed: ${(err as Error).message}`);
    return [];
  }
}

export async function clearAll(): Promise<void> {
  try {
    const db = await openDb();
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).clear();
    await txComplete(tx);
  } catch (err) {
    log(`idb clear failed: ${(err as Error).message}`);
  }
}

export async function countAll(): Promise<number> {
  try {
    const db = await openDb();
    const tx = db.transaction(STORE, 'readonly');
    return new Promise<number>((resolve, reject) => {
      const req = tx.objectStore(STORE).count();
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  } catch {
    return 0;
  }
}

function txComplete(tx: IDBTransaction): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

/** Trigger a download of the entire dataset as JSON. */
export async function downloadJsonExport(): Promise<void> {
  const data = await exportAll();
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `brut-observatory-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  log(`exported ${data.length} records`);
}
