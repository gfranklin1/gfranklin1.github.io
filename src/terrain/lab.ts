import { attachPointerControls } from './env';
import { TerrainView, type Controls } from './view';
import { VARIANTS } from './variants';

/**
 * Lab page wiring: three views on one shared frame budget, one controls panel
 * driving all of them so the variants stay comparable.
 */

const DEFAULTS: Controls = { noiseScale: 34, octaves: 4, slopeCost: 10, gridRes: 48 };

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const coarsePointer = window.matchMedia('(pointer: coarse)').matches;
const small = window.innerWidth < 760;

// Mobile drops grid resolution rather than frame rate, and loses the cursor
// interaction entirely - there is no cursor to follow.
if (small || coarsePointer) DEFAULTS.gridRes = 32;

interface Slot {
  view: TerrainView;
  canvas: HTMLCanvasElement;
  stats: HTMLElement;
  visible: boolean;
  hovered: boolean;
}

const slots: Slot[] = [];
const controls: Controls = { ...DEFAULTS };

function dpr(): number {
  return Math.min(window.devicePixelRatio || 1, 1.75);
}

function fit(slot: Slot): void {
  const rect = slot.canvas.getBoundingClientRect();
  if (rect.width < 1 || rect.height < 1) return;
  slot.view.resize(rect.width, rect.height, dpr());
}

function init(): void {
  for (const variant of VARIANTS) {
    const canvas = document.querySelector<HTMLCanvasElement>(`canvas[data-variant="${variant.id}"]`);
    const stats = document.querySelector<HTMLElement>(`[data-stats="${variant.id}"]`);
    if (!canvas || !stats) continue;

    let view: TerrainView;
    try {
      view = new TerrainView(canvas, variant, reducedMotion, controls);
    } catch (err) {
      stats.textContent = 'webgl unavailable';
      console.error(`[${variant.id}] failed to start`, err);
      continue;
    }

    const slot: Slot = { view, canvas, stats, visible: true, hovered: false };
    slots.push(slot);
    fit(slot);

    attachPointerControls(canvas, view);
    canvas.addEventListener('pointerenter', () => { slot.hovered = true; });
    canvas.addEventListener('pointerleave', () => { slot.hovered = false; });
  }

  if (slots.length === 0) return;

  const io = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        const slot = slots.find((s) => s.canvas === entry.target);
        if (slot) slot.visible = entry.isIntersecting;
      }
    },
    { rootMargin: '120px' },
  );
  for (const slot of slots) io.observe(slot.canvas);

  const ro = new ResizeObserver(() => {
    for (const slot of slots) fit(slot);
  });
  for (const slot of slots) ro.observe(slot.canvas);

  wireControls();
  wireScroll();
  requestAnimationFrame(frame);
}

function wireControls(): void {
  const bind = (id: keyof Controls, format: (v: number) => string) => {
    const input = document.querySelector<HTMLInputElement>(`#ctl-${id}`);
    const readout = document.querySelector<HTMLElement>(`#val-${id}`);
    if (!input) return;

    input.value = String(controls[id]);
    if (readout) readout.textContent = format(controls[id]);

    input.addEventListener('input', () => {
      controls[id] = Number(input.value);
      if (readout) readout.textContent = format(controls[id]);
      for (const slot of slots) slot.view.setControls(controls);
    });
  };

  bind('noiseScale', (v) => `${v} u`);
  bind('octaves', (v) => String(v));
  bind('slopeCost', (v) => v.toFixed(1));
  bind('gridRes', (v) => `${v}³`);
}

function wireScroll(): void {
  if (reducedMotion) return;
  let pending = false;
  const onScroll = () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      const span = document.documentElement.scrollHeight - window.innerHeight;
      const t = span > 0 ? window.scrollY / span : 0;
      for (const slot of slots) slot.view.setScroll(t);
    });
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
}

let lastStats = 0;

function frame(now: number): void {
  // Every visible view advances its own damped state and rebuilds whatever
  // that state changed. There is no cross-view budget any more: a partial
  // rebuild is exactly the thing that put peaks in the air.
  const active = slots.filter((s) => s.visible);
  for (const slot of active) slot.view.tick(now);

  if (now - lastStats > 200) {
    lastStats = now;
    for (const slot of active) {
      const s = slot.view.stats;
      slot.stats.textContent =
        `field ${s.fieldMs.toFixed(1)}ms · mesh ${s.meshMs.toFixed(1)}ms · plan ${s.planMs.toFixed(2)}ms · ` +
        `${(s.triangles / 1000).toFixed(1)}k tris · ${s.expanded.toLocaleString()} nodes expanded`;
    }
  }

  requestAnimationFrame(frame);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init, { once: true });
} else {
  init();
}
