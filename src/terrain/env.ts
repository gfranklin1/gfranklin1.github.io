/**
 * Environment facts the terrain has to respect, read once at startup.
 * Shared by every page that mounts a view so they degrade identically.
 */

export const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
export const coarsePointer = window.matchMedia('(pointer: coarse)').matches;
export const smallScreen = window.innerWidth < 760;

/**
 * Small or touch devices drop grid resolution rather than frame rate, and lose
 * the cursor interaction entirely - there is no cursor to follow.
 */
export const defaultGridRes = smallScreen || coarsePointer ? 32 : 48;

/**
 * True while the page is in its stacked layout. Read live rather than cached:
 * a window can cross the breakpoint, and the two layouts frame the terrain
 * differently. Matches the CSS breakpoint in index.astro.
 */
export function narrowLayout(): boolean {
  return window.matchMedia('(max-width: 900px)').matches;
}

/** Cap the device pixel ratio; a 3x phone does not need 3x of this. */
export function renderScale(): number {
  return Math.min(window.devicePixelRatio || 1, 1.75);
}

/** Size a view to its canvas's CSS box. */
export function fitToCanvas(
  view: { resize(w: number, h: number, dpr: number): void },
  canvas: HTMLCanvasElement,
): void {
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 1 || rect.height < 1) return;
  view.resize(rect.width, rect.height, renderScale());
}

/** Page scroll progress, 0 at the top and 1 at the bottom. */
export function scrollProgress(): number {
  const span = document.documentElement.scrollHeight - window.innerHeight;
  return span > 0 ? window.scrollY / span : 0;
}

/** The slice of TerrainView the pointer wiring needs. */
interface Pointable {
  setPointer(px: number | null, py: number | null): void;
  dotAt(px: number, py: number): 'start' | 'goal' | null;
  beginDotDrag(which: 'start' | 'goal'): void;
  dragDotTo(px: number, py: number): void;
  endDotDrag(): void;
  tap(px: number, py: number): 'activated' | 'placed';
  setCoarsePointer(on: boolean): void;
  readonly isDraggingDot: boolean;
}

/** A pointerup counts as a tap only if it barely moved and did not linger. */
const TAP_SLOP_PX = 12;
const TAP_MAX_MS = 600;

/**
 * Endpoint placement by tapping, for touch. Tapping a dot makes it the active
 * endpoint; tapping the ground moves whichever is active.
 *
 * Dragging is deliberately not offered here: the canvas would have to claim
 * the gesture on pointerdown, before it knows whether the finger started on a
 * dot, and claiming it stops the page scrolling over a canvas that fills half
 * the screen. Tapping needs no such claim, so nothing is preventDefault-ed and
 * scrolling is untouched - a tap is simply a press that went nowhere.
 */
function attachTapControls(canvas: HTMLCanvasElement, view: Pointable): void {
  let downX = 0;
  let downY = 0;
  let downAt = 0;

  canvas.addEventListener('pointerdown', (e) => {
    downX = e.clientX;
    downY = e.clientY;
    downAt = e.timeStamp;
  });

  canvas.addEventListener('pointerup', (e) => {
    const moved = Math.hypot(e.clientX - downX, e.clientY - downY);
    if (moved > TAP_SLOP_PX || e.timeStamp - downAt > TAP_MAX_MS) return;
    const r = canvas.getBoundingClientRect();
    view.tap(e.clientX - r.left, e.clientY - r.top);
  });
}

/**
 * Cursor deformation and endpoint dragging for one canvas, shared so both
 * pages behave identically. Coarse pointers get tap-to-place instead, and no
 * cursor deformation - there is no hover to drive it.
 */
export function attachPointerControls(canvas: HTMLCanvasElement, view: Pointable): void {
  view.setCoarsePointer(coarsePointer);
  if (coarsePointer) {
    attachTapControls(canvas, view);
    return;
  }

  const local = (e: PointerEvent) => {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  canvas.addEventListener('pointerdown', (e) => {
    const { x, y } = local(e);
    const hit = view.dotAt(x, y);
    if (!hit) return;
    view.beginDotDrag(hit);
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = 'grabbing';
    e.preventDefault();
  });

  canvas.addEventListener('pointermove', (e) => {
    const { x, y } = local(e);
    if (view.isDraggingDot) {
      view.dragDotTo(x, y);
      return;
    }
    canvas.style.cursor = view.dotAt(x, y) ? 'grab' : '';
    view.setPointer(x, y);
  });

  const release = (e: PointerEvent) => {
    if (!view.isDraggingDot) return;
    view.endDotDrag();
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    canvas.style.cursor = '';
  };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);

  canvas.addEventListener('pointerleave', () => {
    if (view.isDraggingDot) return;
    canvas.style.cursor = '';
    view.setPointer(null, null);
  });
}
