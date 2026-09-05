import {
  attachPointerControls,
  defaultGridRes,
  fitToCanvas,
  narrowLayout,
  reducedMotion,
  scrollProgress,
} from './env';
import { TerrainView, type Controls } from './view';
import { VARIANTS } from './variants';

/**
 * Homepage terrain: one view, no controls panel. The parameters here are the
 * ones chosen in the lab, frozen.
 */

const CONTROLS: Controls = {
  noiseScale: 34,
  octaves: 4,
  slopeCost: 10,
  gridRes: defaultGridRes,
};

function init(): void {
  const canvas = document.querySelector<HTMLCanvasElement>('#terrain');
  if (!canvas) return;

  const variant = VARIANTS.find((v) => v.id === 'plate');
  if (!variant) return;

  let view: TerrainView;
  try {
    view = new TerrainView(canvas, variant, reducedMotion, CONTROLS);
  } catch (err) {
    // No WebGL: leave the canvas as an empty dark block. The page is readable
    // without it, and the intro never claims the terrain is there.
    console.error('terrain failed to start', err);
    canvas.remove();
    return;
  }

  /*
   * Desktop backs the camera off so the slab clears the type and sits inside
   * the viewport. The narrow layout has the canvas to itself, so it fills it
   * instead - the aspect-fit dolly would otherwise leave the slab marooned in
   * empty ground. Re-applied on resize because the two layouts want different
   * numbers and a window can cross the breakpoint.
   */
  const applyFraming = () => view.setFramingScale(narrowLayout() ? 0.91 : 0.85);
  fitToCanvas(view, canvas);
  applyFraming();
  new ResizeObserver(() => {
    fitToCanvas(view, canvas);
    applyFraming();
  }).observe(canvas);

  let visible = true;
  new IntersectionObserver(
    (entries) => {
      for (const e of entries) visible = e.isIntersecting;
    },
    { rootMargin: '120px' },
  ).observe(canvas);

  // Dragging an endpoint stays available under reduced motion: it is an
  // explicit action, not ambient movement. setPointer is the part that idles.
  attachPointerControls(canvas, view);

  if (!reducedMotion) {
    let pending = false;
    const onScroll = () => {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        view.setScroll(scrollProgress());
      });
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  const frame = (now: number) => {
    if (visible) view.tick(now);
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init, { once: true });
} else {
  init();
}
