import type { TerrainStyle } from './field';

/**
 * Three art directions for the same scene. They differ along the three axes
 * worth deciding first: camera angle, terrain scale, and route contrast.
 * Everything here is fixed per variant - the controls panel drives the field
 * parameters, which are shared, so the variants stay comparable.
 *
 * Colours are only ever the four in the brief: ink #0e0e0e, paper #ece8e1,
 * burnt orange #d9662f, moss #7fb069. Terrain sits on the ink-paper axis.
 */

export const INK = 0x0e0e0e;
export const PAPER = 0xece8e1;
export const ORANGE = 0xd9662f;
export const MOSS = 0x7fb069;

export interface Variant {
  id: string;
  name: string;
  /** Mono caption under the canvas: what was actually chosen and why. */
  note: string;
  style: TerrainStyle;
  camera: {
    position: readonly [number, number, number];
    target: readonly [number, number, number];
    fov: number;
  };
  fog: { near: number; far: number };
  light: {
    dir: readonly [number, number, number];
    intensity: number;
    ambient: number;
  };
  terrainColor: number;
  flatShading: boolean;
  route: {
    color: number;
    /** Screen-space width in CSS pixels. */
    width: number;
    /** How far the line floats above the surface, world units. */
    lift: number;
    dotRadius: number;
  };
}

export const VARIANTS: readonly Variant[] = [
  {
    id: 'ridge',
    name: 'A — Ridge',
    note: 'Low camera, barely above the peaks. Tall ridged noise, sharp crests, raking side light. Route at plain accent weight, ~2px — it disappears behind ridges and comes back, which is either the point or the problem.',
    style: { relief: 25, base: 21, warp: 0, ridged: true, scaleMul: 1.75, seed: 7 },
    camera: { position: [112, 52, 112], target: [0, 12, 0], fov: 32 },
    fog: { near: 110, far: 290 },
    light: { dir: [-0.55, 0.38, 0.3], intensity: 1.15, ambient: 0.045 },
    terrainColor: 0xc4bcb1,
    flatShading: false,
    route: { color: ORANGE, width: 2, lift: 0.7, dotRadius: 0.6 },
  },
  {
    id: 'plate',
    name: 'B — Plate',
    note: 'High three-quarter, close to plan view, both endpoints in frame. Broad dunes, faceted shading, and a low raking key light so the landforms read by their shadow side rather than by colour. Route thin and quiet at ~1.5px — inscribed on a survey model rather than drawn on top of a landscape.',
    style: { relief: 25, base: 17, warp: 0, ridged: false, scaleMul: 1, seed: 21 },
    camera: { position: [61, 172, 104], target: [4, 11, 6], fov: 32 },
    fog: { near: 140, far: 290 },
    light: { dir: [-0.6, 0.52, 0.29], intensity: 1.12, ambient: 0.06 },
    terrainColor: 0xbfb7ac,
    flatShading: true,
    route: { color: ORANGE, width: 1.5, lift: 0.5, dotRadius: 0.55 },
  },
  {
    id: 'trench',
    name: 'C — Trench',
    note: 'Low and close behind the start point, wide lens, heavy fog — the route leaves the foreground and runs away from you into the murk. Medium relief with the 3D term turned up far enough to cut real overhangs, the part a heightmap could not do. Route on the moss accent, ~2.6px, highest contrast.',
    style: { relief: 26, base: 20, warp: 15, ridged: false, scaleMul: 0.85, seed: 43 },
    camera: { position: [-75, 66, -75], target: [2, 9, 2], fov: 46 },
    fog: { near: 55, far: 175 },
    light: { dir: [-0.6, 0.4, 0.68], intensity: 1, ambient: 0.03 },
    terrainColor: 0xc6bdb2,
    flatShading: false,
    route: { color: MOSS, width: 2.6, lift: 0.8, dotRadius: 0.7 },
  },
];

/** Fixed endpoints, in normalised terrain coordinates. Same for every variant. */
export const ROUTE_START = { u: 0.08, v: 0.14 } as const;
export const ROUTE_GOAL = { u: 0.93, v: 0.87 } as const;
