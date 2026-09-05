import { Simplex } from './noise';

/**
 * The scalar field the terrain is carved out of, plus the heightfield the
 * planner walks on.
 *
 * density(x, y, z) = (surface(x, z) - y) + warp, where `warp` is a 3D noise
 * term windowed to a band around the surface. The window is what keeps the
 * 3D term from spawning floating islands: away from the surface the linear
 * (surface - y) term dominates and the sign never flips.
 *
 * The field is diced into chunks so a cursor deformation only re-evaluates and
 * re-meshes the few chunks it actually touches.
 */

export const WORLD_SIZE = 100; // x and z extent, world units
export const WORLD_HEIGHT = 46;
export const CHUNK = 16; // cells per chunk edge

/** Minimum grid cells per noise wavelength before an octave is faded out. */
const NYQUIST_CELLS = 3.2;

/** Density forced onto the outer sample shell to close the mesh. */
const BOUNDARY_AIR = -1;

export interface FieldParams {
  /** Cells along x and z. Vertical resolution is derived. */
  gridRes: number;
  /** World units per noise feature. Larger is broader and smoother. */
  noiseScale: number;
  octaves: number;
  /** Scroll-driven translation through the noise, world units along z. */
  offset: number;
}

/** Per-variant terrain character. Not exposed in the controls panel. */
export interface TerrainStyle {
  /** Peak-to-trough relief, world units. */
  relief: number;
  /** Mean surface height, world units. */
  base: number;
  /** Strength of the 3D term. Above ~4 you start getting real overhangs. */
  warp: number;
  /** Ridged noise gives sharp crests; fbm gives rounded dunes. */
  ridged: boolean;
  /**
   * Per-variant multiplier on the shared noise scale. Lets one variant have
   * broader landforms than another without desynchronising the controls panel,
   * which drives every variant at once so they stay comparable.
   */
  scaleMul: number;
  seed: number;
}

export interface Brush {
  active: boolean;
  x: number;
  z: number;
  radius: number;
  /** Positive raises ground into a mound the route has to negotiate. */
  strength: number;
}

const clampInt = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, Math.round(v)));

export class TerrainField {
  nx = 0;
  ny = 0;
  nz = 0;
  /** Sample grid is (nx+1) * (ny+1) * (nz+1). */
  samples = new Float32Array(0);
  /** World y of the walkable top surface, (nx+1) * (nz+1). */
  heights = new Float32Array(0);

  cx = 0;
  cy = 0;
  cz = 0;
  /** One flag per chunk; set when its samples changed and it needs re-meshing. */
  dirty = new Uint8Array(0);

  readonly params: FieldParams = { gridRes: 48, noiseScale: 34, octaves: 4, offset: 0 };
  readonly brush: Brush = { active: false, x: 0, z: 0, radius: 10, strength: 16 };

  private noise: Simplex;
  private style: TerrainStyle;
  private dx = 0;
  private dy = 0;
  private dz = 0;

  constructor(style: TerrainStyle) {
    this.style = style;
    this.noise = new Simplex(style.seed);
    this.resize(this.params.gridRes);
  }

  get chunkCount(): number {
    return this.cx * this.cy * this.cz;
  }

  /** Cell spacing in world units. */
  get spacing(): number {
    return this.dx;
  }

  resize(gridRes: number): void {
    const n = clampInt(gridRes, 16, 96);
    this.params.gridRes = n;
    this.nx = n;
    this.nz = n;
    this.ny = Math.max(8, Math.round(n * 0.5));

    this.dx = WORLD_SIZE / this.nx;
    this.dz = WORLD_SIZE / this.nz;
    this.dy = WORLD_HEIGHT / this.ny;

    this.samples = new Float32Array((this.nx + 1) * (this.ny + 1) * (this.nz + 1));
    this.heights = new Float32Array((this.nx + 1) * (this.nz + 1));

    this.cx = Math.ceil(this.nx / CHUNK);
    this.cy = Math.ceil(this.ny / CHUNK);
    this.cz = Math.ceil(this.nz / CHUNK);
    this.dirty = new Uint8Array(this.cx * this.cy * this.cz);

    this.evaluateAll();
  }

  /** World position of sample (i, j, k). Origin is centred on x and z. */
  worldX(i: number): number { return i * this.dx - WORLD_SIZE / 2; }
  worldY(j: number): number { return j * this.dy; }
  worldZ(k: number): number { return k * this.dz - WORLD_SIZE / 2; }

  sampleIndex(i: number, j: number, k: number): number {
    return (k * (this.ny + 1) + j) * (this.nx + 1) + i;
  }

  /**
   * Highest octave frequency the grid can carry, in noise space. An octave at
   * frequency f has world wavelength `scale / f`; requiring at least
   * NYQUIST_CELLS samples across it gives the ceiling below.
   */
  private maxFreq(scale: number): number {
    return scale / (NYQUIST_CELLS * this.dx);
  }

  /** Surface height at a world (x, z), before the 3D warp. */
  private surface(x: number, z: number): number {
    const { noiseScale, octaves, offset } = this.params;
    const scale = Math.max(4, noiseScale * this.style.scaleMul);
    const s = 1 / scale;
    const mf = this.maxFreq(scale);
    const n = this.style.ridged
      ? this.noise.ridged(x * s, 0.37, (z + offset) * s, octaves, mf)
      : this.noise.fbm(x * s, 0.37, (z + offset) * s, octaves, mf);
    return this.style.base + n * this.style.relief * 0.5;
  }

  /**
   * Cursor mound contribution at a world (x, z). It is a vertical column, so
   * like the surface it is a function of two coordinates, not three.
   */
  private brushAt(x: number, z: number): number {
    const b = this.brush;
    if (!b.active) return 0;
    const ddx = x - b.x;
    const ddz = z - b.z;
    const r2 = (ddx * ddx + ddz * ddz) / (b.radius * b.radius);
    if (r2 >= 1) return 0;
    const f = 1 - r2;
    return b.strength * f * f;
  }

  /**
   * The 3D term, windowed to a band around the surface so it can carve
   * overhangs without detaching lumps of terrain into the air.
   */
  private warpAt(x: number, y: number, z: number, surf: number, band: number): number {
    const t = (surf - y) / band;
    const window = Math.exp(-t * t);
    if (window <= 0.002) return 0;
    const scale = Math.max(4, this.params.noiseScale * 0.55);
    const s = 1 / scale;
    const w = this.noise.fbm(
      x * s, y * s, (z + this.params.offset) * s,
      Math.min(3, this.params.octaves),
      this.maxFreq(scale),
    );
    return this.style.warp * w * window;
  }

  private markDirtyCell(i: number, j: number, k: number): void {
    const ci = Math.min(this.cx - 1, Math.floor(i / CHUNK));
    const cj = Math.min(this.cy - 1, Math.floor(j / CHUNK));
    const ck = Math.min(this.cz - 1, Math.floor(k / CHUNK));
    this.dirty[(ck * this.cy + cj) * this.cx + ci] = 1;
  }

  /**
   * Re-evaluate samples in an inclusive index box and flag the chunks that own
   * them. A sample on a chunk seam belongs to both neighbours, so the box is
   * widened by one cell before flagging.
   */
  evaluateRegion(i0: number, j0: number, k0: number, i1: number, j1: number, k1: number): void {
    const a0 = clampInt(i0, 0, this.nx), a1 = clampInt(i1, 0, this.nx);
    const b0 = clampInt(j0, 0, this.ny), b1 = clampInt(j1, 0, this.ny);
    const c0 = clampInt(k0, 0, this.nz), c1 = clampInt(k1, 0, this.nz);

    // Column-major on purpose. Both the surface height and the brush depend
    // only on (x, z), so hoisting them out of the vertical loop evaluates the
    // expensive fractal once per column instead of once per sample. With no 3D
    // warp term the whole column is then just an extruded heightfield and no
    // further noise is needed at all - which is the difference between a full
    // rebuild costing ~2k fractal evaluations and ~60k.
    const rowStride = this.nx + 1;
    const colStride = rowStride * (this.ny + 1);
    const warped = this.style.warp > 0;
    const band = this.style.relief * 0.45 + 1;

    for (let k = c0; k <= c1; k++) {
      const z = this.worldZ(k);
      const edgeK = k === 0 || k === this.nz;
      for (let i = a0; i <= a1; i++) {
        const x = this.worldX(i);
        // Forcing the outer shell to air closes the isosurface just inside the
        // domain, so the terrain is a solid slab with a clean chamfered edge
        // instead of a hollow shell you can see the back of.
        const edgeIK = edgeK || i === 0 || i === this.nx;
        const surf = edgeIK ? 0 : this.surface(x, z) + this.brushAt(x, z);

        let idx = k * colStride + b0 * rowStride + i;
        for (let j = b0; j <= b1; j++, idx += rowStride) {
          if (edgeIK || j === 0) {
            this.samples[idx] = BOUNDARY_AIR;
            continue;
          }
          const y = this.worldY(j);
          let d = surf - y;
          if (warped) d += this.warpAt(x, y, z, surf, band);
          this.samples[idx] = d;
        }
      }
    }

    for (let k = Math.max(0, c0 - 1); k <= c1; k++) {
      for (let j = Math.max(0, b0 - 1); j <= b1; j++) {
        for (let i = Math.max(0, a0 - 1); i <= a1; i++) this.markDirtyCell(i, j, k);
      }
    }

    this.updateHeights(a0, c0, a1, c1);
  }

  evaluateAll(): void {
    this.evaluateRegion(0, 0, 0, this.nx, this.ny, this.nz);
  }

  /** Index box (inclusive) covering the brush footprint at every height. */
  brushRegion(): [number, number, number, number, number, number] {
    const b = this.brush;
    const pad = b.radius + this.dx;
    return [
      Math.floor((b.x - pad + WORLD_SIZE / 2) / this.dx),
      0,
      Math.floor((b.z - pad + WORLD_SIZE / 2) / this.dz),
      Math.ceil((b.x + pad + WORLD_SIZE / 2) / this.dx),
      this.ny,
      Math.ceil((b.z + pad + WORLD_SIZE / 2) / this.dz),
    ];
  }

  /**
   * Walk each column top-down and record the first downward zero crossing:
   * the highest solid surface, which is what the planner is allowed to walk on.
   */
  updateHeights(i0: number, k0: number, i1: number, k1: number): void {
    for (let k = k0; k <= k1; k++) {
      for (let i = i0; i <= i1; i++) {
        let h = 0;
        let prev = this.samples[this.sampleIndex(i, this.ny, k)]!;
        for (let j = this.ny - 1; j >= 0; j--) {
          const cur = this.samples[this.sampleIndex(i, j, k)]!;
          if (prev < 0 && cur >= 0) {
            const t = prev / (prev - cur);
            h = this.worldY(j + 1) - t * this.dy;
            break;
          }
          prev = cur;
        }
        this.heights[k * (this.nx + 1) + i] = h;
      }
    }
  }
}
