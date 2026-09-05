import { CHUNK, TerrainField } from './field';
import { CORNER_OFFSETS, EDGE_CORNERS, EDGE_TABLE, TRI_TABLE } from './marchingCubes';

/**
 * Polygonises one chunk of a TerrainField.
 *
 * Output is non-indexed triangle soup written into scratch buffers that grow
 * on demand and are reused across calls, so a steady-state rebuild allocates
 * nothing. Normals come from the analytic field gradient rather than from face
 * normals, which keeps chunk seams smooth: neighbouring chunks compute the
 * same gradient at a shared sample and agree exactly.
 */

export interface MeshResult {
  positions: Float32Array;
  normals: Float32Array;
  /** Vertices written; the buffers above are longer than this. */
  vertexCount: number;
}

export class Mesher {
  private positions = new Float32Array(4096 * 3);
  private normals = new Float32Array(4096 * 3);
  private n = 0;

  // Per-cell scratch, allocated once.
  private readonly cv = new Float32Array(8); // corner values
  private readonly cp = new Float32Array(24); // corner world positions
  private readonly cg = new Float32Array(24); // corner gradients
  private readonly ev = new Float32Array(36); // edge vertex positions
  private readonly en = new Float32Array(36); // edge vertex normals

  constructor(private field: TerrainField) {}

  private ensure(vertices: number): void {
    if (vertices * 3 <= this.positions.length) return;
    let cap = this.positions.length / 3;
    while (cap < vertices) cap *= 2;
    const p = new Float32Array(cap * 3);
    p.set(this.positions);
    this.positions = p;
    const nrm = new Float32Array(cap * 3);
    nrm.set(this.normals);
    this.normals = nrm;
  }

  /** Central-difference gradient of the sample grid, clamped at the border. */
  private gradient(i: number, j: number, k: number, out: Float32Array, o: number): void {
    const f = this.field;
    const s = f.samples;
    const sx = f.nx + 1;
    const sxy = sx * (f.ny + 1);

    const xm = i > 0 ? i - 1 : 0;
    const xp = i < f.nx ? i + 1 : f.nx;
    const ym = j > 0 ? j - 1 : 0;
    const yp = j < f.ny ? j + 1 : f.ny;
    const zm = k > 0 ? k - 1 : 0;
    const zp = k < f.nz ? k + 1 : f.nz;

    const row = k * sxy + j * sx;
    out[o] = (s[row + xp]! - s[row + xm]!) / (xp - xm);
    out[o + 1] = (s[k * sxy + yp * sx + i]! - s[k * sxy + ym * sx + i]!) / (yp - ym);
    out[o + 2] = (s[zp * sxy + j * sx + i]! - s[zm * sxy + j * sx + i]!) / (zp - zm);
  }

  meshChunk(chunkIndex: number): MeshResult {
    const f = this.field;
    this.n = 0;

    const ci = chunkIndex % f.cx;
    const cj = Math.floor(chunkIndex / f.cx) % f.cy;
    const ck = Math.floor(chunkIndex / (f.cx * f.cy));

    const i0 = ci * CHUNK, i1 = Math.min(f.nx, i0 + CHUNK);
    const j0 = cj * CHUNK, j1 = Math.min(f.ny, j0 + CHUNK);
    const k0 = ck * CHUNK, k1 = Math.min(f.nz, k0 + CHUNK);

    const { cv, cp, cg, ev, en } = this;

    for (let k = k0; k < k1; k++) {
      for (let j = j0; j < j1; j++) {
        for (let i = i0; i < i1; i++) {
          // Bit set when the corner is outside the terrain (density < 0),
          // matching the sign convention the case table was built for.
          let cube = 0;
          for (let c = 0; c < 8; c++) {
            const off = CORNER_OFFSETS[c]!;
            const v = f.samples[f.sampleIndex(i + off[0], j + off[1], k + off[2])]!;
            cv[c] = v;
            if (v < 0) cube |= 1 << c;
          }

          const edges = EDGE_TABLE[cube]!;
          if (edges === 0) continue;

          for (let c = 0; c < 8; c++) {
            const off = CORNER_OFFSETS[c]!;
            const gi = i + off[0], gj = j + off[1], gk = k + off[2];
            cp[c * 3] = f.worldX(gi);
            cp[c * 3 + 1] = f.worldY(gj);
            cp[c * 3 + 2] = f.worldZ(gk);
            this.gradient(gi, gj, gk, cg, c * 3);
          }

          for (let e = 0; e < 12; e++) {
            if ((edges & (1 << e)) === 0) continue;
            const [a, b] = EDGE_CORNERS[e]!;
            const va = cv[a]!, vb = cv[b]!;
            const denom = va - vb;
            const t = Math.abs(denom) < 1e-9 ? 0.5 : va / denom;

            const ao = a * 3, bo = b * 3, eo = e * 3;
            ev[eo] = cp[ao]! + t * (cp[bo]! - cp[ao]!);
            ev[eo + 1] = cp[ao + 1]! + t * (cp[bo + 1]! - cp[ao + 1]!);
            ev[eo + 2] = cp[ao + 2]! + t * (cp[bo + 2]! - cp[ao + 2]!);

            // Gradient points into the solid; the outward normal is its negation.
            let gx = -(cg[ao]! + t * (cg[bo]! - cg[ao]!));
            let gy = -(cg[ao + 1]! + t * (cg[bo + 1]! - cg[ao + 1]!));
            let gz = -(cg[ao + 2]! + t * (cg[bo + 2]! - cg[ao + 2]!));
            const len = Math.hypot(gx, gy, gz) || 1;
            en[eo] = gx / len;
            en[eo + 1] = gy / len;
            en[eo + 2] = gz / len;
          }

          const base = cube * 16;
          for (let t = 0; TRI_TABLE[base + t] !== -1; t += 3) {
            const e0 = TRI_TABLE[base + t]! * 3;
            const e1 = TRI_TABLE[base + t + 1]! * 3;
            const e2 = TRI_TABLE[base + t + 2]! * 3;

            this.ensure(this.n + 3);
            const p = this.positions, nrm = this.normals;
            let o = this.n * 3;

            // Emit with the winding that agrees with the analytic normal, so
            // backface culling is correct without trusting the table's order.
            const ax = ev[e1]! - ev[e0]!, ay = ev[e1 + 1]! - ev[e0 + 1]!, az = ev[e1 + 2]! - ev[e0 + 2]!;
            const bx = ev[e2]! - ev[e0]!, by = ev[e2 + 1]! - ev[e0 + 1]!, bz = ev[e2 + 2]! - ev[e0 + 2]!;
            const fx = ay * bz - az * by;
            const fy = az * bx - ax * bz;
            const fz = ax * by - ay * bx;
            const flip = fx * en[e0]! + fy * en[e0 + 1]! + fz * en[e0 + 2]! < 0;

            const s1 = flip ? e2 : e1;
            const s2 = flip ? e1 : e2;

            p[o] = ev[e0]!; p[o + 1] = ev[e0 + 1]!; p[o + 2] = ev[e0 + 2]!;
            nrm[o] = en[e0]!; nrm[o + 1] = en[e0 + 1]!; nrm[o + 2] = en[e0 + 2]!;
            p[o + 3] = ev[s1]!; p[o + 4] = ev[s1 + 1]!; p[o + 5] = ev[s1 + 2]!;
            nrm[o + 3] = en[s1]!; nrm[o + 4] = en[s1 + 1]!; nrm[o + 5] = en[s1 + 2]!;
            p[o + 6] = ev[s2]!; p[o + 7] = ev[s2 + 1]!; p[o + 8] = ev[s2 + 2]!;
            nrm[o + 6] = en[s2]!; nrm[o + 7] = en[s2 + 1]!; nrm[o + 8] = en[s2 + 2]!;

            this.n += 3;
          }
        }
      }
    }

    return { positions: this.positions, normals: this.normals, vertexCount: this.n };
  }
}
