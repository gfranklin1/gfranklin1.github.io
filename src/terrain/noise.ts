/**
 * 3D simplex noise, seeded and allocation-free in the hot path.
 * Ported from Gustavson's reference implementation; the permutation is
 * shuffled from a seed so terrain is reproducible across machines.
 */

const GRAD3 = new Int8Array([
  1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1, 0,
  1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, -1,
  0, 1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1,
]);

const F3 = 1 / 3;
const G3 = 1 / 6;

export class Simplex {
  private readonly p = new Uint8Array(512);
  private readonly pmod12 = new Uint8Array(512);

  constructor(seed = 1) {
    const src = new Uint8Array(256);
    for (let i = 0; i < 256; i++) src[i] = i;

    let s = (seed >>> 0) || 1;
    for (let i = 255; i > 0; i--) {
      s ^= s << 13; s >>>= 0;
      s ^= s >>> 17;
      s ^= s << 5; s >>>= 0;
      const j = s % (i + 1);
      const t = src[i]; src[i] = src[j]; src[j] = t;
    }

    for (let i = 0; i < 512; i++) {
      this.p[i] = src[i & 255]!;
      this.pmod12[i] = this.p[i]! % 12;
    }
  }

  /** Single octave. Output is roughly [-1, 1]. */
  noise(xin: number, yin: number, zin: number): number {
    const p = this.p;
    const pmod12 = this.pmod12;

    const s = (xin + yin + zin) * F3;
    const i = Math.floor(xin + s);
    const j = Math.floor(yin + s);
    const k = Math.floor(zin + s);

    const t = (i + j + k) * G3;
    const x0 = xin - (i - t);
    const y0 = yin - (j - t);
    const z0 = zin - (k - t);

    // Which of the six tetrahedra within the cell are we in?
    let i1: number, j1: number, k1: number;
    let i2: number, j2: number, k2: number;
    if (x0 >= y0) {
      if (y0 >= z0)      { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
      else if (x0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; }
      else               { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; }
    } else {
      if (y0 < z0)       { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; }
      else if (x0 < z0)  { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; }
      else               { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
    }

    const x1 = x0 - i1 + G3,       y1 = y0 - j1 + G3,       z1 = z0 - k1 + G3;
    const x2 = x0 - i2 + 2 * G3,   y2 = y0 - j2 + 2 * G3,   z2 = z0 - k2 + 2 * G3;
    const x3 = x0 - 1 + 3 * G3,    y3 = y0 - 1 + 3 * G3,    z3 = z0 - 1 + 3 * G3;

    const ii = i & 255, jj = j & 255, kk = k & 255;
    const gi0 = pmod12[ii + p[jj + p[kk]!]!]! * 3;
    const gi1 = pmod12[ii + i1 + p[jj + j1 + p[kk + k1]!]!]! * 3;
    const gi2 = pmod12[ii + i2 + p[jj + j2 + p[kk + k2]!]!]! * 3;
    const gi3 = pmod12[ii + 1 + p[jj + 1 + p[kk + 1]!]!]! * 3;

    let n = 0;

    let t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
    if (t0 > 0) {
      t0 *= t0;
      n += t0 * t0 * (GRAD3[gi0]! * x0 + GRAD3[gi0 + 1]! * y0 + GRAD3[gi0 + 2]! * z0);
    }
    let t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
    if (t1 > 0) {
      t1 *= t1;
      n += t1 * t1 * (GRAD3[gi1]! * x1 + GRAD3[gi1 + 1]! * y1 + GRAD3[gi1 + 2]! * z1);
    }
    let t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
    if (t2 > 0) {
      t2 *= t2;
      n += t2 * t2 * (GRAD3[gi2]! * x2 + GRAD3[gi2 + 1]! * y2 + GRAD3[gi2 + 2]! * z2);
    }
    let t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
    if (t3 > 0) {
      t3 *= t3;
      n += t3 * t3 * (GRAD3[gi3]! * x3 + GRAD3[gi3 + 1]! * y3 + GRAD3[gi3 + 2]! * z3);
    }

    return 32 * n;
  }

  /**
   * Weight for one octave under a band limit.
   *
   * An octave whose wavelength approaches the sample spacing cannot be
   * represented by the grid; sampling it anyway turns smooth ground into
   * shards. Octaves fade out over the top half of the usable band instead of
   * being cut off, so raising the grid resolution brings detail in smoothly
   * rather than popping it into existence.
   */
  private static bandWeight(f: number, maxFreq: number): number {
    if (!isFinite(maxFreq) || f <= maxFreq * 0.5) return 1;
    if (f >= maxFreq) return 0;
    const t = (f - maxFreq * 0.5) / (maxFreq * 0.5);
    return 1 - t * t * (3 - 2 * t);
  }

  /**
   * Fractal sum. `octaves` layers, each half the amplitude at twice the
   * frequency. `maxFreq` band-limits the sum to what the sampling grid can
   * actually carry; pass Infinity to disable.
   */
  fbm(
    x: number, y: number, z: number,
    octaves: number,
    maxFreq = Infinity,
    lacunarity = 2.02,
    gain = 0.5,
  ): number {
    let sum = 0, amp = 1, norm = 0, f = 1;
    for (let o = 0; o < octaves; o++) {
      // The base octave always contributes, or a coarse grid would have
      // nothing left and the normalisation would divide by zero.
      const w = o === 0 ? 1 : Simplex.bandWeight(f, maxFreq);
      if (w > 0) {
        sum += amp * w * this.noise(x * f, y * f, z * f);
        norm += amp * w;
      }
      amp *= gain;
      f *= lacunarity;
    }
    return sum / norm;
  }

  /**
   * Ridged multifractal. Each octave is folded about zero so crests stay sharp,
   * then attenuated by how strong the previous octave was at that point.
   *
   * That feedback term is the whole difference between a mountain range and a
   * bed of nails: detail is allowed to accumulate along ridgelines and is
   * suppressed in the valleys, so the crests join up into continuous ranges
   * instead of each octave spiking independently.
   *
   * Output is roughly [-1, 1].
   */
  ridged(
    x: number, y: number, z: number,
    octaves: number,
    maxFreq = Infinity,
    lacunarity = 2.02,
    gain = 0.5,
  ): number {
    let sum = 0, amp = 1, norm = 0, f = 1, weight = 1;
    for (let o = 0; o < octaves; o++) {
      const w = o === 0 ? 1 : Simplex.bandWeight(f, maxFreq);
      if (w > 0) {
        let signal = 1 - Math.abs(this.noise(x * f, y * f, z * f));
        signal *= signal;
        signal *= weight;
        weight = Math.min(1, signal * 2);
        sum += amp * w * signal;
        norm += amp * w;
      }
      amp *= gain;
      f *= lacunarity;
    }
    // Measured over a dense sample of the domain, sum/norm spans very nearly
    // the whole 0..1 interval, so a plain affine map fills the range without
    // clipping. Boosting and clamping here is what turns crests into mesas.
    return (sum / norm) * 2 - 1;
  }
}
