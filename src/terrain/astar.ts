import { TerrainField } from './field';

/**
 * A* over the terrain heightfield, 8-connected.
 *
 * Step cost is horizontal distance scaled by a quadratic slope penalty, so at
 * slopeCost 0 the route is a straight line and at high values it hugs contours.
 * Nothing is impassable: a route always exists, which matters when the thing
 * replans continuously and a "no path" frame would just look broken.
 *
 * Scratch arrays are allocated once per grid size. Instead of clearing them
 * each replan, visited nodes carry a generation stamp that is compared against
 * a counter bumped per search - the arrays are only touched where the search
 * actually goes.
 */

const SQRT2 = Math.SQRT2;

export interface PlanResult {
  /** Node indices from start to goal. Only the first `length` are valid. */
  path: Int32Array;
  length: number;
  /** Nodes popped off the open set. */
  expanded: number;
  ms: number;
}

export class RoutePlanner {
  private w = 0;
  private h = 0;

  private g = new Float32Array(0);
  private cameFrom = new Int32Array(0);
  private stamp = new Uint32Array(0);
  private closed = new Uint8Array(0);
  private generation = 0;

  private heapNode = new Int32Array(0);
  private heapF = new Float32Array(0);
  private heapSize = 0;

  private pathBuf = new Int32Array(0);
  private scratch = new Int32Array(0);

  constructor(private field: TerrainField) {
    this.resize();
  }

  /** Call after the field changes resolution. */
  resize(): void {
    const f = this.field;
    this.w = f.nx + 1;
    this.h = f.nz + 1;
    const n = this.w * this.h;

    this.g = new Float32Array(n);
    this.cameFrom = new Int32Array(n);
    this.stamp = new Uint32Array(n);
    this.closed = new Uint8Array(n);
    this.generation = 0;

    this.heapNode = new Int32Array(n + 1);
    this.heapF = new Float32Array(n + 1);
    this.pathBuf = new Int32Array(n);
    this.scratch = new Int32Array(n);
  }

  nodeAt(i: number, k: number): number {
    return k * this.w + i;
  }

  /** Grid node nearest a normalised (0..1, 0..1) position across the terrain. */
  nodeAtNormalised(u: number, v: number): number {
    const i = Math.max(0, Math.min(this.w - 1, Math.round(u * (this.w - 1))));
    const k = Math.max(0, Math.min(this.h - 1, Math.round(v * (this.h - 1))));
    return this.nodeAt(i, k);
  }

  private push(node: number, f: number): void {
    let i = ++this.heapSize;
    this.heapNode[i] = node;
    this.heapF[i] = f;
    while (i > 1) {
      const parent = i >> 1;
      if (this.heapF[parent]! <= this.heapF[i]!) break;
      this.swap(i, parent);
      i = parent;
    }
  }

  private pop(): number {
    const top = this.heapNode[1]!;
    this.heapNode[1] = this.heapNode[this.heapSize]!;
    this.heapF[1] = this.heapF[this.heapSize]!;
    this.heapSize--;

    let i = 1;
    for (;;) {
      const l = i << 1;
      const r = l + 1;
      let smallest = i;
      if (l <= this.heapSize && this.heapF[l]! < this.heapF[smallest]!) smallest = l;
      if (r <= this.heapSize && this.heapF[r]! < this.heapF[smallest]!) smallest = r;
      if (smallest === i) break;
      this.swap(i, smallest);
      i = smallest;
    }
    return top;
  }

  private swap(a: number, b: number): void {
    const n = this.heapNode[a]!;
    this.heapNode[a] = this.heapNode[b]!;
    this.heapNode[b] = n;
    const f = this.heapF[a]!;
    this.heapF[a] = this.heapF[b]!;
    this.heapF[b] = f;
  }

  plan(start: number, goal: number, slopeCost: number): PlanResult {
    const t0 = performance.now();
    const f = this.field;
    const { w, h } = this;
    const heights = f.heights;
    const dx = f.spacing;

    const gen = ++this.generation;
    this.heapSize = 0;

    const gi = goal % w;
    const gk = (goal - gi) / w;

    // Octile distance in world units. Admissible because the cheapest possible
    // step costs exactly its horizontal length.
    const heuristic = (node: number): number => {
      const i = node % w;
      const k = (node - i) / w;
      const a = Math.abs(i - gi);
      const b = Math.abs(k - gk);
      const lo = a < b ? a : b;
      const hi = a < b ? b : a;
      return (hi - lo + lo * SQRT2) * dx;
    };

    this.g[start] = 0;
    this.stamp[start] = gen;
    this.closed[start] = 0; // `closed` survives across generations; stamp alone is not enough
    this.cameFrom[start] = -1;
    this.push(start, heuristic(start));

    let expanded = 0;
    let found = false;

    while (this.heapSize > 0) {
      const current = this.pop();
      if (this.stamp[current] === gen && this.closed[current] === 1) continue;
      this.closed[current] = 1;
      this.stamp[current] = gen;
      expanded++;

      if (current === goal) {
        found = true;
        break;
      }

      const ci = current % w;
      const ck = (current - ci) / w;
      const ch = heights[current]!;

      for (let dk = -1; dk <= 1; dk++) {
        const nk = ck + dk;
        if (nk < 0 || nk >= h) continue;
        for (let di = -1; di <= 1; di++) {
          if (di === 0 && dk === 0) continue;
          const ni = ci + di;
          if (ni < 0 || ni >= w) continue;

          const next = nk * w + ni;
          if (this.stamp[next] === gen && this.closed[next] === 1) continue;

          const horiz = (di !== 0 && dk !== 0) ? dx * SQRT2 : dx;
          const slope = Math.abs(heights[next]! - ch) / horiz;
          const cost = horiz * (1 + slopeCost * slope * slope);
          const tentative = this.g[current]! + cost;

          const seen = this.stamp[next] === gen;
          if (seen && tentative >= this.g[next]!) continue;

          this.g[next] = tentative;
          this.cameFrom[next] = current;
          this.stamp[next] = gen;
          this.closed[next] = 0;
          this.push(next, tentative + heuristic(next));
        }
      }
    }

    let length = 0;
    if (found) {
      let node = goal;
      let n = 0;
      while (node !== -1 && n < this.scratch.length) {
        this.scratch[n++] = node;
        node = this.cameFrom[node]!;
      }
      for (let i = 0; i < n; i++) this.pathBuf[i] = this.scratch[n - 1 - i]!;
      length = n;
    }

    return { path: this.pathBuf, length, expanded, ms: performance.now() - t0 };
  }
}
