import * as THREE from 'three';
import { Line2 } from 'three/examples/jsm/lines/Line2.js';
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';

import { RoutePlanner } from './astar';
import { CHUNK, TerrainField, WORLD_HEIGHT, WORLD_SIZE } from './field';
import { Mesher } from './mesher';
import { INK, ROUTE_GOAL, ROUTE_START, type Variant } from './variants';

export interface Controls {
  noiseScale: number;
  octaves: number;
  slopeCost: number;
  gridRes: number;
}

export interface ViewStats {
  meshMs: number;
  planMs: number;
  triangles: number;
  expanded: number;
  /** Time re-evaluating the scalar field on the last frame that changed. */
  fieldMs: number;
  /** Chunks re-extracted on the last frame that changed. */
  chunksRebuilt: number;
}

const CHAIKIN_PASSES = 2;

/**
 * The route is always resampled to this many points. A fixed count means the
 * line geometry is allocated once and rewritten in place, instead of throwing
 * away a GPU buffer on every replan - and the replan runs every frame the
 * cursor moves.
 */
const ROUTE_POINTS = 192;

/**
 * The aspect each variant's camera was framed against. A canvas narrower than
 * this has a smaller horizontal field of view and would crop the slab, so the
 * camera dollies back to hold the framing. Wider canvases are left alone -
 * they already show everything, and pulling back would only shrink the subject.
 */
const FRAMED_ASPECT = 1.15;

/**
 * Damping time constant for the field state, milliseconds. Cursor and scroll
 * both write targets; the state eases toward them so the ground never jumps.
 */
const DAMP_TAU_MS = 90;

/**
 * How far the damped state must drift from what the field was last built with
 * before a rebuild is worth it. Every commit rebuilds the whole surface at
 * once, so this deadband is what keeps that affordable: without it a scroll
 * would re-evaluate the entire grid on every single frame.
 */
const OFFSET_DEADBAND = 0.3; // world units
const BRUSH_MOVE_DEADBAND = 0.5; // world units
const BRUSH_AMOUNT_DEADBAND = 0.015; // fraction of full strength

/**
 * Idle creep of the noise offset, world units per second. Slow enough that a
 * still frame looks static and only a couple of seconds of watching gives it
 * away. Suppressed entirely under reduced motion, since stepFieldState returns
 * before the drift is applied.
 */
const IDLE_DRIFT_PER_SEC = 0.8;

/** Peak height the cursor mound reaches once it has fully faded in. */
const BRUSH_STRENGTH = 16;

/** Grab radius around an endpoint dot, CSS pixels. Larger than the dot. */
const DOT_HIT_RADIUS_PX = 18;

/**
 * Cursor influence ramps in with distance from the nearest endpoint: nothing
 * at the grab radius, full at three times it, smoothstepped between.
 *
 * A hard cutoff here is worse than no suppression at all - crossing it drops
 * the mound in one step and the dot visibly jumps. The ramp only sets a
 * target; the easing comes from the damped field state it feeds.
 */
const BRUSH_FADE_NEAR_PX = DOT_HIT_RADIUS_PX;
const BRUSH_FADE_FAR_PX = DOT_HIT_RADIUS_PX * 3;

/** Hermite ramp, 0 below edge0 and 1 above edge1. */
function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/**
 * Touch affordances. A finger needs a bigger mark than a cursor does, and it
 * needs to be obvious which endpoint the next tap will move, so both dots draw
 * larger with a thin accent ring and the active one carries a wider, brighter
 * ring than the idle one.
 */
const TOUCH_DOT_SCALE = 1.5;
const RING_RADIUS_SCALE = 4.3;
const ACTIVE_RING_SCALE = 1.5;
const IDLE_RING_OPACITY = 0.3;

/**
 * Tap radius on coarse pointers: a 44px target, the usual floor for a finger,
 * regardless of how small the dot is drawn.
 */
const TOUCH_HIT_RADIUS_PX = 22;

/**
 * How long the route takes to travel from its old shape to a new one. Short
 * enough to feel immediate, long enough that a replan reads as the line moving
 * rather than as one line being swapped for another.
 */
const ROUTE_EASE_MS = 150;

/** Keep dragged endpoints this far inside the slab, as a fraction of it. */
const ENDPOINT_MARGIN = 0.05;

/** One canvas: terrain, route, and the loop that keeps them in sync. */
export class TerrainView {
  readonly stats: ViewStats = {
    meshMs: 0, planMs: 0, fieldMs: 0, triangles: 0, expanded: 0, chunksRebuilt: 0,
  };

  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private field: TerrainField;
  private mesher: Mesher;
  private planner: RoutePlanner;

  private terrainGroup = new THREE.Group();
  private material: THREE.MeshLambertMaterial;
  private meshes: THREE.Mesh[] = [];

  private routeLine: Line2;
  private routeMaterial: LineMaterial;
  private routeGeometry: LineGeometry;
  private routeSegments: THREE.InstancedInterleavedBuffer;
  private startDot: THREE.Mesh;
  private goalDot: THREE.Mesh;
  private dotGeometry: THREE.SphereGeometry;
  private dotMaterial: THREE.MeshBasicMaterial;
  private startRing: THREE.Mesh;
  private goalRing: THREE.Mesh;
  private ringGeometry: THREE.RingGeometry;
  private activeRingMaterial: THREE.MeshBasicMaterial;
  private idleRingMaterial: THREE.MeshBasicMaterial;
  /** Touch affordances are off until a coarse-pointer page turns them on. */
  private coarse = false;

  private slopeCost = 6;
  private needsPlan = true;

  // The single damped field state. Inputs only ever write the target half;
  // nothing outside stepFieldState is allowed to touch the field itself.
  private offsetTarget = 0;
  private dampedOffset = 0;
  /** Scroll and idle creep are summed into offsetTarget once per frame. */
  private scrollOffset = 0;
  private idleOffset = 0;
  private brushTargetX = 0;
  private brushTargetZ = 0;
  private brushTargetAmount = 0;
  private dampedBrushX = 0;
  private dampedBrushZ = 0;
  private dampedBrushAmount = 0;
  private lastTickMs = 0;

  // Route endpoints, in normalised terrain coordinates so they survive a grid
  // resolution change. Seeded from the variant defaults, then dragged.
  private readonly startUV: { u: number; v: number } = { u: ROUTE_START.u, v: ROUTE_START.v };
  private readonly goalUV: { u: number; v: number } = { u: ROUTE_GOAL.u, v: ROUTE_GOAL.v };
  private dragging: 'start' | 'goal' | null = null;
  /**
   * Which endpoint a ground tap moves. Sticky: it stays put until the other
   * dot is tapped, so repeated taps keep moving the same endpoint. The goal is
   * the default, and both names refer to the same dots the desktop drags -
   * never to whichever happens to be nearer the finger.
   */
  private active: 'start' | 'goal' = 'goal';
  private readonly projScratch = new THREE.Vector3();
  private pointerPlane: THREE.Plane;
  private raycaster = new THREE.Raycaster();
  private pointerNdc = new THREE.Vector2();
  private hitPoint = new THREE.Vector3();

  // Route working buffers, reused every replan.
  private rawPath: number[] = [];
  private smoothA: number[] = [];
  private smoothB: number[] = [];
  private resampled = new Float32Array(ROUTE_POINTS * 3);
  /** Endpoints of the route ease: where the line was, and where it is heading. */
  private routeFrom = new Float32Array(ROUTE_POINTS * 3);
  private routeTo = new Float32Array(ROUTE_POINTS * 3);
  /** Timestamp the current ease began, or -1 when the line is settled. */
  private routeEaseStart = -1;
  /** False until the first plan, which has nothing to ease from. */
  private hasRoute = false;

  private width = 1;
  private height = 1;
  private dpr = 1;
  /** Below 1 shrinks the subject in frame without changing the canvas box. */
  private framingScale = 1;
  private disposed = false;

  /** Camera rig as authored, before any aspect-fit dolly. */
  private readonly baseEye = new THREE.Vector3();
  private readonly lookAt = new THREE.Vector3();

  constructor(
    canvas: HTMLCanvasElement,
    readonly variant: Variant,
    private reducedMotion: boolean,
    controls: Controls,
  ) {
    this.field = new TerrainField(variant.style);
    this.field.params.noiseScale = controls.noiseScale;
    this.field.params.octaves = controls.octaves;
    this.slopeCost = controls.slopeCost;
    this.field.resize(controls.gridRes);

    this.mesher = new Mesher(this.field);
    this.planner = new RoutePlanner(this.field);

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setClearColor(INK, 1);

    this.scene.fog = new THREE.Fog(INK, variant.fog.near, variant.fog.far);

    this.camera = new THREE.PerspectiveCamera(variant.camera.fov, 1, 0.5, 500);
    this.baseEye.set(...variant.camera.position);
    this.lookAt.set(...variant.camera.target);
    this.camera.position.copy(this.baseEye);
    this.camera.lookAt(this.lookAt);

    const dir = new THREE.DirectionalLight(0xffffff, variant.light.intensity);
    dir.position.set(...variant.light.dir).normalize().multiplyScalar(100);
    this.scene.add(dir);
    this.scene.add(new THREE.AmbientLight(0xffffff, variant.light.ambient));

    this.material = new THREE.MeshLambertMaterial({
      color: variant.terrainColor,
      flatShading: variant.flatShading,
      side: THREE.FrontSide,
    });
    this.scene.add(this.terrainGroup);
    this.buildChunkMeshes();

    this.routeMaterial = new LineMaterial({
      color: variant.route.color,
      linewidth: variant.route.width,
      worldUnits: false,
      dashed: false,
    });
    this.routeGeometry = new LineGeometry();
    this.routeGeometry.setPositions(new Float32Array(ROUTE_POINTS * 3));
    this.routeSegments = (
      this.routeGeometry.getAttribute('instanceStart') as THREE.InterleavedBufferAttribute
    ).data as THREE.InstancedInterleavedBuffer;
    this.routeLine = new Line2(this.routeGeometry, this.routeMaterial);
    this.routeLine.frustumCulled = false;
    this.scene.add(this.routeLine);

    const dotR = variant.route.dotRadius;
    this.dotGeometry = new THREE.SphereGeometry(dotR, 12, 8);
    this.dotMaterial = new THREE.MeshBasicMaterial({ color: variant.route.color, fog: false });
    this.startDot = new THREE.Mesh(this.dotGeometry, this.dotMaterial);
    this.goalDot = new THREE.Mesh(this.dotGeometry, this.dotMaterial);
    this.scene.add(this.startDot, this.goalDot);

    // A thin annulus, billboarded to the camera. Hidden entirely on a fine
    // pointer, where the cursor change over a dot already says enough.
    const ringR = dotR * RING_RADIUS_SCALE;
    this.ringGeometry = new THREE.RingGeometry(ringR * 0.9, ringR, 48);
    this.activeRingMaterial = new THREE.MeshBasicMaterial({
      color: variant.route.color, fog: false, side: THREE.DoubleSide, transparent: true,
    });
    this.idleRingMaterial = new THREE.MeshBasicMaterial({
      color: variant.route.color, fog: false, side: THREE.DoubleSide,
      transparent: true, opacity: IDLE_RING_OPACITY,
    });
    this.startRing = new THREE.Mesh(this.ringGeometry, this.idleRingMaterial);
    this.goalRing = new THREE.Mesh(this.ringGeometry, this.activeRingMaterial);
    this.startRing.visible = false;
    this.goalRing.visible = false;
    this.startRing.frustumCulled = false;
    this.goalRing.frustumCulled = false;
    this.scene.add(this.startRing, this.goalRing);

    this.pointerPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -variant.style.base);

    // First build is synchronous so the first frame on screen is complete.
    this.rebuildAll();
    this.replan();
  }

  // ---------------------------------------------------------------- terrain

  private buildChunkMeshes(): void {
    for (const m of this.meshes) {
      m.geometry.dispose();
      this.terrainGroup.remove(m);
    }
    this.meshes = [];

    const f = this.field;
    for (let c = 0; c < f.chunkCount; c++) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3 * 3), 3));
      geo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(3 * 3), 3));
      geo.setDrawRange(0, 0);

      // Bounding sphere is the chunk's world box, set once. Recomputing it on
      // every rebuild would cost more than the culling saves.
      const ci = c % f.cx;
      const cj = Math.floor(c / f.cx) % f.cy;
      const ck = Math.floor(c / (f.cx * f.cy));
      const sx = (WORLD_SIZE / f.nx) * CHUNK;
      const sy = (WORLD_HEIGHT / f.ny) * CHUNK;
      const sz = (WORLD_SIZE / f.nz) * CHUNK;
      geo.boundingSphere = new THREE.Sphere(
        new THREE.Vector3(
          (ci + 0.5) * sx - WORLD_SIZE / 2,
          (cj + 0.5) * sy,
          (ck + 0.5) * sz - WORLD_SIZE / 2,
        ),
        Math.hypot(sx, sy, sz) * 0.5,
      );

      const mesh = new THREE.Mesh(geo, this.material);
      this.meshes.push(mesh);
      this.terrainGroup.add(mesh);
    }
  }

  private uploadChunk(chunkIndex: number): void {
    const res = this.mesher.meshChunk(chunkIndex);
    const geo = this.meshes[chunkIndex]!.geometry;
    let pos = geo.getAttribute('position') as THREE.BufferAttribute;
    let nrm = geo.getAttribute('normal') as THREE.BufferAttribute;

    if (pos.count < res.vertexCount) {
      let cap = Math.max(pos.count, 1);
      while (cap < res.vertexCount) cap *= 2;
      geo.deleteAttribute('position');
      geo.deleteAttribute('normal');
      pos = new THREE.BufferAttribute(new Float32Array(cap * 3), 3);
      nrm = new THREE.BufferAttribute(new Float32Array(cap * 3), 3);
      geo.setAttribute('position', pos);
      geo.setAttribute('normal', nrm);
    }

    (pos.array as Float32Array).set(res.positions.subarray(0, res.vertexCount * 3));
    (nrm.array as Float32Array).set(res.normals.subarray(0, res.vertexCount * 3));
    pos.needsUpdate = true;
    nrm.needsUpdate = true;
    geo.setDrawRange(0, res.vertexCount);
  }

  private countTriangles(): void {
    let tris = 0;
    for (const m of this.meshes) tris += m.geometry.drawRange.count / 3;
    this.stats.triangles = tris;
  }

  private rebuildAll(): void {
    const t0 = performance.now();
    for (let c = 0; c < this.field.chunkCount; c++) {
      this.uploadChunk(c);
      this.field.dirty[c] = 0;
    }
    this.stats.meshMs = performance.now() - t0;
    this.stats.chunksRebuilt = this.field.chunkCount;
    this.countTriangles();
  }

  /**
   * Re-extract every dirty chunk. Deliberately unbudgeted: spreading an
   * extraction across frames leaves neighbouring chunks holding surfaces from
   * different field states, which is what put peaks in the air with nothing
   * under them. Either the whole surface moves this frame or none of it does.
   */
  private drainDirty(): void {
    const f = this.field;
    let done = 0;
    const t0 = performance.now();

    for (let c = 0; c < f.chunkCount; c++) {
      if (f.dirty[c] !== 1) continue;
      this.uploadChunk(c);
      f.dirty[c] = 0;
      done++;
    }

    if (done > 0) {
      this.stats.meshMs = performance.now() - t0;
      this.countTriangles();
    }
    this.stats.chunksRebuilt = done;
  }

  // ------------------------------------------------------------------ route

  /** Bilinear height lookup in world (x, z). */
  private heightAt(x: number, z: number): number {
    const f = this.field;
    const w = f.nx + 1;
    const fx = Math.max(0, Math.min(f.nx, (x + WORLD_SIZE / 2) / f.spacing));
    const fz = Math.max(0, Math.min(f.nz, (z + WORLD_SIZE / 2) / f.spacing));
    const i0 = Math.floor(fx), k0 = Math.floor(fz);
    const i1 = Math.min(f.nx, i0 + 1), k1 = Math.min(f.nz, k0 + 1);
    const tx = fx - i0, tz = fz - k0;
    const h00 = f.heights[k0 * w + i0]!;
    const h10 = f.heights[k0 * w + i1]!;
    const h01 = f.heights[k1 * w + i0]!;
    const h11 = f.heights[k1 * w + i1]!;
    return (h00 * (1 - tx) + h10 * tx) * (1 - tz) + (h01 * (1 - tx) + h11 * tx) * tz;
  }

  /** One Chaikin corner-cutting pass over an (x, _, z) triple list. */
  private static chaikin(src: number[], dst: number[]): void {
    dst.length = 0;
    dst.push(src[0]!, 0, src[2]!);
    for (let n = 0; n + 5 < src.length; n += 3) {
      const x0 = src[n]!, z0 = src[n + 2]!;
      const x1 = src[n + 3]!, z1 = src[n + 5]!;
      dst.push(x0 * 0.75 + x1 * 0.25, 0, z0 * 0.75 + z1 * 0.25);
      dst.push(x0 * 0.25 + x1 * 0.75, 0, z0 * 0.25 + z1 * 0.75);
    }
    dst.push(src[src.length - 3]!, 0, src[src.length - 1]!);
  }

  /** Even arc-length resample of the smoothed path into the ease target. */
  private resample(src: number[]): void {
    const out = this.routeTo;
    const count = src.length / 3;

    let total = 0;
    for (let n = 0; n + 5 < src.length; n += 3) {
      total += Math.hypot(src[n + 3]! - src[n]!, src[n + 5]! - src[n + 2]!);
    }
    if (total <= 0) {
      for (let p = 0; p < ROUTE_POINTS; p++) {
        out[p * 3] = src[0]!;
        out[p * 3 + 2] = src[2]!;
      }
      return;
    }

    const step = total / (ROUTE_POINTS - 1);
    let seg = 0;
    let segStart = 0;
    let segLen = Math.hypot(src[3]! - src[0]!, src[5]! - src[2]!);

    for (let p = 0; p < ROUTE_POINTS; p++) {
      const target = Math.min(total, p * step);
      while (segStart + segLen < target && seg < count - 2) {
        segStart += segLen;
        seg++;
        segLen = Math.hypot(
          src[(seg + 1) * 3]! - src[seg * 3]!,
          src[(seg + 1) * 3 + 2]! - src[seg * 3 + 2]!,
        );
      }
      const t = segLen > 0 ? Math.min(1, (target - segStart) / segLen) : 0;
      const x = src[seg * 3]! + t * (src[(seg + 1) * 3]! - src[seg * 3]!);
      const z = src[seg * 3 + 2]! + t * (src[(seg + 1) * 3 + 2]! - src[seg * 3 + 2]!);
      out[p * 3] = x;
      out[p * 3 + 1] = this.heightAt(x, z) + this.variant.route.lift;
      out[p * 3 + 2] = z;
    }
  }

  /**
   * Write the resampled polyline into the line's interleaved segment buffer.
   * Layout matches LineGeometry.setPositions: six floats per segment, start
   * triple then end triple.
   */
  private uploadRoute(): void {
    const out = this.resampled;
    const arr = this.routeSegments.array as Float32Array;
    for (let s = 0; s < ROUTE_POINTS - 1; s++) {
      const o = s * 6;
      const a = s * 3;
      arr[o] = out[a]!; arr[o + 1] = out[a + 1]!; arr[o + 2] = out[a + 2]!;
      arr[o + 3] = out[a + 3]!; arr[o + 4] = out[a + 4]!; arr[o + 5] = out[a + 5]!;
    }
    this.routeSegments.needsUpdate = true;

    this.startDot.position.set(out[0]!, out[1]!, out[2]!);
    this.startRing.position.copy(this.startDot.position);
    this.startRing.quaternion.copy(this.camera.quaternion);
    this.goalRing.quaternion.copy(this.camera.quaternion);
    const last = (ROUTE_POINTS - 1) * 3;
    this.goalDot.position.set(out[last]!, out[last + 1]!, out[last + 2]!);
    this.goalRing.position.copy(this.goalDot.position);
  }

  // ------------------------------------------------------------ dot dragging

  /** Canvas-space position of a world point, in CSS pixels. */
  private projectToScreen(p: THREE.Vector3, out: THREE.Vector3): THREE.Vector3 {
    out.copy(p).project(this.camera);
    out.x = ((out.x + 1) / 2) * this.width;
    out.y = ((-out.y + 1) / 2) * this.height;
    return out;
  }

  /** Screen-space distance from a point to one named dot. */
  private distToDot(which: 'start' | 'goal', px: number, py: number): number {
    const dot = which === 'start' ? this.startDot : this.goalDot;
    const p = this.projectToScreen(dot.position, this.projScratch);
    return Math.hypot(p.x - px, p.y - py);
  }

  /** Nearest endpoint dot to a canvas-space point, and how far off it is. */
  private nearestDot(px: number, py: number): { which: 'start' | 'goal'; dist: number } {
    const ds = this.distToDot('start', px, py);
    const dg = this.distToDot('goal', px, py);
    return ds <= dg ? { which: 'start', dist: ds } : { which: 'goal', dist: dg };
  }

  /** Which endpoint dot, if any, is under a canvas-space point. */
  dotAt(px: number, py: number, radiusPx = DOT_HIT_RADIUS_PX): 'start' | 'goal' | null {
    const near = this.nearestDot(px, py);
    return near.dist <= radiusPx ? near.which : null;
  }

  get isDraggingDot(): boolean {
    return this.dragging !== null;
  }

  beginDotDrag(which: 'start' | 'goal'): void {
    this.dragging = which;
    // Do not raise ground under a dot the user is placing.
    this.brushTargetAmount = 0;
  }

  endDotDrag(): void {
    this.dragging = null;
  }

  /** Put an endpoint at a canvas-space point. Shared by dragging and tapping. */
  private moveEndpoint(which: 'start' | 'goal', px: number, py: number): void {
    const ground = this.groundAt(px, py);
    if (!ground) return;

    // Keep endpoints off the chamfered rim, where the closed field gives the
    // outer columns no surface and so no sane height to plan from.
    const u = Math.min(1 - ENDPOINT_MARGIN, Math.max(ENDPOINT_MARGIN, (ground.x + WORLD_SIZE / 2) / WORLD_SIZE));
    const v = Math.min(1 - ENDPOINT_MARGIN, Math.max(ENDPOINT_MARGIN, (ground.z + WORLD_SIZE / 2) / WORLD_SIZE));

    const target = which === 'start' ? this.startUV : this.goalUV;
    if (Math.abs(target.u - u) < 1e-4 && Math.abs(target.v - v) < 1e-4) return;
    target.u = u;
    target.v = v;
    this.needsPlan = true;
  }

  /**
   * Move the dragged endpoint. Replanning happens in the same frame via
   * needsPlan, so the route follows the dot rather than snapping when released.
   */
  dragDotTo(px: number, py: number): void {
    if (!this.dragging) return;
    this.moveEndpoint(this.dragging, px, py);
  }

  /** Which endpoint is armed for placing, if any. */
  /** Which endpoint a ground tap will move. */
  get activeEndpoint(): 'start' | 'goal' {
    return this.active;
  }

  /**
   * Turn on the touch affordances: larger dots, and rings that say which
   * endpoint is active. A fine pointer gets neither.
   */
  setCoarsePointer(on: boolean): void {
    this.coarse = on;
    const scale = on ? TOUCH_DOT_SCALE : 1;
    this.startDot.scale.setScalar(scale);
    this.goalDot.scale.setScalar(scale);
    this.startRing.visible = on;
    this.goalRing.visible = on;
    this.applyActiveMarking();
  }

  private applyActiveMarking(): void {
    const startActive = this.active === 'start';
    this.startRing.material = startActive ? this.activeRingMaterial : this.idleRingMaterial;
    this.goalRing.material = startActive ? this.idleRingMaterial : this.activeRingMaterial;
    // The active ring also sits wider, so the two read apart at a glance and
    // not only by brightness.
    this.startRing.scale.setScalar(startActive ? ACTIVE_RING_SCALE : 1);
    this.goalRing.scale.setScalar(startActive ? 1 : ACTIVE_RING_SCALE);
  }

  private setActive(which: 'start' | 'goal'): void {
    if (this.active === which) return;
    this.active = which;
    this.applyActiveMarking();
  }

  /**
   * Touch interaction. Tapping a dot makes that endpoint active and moves
   * nothing; tapping the ground moves whichever endpoint is active. The choice
   * is sticky, so several taps in a row keep moving the same endpoint until
   * the other dot is tapped.
   *
   * Both dots are hit-tested by name rather than by proximity, so the goal is
   * always the same dot the desktop drag calls the goal.
   */
  tap(px: number, py: number): 'activated' | 'placed' {
    const ds = this.distToDot('start', px, py);
    const dg = this.distToDot('goal', px, py);
    const hitStart = ds <= TOUCH_HIT_RADIUS_PX;
    const hitGoal = dg <= TOUCH_HIT_RADIUS_PX;

    if (hitStart || hitGoal) {
      // Only when both targets overlap does distance decide between them.
      this.setActive(hitStart && (!hitGoal || ds <= dg) ? 'start' : 'goal');
      return 'activated';
    }

    this.moveEndpoint(this.active, px, py);
    return 'placed';
  }

  private replan(): void {
    const f = this.field;
    const start = this.planner.nodeAtNormalised(this.startUV.u, this.startUV.v);
    const goal = this.planner.nodeAtNormalised(this.goalUV.u, this.goalUV.v);
    const res = this.planner.plan(start, goal, this.slopeCost);
    this.stats.planMs = res.ms;
    this.stats.expanded = res.expanded;
    if (res.length < 2) return;

    const w = f.nx + 1;
    const pts = this.rawPath;
    pts.length = 0;
    for (let n = 0; n < res.length; n++) {
      const node = res.path[n]!;
      const i = node % w;
      const k = (node - i) / w;
      pts.push(f.worldX(i), 0, f.worldZ(k));
    }

    // A* on an 8-connected grid produces a staircase. Two corner-cutting
    // passes turn it into something that reads as a planned route; heights are
    // resampled afterwards so the smoothed line still follows the ground.
    let src = pts;
    for (let pass = 0; pass < CHAIKIN_PASSES; pass++) {
      const dst = src === this.smoothA ? this.smoothB : this.smoothA;
      TerrainView.chaikin(src, dst);
      src = dst;
    }

    // resample() writes the new shape into routeTo, never straight to the
    // displayed line, so the ease is the only thing that ever moves it.
    this.resample(src);

    // A drag is direct manipulation: the line has to track the cursor exactly,
    // so it is the one case that skips the ease. Same for the very first plan,
    // which has nothing to travel from.
    if (this.dragging !== null || !this.hasRoute) {
      this.resampled.set(this.routeTo);
      this.routeEaseStart = -1;
      this.hasRoute = true;
      this.uploadRoute();
      return;
    }

    // Start from wherever the line is right now, which mid-ease is a partly
    // travelled shape. That keeps a replan during an ease continuous instead
    // of snapping back to the shape the ease began from.
    this.routeFrom.set(this.resampled);
    this.routeEaseStart = performance.now();
  }

  /** Advance the route ease. Called once per frame, after any replan. */
  private advanceRouteEase(nowMs: number): void {
    if (this.routeEaseStart < 0) return;
    const t = Math.min(1, (nowMs - this.routeEaseStart) / ROUTE_EASE_MS);
    const k = t * t * (3 - 2 * t);
    const from = this.routeFrom, to = this.routeTo, out = this.resampled;
    for (let i = 0; i < out.length; i++) out[i] = from[i]! + (to[i]! - from[i]!) * k;
    if (t >= 1) this.routeEaseStart = -1;
    this.uploadRoute();
  }

  // ----------------------------------------------------------------- inputs

  setControls(c: Controls): void {
    const f = this.field;

    if (c.gridRes !== f.params.gridRes) {
      f.params.noiseScale = c.noiseScale;
      f.params.octaves = c.octaves;
      this.slopeCost = c.slopeCost;
      f.resize(c.gridRes);
      this.planner.resize();
      this.mesher = new Mesher(f);
      this.buildChunkMeshes();
      this.rebuildAll();
      this.needsPlan = true;
      return;
    }

    if (c.noiseScale !== f.params.noiseScale || c.octaves !== f.params.octaves) {
      f.params.noiseScale = c.noiseScale;
      f.params.octaves = c.octaves;
      f.evaluateAll();
      this.needsPlan = true;
    }
    if (c.slopeCost !== this.slopeCost) {
      this.slopeCost = c.slopeCost;
      this.needsPlan = true;
    }
  }

  /**
   * Scroll progress through the page, 0..1. Records a target only; the field
   * is not touched until tick() commits a state. Static under reduced motion.
   */
  setScroll(t: number): void {
    if (this.reducedMotion) return;
    this.scrollOffset = t * WORLD_SIZE * 1.6;
  }

  /**
   * Pointer in CSS pixels relative to the canvas, or null when it leaves.
   * Records a target only; tick() is the one place the field changes.
   */
  setPointer(px: number | null, py: number | null): void {
    if (this.reducedMotion) return;

    if (px === null || py === null) {
      this.brushTargetAmount = 0;
      return;
    }

    // Approaching a dot raises terrain under it, so the dot rides up and the
    // grab lands on empty air. Fading the cursor's influence out as it nears a
    // dot fixes that; doing it as a ramp rather than a cutoff keeps the ground
    // from dropping in one step. Checked here, not in the caller, so nothing
    // can bypass it.
    const influence = smoothstep(BRUSH_FADE_NEAR_PX, BRUSH_FADE_FAR_PX, this.nearestDot(px, py).dist);
    if (influence <= 0) {
      // Leave the mound where it is and let it decay in place rather than
      // dragging a collapsing lump along with the cursor.
      this.brushTargetAmount = 0;
      return;
    }

    const ground = this.groundAt(px, py);
    if (!ground) return;

    if (this.dampedBrushAmount <= 0) {
      // Entering fresh: start the mound under the cursor rather than sweeping
      // it in from wherever the pointer left the canvas last time.
      this.dampedBrushX = ground.x;
      this.dampedBrushZ = ground.z;
    }
    this.brushTargetAmount = influence;
    this.brushTargetX = ground.x;
    this.brushTargetZ = ground.z;
  }

  /**
   * Project a canvas-space point onto the ground plane. A flat plane at the
   * mean terrain height is enough for a vertical brush column and for dot
   * dragging, and avoids raycasting tens of thousands of triangles per frame.
   */
  private groundAt(px: number, py: number): { x: number; z: number } | null {
    this.pointerNdc.set((px / this.width) * 2 - 1, -(py / this.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointerNdc, this.camera);
    if (!this.raycaster.ray.intersectPlane(this.pointerPlane, this.hitPoint)) return null;
    return { x: this.hitPoint.x, z: this.hitPoint.z };
  }

  /**
   * Advance the damped state and, when it has drifted far enough to matter,
   * commit it to the field. Everything a commit touches is re-evaluated before
   * this returns, so the caller can extract the whole surface from one state.
   */
  private stepFieldState(nowMs: number): void {
    const f = this.field;
    const dt = this.lastTickMs === 0 ? 16 : Math.min(64, nowMs - this.lastTickMs);
    this.lastTickMs = nowMs;
    if (this.reducedMotion) return;

    // The scene should read as live before anyone scrolls, so the noise offset
    // creeps on its own. It joins the scroll contribution to form one target
    // and goes through the same damping, so there is still exactly one field
    // state and no second animation path to keep in sync.
    this.idleOffset += (IDLE_DRIFT_PER_SEC * dt) / 1000;
    this.offsetTarget = this.scrollOffset + this.idleOffset;

    const k = 1 - Math.exp(-dt / DAMP_TAU_MS);
    this.dampedOffset += (this.offsetTarget - this.dampedOffset) * k;
    this.dampedBrushX += (this.brushTargetX - this.dampedBrushX) * k;
    this.dampedBrushZ += (this.brushTargetZ - this.dampedBrushZ) * k;
    this.dampedBrushAmount += (this.brushTargetAmount - this.dampedBrushAmount) * k;
    if (this.dampedBrushAmount < 0.004) this.dampedBrushAmount = 0;

    const b = f.brush;
    const brushOn = this.dampedBrushAmount > 0;
    const offsetMoved = Math.abs(this.dampedOffset - f.params.offset) > OFFSET_DEADBAND;
    const brushMoved =
      brushOn !== b.active ||
      (brushOn &&
        (Math.hypot(this.dampedBrushX - b.x, this.dampedBrushZ - b.z) > BRUSH_MOVE_DEADBAND ||
          Math.abs(this.dampedBrushAmount * BRUSH_STRENGTH - b.strength) / BRUSH_STRENGTH >
            BRUSH_AMOUNT_DEADBAND));

    if (!offsetMoved && !brushMoved) return;

    // The brush is part of the same state, so it is written before either kind
    // of re-evaluation, never in between.
    const before = b.active ? f.brushRegion() : null;
    b.active = brushOn;
    b.x = this.dampedBrushX;
    b.z = this.dampedBrushZ;
    b.strength = this.dampedBrushAmount * BRUSH_STRENGTH;

    const t0 = performance.now();
    if (offsetMoved) {
      f.params.offset = this.dampedOffset;
      f.evaluateAll();
    } else {
      const after = f.brushRegion();
      const i0 = before ? Math.min(before[0], after[0]) : after[0];
      const k0 = before ? Math.min(before[2], after[2]) : after[2];
      const i1 = before ? Math.max(before[3], after[3]) : after[3];
      const k1 = before ? Math.max(before[5], after[5]) : after[5];
      f.evaluateRegion(i0, 0, k0, i1, f.ny, k1);
    }
    this.stats.fieldMs = performance.now() - t0;
    this.needsPlan = true;
  }

  /**
   * Shrink or grow the subject in frame. 0.85 renders the slab 15% smaller,
   * leaving margin on every side without touching the canvas geometry.
   */
  setFramingScale(scale: number): void {
    this.framingScale = Math.max(0.2, scale);
    this.resize(this.width, this.height, this.dpr);
  }

  resize(width: number, height: number, dpr: number): void {
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);
    this.dpr = dpr;
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(this.width, this.height, false);
    const aspect = this.width / this.height;
    this.camera.aspect = aspect;

    // Dolly straight back along the view axis so the framing is preserved
    // rather than re-aimed; the composition stays the one that was chosen.
    // framingScale below 1 pulls back further still, which is how a caller asks
    // for a smaller subject without narrowing the canvas to get it.
    const fit = aspect < FRAMED_ASPECT ? Math.min(2, FRAMED_ASPECT / aspect) : 1;
    const dolly = fit / this.framingScale;
    this.camera.position
      .copy(this.baseEye)
      .sub(this.lookAt)
      .multiplyScalar(dolly)
      .add(this.lookAt);
    this.camera.lookAt(this.lookAt);
    this.camera.updateProjectionMatrix();

    // Fog is measured from the camera, so it has to travel with it. Without
    // this the dolly pushes the whole slab toward the far plane and drains it
    // to the background colour.
    const fog = this.scene.fog as THREE.Fog;
    fog.near = this.variant.fog.near * dolly;
    fog.far = this.variant.fog.far * dolly;
    this.routeMaterial.resolution.set(this.width * dpr, this.height * dpr);
  }

  /**
   * One frame: advance and commit the damped state, extract every chunk that
   * changed, replan, draw. The order is the guarantee - the surface drawn here
   * is always the surface the field describes right now.
   */
  tick(nowMs: number): void {
    if (this.disposed) return;
    this.stepFieldState(nowMs);
    this.drainDirty();
    if (this.needsPlan) {
      this.replan();
      this.needsPlan = false;
    }
    this.advanceRouteEase(nowMs);
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this.disposed = true;
    for (const m of this.meshes) m.geometry.dispose();
    this.material.dispose();
    this.routeGeometry.dispose();
    this.routeMaterial.dispose();
    this.dotGeometry.dispose();
    this.dotMaterial.dispose();
    this.ringGeometry.dispose();
    this.activeRingMaterial.dispose();
    this.idleRingMaterial.dispose();
    this.renderer.dispose();
  }
}
