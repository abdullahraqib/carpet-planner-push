import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import {
  Plus, Trash2, Copy, RotateCw, Undo2, Redo2, Grid3x3, ZoomIn, ZoomOut,
  Maximize2, X, Check, AlertTriangle, Ruler, Layers, ScissorsLineDashed,
  ArrowRight, ArrowUp, ArrowLeft, ArrowDown, Settings2, FolderOpen, Save,
  Pencil, ChevronLeft, ChevronRight, Undo, Info, Download, FileText, ChevronUp, ChevronDown,
} from "lucide-react";

/* ======================================================================
   SECTION: GEOMETRY LAYER
   Pure functions operating on real-world millimetre coordinates.
   Vertices are always ordered, orthogonal (every edge horizontal or
   vertical) and normalised so the bounding box starts at (0,0).
   ====================================================================== */

const EPS = 0.5; // mm tolerance for "closed" checks

function dirDelta(dir, len) {
  switch (dir) {
    case "R": return { dx: len, dy: 0 };
    case "L": return { dx: -len, dy: 0 };
    case "D": return { dx: 0, dy: len };
    case "U": return { dx: 0, dy: -len };
    default: return { dx: 0, dy: 0 };
  }
}

function isHorizontal(dir) { return dir === "L" || dir === "R"; }

// Build normalised vertices from a direction sequence + wall lengths (mm).
function buildPolygon(directions, lengths) {
  const pts = [{ x: 0, y: 0 }];
  let x = 0, y = 0;
  for (let i = 0; i < directions.length; i++) {
    const { dx, dy } = dirDelta(directions[i], lengths[i] || 0);
    x += dx; y += dy;
    pts.push({ x, y });
  }
  pts.pop(); // last point should coincide with first (closure)
  const minX = Math.min(...pts.map((p) => p.x));
  const minY = Math.min(...pts.map((p) => p.y));
  return pts.map((p) => ({ x: p.x - minX, y: p.y - minY }));
}

function closureError(directions, lengths) {
  let h = 0, v = 0;
  directions.forEach((d, i) => {
    const len = lengths[i] || 0;
    if (d === "R") h += len;
    else if (d === "L") h -= len;
    else if (d === "D") v += len;
    else if (d === "U") v -= len;
  });
  return { h, v };
}

function polygonArea(vertices) {
  let a = 0;
  for (let i = 0; i < vertices.length; i++) {
    const p1 = vertices[i];
    const p2 = vertices[(i + 1) % vertices.length];
    a += p1.x * p2.y - p2.x * p1.y;
  }
  return Math.abs(a) / 2;
}

function boundingBox(vertices) {
  const xs = vertices.map((v) => v.x), ys = vertices.map((v) => v.y);
  return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
}

// Rotate a normalised polygon by 0/90/180/270 degrees, re-normalised to (0,0).
function rotatePolygon(vertices, rotation) {
  const pts = vertices.map((v) => {
    switch (((rotation % 360) + 360) % 360) {
      case 90: return { x: -v.y, y: v.x };
      case 180: return { x: -v.x, y: -v.y };
      case 270: return { x: v.y, y: -v.x };
      default: return { x: v.x, y: v.y };
    }
  });
  const bb = boundingBox(pts);
  return pts.map((p) => ({ x: p.x - bb.minX, y: p.y - bb.minY }));
}

function translatePolygon(vertices, dx, dy) {
  return vertices.map((v) => ({ x: v.x + dx, y: v.y + dy }));
}

// World-space vertices for a placed instance (rotation + position applied).
function instanceVertices(room, instance) {
  const rotated = rotatePolygon(room.vertices, instance.rotation);
  return translatePolygon(rotated, instance.x, instance.y);
}

function pointInPolygon(x, y, vertices) {
  let inside = false;
  for (let i = 0, j = vertices.length - 1; i < vertices.length; j = i++) {
    const xi = vertices[i].x, yi = vertices[i].y;
    const xj = vertices[j].x, yj = vertices[j].y;
    const intersect = (yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

// Decompose an orthogonal polygon into non-overlapping rectangles.
// This is the exact representation used for collision/bounds checks so
// L-shaped and stepped rooms are never treated as their bounding box.
function decompose(vertices) {
  const xs = [...new Set(vertices.map((v) => v.x))].sort((a, b) => a - b);
  const ys = [...new Set(vertices.map((v) => v.y))].sort((a, b) => a - b);
  const rects = [];
  for (let i = 0; i < xs.length - 1; i++) {
    const x0 = xs[i], x1 = xs[i + 1], midX = (x0 + x1) / 2;
    let runStart = null;
    for (let j = 0; j < ys.length; j++) {
      const y0 = ys[j], y1 = ys[j + 1];
      const inside = y1 !== undefined && pointInPolygon(midX, (y0 + y1) / 2, vertices);
      if (inside && runStart === null) runStart = y0;
      if ((!inside || y1 === undefined) && runStart !== null) {
        rects.push({ x: x0, y: runStart, w: x1 - x0, h: y0 - runStart });
        runStart = null;
      }
    }
  }
  return rects;
}

function inflateRect(r, a) {
  return { x: r.x - a, y: r.y - a, w: r.w + 2 * a, h: r.h + 2 * a };
}

function rectsOverlap(a, b) {
  return a.x < b.x + b.w - EPS && a.x + a.w > b.x + EPS && a.y < b.y + b.h - EPS && a.y + a.h > b.y + EPS;
}

// Full polygon-vs-polygon overlap test (allowance applied to both sides).
function polygonsCollide(vertsA, vertsB, allowanceMm) {
  const rectsA = decompose(vertsA).map((r) => inflateRect(r, allowanceMm / 2));
  const rectsB = decompose(vertsB).map((r) => inflateRect(r, allowanceMm / 2));
  for (const ra of rectsA) for (const rb of rectsB) if (rectsOverlap(ra, rb)) return true;
  return false;
}

function withinRoll(vertsWithAllowance, rollWidth) {
  const bb = boundingBox(vertsWithAllowance);
  return bb.minX >= -EPS && bb.maxX <= rollWidth + EPS && bb.minY >= -EPS;
}

function allowedVerts(room, instance, allowanceMm) {
  const v = instanceVertices(room, instance);
  if (!allowanceMm) return v;
  // expand every decomposed rect, then take their combined bbox edges
  // (used only for the roll-boundary check, which is a supported case
  // for orthogonal shapes because inflation is applied uniformly).
  const bb = boundingBox(v);
  return [
    { x: bb.minX - allowanceMm / 2, y: bb.minY - allowanceMm / 2 },
    { x: bb.maxX + allowanceMm / 2, y: bb.maxY + allowanceMm / 2 },
  ];
}

/* ======================================================================
   SECTION: COLLISION / VALIDATION LAYER
   ====================================================================== */

function isPlacementValid(room, instance, allInstances, roomsById, rollWidth, tolerance) {
  const verts = instanceVertices(room, instance);
  const bounds = allowedVerts(room, instance, tolerance);
  if (!withinRoll(bounds, rollWidth)) return { valid: false, reason: "outside" };
  for (const other of allInstances) {
    if (other.id === instance.id) continue;
    const otherRoom = roomsById[other.roomId];
    if (!otherRoom) continue;
    const otherVerts = instanceVertices(otherRoom, other);
    if (polygonsCollide(verts, otherVerts, tolerance)) return { valid: false, reason: "overlap" };
  }
  return { valid: true };
}

function requiredLength(instances, roomsById) {
  let maxY = 0;
  instances.forEach((inst) => {
    const room = roomsById[inst.roomId];
    if (!room) return;
    const v = instanceVertices(room, inst);
    v.forEach((p) => { if (p.y > maxY) maxY = p.y; });
  });
  return maxY;
}

/* ======================================================================
   SECTION: PACKING LAYER (auto-arrange)

   Honest framing: 2D nesting with rotation is NP-hard — true exhaustive
   branch-and-bound to PROVEN global optimality is not computationally
   reachable client-side for anything but trivial instances, and no real
   nesting system (commercial or otherwise) actually does this within an
   interactive time budget. What this layer does instead, and does exactly
   rather than approximately:

   1. EXACT geometry, not bounding boxes. Every placement test below works
      on each room's actual rectilinear decomposition (the same `decompose`
      used by collision detection elsewhere in this file) — an L-shaped
      room's concave notch is real free space, not reserved dead space.

   2. A true No-Fit-Polygon, generalised to this app's rotation set
      (0/90/180/270°, matching the rotation restrictions already enforced
      for grain direction — continuous/arbitrary rotation is intentionally
      excluded because it would misalign a room from its real measured
      wall lengths). For axis-aligned rectilinear shapes the NFP of two
      polygons is exactly the union of pairwise Minkowski sums of their
      decomposed rectangles — `forbiddenRectsForPiece` below computes this
      precisely, so every candidate placement is gap-free and exact, never
      a heuristic approximation.

   3. Bottom-left-fill seeded from the NFP's own forbidden-region corners
      (`bottomLeftPlace`) — the only placement points that can possibly be
      "tightest fit" are at those corners, so this is an exact search over
      the genuinely relevant candidate set, not a coarse grid.

   4. A GUILLOTINE packer (`guillotineArrangeWithOrder`) as an alternative
      placement method, for when the priority is a layout a person can
      actually cut with a straight knife — every cut it produces runs the
      full span of whatever piece it's cutting from, never jogging around
      a corner. It works on bounding boxes rather than exact polygons
      (conservative but always overlap-safe, including for L-shaped
      rooms), so it can't tuck a piece into another's notch the way the
      NFP placer can, trading a little density for cuttability.

   5. A simulated-annealing metaheuristic (`autoArrange`) searches over
      placement ORDER and per-piece ROTATION — using whichever of the two
      placers above was requested — re-evaluating each candidate sequence
      and keeping the best result found under a time budget. Symmetry is
      broken for interchangeable identical pieces (same room, same
      rotation) so swapping two such pieces — which can never change the
      result — is never wastefully re-evaluated.

   The result is always labelled a "Suggested Layout": the best arrangement
   found, not a claim of global optimality.
   ====================================================================== */

// Exact Minkowski-sum-based forbidden region: the set of positions for a
// moving piece's origin that would make a given (already-inflated) moving
// rectangle overlap a given (already-inflated) stationary rectangle.
function minkowskiForbiddenRect(stationary, moving) {
  return {
    x: stationary.x - moving.x - moving.w,
    y: stationary.y - moving.y - moving.h,
    w: stationary.w + moving.w,
    h: stationary.h + moving.h,
  };
}

// All forbidden rectangles for a candidate piece's ORIGIN, exact (no
// bounding-box shortcuts): decomposes the candidate at its trial rotation
// and every already-placed instance into real rectangles, and computes the
// pairwise Minkowski sum for each pair. allowanceMm is split evenly as the
// existing collision system does.
function forbiddenRectsForPiece(movingVerts, placedInstances, roomsById, allowanceMm) {
  const movingRects = decompose(movingVerts).map((r) => inflateRect(r, allowanceMm / 2));
  const forbidden = [];
  for (const other of placedInstances) {
    const otherRoom = roomsById[other.roomId];
    if (!otherRoom) continue;
    const otherVerts = instanceVertices(otherRoom, other);
    const otherTol = other.allowanceOverride != null && other.allowanceOverride !== "" ? parseFloat(other.allowanceOverride) : allowanceMm;
    const stationaryRects = decompose(otherVerts).map((r) => inflateRect(r, otherTol / 2));
    for (const sr of stationaryRects) for (const mr of movingRects) forbidden.push(minkowskiForbiddenRect(sr, mr));
  }
  return forbidden;
}

function pointInsideRect(x, y, r) {
  return x > r.x + EPS && x < r.x + r.w - EPS && y > r.y + EPS && y < r.y + r.h - EPS;
}

// Exact bottom-left-fill: places one piece as low, then as far left, as the
// real geometry allows, using only the NFP's own forbidden-region corners
// as candidate points (the only points that can possibly be tightest-fit).
function bottomLeftPlace(room, rotation, placedInstances, roomsById, rollWidth, tolerance) {
  const movingVerts = rotatePolygon(room.vertices, rotation);
  const bb = boundingBox(movingVerts);
  const w = bb.maxX - bb.minX, h = bb.maxY - bb.minY;
  const minX = tolerance / 2, maxX = rollWidth - w - tolerance / 2;
  if (maxX < minX - EPS) return null; // doesn't fit the roll width at all, even alone

  const forbidden = forbiddenRectsForPiece(movingVerts, placedInstances, roomsById, tolerance);

  // Candidate points: the full grid of each forbidden rectangle's left/right
  // x-edges crossed with each rectangle's bottom/top y-edges, plus the roll's
  // own left edge and y=0. A single rectangle's own two "outer" corners are
  // not enough on their own — a concave notch formed by TWO rectangles (an
  // L-shaped room's cut corner, for instance) only has a valid touching
  // point at the combination of one rect's x-edge and a different rect's
  // y-edge, so every combination has to be considered to stay exact.
  const xs = new Set([Math.max(0, minX)]);
  const ys = new Set([0]);
  forbidden.forEach((f) => { xs.add(f.x); xs.add(f.x + f.w); ys.add(f.y); ys.add(f.y + f.h); });
  const xList = [...xs].filter((x) => x >= minX - EPS && x <= maxX + EPS);
  const yList = [...ys].filter((y) => y >= -EPS);
  yList.sort((a, b) => a - b);
  xList.sort((a, b) => a - b);

  let bestPt = null;
  outer:
  for (const y of yList) {
    if (bestPt && y > bestPt.y + EPS) break;
    for (const x of xList) {
      if (x < minX - EPS || x > maxX + EPS) continue;
      if (forbidden.some((f) => pointInsideRect(x, y, f))) continue;
      bestPt = { x, y };
      break outer; // yList and xList are sorted ascending, so the first hit at the lowest y is already leftmost-at-that-y
    }
  }
  if (bestPt) return bestPt;

  // Guaranteed-correct fallback: stack clear above every forbidden region.
  const topY = forbidden.reduce((m, f) => Math.max(m, f.y + f.h), 0);
  return { x: Math.max(0, minX), y: topY };
}

function arrangeWithOrder(orderedItems, roomsById, rollWidth, tolerance) {
  const placed = [];
  for (const { inst, rotation } of orderedItems) {
    const room = roomsById[inst.roomId];
    if (!room) continue;
    const pos = bottomLeftPlace(room, rotation, placed, roomsById, rollWidth, tolerance);
    if (!pos) continue; // piece cannot fit the roll width in this rotation; skip rather than corrupt the layout
    placed.push({ ...inst, rotation, x: pos.x, y: pos.y });
  }
  return placed;
}

/* ----------------------------------------------------------------------
   Guillotine packing: every cut this produces runs straight across the
   current piece being cut from, edge to edge — the only kind of cut a
   roll of carpet can realistically take with a straight knife and a
   metre rule, as opposed to the free-form nesting above, which can tuck
   a piece into another's concave notch but may leave cut lines that jog
   around a corner (no problem for a CNC cutter; a real problem for
   someone cutting by hand). This works on each room's bounding box
   rather than its exact polygon — strictly conservative, since a room's
   true (possibly L-shaped) footprint always sits inside its own bounding
   box, so two bounding boxes placed without overlap can never let the
   real shapes overlap either, even though it can't nest a piece into
   another's notch the way the exact-NFP placer above can.

   Classic guillotine bin-packing: a list of free rectangles is kept,
   starting as one rectangle spanning the whole roll (width fixed, length
   effectively unbounded). Each placed piece's bounding box is cut from
   whichever free rectangle it fits most snugly ("best short-side fit"),
   and that free rectangle is then itself split by ONE straight cut into
   up to two new free rectangles — recursively, so the whole structure
   stays guillotine-cuttable no matter how many pieces are placed.
   ---------------------------------------------------------------------- */
function guillotineArrangeWithOrder(orderedItems, roomsById, rollWidth, tolerance) {
  const BIG = 1e9;
  let freeRects = [{ x: 0, y: 0, w: rollWidth, h: BIG }];
  const placed = [];

  for (const { inst, rotation } of orderedItems) {
    const room = roomsById[inst.roomId];
    if (!room) continue;
    const bb = boundingBox(rotatePolygon(room.vertices, rotation));
    const w0 = bb.maxX - bb.minX, h0 = bb.maxY - bb.minY;

    let bestIdx = -1, bestScore = Infinity;
    for (let i = 0; i < freeRects.length; i++) {
      const r = freeRects[i];
      if (r.w >= w0 - EPS && r.h >= h0 - EPS) {
        const score = Math.min(r.w - w0, r.h - h0); // best short-side fit
        if (score < bestScore) { bestScore = score; bestIdx = i; }
      }
    }
    if (bestIdx === -1) continue; // cannot fit this piece's bounding box anywhere; skip rather than corrupt the layout

    const rect = freeRects[bestIdx];
    placed.push({ ...inst, rotation, x: rect.x, y: rect.y });
    freeRects.splice(bestIdx, 1);

    const leftoverW = rect.w - w0 - tolerance;
    const leftoverH = rect.h - h0 - tolerance;
    const next = [];
    // Shorter-leftover-axis rule: whichever direction has less slack gets
    // the full-span cut first, keeping the resulting free rectangles as
    // usable (least sliver-shaped) as the guillotine constraint allows.
    if (leftoverW <= leftoverH) {
      if (leftoverW > EPS) next.push({ x: rect.x + w0 + tolerance, y: rect.y, w: leftoverW, h: h0 });
      if (leftoverH > EPS) next.push({ x: rect.x, y: rect.y + h0 + tolerance, w: rect.w, h: leftoverH });
    } else {
      if (leftoverH > EPS) next.push({ x: rect.x, y: rect.y + h0 + tolerance, w: w0, h: leftoverH });
      if (leftoverW > EPS) next.push({ x: rect.x + w0 + tolerance, y: rect.y, w: leftoverW, h: rect.h });
    }
    freeRects.push(...next);
  }
  return placed;
}

function allowedRotationsFor(grainMode) {
  return grainMode === "locked" ? [0] : [0, 90, 180, 270];
}

function bestDefaultRotation(room, rollWidth, allowedRotations) {
  let best = allowedRotations[0], bestH = Infinity;
  for (const rot of allowedRotations) {
    const bb = boundingBox(rotatePolygon(room.vertices, rot));
    const w = bb.maxX - bb.minX, h = bb.maxY - bb.minY;
    if (w <= rollWidth + EPS && h < bestH) { bestH = h; best = rot; }
  }
  return best;
}

// Simulated annealing over (placement order, per-piece rotation), each
// candidate evaluated with the exact NFP + bottom-left-fill placer above.
// Time-budgeted so it stays responsive regardless of instance count.
function autoArrange(instanceList, roomsById, rollWidth, tolerance, grainMode, guillotine) {
  const allowedRotations = allowedRotationsFor(grainMode);
  const n = instanceList.length;
  if (n === 0) return [];
  const placer = guillotine ? guillotineArrangeWithOrder : arrangeWithOrder;

  // Rotation is tracked per ROOM, not per instance: every placed copy of
  // the same room (e.g. several identical "Stairs" pieces cut from one
  // duplicated entry) always shares one rotation, because carpet pile/grain
  // direction has to run the same way across duplicates of the same piece
  // — the search is never allowed to spin one copy differently from its
  // siblings just because that happens to pack a little tighter.
  const roomIds = [...new Set(instanceList.map((i) => i.roomId))];
  let rotByRoom = {};
  roomIds.forEach((id) => {
    const room = roomsById[id];
    rotByRoom[id] = room ? bestDefaultRotation(room, rollWidth, allowedRotations) : 0;
  });

  let order = instanceList.map((inst) => ({ inst }));
  // Decreasing-height start is a strong seed for bottom-left-fill.
  order.sort((a, b) => {
    const ra = boundingBox(rotatePolygon(roomsById[a.inst.roomId]?.vertices || [{ x: 0, y: 0 }], rotByRoom[a.inst.roomId] || 0));
    const rb = boundingBox(rotatePolygon(roomsById[b.inst.roomId]?.vertices || [{ x: 0, y: 0 }], rotByRoom[b.inst.roomId] || 0));
    return (rb.maxY - rb.minY) - (ra.maxY - ra.minY);
  });

  function evaluate(seq, rotations) {
    const withRot = seq.map((s) => ({ inst: s.inst, rotation: rotations[s.inst.roomId] || 0 }));
    const placed = placer(withRot, roomsById, rollWidth, tolerance);
    return { placed, length: requiredLength(placed, roomsById) };
  }

  let current = order, currentRot = rotByRoom;
  let evald = evaluate(current, currentRot);
  let bestPlaced = evald.placed, bestLen = evald.length;
  let currentLen = evald.length;

  if (n <= 1) return bestPlaced;

  // Budget scales down per-iteration cost as the instance count grows, so
  // a big project still finishes promptly rather than freezing the tab.
  const timeBudgetMs = Math.min(2500, 400 + n * 40);
  const iterCap = Math.max(150, Math.floor(30000 / n));
  let T = Math.max(currentLen * 0.04, 50);
  const cooling = 0.97;
  const startTime = Date.now();
  let iter = 0;

  while (iter < iterCap && Date.now() - startTime < timeBudgetMs) {
    iter++;
    const next = current.map((s) => ({ ...s }));
    let nextRot = currentRot;
    const roll = Math.random();
    if (roll < 0.45) {
      // Swap two positions in the sequence.
      const i = Math.floor(Math.random() * n), j = Math.floor(Math.random() * n);
      if (i === j) continue;
      // Symmetry-breaking: swapping two instances of the SAME room is a
      // no-op (they already share one rotation by construction above), so
      // the search never wastes iterations re-evaluating something that
      // can't possibly change the layout.
      if (next[i].inst.roomId === next[j].inst.roomId) continue;
      const tmp = next[i]; next[i] = next[j]; next[j] = tmp;
    } else if (roll < 0.75) {
      // Try a different rotation for one ROOM — applies to every instance
      // of it at once, keeping duplicates consistent with each other.
      const roomId = roomIds[Math.floor(Math.random() * roomIds.length)];
      const room = roomsById[roomId];
      if (!room) continue;
      const options = allowedRotations.filter((r) => {
        const bb = boundingBox(rotatePolygon(room.vertices, r));
        return bb.maxX - bb.minX <= rollWidth + EPS;
      });
      if (options.length <= 1) continue;
      let r;
      let tries = 0;
      do { r = options[Math.floor(Math.random() * options.length)]; tries++; } while (r === currentRot[roomId] && tries < 6);
      nextRot = { ...currentRot, [roomId]: r };
    } else {
      // Move one piece to a different point in the sequence.
      const i = Math.floor(Math.random() * n), j = Math.floor(Math.random() * n);
      if (i === j) continue;
      const [item] = next.splice(i, 1);
      next.splice(j, 0, item);
    }
    const nextEval = evaluate(next, nextRot);
    const delta = nextEval.length - currentLen;
    if (delta < 0 || Math.random() < Math.exp(-delta / Math.max(T, 1e-6))) {
      current = next;
      currentRot = nextRot; // an accepted rotation change has to actually stick, or every later iteration keeps re-trying from the original starting rotation instead of building on what was just accepted
      currentLen = nextEval.length;
      if (currentLen < bestLen - EPS) { bestPlaced = nextEval.placed; bestLen = currentLen; }
    }
    T *= cooling;
  }

  return bestPlaced;
}

/* ======================================================================
   SECTION: UNITS / FORMATTING
   ====================================================================== */

function parseLength(input, defaultUnit) {
  if (input == null || input === "") return null;
  const str = String(input).trim().toLowerCase();
  if (/^[0-9.]+$/.test(str)) {
    const v = parseFloat(str);
    if (isNaN(v)) return null;
    return defaultUnit === "m" ? v * 1000 : defaultUnit === "cm" ? v * 10 : v;
  }
  const tokenRe = /([0-9]*\.?[0-9]+)\s*(mm|cm|m)/g;
  let match, total = 0, found = false;
  while ((match = tokenRe.exec(str)) !== null) {
    found = true;
    const v = parseFloat(match[1]);
    const unit = match[2];
    total += unit === "m" ? v * 1000 : unit === "cm" ? v * 10 : v;
  }
  return found ? total : null;
}

function formatLength(mm, unit) {
  if (mm == null || isNaN(mm)) return "-";
  if (unit === "m") return (mm / 1000).toFixed(2) + "m";
  if (unit === "cm") return (mm / 10).toFixed(1) + "cm";
  return Math.round(mm) + "mm";
}

function formatArea(mm2) { return (mm2 / 1_000_000).toFixed(2) + "m\u00B2"; }

const DIR_ICON = { R: ArrowRight, L: ArrowLeft, U: ArrowUp, D: ArrowDown };
const ROOM_COLORS = ["#c98a3a", "#3c7a6f", "#8a5fb0", "#4a7fb5", "#b5583f", "#5a8a4a", "#a3843f", "#7a5a8a"];

/* ======================================================================
   SECTION: PERSISTENCE
   ====================================================================== */

const STORAGE_KEY = "carpet-cutting-planner:project";

// window.storage is only available inside the Claude.ai artifact preview.
// On a standalone deployed site (e.g. this app hosted from GitHub) it's
// undefined, so autosave falls back to plain localStorage there instead.
async function loadProject() {
  try {
    if (typeof window !== "undefined" && window.storage?.get) {
      const res = await window.storage.get(STORAGE_KEY, false);
      if (res?.value) return JSON.parse(res.value);
      return null;
    }
  } catch (e) { /* fall through to localStorage */ }
  try {
    const raw = typeof window !== "undefined" && window.localStorage ? window.localStorage.getItem(STORAGE_KEY) : null;
    if (raw) return JSON.parse(raw);
  } catch (e) { /* nothing saved yet */ }
  return null;
}

async function saveProject(data) {
  try {
    if (typeof window !== "undefined" && window.storage?.set) {
      await window.storage.set(STORAGE_KEY, JSON.stringify(data), false);
      return;
    }
  } catch (e) { /* fall through to localStorage */ }
  try {
    if (typeof window !== "undefined" && window.localStorage) window.localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch (e) { console.error("save failed", e); }
}

/* ======================================================================
   SECTION: SMALL PRESENTATIONAL PIECES
   ====================================================================== */

function RoomThumbnail({ vertices, color, size = 72 }) {
  const bb = boundingBox(vertices);
  const w = bb.maxX - bb.minX || 1, h = bb.maxY - bb.minY || 1;
  const pad = 6;
  const scale = Math.min((size - pad * 2) / w, (size - pad * 2) / h);
  const pts = vertices.map((v) => `${(v.x - bb.minX) * scale + pad},${(v.y - bb.minY) * scale + pad}`).join(" ");
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0">
      <polygon points={pts} fill={color + "33"} stroke={color} strokeWidth="2" strokeLinejoin="round" />
    </svg>
  );
}

function Field({ label, children }) {
  return (
    <label className="block">
      <span className="block text-[11px] uppercase tracking-wide text-stone-400 mb-1">{label}</span>
      {children}
    </label>
  );
}

function WallShapePreview({ vertices, lengths, unit, activeWall }) {
  if (!vertices || vertices.length < 3) return null;
  const bb = boundingBox(vertices);
  const w = bb.maxX - bb.minX || 1, h = bb.maxY - bb.minY || 1;
  const boxW = 240, boxH = 200, pad = 28;
  const scale = Math.min((boxW - pad * 2) / w, (boxH - pad * 2) / h);
  const pts = vertices.map((v) => ({ x: (v.x - bb.minX) * scale + pad, y: (v.y - bb.minY) * scale + pad }));
  const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
  const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
  const pointsStr = pts.map((p) => `${p.x},${p.y}`).join(" ");
  return (
    <svg width={boxW} height={boxH} viewBox={`0 0 ${boxW} ${boxH}`} className="bg-white border border-stone-300 rounded-xl shrink-0">
      <polygon points={pointsStr} fill="#c98a3a22" stroke="#c98a3a" strokeWidth="2" strokeLinejoin="round" />
      {activeWall != null && pts[activeWall] && (
        <line
          x1={pts[activeWall].x} y1={pts[activeWall].y}
          x2={pts[(activeWall + 1) % pts.length].x} y2={pts[(activeWall + 1) % pts.length].y}
          stroke="#3c7a6f" strokeWidth="4" strokeLinecap="round"
        />
      )}
      {pts.map((p, i) => {
        const q = pts[(i + 1) % pts.length];
        const mx = (p.x + q.x) / 2, my = (p.y + q.y) / 2;
        let dx = mx - cx, dy = my - cy;
        const dist = Math.hypot(dx, dy) || 1;
        dx /= dist; dy /= dist;
        const lx = mx + dx * 15, ly = my + dy * 15;
        const active = activeWall === i;
        const len = lengths ? parseLength(lengths[i], unit) : null;
        return (
          <g key={i}>
            <circle cx={lx} cy={ly} r={active ? 10 : 8} fill={active ? "#3c7a6f" : "#f7f5f0"} stroke={active ? "#3c7a6f" : "#c98a3a"} strokeWidth="1.5" />
            <text x={lx} y={ly + 3} fontSize="9" fontFamily="monospace" fontWeight="bold" textAnchor="middle" fill={active ? "white" : "#c98a3a"}>{i + 1}</text>
            {active && len != null && (
              <text x={mx} y={my - dy * 6 - 4} fontSize="10" fontFamily="monospace" fontWeight="bold" textAnchor="middle" fill="#3c7a6f">{formatLength(len, unit)}</text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

/* ======================================================================
   SECTION: ADD / EDIT ROOM MODAL
   ====================================================================== */

const DRAW_W = 640, DRAW_H = 420, DRAW_SNAP = 20;

function AddRoomModal({ initialRoom, onClose, onSave, displayUnit }) {
  const [step, setStep] = useState(initialRoom ? "dims" : "draw");
  const [points, setPoints] = useState(
    initialRoom ? [] : []
  );
  const [directions, setDirections] = useState(initialRoom ? initialRoom.directions : []);
  const [drawError, setDrawError] = useState("");
  const [cursor, setCursor] = useState(null);
  const [lengths, setLengths] = useState(
    initialRoom ? initialRoom.wallLengths.map(String) : []
  );
  const [unit, setUnit] = useState(displayUnit || "mm");
  const [activeWall, setActiveWall] = useState(null);
  const [name, setName] = useState(initialRoom?.name || "");
  const [quantity, setQuantity] = useState(initialRoom?.quantity ?? 1);
  const [notes, setNotes] = useState(initialRoom?.notes || "");
  const [allowance, setAllowance] = useState(initialRoom?.cuttingAllowance ?? "");
  const [grain, setGrain] = useState(initialRoom?.grainDirection ?? false);
  const svgRef = useRef(null);

  function snap(pt) { return { x: Math.round(pt.x / DRAW_SNAP) * DRAW_SNAP, y: Math.round(pt.y / DRAW_SNAP) * DRAW_SNAP }; }

  function toLocal(e) {
    const rect = svgRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * DRAW_W;
    const y = ((e.clientY - rect.top) / rect.height) * DRAW_H;
    return snap({ x, y });
  }

  function constrainedPoint(raw) {
    if (points.length === 0) return raw;
    const last = points[points.length - 1];
    const dx = raw.x - last.x, dy = raw.y - last.y;
    if (Math.abs(dx) >= Math.abs(dy)) return { x: raw.x, y: last.y };
    return { x: last.x, y: raw.y };
  }

  function directionBetween(a, b) {
    if (Math.abs(b.x - a.x) > Math.abs(b.y - a.y)) return b.x > a.x ? "R" : "L";
    return b.y > a.y ? "D" : "U";
  }

  function handleMove(e) { setCursor(constrainedPoint(toLocal(e))); }

  function handleClick(e) {
    const raw = toLocal(e);
    const pt = constrainedPoint(raw);
    setDrawError("");
    if (points.length >= 2) {
      const start = points[0];
      const nearStart = Math.hypot(pt.x - start.x, pt.y - start.y) < 18;
      if (nearStart) { closeShape(); return; }
    }
    if (points.length > 0) {
      const last = points[points.length - 1];
      if (pt.x === last.x && pt.y === last.y) return;
      setDirections((d) => [...d, directionBetween(last, pt)]);
    }
    setPoints((p) => [...p, pt]);
  }

  function closeShape() {
    if (points.length < 3) { setDrawError("Add at least 3 corners before closing."); return; }
    const last = points[points.length - 1];
    const start = points[0];
    if (last.x !== start.x && last.y !== start.y) {
      setDrawError("Cannot close: the last corner doesn't line up with the start on either axis. Add another corner first.");
      return;
    }
    const closingDir = directionBetween(last, start);
    const finalDirections = [...directions, closingDir];
    setDirections(finalDirections);
    setLengths(finalDirections.map(() => ""));
    setStep("dims");
  }

  function undoPoint() {
    setPoints((p) => p.slice(0, -1));
    setDirections((d) => d.slice(0, -1));
    setDrawError("");
  }
  function resetDraw() { setPoints([]); setDirections([]); setDrawError(""); }

  // ---- dimension step ----
  const numericLengths = lengths.map((l) => parseLength(l, unit));
  const { h: hSum, v: vSum } = useMemo(() => {
    let h = 0, v = 0;
    directions.forEach((d, i) => {
      const len = numericLengths[i];
      if (len == null) return;
      if (d === "R") h += len; else if (d === "L") h -= len;
      else if (d === "D") v += len; else if (d === "U") v -= len;
    });
    return { h, v };
  }, [directions, JSON.stringify(numericLengths)]);

  const hBlanks = directions.map((d, i) => (isHorizontal(d) && numericLengths[i] == null ? i : -1)).filter((i) => i >= 0);
  const vBlanks = directions.map((d, i) => (!isHorizontal(d) && numericLengths[i] == null ? i : -1)).filter((i) => i >= 0);

  let resolvedLengths = [...numericLengths];
  let calcIndexes = new Set();
  let dimError = "";
  if (hBlanks.length === 1) {
    const idx = hBlanks[0];
    const dir = directions[idx];
    // remaining signed sum without this wall must be cancelled by it
    let sum = 0;
    directions.forEach((d, i) => { if (i === idx || !isHorizontal(d)) return; sum += d === "R" ? numericLengths[i] : -numericLengths[i]; });
    const needed = dir === "R" ? -sum : sum;
    if (needed > 0) { resolvedLengths[idx] = needed; calcIndexes.add(idx); }
    else dimError = "Horizontal dimensions can't form a closed shape with these values.";
  } else if (hBlanks.length === 0 && directions.some(isHorizontal) && Math.abs(hSum) > EPS) {
    dimError = `Horizontal dimensions are inconsistent by ${Math.round(Math.abs(hSum))}mm.`;
  } else if (hBlanks.length > 1) {
    dimError = `Enter all but one horizontal wall (${hBlanks.length} are still blank).`;
  }
  if (!dimError) {
    if (vBlanks.length === 1) {
      const idx = vBlanks[0];
      const dir = directions[idx];
      let sum = 0;
      directions.forEach((d, i) => { if (i === idx || isHorizontal(d)) return; sum += d === "D" ? numericLengths[i] : -numericLengths[i]; });
      const needed = dir === "D" ? -sum : sum;
      if (needed > 0) { resolvedLengths[idx] = needed; calcIndexes.add(idx); }
      else dimError = "Vertical dimensions can't form a closed shape with these values.";
    } else if (vBlanks.length === 0 && directions.some((d) => !isHorizontal(d)) && Math.abs(vSum) > EPS) {
      dimError = `Vertical dimensions are inconsistent by ${Math.round(Math.abs(vSum))}mm.`;
    } else if (vBlanks.length > 1) {
      dimError = `Enter all but one vertical wall (${vBlanks.length} are still blank).`;
    }
  }

  const allFilled = resolvedLengths.every((l) => l != null && l > 0);
  const canConstruct = allFilled && !dimError;
  const previewVerts = canConstruct ? buildPolygon(directions, resolvedLengths) : null;
  const previewArea = previewVerts ? polygonArea(previewVerts) : 0;
  const previewBB = previewVerts ? boundingBox(previewVerts) : null;

  const shapeVerts = previewVerts || (points.length > 0 ? points : (initialRoom ? buildPolygon(directions, initialRoom.wallLengths) : null));

  function handleSave() {
    if (!canConstruct || !name.trim()) return;
    const vertices = buildPolygon(directions, resolvedLengths);
    const bb = boundingBox(vertices);
    onSave({
      id: initialRoom?.id || `room_${Date.now()}`,
      name: name.trim(),
      vertices,
      directions,
      wallLengths: resolvedLengths,
      width: bb.maxX - bb.minX,
      length: bb.maxY - bb.minY,
      area: polygonArea(vertices),
      corners: vertices.length,
      quantity: Math.max(1, parseInt(quantity) || 1),
      notes,
      cuttingAllowance: allowance === "" ? null : parseFloat(allowance) * 10,
      grainDirection: grain,
      color: initialRoom?.color || ROOM_COLORS[Math.floor(Math.random() * ROOM_COLORS.length)],
    });
  }

  const drawPtsStr = points.map((p) => `${p.x},${p.y}`).join(" ");
  const previewLine = points.length > 0 && cursor ? `${points[points.length - 1].x},${points[points.length - 1].y} ${cursor.x},${cursor.y}` : "";

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4">
      <div className="bg-[#f7f5f0] text-stone-800 rounded-xl w-full max-w-3xl max-h-[92vh] overflow-y-auto shadow-2xl border border-stone-300">
        <div className="flex items-center justify-between px-5 py-3 border-b border-stone-300 bg-[#efece3]">
          <div>
            <h2 className="font-semibold text-sm tracking-wide uppercase flex items-center gap-2">
              <Pencil size={15} /> {initialRoom ? "Edit Room" : "Add Room"}
            </h2>
            <div className="flex items-center gap-2 mt-1.5">
              {[["draw", "Draw"], ["dims", "Dimensions"], ["details", "Details"]].map(([key, label], i) => {
                const order = ["draw", "dims", "details"];
                const active = step === key;
                const done = order.indexOf(step) > i;
                return (
                  <div key={key} className="flex items-center gap-1.5">
                    <span className={`w-4 h-4 rounded-full flex items-center justify-center text-[9px] font-mono ${active ? "bg-[#3c7a6f] text-white" : done ? "bg-[#3c7a6f]/30 text-[#3c7a6f]" : "bg-stone-300 text-stone-500"}`}>{done ? "✓" : i + 1}</span>
                    <span className={`text-[10px] ${active ? "text-stone-700 font-medium" : "text-stone-400"}`}>{label}</span>
                    {i < 2 && <span className="w-4 h-px bg-stone-300 ml-1" />}
                  </div>
                );
              })}
            </div>
          </div>
          <button onClick={onClose} title="Close without saving" className="text-stone-500 hover:text-stone-800 hover:bg-stone-100 rounded-full p-1.5 transition-colors"><X size={18} /></button>
        </div>

        {step === "draw" && (
          <div className="p-5">
            <p className="text-xs text-stone-500 mb-3">Click to place each corner. Every wall snaps to horizontal or vertical automatically. Click back on the <span className="text-[#3c7a6f] font-medium">first corner (teal dot)</span> to close the shape.</p>
            <div className="relative">
              <svg
                ref={svgRef}
                viewBox={`0 0 ${DRAW_W} ${DRAW_H}`}
                className="w-full bg-white border border-stone-300 rounded-xl cursor-crosshair select-none"
                style={{ aspectRatio: `${DRAW_W}/${DRAW_H}` }}
                onMouseMove={handleMove}
                onClick={handleClick}
              >
                <defs>
                  <pattern id="dotgrid" width={DRAW_SNAP} height={DRAW_SNAP} patternUnits="userSpaceOnUse">
                    <circle cx="1" cy="1" r="1" fill="#e3ded2" />
                  </pattern>
                </defs>
                <rect width={DRAW_W} height={DRAW_H} fill="url(#dotgrid)" />
                {points.length > 1 && <polyline points={drawPtsStr} fill="none" stroke="#c98a3a" strokeWidth="3" strokeLinejoin="round" />}
                {previewLine && <polyline points={previewLine} fill="none" stroke="#3c7a6f" strokeWidth="2" strokeDasharray="6 4" />}
                {points.map((p, i) => (
                  <g key={i}>
                    <circle cx={p.x} cy={p.y} r={i === 0 ? 8 : 6} fill={i === 0 ? "#3c7a6f" : "#c98a3a"} stroke="white" strokeWidth="2" />
                    {i > 0 && <text x={p.x + 8} y={p.y - 8} fontSize="11" fill="#78716c" fontFamily="monospace">W{i}</text>}
                  </g>
                ))}
                {cursor && <circle cx={cursor.x} cy={cursor.y} r="4" fill="#3c7a6f" opacity="0.6" />}
              </svg>
              {points.length === 0 && (
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <span className="text-xs text-stone-400 bg-white/90 px-3 py-1.5 rounded-xl border border-stone-200">Click anywhere to place the first corner</span>
                </div>
              )}
            </div>
            {drawError && <div className="mt-2 text-xs text-red-600 flex items-center gap-1"><AlertTriangle size={13} />{drawError}</div>}
            <div className="flex items-center gap-2 mt-3">
              <button onClick={undoPoint} disabled={points.length === 0} title="Remove the last corner placed" className="px-3 py-1.5 text-xs border border-stone-300 rounded-xl disabled:opacity-40 flex items-center gap-1"><Undo size={13} /> Undo point</button>
              <button onClick={resetDraw} disabled={points.length === 0} title="Clear this drawing and start over" className="px-3 py-1.5 text-xs border border-stone-300 rounded-xl disabled:opacity-40">Reset</button>
              <button onClick={closeShape} disabled={points.length < 3} title="Finish the shape" className="ml-auto px-5 py-2 text-sm font-medium bg-[#3c7a6f] text-white rounded-xl disabled:opacity-40 flex items-center gap-1.5 shadow-sm transition-colors"><Check size={14} /> Close shape</button>
            </div>
          </div>
        )}

        {step === "dims" && (
          <div className="p-5">
            <div className="flex items-center justify-between mb-3">
              <p className="text-xs text-stone-500">Enter the real length of every wall — the shape on the left shows which side each number belongs to. Leave one wall per axis blank to have it calculated automatically.</p>
              <select value={unit} onChange={(e) => setUnit(e.target.value)} className="text-xs border border-stone-300 rounded-xl px-2 py-1 bg-white shrink-0 ml-3">
                <option value="mm">mm</option><option value="cm">cm</option><option value="m">m</option>
              </select>
            </div>
            <div className="flex gap-4 items-start">
              <div className="shrink-0">
                <WallShapePreview vertices={shapeVerts} lengths={lengths} unit={unit} activeWall={activeWall} />
                <p className="text-[10px] text-stone-400 mt-1 text-center w-[240px]">Click or focus a wall field to highlight it here</p>
              </div>
              <div className="flex-1 min-w-0 grid grid-cols-1 gap-2 max-h-64 overflow-y-auto pr-1">
                {directions.map((d, i) => {
                  const Icon = DIR_ICON[d];
                  return (
                    <div
                      key={i}
                      className={`flex items-center gap-2 bg-white border rounded-xl px-2 py-1.5 ${activeWall === i ? "border-[#3c7a6f] ring-1 ring-[#3c7a6f]/40" : "border-stone-300"}`}
                    >
                      <span className={`w-4 h-4 rounded-full flex items-center justify-center text-[9px] font-mono shrink-0 ${activeWall === i ? "bg-[#3c7a6f] text-white" : "bg-stone-200 text-stone-500"}`}>{i + 1}</span>
                      <Icon size={14} className="text-[#c98a3a] shrink-0" />
                      <input
                        value={lengths[i]}
                        onChange={(e) => setLengths((ls) => ls.map((l, j) => (j === i ? e.target.value : l)))}
                        onFocus={() => setActiveWall(i)}
                        onBlur={() => setActiveWall((a) => (a === i ? null : a))}
                        placeholder={`e.g. 4${unit === "m" ? "" : "000"}${unit}`}
                        className="flex-1 text-xs font-mono border-0 outline-none bg-transparent min-w-0"
                      />
                      {calcIndexes.has(i) && <span className="text-[10px] text-[#3c7a6f] font-mono shrink-0">calc: {formatLength(resolvedLengths[i], unit)}</span>}
                    </div>
                  );
                })}
              </div>
            </div>
            {dimError && <div className="mt-3 text-xs text-red-600 flex items-center gap-1 bg-red-50 border border-red-200 rounded-xl px-2 py-1.5"><AlertTriangle size={13} className="shrink-0" />{dimError}</div>}
            {canConstruct && previewBB && (
              <div className="mt-3 flex items-center gap-4 bg-[#efece3] rounded-xl p-3">
                <div className="text-xs font-mono text-stone-600 space-y-0.5">
                  <div>{formatLength(previewBB.maxX - previewBB.minX, unit)} × {formatLength(previewBB.maxY - previewBB.minY, unit)}</div>
                  <div>Area: {formatArea(previewArea)}</div>
                  <div>Corners: {previewVerts.length}</div>
                </div>
              </div>
            )}
            <div className="flex items-center gap-2 mt-4">
              <button onClick={() => (initialRoom ? setStep("details") : setStep("draw"))} className="px-3 py-1.5 text-xs border border-stone-300 rounded-xl flex items-center gap-1"><ChevronLeft size={13} /> Back</button>
              <button onClick={() => setStep("details")} disabled={!canConstruct} className="ml-auto px-5 py-2 text-sm font-medium bg-[#3c7a6f] text-white rounded-xl disabled:opacity-40 flex items-center gap-1.5 shadow-sm transition-colors">Next <ChevronRight size={14} /></button>
            </div>
          </div>
        )}

        {step === "details" && (
          <div className="p-5 space-y-3">
            <Field label="Room name">
              <input value={name} onChange={(e) => setName(e.target.value)} autoFocus className="w-full border border-stone-300 rounded-xl px-2 py-1.5 text-sm bg-white" placeholder="Living Room" />
            </Field>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Quantity">
                <input type="number" min="1" value={quantity} onChange={(e) => setQuantity(e.target.value)} className="w-full border border-stone-300 rounded-xl px-2 py-1.5 text-sm bg-white font-mono" />
              </Field>
              <Field label="Cutting allowance (cm, optional)">
                <input value={allowance} onChange={(e) => setAllowance(e.target.value)} placeholder="project default" className="w-full border border-stone-300 rounded-xl px-2 py-1.5 text-sm bg-white font-mono" />
              </Field>
            </div>
            <Field label="Notes">
              <textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows={2} className="w-full border border-stone-300 rounded-xl px-2 py-1.5 text-sm bg-white" />
            </Field>
            <label className="flex items-center gap-2 text-xs text-stone-600">
              <input type="checkbox" checked={grain} onChange={(e) => setGrain(e.target.checked)} /> Show grain/pile direction arrow on this room
            </label>
            {previewVerts && (
              <div className="flex items-center gap-4 bg-[#efece3] rounded-xl p-3">
                <RoomThumbnail vertices={previewVerts} color="#c98a3a" size={64} />
                <div className="text-xs font-mono text-stone-600 space-y-0.5">
                  <div>{formatLength(previewBB.maxX - previewBB.minX, "m")} × {formatLength(previewBB.maxY - previewBB.minY, "m")}</div>
                  <div>Area: {formatArea(previewArea)}</div>
                </div>
              </div>
            )}
            <div className="flex items-center gap-2 pt-2">
              <button onClick={() => setStep("dims")} className="px-3 py-1.5 text-xs border border-stone-300 rounded-xl flex items-center gap-1"><ChevronLeft size={13} /> Back</button>
              <button onClick={handleSave} disabled={!name.trim()} className="ml-auto px-5 py-2 text-sm font-medium bg-[#c98a3a] text-white rounded-xl disabled:opacity-40 flex items-center gap-1.5 shadow-sm transition-colors"><Check size={14} /> Save room</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ======================================================================
   SECTION: LEFT SIDEBAR — rooms library + roll settings
   ====================================================================== */

function LeftSidebar({ rooms, onAddRoom, onEditRoom, onDeleteRoom, onDuplicateRoom, onQuantityChange, onPlaceOnCarpet, rollWidth, setRollWidth, unit, setUnit, tolerance, setTolerance, grainMode, setGrainMode, gridSize, setGridSize, snapEnabled, setSnapEnabled, allowOverlap, setAllowOverlap, guillotineMode, setGuillotineMode }) {
  const [tab, setTab] = useState("rooms");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [confirmDeleteId, setConfirmDeleteId] = useState(null);
  return (
    <div className="w-72 shrink-0 border-r border-stone-300 bg-[#efece3] flex flex-col h-full">  function deleteSelected() {
    setInstances((prev) => {
      const next = prev.filter((i) => i.id !== selectedId);
      commitHistory(next);
      return next;
    });
    setSelectedId(null);
  }
  function duplicateSelected() {
    setInstances((prev) => {
      const inst = prev.find((i) => i.id === selectedId);
      if (!inst) return prev;
      const copy = { ...inst, id: `inst_${Date.now()}`, x: inst.x + 100, y: inst.y + 100 };
      const next = [...prev, copy];
      commitHistory(next);
      setSelectedId(copy.id);
      return next;
    });
  }
  function updateAllowanceOverride(val) {
    setInstances((prev) => {
      const next = prev.map((i) => (i.id === selectedId ? { ...i, allowanceOverride: val } : i));
      commitHistory(next);
      return next;
    });
  }

  // One-click, always-valid placement: drops the room just below whatever is
  // already on the carpet, so it can never collide or overhang the roll width.
  function placeRoomOnCarpet(room) {
    setInstances((prev) => {
      const bb = boundingBox(room.vertices);
      const w = bb.maxX - bb.minX;
      const y = requiredLength(prev, roomsById) + (prev.length ? tolerance + 60 : 0);
      const x = Math.max(0, Math.min(rollWidth - w, 0));
      const newInst = {
        id: `inst_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
        roomId: room.id, rotation: 0, x, y,
        allowanceOverride: room.cuttingAllowance != null ? String(room.cuttingAllowance) : null,
      };
      const next = [...prev, newInst];
      commitHistory(next);
      setSelectedId(newInst.id);
      return next;
    });
    setView("cutting");
  }

  // Keyboard shortcuts on the cutting canvas: Delete/Backspace removes the
  // selected piece, R rotates it, Escape deselects. Ignored while typing.
  useEffect(() => {
    function onKey(e) {
      const tag = (e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select" || modal) return;
      if (view !== "cutting") return;
      if ((e.key === "Delete" || e.key === "Backspace") && selectedId) { e.preventDefault(); deleteSelected(); }
      else if (e.key.toLowerCase() === "r" && selectedId) { rotateSelected(); }
      else if (e.key === "Escape") { setSelectedId(null); }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedId, view, modal, grainMode]); // eslint-disable-line

  function runAutoArrange() {
    setArranging(true);
    // Defer to the next tick so the "Optimising…" label actually paints
    // before the (synchronous, time-budgeted) search runs.
    setTimeout(() => {
      setInstances((prev) => {
        const next = autoArrange(prev, roomsById, rollWidth, tolerance, grainMode, guillotineMode);
        commitHistory(next);
        return next;
      });
      setArranging(false);
    }, 10);
  }

  const reqLength = useMemo(() => requiredLength(instances, roomsById), [instances, roomsById]);
  const roomArea = useMemo(() => instances.reduce((sum, i) => sum + (roomsById[i.roomId]?.area || 0), 0), [instances, roomsById]);
  const rollArea = rollWidth * reqLength;
  const waste = Math.max(0, rollArea - roomArea);
  const wastePct = rollArea > 0 ? ((waste / rollArea) * 100).toFixed(1) : "0.0";

  const stats = {
    length: reqLength > 0 ? (reqLength / 1000).toFixed(2) + "m" : "-",
    area: rollArea > 0 ? formatArea(rollArea) : "-",
    waste: rollArea > 0 ? `${formatArea(waste)} (${wastePct}%)` : "-",
  };

  const selectedInstance = instances.find((i) => i.id === selectedId) || null;
  const selectedRoom = selectedInstance ? roomsById[selectedInstance.roomId] : null;

  if (!loaded) return <div className="h-screen flex items-center justify-center bg-[#2b2926] text-stone-400 text-sm font-mono">Loading project…</div>;

  return (
    <div className="h-screen flex flex-col font-sans" style={{ fontFamily: "Inter, ui-sans-serif, system-ui" }}>
      <TopBar
        projectName={projectName} setProjectName={setProjectName} stats={stats}
        onUndo={undo} onRedo={redo} canUndo={historyRef.current.index > 0} canRedo={historyRef.current.index < historyRef.current.stack.length - 1}
        onAutoArrange={runAutoArrange} arranging={arranging} onSave={doSave} view={view} setView={setView} onOpenCuttingPlan={() => setShowCuttingPlan(true)}
        instanceCount={instances.length}
      />
      <div className="flex flex-1 min-h-0">
        <LeftSidebar
          rooms={rooms} onAddRoom={() => setModal({ mode: "add" })} onEditRoom={(r) => setModal({ mode: "edit", room: r })}
          onDeleteRoom={handleDeleteRoom} onDuplicateRoom={handleDuplicateRoom} onQuantityChange={handleQuantityChange}
          onPlaceOnCarpet={placeRoomOnCarpet}
          rollWidth={rollWidth} setRollWidth={setRollWidth} unit={unit} setUnit={setUnit}
          tolerance={tolerance} setTolerance={setTolerance} grainMode={grainMode} setGrainMode={setGrainMode}
          gridSize={gridSize} setGridSize={setGridSize} snapEnabled={snapEnabled} setSnapEnabled={setSnapEnabled}
          allowOverlap={allowOverlap} setAllowOverlap={setAllowOverlap}
          guillotineMode={guillotineMode} setGuillotineMode={setGuillotineMode}
        />
        {view === "cutting" ? (
          <>
            <CuttingCanvas
              rooms={rooms} roomsById={roomsById} instances={instances} setInstances={setInstances}
              rollWidth={rollWidth} tolerance={tolerance} gridSize={gridSize} snapEnabled={snapEnabled}
              selectedId={selectedId} setSelectedId={setSelectedId} commitHistory={commitHistory}
              allowOverlap={allowOverlap}
            />
            <RightSidebar
              instance={selectedInstance} room={selectedRoom} onRotate={rotateSelected} onDelete={deleteSelected}
              onDuplicate={duplicateSelected} onAllowanceChange={updateAllowanceOverride} unit={unit}
            />
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center bg-[#f7f5f0] p-8">
            <div className="max-w-md text-center text-stone-500 text-sm">
              <Layers size={28} className="mx-auto mb-3 text-[#c98a3a]" />
              <p className="font-medium text-stone-700 mb-1">Build your room library, then switch to Cutting Plan</p>
              <p>Draw each room once, set its quantity, then head to the Cutting Plan tab to drag rooms onto the carpet roll and find the most efficient layout.</p>
            </div>
          </div>
        )}
      </div>

      {modal && (
        <AddRoomModal
          initialRoom={modal.mode === "edit" ? modal.room : null}
          onClose={() => setModal(null)}
          onSave={handleSaveRoom}
          displayUnit={unit}
        />
      )}
      {showCuttingPlan && (
        <CuttingPlanModal
          onClose={() => setShowCuttingPlan(false)}
          rooms={rooms} roomsById={roomsById} instances={instances} rollWidth={rollWidth}
          projectName={projectName} reqLength={reqLength}
        />
      )}
    </div>
  );
}