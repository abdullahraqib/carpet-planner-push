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
  forbidden.forEach((f) => { xs.add(f.x); xs.add(f.x + f.w); ys.add(f.y); ys.add(f.y + f.h); });  const xList = [...xs].filter((x) => x >= minX - EPS && x <= maxX + EPS);
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
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0">      <polygon points={pts} fill={color + "33"} stroke={color} strokeWidth="2" strokeLinejoin="round" />
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
            <div className="flex items-center justify-between mb-3">              <p className="text-xs text-stone-500">Enter the real length of every wall — the shape on the left shows which side each number belongs to. Leave one wall per axis blank to have it calculated automatically.</p>
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
    <div className="w-72 shrink-0 border-r border-stone-300 bg-[#efece3] flex flex-col h-full">
      <div className="flex border-b border-stone-300 text-xs font-medium uppercase tracking-wide">
        <button onClick={() => setTab("rooms")} className={`flex-1 py-2.5 ${tab === "rooms" ? "bg-[#f7f5f0] text-stone-800" : "text-stone-500"}`}>Rooms</button>
        <button onClick={() => setTab("roll")} className={`flex-1 py-2.5 ${tab === "roll" ? "bg-[#f7f5f0] text-stone-800" : "text-stone-500"}`}>Roll Settings</button>
      </div>

      {tab === "rooms" && (
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          <button onClick={onAddRoom} title="Draw a new room shape" className="w-full flex items-center justify-center gap-2 py-3 text-sm font-semibold bg-[#3c7a6f] text-white rounded-xl hover:bg-[#336a60] shadow-sm transition-colors">
            <Plus size={16} /> Add Room
          </button>
          {rooms.length === 0 && (
            <div className="mt-3 bg-white border border-dashed border-stone-300 rounded-xl p-3.5">
              <p className="text-xs font-semibold text-stone-700 mb-2">Get started in 3 steps</p>
              <ol className="text-[11px] text-stone-500 space-y-1.5 list-decimal list-inside">
                <li>Tap <b className="text-stone-700">Add Room</b> and draw its shape corner by corner</li>
                <li>Type in each wall's real length</li>
                <li>Set your roll width, then drag or place rooms onto the Cutting Plan</li>
              </ol>
            </div>
          )}
          {rooms.map((room) => (
            <div key={room.id} className="bg-white border border-stone-300 rounded-xl p-2.5">
              <div className="flex gap-2.5">
                <div draggable onDragStart={(e) => e.dataTransfer.setData("roomId", room.id)} title="Drag onto the carpet canvas" className="cursor-grab active:cursor-grabbing relative">
                  <RoomThumbnail vertices={room.vertices} color={room.color} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full shrink-0" style={{ background: room.color }} />
                    <div className="text-sm font-medium truncate">{room.name}</div>
                  </div>
                  <div className="text-[11px] font-mono text-stone-500">{(room.width / 1000).toFixed(2)}m × {(room.length / 1000).toFixed(2)}m</div>
                  <div className="text-[11px] font-mono text-stone-500">{formatArea(room.area)} · {room.corners} corners</div>
                  <div className="flex items-center gap-1 mt-1.5">
                    <span className="text-[10px] text-stone-400">Qty</span>
                    <div className="flex items-center border border-stone-200 rounded-xl overflow-hidden">
                      <button onClick={() => onQuantityChange(room.id, Math.max(1, room.quantity - 1))} title="Decrease quantity" className="px-1.5 py-0.5 text-stone-500 hover:bg-stone-100 leading-none">−</button>
                      <input type="number" min="1" value={room.quantity} onChange={(e) => onQuantityChange(room.id, Math.max(1, parseInt(e.target.value) || 1))} className="w-8 text-[11px] font-mono text-center border-x border-stone-200 py-0.5" />
                      <button onClick={() => onQuantityChange(room.id, room.quantity + 1)} title="Increase quantity" className="px-1.5 py-0.5 text-stone-500 hover:bg-stone-100 leading-none">+</button>
                    </div>
                  </div>
                </div>
              </div>
              <button onClick={() => onPlaceOnCarpet(room)} title="Add one copy to the Cutting Plan canvas" className="w-full mt-2.5 flex items-center justify-center gap-1.5 py-2 text-xs font-medium bg-[#efece3] hover:bg-[#e5e0d3] text-stone-700 rounded-xl transition-colors">
                <Layers size={13} /> Place on Carpet
              </button>
              <div className="flex items-center gap-2.5 mt-1.5 pt-1.5 border-t border-stone-100">
                <button onClick={() => onEditRoom(room)} title="Edit this room" className="text-[11px] text-stone-500 hover:text-stone-800 flex items-center gap-1"><Pencil size={11} /> Edit</button>
                <button onClick={() => onDuplicateRoom(room)} title="Duplicate this room" className="text-[11px] text-stone-500 hover:text-stone-800 flex items-center gap-1"><Copy size={11} /> Duplicate</button>
                {confirmDeleteId === room.id ? (
                  <span className="ml-auto flex items-center gap-1.5 text-[11px]">
                    <span className="text-stone-500">Delete?</span>
                    <button onClick={() => { onDeleteRoom(room.id); setConfirmDeleteId(null); }} className="text-red-600 font-semibold hover:underline">Yes</button>
                    <button onClick={() => setConfirmDeleteId(null)} className="text-stone-400 hover:underline">No</button>
                  </span>
                ) : (
                  <button onClick={() => setConfirmDeleteId(room.id)} title="Delete this room" className="ml-auto text-[11px] text-red-500 hover:text-red-700 flex items-center gap-1"><Trash2 size={11} /> Delete</button>
                )}
              </div>
            </div>
          ))}
          {rooms.length > 0 && <p className="text-[10px] text-stone-400 pt-2">Drag a card onto the carpet, or use "Place on Carpet" for a quick, always-valid drop you can then fine-tune.</p>}
        </div>
      )}

      {tab === "roll" && (
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <Field label="Carpet roll width">
            <div className="flex items-center gap-2">
              <input
                type="number" step="0.01" value={rollWidth / 1000}
                onChange={(e) => setRollWidth(Math.max(500, Math.round((parseFloat(e.target.value) || 0) * 1000)))}
                className="flex-1 border border-stone-300 rounded-xl px-3 py-2 text-sm font-mono bg-white"
              />
              <span className="text-sm text-stone-500">m</span>
            </div>
            <div className="flex gap-1.5 mt-2">
              {[3.66, 4.0, 5.0].map((w) => (
                <button key={w} onClick={() => setRollWidth(Math.round(w * 1000))} title={`Set roll width to ${w}m`} className="text-xs font-medium px-2.5 py-1 rounded-full bg-stone-100 hover:bg-stone-200 transition-colors">{w}m</button>
              ))}
            </div>
          </Field>
          <Field label="Display unit">
            <select value={unit} onChange={(e) => setUnit(e.target.value)} className="w-full border border-stone-300 rounded-xl px-3 py-2 text-sm bg-white">
              <option value="mm">Millimetres</option><option value="cm">Centimetres</option><option value="m">Metres</option>
            </select>
          </Field>

          <button
            onClick={() => setShowAdvanced((s) => !s)}
            className="w-full flex items-center justify-between text-xs font-medium text-stone-500 hover:text-stone-700 pt-2 border-t border-stone-200"
          >
            <span>More settings</span>
            {showAdvanced ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>

          {showAdvanced && (
            <div className="space-y-4 pt-1">
              <Field label="Extra allowance around each piece">
                <select value={tolerance} onChange={(e) => setTolerance(parseInt(e.target.value))} className="w-full border border-stone-300 rounded-xl px-3 py-2 text-sm bg-white">
                  {[0, 1, 2.5, 5, 10].map((cm) => <option key={cm} value={cm * 10}>{cm === 0 ? "None" : `${cm}cm`}</option>)}
                </select>
              </Field>
              <Field label="Rotating pieces">
                <select value={grainMode} onChange={(e) => setGrainMode(e.target.value)} className="w-full border border-stone-300 rounded-xl px-3 py-2 text-sm bg-white">
                  <option value="free">Turn any amount</option>
                  <option value="90">Turn in 90° steps only</option>
                  <option value="locked">Don't allow turning</option>
                </select>
              </Field>
              <Field label="Snap-to grid">
                <div className="flex items-center gap-2">
                  <select value={gridSize} onChange={(e) => setGridSize(parseInt(e.target.value))} className="flex-1 border border-stone-300 rounded-xl px-3 py-2 text-sm bg-white">
                    {[10, 25, 50, 100].map((v) => <option key={v} value={v}>{v}mm</option>)}
                  </select>
                  <label className="flex items-center gap-1.5 text-xs text-stone-600">
                    <input type="checkbox" checked={snapEnabled} onChange={(e) => setSnapEnabled(e.target.checked)} /> On
                  </label>
                </div>
              </Field>
              <div className="pt-1 border-t border-stone-200">
                <label className="flex items-center gap-2 text-xs text-stone-700 pt-3 cursor-pointer">
                  <input type="checkbox" checked={guillotineMode} onChange={(e) => setGuillotineMode(e.target.checked)} />
                  Auto Arrange: keep cuts easy to make by hand
                </label>
                <p className="text-[10px] text-stone-400 mt-1">
                  {guillotineMode
                    ? "On by default: every cut Auto Arrange produces runs straight across the piece it's cutting from — easy with a knife and rule, though it may use a little more carpet."
                    : "Pieces can nest into any gap, including the notch of an L-shaped room, for the tightest possible fit — but the cut lines between pieces may not be straight."}
                </p>
              </div>
              <div>
                <label className="flex items-center gap-2 text-xs text-stone-700 cursor-pointer">
                  <input type="checkbox" checked={allowOverlap} onChange={(e) => setAllowOverlap(e.target.checked)} />
                  Allow pieces to overlap
                </label>
                <p className="text-[10px] text-stone-400 mt-1">
                  {allowOverlap
                    ? "Pieces can be placed on top of each other — overlapping ones are outlined red until you move them apart."
                    : "Off by default: pieces snap back to the last clear spot if they'd overlap another room."}
                </p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ======================================================================
   SECTION: RIGHT SIDEBAR — selected instance inspector
   ====================================================================== */

function RightSidebar({ instance, room, onRotate, onDelete, onDuplicate, onAllowanceChange, unit }) {
  if (!instance || !room) {
    return (
      <div className="w-64 shrink-0 border-l border-stone-300 bg-[#efece3] p-4">
        <p className="text-xs text-stone-500">Select a room piece on the carpet to see its details, or drag a room from the library onto the canvas.</p>
      </div>
    );
  }
  const verts = instanceVertices(room, instance);
  const bb = boundingBox(verts);
  return (
    <div className="w-64 shrink-0 border-l border-stone-300 bg-[#efece3] p-4 space-y-3">
      <div className="flex items-center gap-2">
        <span className="w-3 h-3 rounded-xl" style={{ background: room.color }} />
        <h3 className="text-sm font-semibold truncate">{room.name}</h3>
      </div>
      <div className="text-xs font-mono text-stone-600 space-y-1 bg-white border border-stone-200 rounded-xl p-2.5">
        <div className="flex justify-between"><span className="text-stone-400">Size</span><span>{formatLength(bb.maxX - bb.minX, unit)} × {formatLength(bb.maxY - bb.minY, unit)}</span></div>
        <div className="flex justify-between"><span className="text-stone-400">Area</span><span>{formatArea(room.area)}</span></div>
        <div className="flex justify-between"><span className="text-stone-400">Position</span><span>{Math.round(instance.x)}, {Math.round(instance.y)}mm</span></div>
        <div className="flex justify-between"><span className="text-stone-400">Rotation</span><span>{instance.rotation}°</span></div>
      </div>
      <Field label="Cutting allowance override (cm)">
        <input
          value={instance.allowanceOverride != null && instance.allowanceOverride !== "" ? parseFloat(instance.allowanceOverride) / 10 : ""}
          onChange={(e) => onAllowanceChange(e.target.value === "" ? "" : String(parseFloat(e.target.value) * 10))}
          placeholder="project default" className="w-full border border-stone-300 rounded-xl px-2 py-1.5 text-xs font-mono bg-white"
        />
      </Field>
      {room.grainDirection && <div className="text-[11px] text-[#3c7a6f] flex items-center gap-1"><ArrowUp size={12} style={{ transform: `rotate(${instance.rotation}deg)` }} /> Grain direction shown on canvas</div>}
      <div className="grid grid-cols-2 gap-2 pt-1">
        <button onClick={onRotate} title="Rotate 90° (shortcut: R)" className="flex items-center justify-center gap-1.5 text-xs font-medium rounded-xl py-2 bg-stone-100 hover:bg-stone-200 transition-colors"><RotateCw size={14} /> Rotate 90°</button>
        <button onClick={onDuplicate} title="Duplicate this piece" className="flex items-center justify-center gap-1.5 text-xs font-medium rounded-xl py-2 bg-stone-100 hover:bg-stone-200 transition-colors"><Copy size={14} /> Duplicate</button>
        <button onClick={onDelete} title="Remove (shortcut: Delete)" className="col-span-2 flex items-center justify-center gap-1.5 text-xs font-medium text-red-600 rounded-xl py-2 bg-red-50 hover:bg-red-100 transition-colors"><Trash2 size={14} /> Remove from carpet</button>
      </div>
    </div>
  );
}

/* ======================================================================
   SECTION: CUTTING CANVAS
   ====================================================================== */

function CuttingCanvas({ rooms, roomsById, instances, setInstances, rollWidth, tolerance, gridSize, snapEnabled, selectedId, setSelectedId, commitHistory, allowOverlap }) {
  const containerRef = useRef(null);
  const [pxPerMm, setPxPerMm] = useState(0.09);  const [pan, setPan] = useState({ x: 0, y: 0 });
  const dragState = useRef(null);
  const rafId = useRef(null);
  const pendingMouse = useRef(null);

  const reqLength = useMemo(() => requiredLength(instances, roomsById), [instances, roomsById]);
  const canvasLenMm = Math.max(reqLength + 1500, 3000);

  function screenToMm(clientX, clientY) {
    const el = containerRef.current;
    const rect = el.getBoundingClientRect();
    // Scroll offset has to be added back in — the bounding rect is always
    // relative to the viewport, regardless of how far the container itself
    // has been scrolled, so without this a scrolled view would misplace
    // every drag by exactly however much had been scrolled.
    const x = (clientX - rect.left - pan.x + el.scrollLeft) / pxPerMm;
    const y = (clientY - rect.top - pan.y + el.scrollTop) / pxPerMm;
    return { x, y };
  }

  function snapVal(v) { return snapEnabled ? Math.round(v / gridSize) * gridSize : v; }

  function effectiveTolerance(inst) { return inst.allowanceOverride != null && inst.allowanceOverride !== "" ? parseFloat(inst.allowanceOverride) : tolerance; }

  // Every placed piece that currently overlaps another piece, or falls
  // outside the roll width, is flagged persistently — not just while it's
  // being dragged — so collisions stay visible until resolved.
  const invalidIds = useMemo(() => {
    const s = new Set();
    instances.forEach((inst) => {
      const room = roomsById[inst.roomId];
      if (!room) return;
      const check = isPlacementValid(room, inst, instances, roomsById, rollWidth, effectiveTolerance(inst));
      if (!check.valid) s.add(inst.id);
    });
    return s;
  }, [instances, roomsById, rollWidth, tolerance]); // eslint-disable-line

  // Grid lines are cheap individually but there can be hundreds of them —
  // memoised so a drag (which updates instance positions, not the grid)
  // never has to rebuild this list on every frame.
  const gridLinesX = useMemo(() => {
    if (!snapEnabled) return [];
    return Array.from({ length: Math.floor(rollWidth / gridSize) }, (_, i) => i);
  }, [snapEnabled, rollWidth, gridSize]);
  const gridLinesY = useMemo(() => {
    if (!snapEnabled) return [];
    return Array.from({ length: Math.floor(canvasLenMm / gridSize) }, (_, i) => i);
  }, [snapEnabled, canvasLenMm, gridSize]);

  // Magnetic snap to the roll's boundaries (left edge x=0, right edge
  // x=rollWidth, start of the roll y=0) AND to the edges of other pieces
  // already on the carpet, so two pieces dragged close together lock
  // together with their edges exactly touching rather than left with an
  // awkward sliver gap or nudged into a (rejected) overlap. Each axis snaps
  // independently to whichever candidate is nearest, within a small
  // real-mm threshold that's independent of grid snapping.
  const EDGE_SNAP_MM = 40;
  function snapToEdges(x, y, w, h, excludeId) {
    let nx = x, ny = y, bestXDist = EDGE_SNAP_MM, bestYDist = EDGE_SNAP_MM;
    if (Math.abs(x) <= bestXDist) { bestXDist = Math.abs(x); nx = 0; }
    if (Math.abs(x + w - rollWidth) <= bestXDist) { bestXDist = Math.abs(x + w - rollWidth); nx = rollWidth - w; }
    if (Math.abs(y) <= bestYDist) { bestYDist = Math.abs(y); ny = 0; }

    instances.forEach((other) => {
      if (other.id === excludeId) return;
      const otherRoom = roomsById[other.roomId];
      if (!otherRoom) return;
      const obb = boundingBox(instanceVertices(otherRoom, other));
      const yOverlaps = y < obb.maxY && y + h > obb.minY;
      const xOverlaps = x < obb.maxX && x + w > obb.minX;
      if (yOverlaps) {
        const dRight = Math.abs(x - obb.maxX); // this piece's left touching the other's right
        if (dRight <= bestXDist) { bestXDist = dRight; nx = obb.maxX; }
        const dLeft = Math.abs(x + w - obb.minX); // this piece's right touching the other's left
        if (dLeft <= bestXDist) { bestXDist = dLeft; nx = obb.minX - w; }
      }
      if (xOverlaps) {
        const dBottom = Math.abs(y - obb.maxY); // this piece's top touching the other's bottom
        if (dBottom <= bestYDist) { bestYDist = dBottom; ny = obb.maxY; }
        const dTop = Math.abs(y + h - obb.minY); // this piece's bottom touching the other's top
        if (dTop <= bestYDist) { bestYDist = dTop; ny = obb.minY - h; }
      }
    });
    return { x: nx, y: ny };
  }

  function onDropRoom(e) {
    e.preventDefault();
    const roomId = e.dataTransfer.getData("roomId");
    const room = roomsById[roomId];
    if (!room) return;
    const { x, y } = screenToMm(e.clientX, e.clientY);
    const bb = boundingBox(room.vertices);
    const w = bb.maxX - bb.minX, h = bb.maxY - bb.minY;
    let px = Math.max(0, Math.min(rollWidth - w, snapVal(x - w / 2)));
    let py = Math.max(0, snapVal(y - h / 2));
    const snapped = snapToEdges(px, py, w, h, null);
    px = snapped.x; py = snapped.y;
    let newInst = {
      id: `inst_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      roomId, rotation: 0, x: px, y: py,
      allowanceOverride: room.cuttingAllowance != null ? String(room.cuttingAllowance) : null,
    };
    if (!allowOverlap) {
      const check = isPlacementValid(room, newInst, instances, roomsById, rollWidth, effectiveTolerance(newInst));
      if (!check.valid) {
        // Overlap not allowed: fall back to a guaranteed-clear spot below
        // everything already on the carpet instead of rejecting the drop.
        const gap = instances.length ? tolerance + 60 : 0;
        newInst = { ...newInst, x: 0, y: requiredLength(instances, roomsById) + gap };
      }
    }
    const next = [...instances, newInst];
    setInstances(next); commitHistory(next); setSelectedId(newInst.id);
  }

  function beginDrag(e, inst) {
    e.stopPropagation();
    setSelectedId(inst.id);
    const startMm = screenToMm(e.clientX, e.clientY);
    dragState.current = { id: inst.id, offsetX: startMm.x - inst.x, offsetY: startMm.y - inst.y, lastValid: { x: inst.x, y: inst.y } };
    pendingMouse.current = { clientX: e.clientX, clientY: e.clientY };
    window.addEventListener("mousemove", onDragMove);
    window.addEventListener("mouseup", onDragEnd);
    // A continuous per-frame loop (rather than only reacting to mousemove)
    // is what lets autoscroll keep going even while the mouse holds still
    // near the bottom edge — a mousemove-only loop would stop scrolling the
    // instant the cursor stops moving, even though it's still past the edge.
    rafId.current = requestAnimationFrame(processDragFrame);
  }

  // Fast mouse movement can fire far more mousemove events than the app can
  // usefully process (each move re-checks collisions against every other
  // room), so only the latest position is kept here; the actual work runs
  // in the continuous per-frame loop below instead of once per raw event.
  function onDragMove(e) {
    if (!dragState.current) return;
    pendingMouse.current = { clientX: e.clientX, clientY: e.clientY };
  }

  const AUTOSCROLL_EDGE_PX = 48, AUTOSCROLL_MAX_SPEED = 22;

  function processDragFrame() {
    const ds = dragState.current;
    const mouse = pendingMouse.current;
    if (!ds || !mouse) { rafId.current = null; return; }

    // Autoscroll: if the cursor is held near the top/bottom (or left/right)
    // edge of the visible canvas while dragging, scroll that direction so
    // the piece can keep moving toward carpet that's off-screen below.
    const el = containerRef.current;
    if (el) {
      const rect = el.getBoundingClientRect();
      let dy = 0, dx = 0;
      if (mouse.clientY > rect.bottom - AUTOSCROLL_EDGE_PX) {
        dy = AUTOSCROLL_MAX_SPEED * Math.min(1, (mouse.clientY - (rect.bottom - AUTOSCROLL_EDGE_PX)) / AUTOSCROLL_EDGE_PX);
      } else if (mouse.clientY < rect.top + AUTOSCROLL_EDGE_PX) {
        dy = -AUTOSCROLL_MAX_SPEED * Math.min(1, ((rect.top + AUTOSCROLL_EDGE_PX) - mouse.clientY) / AUTOSCROLL_EDGE_PX);
      }
      if (mouse.clientX > rect.right - AUTOSCROLL_EDGE_PX) {
        dx = AUTOSCROLL_MAX_SPEED * Math.min(1, (mouse.clientX - (rect.right - AUTOSCROLL_EDGE_PX)) / AUTOSCROLL_EDGE_PX);
      } else if (mouse.clientX < rect.left + AUTOSCROLL_EDGE_PX) {
        dx = -AUTOSCROLL_MAX_SPEED * Math.min(1, ((rect.left + AUTOSCROLL_EDGE_PX) - mouse.clientX) / AUTOSCROLL_EDGE_PX);
      }
      if (dy) el.scrollTop = Math.max(0, el.scrollTop + dy);
      if (dx) el.scrollLeft = Math.max(0, el.scrollLeft + dx);
    }

    const mm = screenToMm(mouse.clientX, mouse.clientY);
    let nx = snapVal(mm.x - ds.offsetX);
    let ny = Math.max(0, snapVal(mm.y - ds.offsetY));
    setInstances((prev) => {
      // Re-check: the drag may have ended (or the piece been deleted) while
      // this frame's work was queued, so never trust the outer closure alone.
      if (!dragState.current || dragState.current.id !== ds.id) return prev;
      const instPrev = prev.find((i) => i.id === ds.id);
      if (!instPrev) return prev;
      const room = roomsById[instPrev.roomId];
      if (!room) return prev;
      const rotBB = boundingBox(rotatePolygon(room.vertices, instPrev.rotation));
      const w = rotBB.maxX - rotBB.minX, h = rotBB.maxY - rotBB.minY;
      const snapped = snapToEdges(nx, ny, w, h, ds.id);
      nx = snapped.x; ny = snapped.y;
      const next = prev.map((i) => (i.id === ds.id ? { ...i, x: nx, y: ny } : i));
      const inst = next.find((i) => i.id === ds.id);
      const check = isPlacementValid(room, inst, next, roomsById, rollWidth, effectiveTolerance(inst));
      if (check.valid && dragState.current) dragState.current.lastValid = { x: nx, y: ny };
      return next;
    });

    if (dragState.current) rafId.current = requestAnimationFrame(processDragFrame);
    else rafId.current = null;
  }

  function onDragEnd() {
    window.removeEventListener("mousemove", onDragMove);
    window.removeEventListener("mouseup", onDragEnd);
    if (rafId.current != null) { cancelAnimationFrame(rafId.current); rafId.current = null; }
    pendingMouse.current = null;
    const ds = dragState.current;
    dragState.current = null; // clear before any async state work touches it
    if (!ds) return;
    const { id, lastValid } = ds;
    if (!allowOverlap) {
      // Strict mode: snap back to the last spot that didn't collide.
      setInstances((prev) => {
        const next = prev.map((i) => (i.id === id ? { ...i, x: lastValid.x, y: lastValid.y } : i));
        commitHistory(next);
        return next;
      });
    } else {
      // Overlap allowed: keep wherever it was dropped, collision or not —
      // it stays flagged red via invalidIds until the user resolves it.
      setInstances((prev) => { commitHistory(prev); return prev; });
    }
  }

  function zoom(delta) { setPxPerMm((z) => Math.min(0.4, Math.max(0.02, z * delta))); }
  function fitToScreen() {
    const rect = containerRef.current.getBoundingClientRect();
    setPxPerMm(Math.max(0.02, (rect.width - 60) / rollWidth));
    setPan({ x: 30, y: 30 });
  }
  useEffect(() => { fitToScreen(); }, []); // eslint-disable-line

  const widthPx = rollWidth * pxPerMm;
  const lenPx = canvasLenMm * pxPerMm;
  const metreTicks = [];
  for (let m = 0; m * 1000 <= rollWidth; m++) metreTicks.push(m);
  const lengthTicks = [];
  for (let m = 0; m * 1000 <= canvasLenMm; m++) lengthTicks.push(m);

  return (
    <div className="flex-1 flex flex-col bg-[#2b2926] min-w-0">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-black/30 bg-[#242220] text-stone-300">
        <span className="text-[11px] font-mono uppercase tracking-wider text-[#e0a527]">Roll width: {(rollWidth / 1000).toFixed(2)}m</span>
        <span className="text-[11px] font-mono text-stone-400">Required length: {(reqLength / 1000).toFixed(2)}m</span>
        <span className="text-[10px] text-stone-500 ml-3 hidden md:inline">Click a piece to select · drag to move · R to rotate · Delete to remove</span>
        {invalidIds.size > 0 && (
          <span title={allowOverlap ? "Overlap is allowed — resolve when ready" : "These will snap clear once you move them"} className="text-[11px] font-mono text-red-400 flex items-center gap-1 ml-3">
            <AlertTriangle size={12} /> {invalidIds.size} overlapping
          </span>
        )}
        <div className="ml-auto flex items-center gap-1">
          <button onClick={() => zoom(0.85)} title="Zoom out" className="p-2 hover:bg-white/10 rounded-full transition-colors"><ZoomOut size={14} /></button>
          <button onClick={() => zoom(1.18)} title="Zoom in" className="p-2 hover:bg-white/10 rounded-full transition-colors"><ZoomIn size={14} /></button>
          <button onClick={fitToScreen} title="Fit to screen" className="p-2 hover:bg-white/10 rounded-full transition-colors"><Maximize2 size={14} /></button>
        </div>
      </div>
      <div
        ref={containerRef}
        className="flex-1 overflow-auto relative"
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDropRoom}
        onMouseDown={() => setSelectedId(null)}
        style={{
          backgroundImage: "repeating-linear-gradient(-45deg, rgba(255,255,255,0.015) 0px, rgba(255,255,255,0.015) 1px, transparent 1px, transparent 6px)",
        }}
      >
        {instances.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
            <div className="text-center text-stone-400 bg-[#242220]/90 border border-dashed border-stone-600 rounded-xl px-6 py-5">
              <Layers size={22} className="mx-auto mb-2 text-[#e0a527]" />
              <p className="text-xs font-medium text-stone-300">Drag a room here from the library</p>
              <p className="text-[11px] mt-1">or click "Place on Carpet" on a room card</p>
            </div>
          </div>
        )}
        <svg width={widthPx + 90} height={lenPx + 60} className="select-none">
          <g transform={`translate(${pan.x},${pan.y})`}>
            {/* carpet body */}
            <rect x={0} y={0} width={widthPx} height={lenPx} fill="#3a3631" stroke="#55504a" strokeWidth="1" />
            <rect x={0} y={0} width={widthPx} height={lenPx} fill="url(#weave)" opacity="0.5" />
            <defs>
              <pattern id="weave" width="8" height="8" patternUnits="userSpaceOnUse">
                <path d="M0 0L8 8M8 0L0 8" stroke="#4a453f" strokeWidth="0.6" />
              </pattern>
            </defs>
            {/* grid */}
            {gridLinesX.map((i) => (
              <line key={"gx" + i} x1={i * gridSize * pxPerMm} y1={0} x2={i * gridSize * pxPerMm} y2={lenPx} stroke="#4a453f" strokeWidth="0.5" opacity="0.4" />
            ))}
            {gridLinesY.map((i) => (
              <line key={"gy" + i} x1={0} y1={i * gridSize * pxPerMm} x2={widthPx} y2={i * gridSize * pxPerMm} stroke="#4a453f" strokeWidth="0.5" opacity="0.4" />
            ))}
            {/* required-length marker */}
            <line x1={0} y1={reqLength * pxPerMm} x2={widthPx} y2={reqLength * pxPerMm} stroke="#e0a527" strokeWidth="1.5" strokeDasharray="5 3" />
            {/* top ruler */}
            {metreTicks.map((m) => (
              <g key={"tm" + m}>
                <line x1={m * 1000 * pxPerMm} y1={-8} x2={m * 1000 * pxPerMm} y2={0} stroke="#e0a527" strokeWidth="1.5" />
                <text x={m * 1000 * pxPerMm + 3} y={-10} fontSize="10" fontFamily="monospace" fill="#e0a527">{m}m</text>
              </g>
            ))}
            {/* left ruler */}
            {lengthTicks.map((m) => (
              <g key={"lm" + m}>
                <line x1={-8} y1={m * 1000 * pxPerMm} x2={0} y2={m * 1000 * pxPerMm} stroke="#7a756c" strokeWidth="1" />
                <text x={-32} y={m * 1000 * pxPerMm + 3} fontSize="9" fontFamily="monospace" fill="#7a756c">{m}m</text>              </g>
            ))}

            {instances.map((inst) => {
              const room = roomsById[inst.roomId];
              if (!room) return null;
              const verts = instanceVertices(room, inst);
              const pts = verts.map((v) => `${v.x * pxPerMm},${v.y * pxPerMm}`).join(" ");
              const isSelected = selectedId === inst.id;
              const isInvalid = invalidIds.has(inst.id);
              const bb = boundingBox(verts);
              const cx = (bb.minX + bb.maxX) / 2 * pxPerMm, cy = (bb.minY + bb.maxY) / 2 * pxPerMm;
              return (
                <g key={inst.id} onMouseDown={(e) => beginDrag(e, inst)} style={{ cursor: "grab" }}>
                  <polygon
                    points={pts}
                    fill={isInvalid ? "#c9403066" : room.color + "77"}
                    stroke={isInvalid ? "#e5493a" : isSelected ? "#e0a527" : room.color}
                    strokeWidth={isSelected ? 2.5 : isInvalid ? 2 : 1.5}
                    strokeDasharray={isInvalid ? "7 4" : undefined}
                  />
                  {room.grainDirection && (
                    <line x1={cx} y1={cy + 14} x2={cx} y2={cy - 14} stroke="#f7f5f0" strokeWidth="1.5" markerEnd="url(#arrow)" transform={`rotate(${0} ${cx} ${cy})`} />
                  )}
                  <text x={cx} y={cy} fontSize="11" fontFamily="monospace" fill="#f7f5f0" textAnchor="middle" style={{ pointerEvents: "none" }}>{room.name}</text>
                  {isInvalid && (
                    <g transform={`translate(${bb.minX * pxPerMm + 4},${bb.minY * pxPerMm + 4})`} style={{ pointerEvents: "none" }}>
                      <circle r="8" fill="#e5493a" />
                      <text x="0" y="3" fontSize="10" fontWeight="bold" textAnchor="middle" fill="white">!</text>
                    </g>
                  )}
                </g>
              );
            })}
            <defs>
              <marker id="arrow" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#f7f5f0" /></marker>
            </defs>
          </g>
        </svg>
      </div>
    </div>
  );
}

/* ======================================================================
   SECTION: TOP BAR
   ====================================================================== */

function TopBar({ projectName, setProjectName, stats, onUndo, onRedo, canUndo, canRedo, onAutoArrange, arranging, onSave, view, setView, onOpenCuttingPlan, instanceCount }) {
  return (
    <div className="flex items-center gap-3 px-4 py-2.5 bg-[#242220] text-stone-200 border-b border-black/40 flex-wrap">
      <ScissorsLineDashed size={16} className="text-[#e0a527]" />
      <input value={projectName} onChange={(e) => setProjectName(e.target.value)} title="Project name" className="bg-transparent text-sm font-semibold outline-none border-b border-transparent focus:border-stone-500 w-40" />
      <div className="flex bg-[#1c1a18] rounded-xl p-0.5 text-xs ml-2">
        <button onClick={() => setView("rooms")} className={`px-3 py-1 rounded-xl ${view === "rooms" ? "bg-[#3c7a6f] text-white" : "text-stone-400"}`}>Rooms</button>
        <button onClick={() => setView("cutting")} className={`px-3 py-1 rounded-xl flex items-center gap-1.5 ${view === "cutting" ? "bg-[#3c7a6f] text-white" : "text-stone-400"}`}>
          Cutting Plan
          {instanceCount > 0 && <span className={`text-[9px] rounded-full px-1.5 ${view === "cutting" ? "bg-white/25" : "bg-white/10"}`}>{instanceCount}</span>}
        </button>
      </div>
      <div className="flex items-center gap-4 ml-4 text-[11px] font-mono text-stone-400">
        <span>Length <b className="text-[#e0a527]">{stats.length}</b></span>
        <span>Area <b className="text-stone-200">{stats.area}</b></span>
        <span>Waste <b className="text-stone-200">{stats.waste}</b></span>
      </div>
      <div className="ml-auto flex items-center gap-1.5">
        <button onClick={onUndo} disabled={!canUndo} title="Undo" className="p-2 hover:bg-white/10 rounded-full disabled:opacity-30 transition-colors"><Undo2 size={15} /></button>
        <button onClick={onRedo} disabled={!canRedo} title="Redo" className="p-2 hover:bg-white/10 rounded-full disabled:opacity-30 transition-colors"><Redo2 size={15} /></button>
        <button onClick={onAutoArrange} disabled={arranging} title="Search for an efficient layout using exact piece geometry" className="flex items-center gap-1.5 text-sm font-medium bg-[#c98a3a] text-white px-4 py-2 rounded-xl hover:bg-[#b57a2f] shadow-sm transition-colors disabled:opacity-60"><Grid3x3 size={14} className={arranging ? "animate-spin" : ""} /> {arranging ? "Optimising…" : "Auto Arrange"}</button>

        <button onClick={onOpenCuttingPlan} title="Open a clean, printable cutting plan" className="flex items-center gap-1.5 text-sm font-medium px-4 py-2 rounded-xl hover:bg-white/10 transition-colors"><Layers size={14} /> Cutting Plan</button>
        <button onClick={onSave} title="Project autosaves — click to save immediately" className="flex items-center gap-1.5 text-sm font-medium px-4 py-2 rounded-xl hover:bg-white/10 transition-colors"><Save size={14} /> Save</button>
      </div>
    </div>
  );
}

/* ======================================================================
   SECTION: CUTTING PLAN VIEW
   ====================================================================== */

/* ======================================================================
   SECTION: EXPORT HELPERS (PNG / PDF)
   A minimal, dependency-free PDF writer: no external library is available
   in this environment, so a single-page PDF is hand-built by embedding a
   JPEG snapshot of the plan directly via the DCTDecode filter, which every
   standard PDF viewer can decode natively without re-compression.
   ====================================================================== */

function safeFileName(name) {
  const s = (name || "cutting-plan").trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
  return s || "cutting-plan";
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.download = filename;
  link.href = url;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

// Rasterises an <svg> element (given its declared pixel width/height) onto
// a white-backed canvas at the requested resolution multiplier.
// Renders the cutting plan directly with the Canvas 2D API — never via an
// <img>-loaded SVG. Drawing an SVG-sourced image onto a canvas is known to
// taint it in some browsers (Firefox in particular), which makes
// canvas.toDataURL() throw even though canvas.toBlob() silently no-ops —
// so PDF export (which needs toDataURL for the JPEG) would fail exactly
// where PNG export (toBlob) might appear to work. Drawing natively avoids
// that risk entirely, for both export paths.
function renderPlanCanvas({ instances, roomsById, rollWidth, reqLength, projectName, pxPerMm, headerH, scale }) {
  const svgW = rollWidth * pxPerMm + 20;
  const svgH = reqLength * pxPerMm + 20 + headerH;
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(svgW * scale));
  canvas.height = Math.max(1, Math.round(svgH * scale));
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, svgW, svgH);

  ctx.textBaseline = "alphabetic";
  ctx.textAlign = "left";
  ctx.fillStyle = "#1c1a18";
  ctx.font = "bold 13px sans-serif";
  ctx.fillText(`${projectName} — Cutting Plan`, 10, 18);
  ctx.fillStyle = "#78716c";
  ctx.font = "9.5px monospace";
  ctx.fillText(
    `Roll ${(rollWidth / 1000).toFixed(2)}m wide \u00B7 Required length ${(reqLength / 1000).toFixed(2)}m \u00B7 ${instances.length} piece(s) \u00B7 ${new Date().toLocaleDateString()}`,
    10, 32
  );

  const offX = 10, offY = headerH + 10;
  const rollW = rollWidth * pxPerMm, rollH = reqLength * pxPerMm;
  ctx.fillStyle = "#fafaf7";
  ctx.fillRect(offX, offY, rollW, rollH);
  ctx.strokeStyle = "#333333";
  ctx.lineWidth = 1;
  ctx.strokeRect(offX, offY, rollW, rollH);

  instances.forEach((inst) => {
    const room = roomsById[inst.roomId];
    if (!room) return;
    const verts = instanceVertices(room, inst);
    if (!verts.length) return;
    ctx.beginPath();
    verts.forEach((v, i) => {
      const x = offX + v.x * pxPerMm, y = offY + v.y * pxPerMm;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = room.color + "33";
    ctx.fill();
    ctx.strokeStyle = room.color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    const bb = boundingBox(verts);
    const cx = offX + ((bb.minX + bb.maxX) / 2) * pxPerMm;
    const cy = offY + ((bb.minY + bb.maxY) / 2) * pxPerMm;
    const sizeLabel = `${formatLength(bb.maxX - bb.minX, "m")} × ${formatLength(bb.maxY - bb.minY, "m")}`;
    ctx.textAlign = "center";
    ctx.fillStyle = "#1c1a18";
    ctx.font = "bold 9px monospace";
    ctx.fillText(room.name, cx, cy - 5);
    ctx.fillStyle = "#555048";
    ctx.font = "8px monospace";
    ctx.fillText(sizeLabel, cx, cy + 7);
    ctx.textAlign = "left";
  });

  return canvas;
}

function base64ToUint8Array(base64) {
  const binStr = atob(base64);
  const bytes = new Uint8Array(binStr.length);
  for (let i = 0; i < binStr.length; i++) bytes[i] = binStr.charCodeAt(i);
  return bytes;
}

// Builds a valid, minimal single-page PDF wrapping one JPEG image, scaled
// to fill a page of pageW x pageH points. imgPxW/imgPxH are the JPEG's own
// pixel dimensions (kept higher than the page size for a crisp render).
function buildSinglePageImagePDF(jpegBytes, imgPxW, imgPxH, pageW, pageH) {
  const enc = new TextEncoder();
  const parts = [];
  let pos = 0;
  const offsets = {};
  function writeStr(s) { const b = enc.encode(s); parts.push(b); pos += b.length; }
  function writeBytes(b) { parts.push(b); pos += b.length; }
  function beginObj(n) { offsets[n] = pos; writeStr(`${n} 0 obj\n`); }
  function endObj() { writeStr("endobj\n"); }

  writeStr("%PDF-1.4\n");

  beginObj(1);
  writeStr("<< /Type /Catalog /Pages 2 0 R >>\n");
  endObj();

  beginObj(2);
  writeStr("<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n");
  endObj();

  beginObj(3);
  writeStr(`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${pageW} ${pageH}] /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>\n`);
  endObj();

  beginObj(4);
  writeStr(`<< /Type /XObject /Subtype /Image /Width ${imgPxW} /Height ${imgPxH} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${jpegBytes.length} >>\nstream\n`);
  writeBytes(jpegBytes);
  writeStr("\nendstream\n");
  endObj();

  const contentBytes = enc.encode(`q\n${pageW} 0 0 ${pageH} 0 0 cm\n/Im0 Do\nQ`);
  beginObj(5);
  writeStr(`<< /Length ${contentBytes.length} >>\nstream\n`);
  writeBytes(contentBytes);
  writeStr("\nendstream\n");
  endObj();

  const xrefStart = pos;
  writeStr("xref\n0 6\n");
  writeStr("0000000000 65535 f \n");
  for (let i = 1; i <= 5; i++) writeStr(`${String(offsets[i]).padStart(10, "0")} 00000 n \n`);
  writeStr(`trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n${xrefStart}\n%%EOF`);

  const total = parts.reduce((s, p) => s + p.length, 0);
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

function CuttingPlanModal({ onClose, rooms, roomsById, instances, rollWidth, projectName, reqLength }) {
  const pxPerMm = Math.min(0.5, 780 / rollWidth);
  const svgRef = useRef(null);
  const [exportState, setExportState] = useState("idle"); // idle | working | error
  const [pdfState, setPdfState] = useState("idle"); // idle | working | error
  const headerH = 40;
  const svgW = rollWidth * pxPerMm + 20;
  const svgH = reqLength * pxPerMm + 20 + headerH;

  async function exportAsImage() {
    setExportState("working");
    try {
      const canvas = renderPlanCanvas({ instances, roomsById, rollWidth, reqLength, projectName, pxPerMm, headerH, scale: 2 });
      canvas.toBlob((blob) => {
        if (!blob) { setExportState("error"); return; }
        downloadBlob(blob, `${safeFileName(projectName)}.png`);
        setExportState("idle");
      }, "image/png");
    } catch (err) {
      setExportState("error");
    }
  }

  async function exportAsPDF() {
    setPdfState("working");
    try {
      const scale = 2;
      const canvas = renderPlanCanvas({ instances, roomsById, rollWidth, reqLength, projectName, pxPerMm, headerH, scale });
      const jpegDataUrl = canvas.toDataURL("image/jpeg", 0.92);
      const jpegBytes = base64ToUint8Array(jpegDataUrl.split(",")[1]);
      const pdfBytes = buildSinglePageImagePDF(jpegBytes, canvas.width, canvas.height, svgW, svgH);
      downloadBlob(new Blob([pdfBytes], { type: "application/pdf" }), `${safeFileName(projectName)}.pdf`);
      setPdfState("idle");
    } catch (err) {
      setPdfState("error");
    }
  }

  return (
    <div
      className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4"
      style={{ position: "fixed", inset: 0 }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        className="bg-white rounded-xl w-full max-w-4xl shadow-2xl"
        style={{ display: "flex", flexDirection: "column", maxHeight: "88vh", height: "88vh" }}
      >
        <div className="flex items-center justify-between px-5 py-3 border-b border-stone-200" style={{ flex: "0 0 auto" }}>
          <h2 className="font-semibold text-sm">{projectName} — Cutting Plan</h2>
          <div className="flex items-center gap-3">
            <button
              onClick={exportAsPDF}
              disabled={pdfState === "working"}
              title="Download this cutting plan as a PDF"
              className="flex items-center gap-1.5 text-sm font-medium text-[#3c7a6f] px-4 py-2 rounded-xl bg-[#3c7a6f]/10 hover:bg-[#3c7a6f]/20 disabled:opacity-50 transition-colors"
            >
              <FileText size={14} /> {pdfState === "working" ? "Preparing…" : "Save as PDF"}
            </button>
            <button
              onClick={exportAsImage}
              disabled={exportState === "working"}              title="Download this cutting plan as a PNG image"
              className="flex items-center gap-1.5 text-sm font-medium bg-[#3c7a6f] text-white px-4 py-2 rounded-xl hover:bg-[#336a60] shadow-sm disabled:opacity-50 transition-colors"
            >
              <Download size={14} /> {exportState === "working" ? "Preparing…" : "Save as Image"}
            </button>
            <button onClick={onClose} title="Close" className="text-stone-500 hover:text-stone-800 hover:bg-stone-100 rounded-full p-1.5 transition-colors"><X size={18} /></button>
          </div>
        </div>
        <div className="p-5" style={{ flex: "1 1 auto", minHeight: 0, overflowY: "scroll", overflowX: "auto" }}>
          {pdfState === "error" && (
            <div className="mb-3 text-xs text-red-600 flex items-center gap-1 bg-red-50 border border-red-200 rounded-xl px-2 py-1.5">
              <AlertTriangle size={13} /> Couldn't generate the PDF in this browser. Try "Save as Image" instead, or print this view and choose "Save as PDF" in your browser's print dialog.
            </div>
          )}
          {exportState === "error" && (
            <div className="mb-3 text-xs text-red-600 flex items-center gap-1 bg-red-50 border border-red-200 rounded-xl px-2 py-1.5">
              <AlertTriangle size={13} /> Couldn't generate the image in this browser. Try a right-click → "Save image as" on the plan below instead.
            </div>
          )}
          <div className="flex gap-6 text-xs font-mono text-stone-600 mb-4">
            <span>Roll width: {(rollWidth / 1000).toFixed(2)}m</span>
            <span>Required length: {(reqLength / 1000).toFixed(2)}m</span>
            <span>Pieces: {instances.length}</span>
          </div>
          <svg ref={svgRef} width={svgW} height={svgH} viewBox={`0 0 ${svgW} ${svgH}`} style={{ maxWidth: "100%", height: "auto" }} className="border border-stone-300 block">            <rect width={svgW} height={svgH} fill="#ffffff" />
            <text x={10} y={18} fontSize="13" fontWeight="bold" fontFamily="ui-sans-serif, system-ui, sans-serif" fill="#1c1a18">{projectName} — Cutting Plan</text>
            <text x={10} y={32} fontSize="9.5" fontFamily="monospace" fill="#78716c">
              {`Roll ${(rollWidth / 1000).toFixed(2)}m wide · Required length ${(reqLength / 1000).toFixed(2)}m · ${instances.length} piece(s) · ${new Date().toLocaleDateString()}`}
            </text>
            <g transform={`translate(10,${headerH + 10})`}>
              <rect x={0} y={0} width={rollWidth * pxPerMm} height={reqLength * pxPerMm} fill="#fafaf7" stroke="#333" />
              {instances.map((inst) => {
                const room = roomsById[inst.roomId];
                if (!room) return null;
                const verts = instanceVertices(room, inst);
                const pts = verts.map((v) => `${v.x * pxPerMm},${v.y * pxPerMm}`).join(" ");
                const bb = boundingBox(verts);
                const midX = ((bb.minX + bb.maxX) / 2) * pxPerMm, midY = ((bb.minY + bb.maxY) / 2) * pxPerMm;
                return (
                  <g key={inst.id}>
                    <polygon points={pts} fill={room.color + "33"} stroke={room.color} strokeWidth="1.5" />
                    <text x={midX} y={midY - 4} fontSize="9" fontWeight="bold" textAnchor="middle" fontFamily="monospace" fill="#1c1a18">{room.name}</text>
                    <text x={midX} y={midY + 8} fontSize="8" textAnchor="middle" fontFamily="monospace" fill="#555048">
                      {formatLength(bb.maxX - bb.minX, "m")} × {formatLength(bb.maxY - bb.minY, "m")}
                    </text>
                  </g>
                );
              })}
            </g>
          </svg>
          <table className="w-full text-xs font-mono mt-5 border-t border-stone-200">
            <thead><tr className="text-stone-400 text-left"><th className="py-1.5">Room</th><th>Size</th><th>Area</th><th>Qty on carpet</th></tr></thead>
            <tbody>
              {Object.entries(
                instances.reduce((acc, inst) => {
                  if (!roomsById[inst.roomId]) return acc;
                  acc[inst.roomId] = (acc[inst.roomId] || 0) + 1;
                  return acc;
                }, {})
              ).map(([roomId, count]) => {
                const room = roomsById[roomId];
                return (
                  <tr key={roomId} className="border-t border-stone-100">
                    <td className="py-1.5">{room.name}</td>
                    <td>{formatLength(room.width, "m")} × {formatLength(room.length, "m")}</td>
                    <td>{formatArea(room.area)}</td>
                    <td>× {count}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="text-[11px] text-stone-400 mt-4 flex items-center gap-1"><Info size={12} /> Both exports are rendered client-side at 2x resolution for a crisp, printable result.</p>
        </div>
      </div>
    </div>
  );
}

/* ======================================================================
   SECTION: APP ROOT
   ====================================================================== */

export default function App() {
  const [rooms, setRooms] = useState([]);
  const [instances, setInstances] = useState([]);
  const [rollWidth, setRollWidth] = useState(4000);
  const [unit, setUnit] = useState("m");
  const [tolerance, setTolerance] = useState(0);
  const [grainMode, setGrainMode] = useState("90");
  const [gridSize, setGridSize] = useState(25);
  const [snapEnabled, setSnapEnabled] = useState(true);
  const [allowOverlap, setAllowOverlap] = useState(false);
  const [guillotineMode, setGuillotineMode] = useState(true);
  const [arranging, setArranging] = useState(false);
  const [projectName, setProjectName] = useState("Untitled Project");
  const [view, setView] = useState("rooms");
  const [selectedId, setSelectedId] = useState(null);
  const [modal, setModal] = useState(null); // { mode: 'add'|'edit', room? }
  const [showCuttingPlan, setShowCuttingPlan] = useState(false);
  const [loaded, setLoaded] = useState(false);

  const historyRef = useRef({ stack: [], index: -1 });
  const [historyTick, setHistoryTick] = useState(0);

  const roomsById = useMemo(() => Object.fromEntries(rooms.map((r) => [r.id, r])), [rooms]);

  function commitHistory(nextInstances) {
    const h = historyRef.current;
    h.stack = h.stack.slice(0, h.index + 1);
    h.stack.push(nextInstances);
    h.index = h.stack.length - 1;
    setHistoryTick((t) => t + 1);
  }
  function undo() {
    const h = historyRef.current;
    if (h.index <= 0) return;
    h.index -= 1;
    setInstances(h.stack[h.index]);
    setHistoryTick((t) => t + 1);
  }
  function redo() {
    const h = historyRef.current;
    if (h.index >= h.stack.length - 1) return;
    h.index += 1;
    setInstances(h.stack[h.index]);
    setHistoryTick((t) => t + 1);
  }

  // initial load
  useEffect(() => {
    (async () => {
      const data = await loadProject();
      if (data) {
        setRooms(data.rooms || []);
        setInstances(data.instances || []);
        setRollWidth(data.rollWidth || 4000);
        setUnit(data.unit || "mm");
        setTolerance(data.tolerance ?? 0);
        setGrainMode(data.grainMode || "90");
        setGridSize(data.gridSize || 25);
        setSnapEnabled(data.snapEnabled ?? true);
        setAllowOverlap(data.allowOverlap ?? false);
        setGuillotineMode(data.guillotineMode ?? true);
        setProjectName(data.projectName || "Untitled Project");
        historyRef.current = { stack: [data.instances || []], index: 0 };
      } else {
        historyRef.current = { stack: [[]], index: 0 };
      }
      setLoaded(true);
    })();
  }, []);

  function doSave() {
    saveProject({ rooms, instances, rollWidth, unit, tolerance, grainMode, gridSize, snapEnabled, allowOverlap, guillotineMode, projectName });
  }
  useEffect(() => {
    if (!loaded) return;
    const t = setTimeout(doSave, 700);
    return () => clearTimeout(t);
  }, [rooms, instances, rollWidth, unit, tolerance, grainMode, gridSize, snapEnabled, allowOverlap, guillotineMode, projectName, loaded]); // eslint-disable-line

  function handleSaveRoom(roomData) {
    setRooms((prev) => {
      const exists = prev.some((r) => r.id === roomData.id);
      return exists ? prev.map((r) => (r.id === roomData.id ? roomData : r)) : [...prev, roomData];
    });
    setModal(null);
  }

  function handleDeleteRoom(id) {
    setRooms((prev) => prev.filter((r) => r.id !== id));
    setInstances((prev) => {
      const next = prev.filter((i) => i.roomId !== id);
      commitHistory(next);
      return next;
    });
  }

  function handleDuplicateRoom(room) {
    const copy = { ...room, id: `room_${Date.now()}`, name: room.name + " copy" };
    setRooms((prev) => [...prev, copy]);
  }

  function handleQuantityChange(id, qty) {
    setRooms((prev) => prev.map((r) => (r.id === id ? { ...r, quantity: qty } : r)));
  }

  function rotateSelected() {
    setInstances((prev) => {
      const next = prev.map((i) => {
        if (i.id !== selectedId) return i;
        if (grainMode === "locked") return i;
        const step = 90;
        return { ...i, rotation: (i.rotation + step) % 360 };
      });
      commitHistory(next);
      return next;
    });
  }
  function deleteSelected() {
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
        onAutoArrange={runAutoArrange} arranging={arranging} onSave={doSave} view={view} setView={setView} onOpenCuttingPlan={() => setShowCuttingPlan(true)}        instanceCount={instances.length}
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