/**
 * DOM wiring for `CameraController`: pointer, wheel, keyboard and double-click on a canvas.
 *
 * - left drag: orbit; right or middle drag, or shift + left drag: pan; wheel (and ctrl + wheel, which is how
 *   browsers report a trackpad pinch): dolly; two pointers: pan with the centroid and dolly with the spread.
 * - double-click: animated reset; alt + double-click (or a click while "pick pivot" mode is on): the pivot
 *   becomes the depth under the cursor.
 * - keyboard when the element has focus: arrows orbit (shift = 5°), `+` / `-` dolly, `0` / Home reset, `p`
 *   toggles pick mode.
 * All coordinates passed to the controller are canvas (device) pixels: CSS pixels × `pixelRatio()`.
 * Every listener is removed by the returned `detach()`; nothing is attached to `window`.
 */
import type { CameraController } from "./controller";

/** The few element members used, so tests can supply a fake. */
export interface InputTarget {
  addEventListener(
    type: string,
    listener: EventListener,
    options?: AddEventListenerOptions | boolean,
  ): void;
  removeEventListener(
    type: string,
    listener: EventListener,
    options?: EventListenerOptions | boolean,
  ): void;
  setPointerCapture?(id: number): void;
  releasePointerCapture?(id: number): void;
  getBoundingClientRect(): { left: number; top: number; width: number; height: number };
  style: { cursor: string; touchAction?: string; userSelect?: string; outlineOffset?: string };
  tabIndex: number;
  focus?(): void;
}

export interface InputOptions {
  /** Device pixels per CSS pixel. Default: `devicePixelRatio`. */
  pixelRatio?: () => number;
}

interface PointerLike {
  pointerId: number;
  clientX: number;
  clientY: number;
  button: number;
  buttons?: number;
  shiftKey: boolean;
  altKey?: boolean;
  preventDefault?: () => void;
}
interface WheelLike {
  deltaY: number;
  deltaMode?: number;
  ctrlKey?: boolean;
  preventDefault: () => void;
}
interface KeyLike {
  key: string;
  shiftKey: boolean;
  ctrlKey?: boolean;
  metaKey?: boolean;
  altKey?: boolean;
  preventDefault: () => void;
}
interface MouseLike {
  clientX: number;
  clientY: number;
  altKey: boolean;
}

type Mode = "orbit" | "pan" | "pinch" | "none";

export function attachControllerInput(
  controller: CameraController,
  el: InputTarget,
  options: InputOptions = {},
): () => void {
  const ratio =
    options.pixelRatio ??
    ((): number => (typeof devicePixelRatio === "number" ? devicePixelRatio : 1));
  const pointers = new Map<number, { x: number; y: number }>();
  let mode: Mode = "none";
  let moved = 0;
  let last = { x: 0, y: 0 };
  let lastSpread = 0;
  let lastCentre = { x: 0, y: 0 };

  el.tabIndex = 0;
  el.style.touchAction = "none";
  el.style.userSelect = "none";
  const idleCursor = (): string => (controller.pickMode ? "crosshair" : "grab");
  const setCursor = (c: string): void => {
    el.style.cursor = c;
  };
  setCursor(idleCursor());

  const local = (e: { clientX: number; clientY: number }): { x: number; y: number } => {
    const r = el.getBoundingClientRect();
    const k = ratio();
    return { x: (e.clientX - r.left) * k, y: (e.clientY - r.top) * k };
  };
  const spreadAndCentre = (): { spread: number; centre: { x: number; y: number } } => {
    const [a, b] = [...pointers.values()];
    if (!a || !b) return { spread: 0, centre: last };
    return {
      spread: Math.hypot(a.x - b.x, a.y - b.y),
      centre: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 },
    };
  };

  const onDown = (e: PointerLike): void => {
    const p = local(e);
    pointers.set(e.pointerId, p);
    el.setPointerCapture?.(e.pointerId);
    el.focus?.();
    controller.beginGesture();
    moved = 0;
    last = p;
    if (pointers.size >= 2) {
      mode = "pinch";
      const s = spreadAndCentre();
      lastSpread = s.spread;
      lastCentre = s.centre;
      setCursor("move");
    } else if (e.button === 2 || e.button === 1 || e.shiftKey) {
      mode = "pan";
      setCursor("move");
    } else if (e.button === 0) {
      mode = "orbit";
      setCursor(controller.pickMode ? "crosshair" : "grabbing");
    }
    e.preventDefault?.();
  };

  const onMove = (e: PointerLike): void => {
    if (!pointers.has(e.pointerId)) return;
    const p = local(e);
    pointers.set(e.pointerId, p);
    if (mode === "pinch" && pointers.size >= 2) {
      const s = spreadAndCentre();
      if (lastSpread > 0 && s.spread > 0) controller.pinch(s.spread / lastSpread);
      controller.panDrag(s.centre.x - lastCentre.x, s.centre.y - lastCentre.y);
      lastSpread = s.spread;
      lastCentre = s.centre;
      return;
    }
    const dx = p.x - last.x;
    const dy = p.y - last.y;
    last = p;
    moved += Math.abs(dx) + Math.abs(dy);
    if (mode === "orbit" && !controller.pickMode) controller.orbitDrag(dx, dy);
    else if (mode === "pan") controller.panDrag(dx, dy);
  };

  const onUp = (e: PointerLike): void => {
    if (!pointers.has(e.pointerId)) return;
    const wasPinch = mode === "pinch";
    pointers.delete(e.pointerId);
    el.releasePointerCapture?.(e.pointerId);
    if (pointers.size > 0) {
      // one finger left after a pinch: carry on as a pan without a jump
      const only = [...pointers.values()][0];
      if (only) last = only;
      mode = "pan";
      return;
    }
    if (mode === "orbit") {
      if (controller.pickMode && moved < 4) controller.pickPivotAt(last.x, last.y);
      else controller.endOrbitGesture();
    } else if (mode === "pan" || wasPinch) {
      controller.endGesture();
    }
    mode = "none";
    setCursor(idleCursor());
  };

  const onWheel = (e: WheelLike): void => {
    e.preventDefault();
    const unit = e.deltaMode === 1 ? 33 : e.deltaMode === 2 ? 400 : 1; // lines, pages -> pixels
    controller.beginGesture();
    controller.wheel(e.deltaY * unit * (e.ctrlKey ? 3 : 1)); // pinch gestures arrive as small ctrl+wheel deltas
    controller.endGesture();
  };

  const onDblClick = (e: MouseLike): void => {
    if (e.altKey) {
      const p = local(e);
      controller.pickPivotAt(p.x, p.y);
    } else {
      controller.reset();
    }
  };

  const onKey = (e: KeyLike): void => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (controller.key(e.key, e.shiftKey)) {
      e.preventDefault();
      setCursor(idleCursor());
    }
  };

  const onContextMenu = (e: { preventDefault: () => void }): void => e.preventDefault(); // right-drag pans

  const listeners: [string, EventListener, AddEventListenerOptions | undefined][] = [
    ["pointerdown", onDown as unknown as EventListener, undefined],
    ["pointermove", onMove as unknown as EventListener, undefined],
    ["pointerup", onUp as unknown as EventListener, undefined],
    ["pointercancel", onUp as unknown as EventListener, undefined],
    ["wheel", onWheel as unknown as EventListener, { passive: false }],
    ["dblclick", onDblClick as unknown as EventListener, undefined],
    ["keydown", onKey as unknown as EventListener, undefined],
    ["contextmenu", onContextMenu as unknown as EventListener, undefined],
  ];
  for (const [type, fn, opts] of listeners) el.addEventListener(type, fn, opts);

  return () => {
    for (const [type, fn, opts] of listeners) el.removeEventListener(type, fn, opts);
    pointers.clear();
    mode = "none";
  };
}
