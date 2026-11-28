import { signal } from "@preact/signals";
import { useEffect, useRef, useState } from "preact/hooks";
import {
  CameraController,
  attachControllerInput,
  medianDepthAround,
  type OrbitParams,
} from "../camera";
import { assetUrl } from "../api/sessions";
import { PreviewRenderer, RendererUnsupportedError, fetchAssets } from "../gl";
import { showError } from "../state/session";
import { holdCompare, renderState, startRender } from "../state/render";
import { ViewerLogic } from "../state/viewer";

/** The live controller and pose, for the panel (reset button, pose readout, pick toggle). */
export const controllerSignal = signal<CameraController | null>(null);
export const poseSignal = signal<OrbitParams | null>(null);
export const pickModeSignal = signal(false);

interface Props {
  id: string;
  /** Original file name, for the download name. */
  filename: string;
  /** Mesh columns: 512 = 1×, 1024 = 2×. The renderer is rebuilt when it changes. */
  meshCols: number;
}

/**
 * The 3D preview: renderer + controller on one canvas. Owns every resource it creates and frees them when the
 * session or the quality changes or the component unmounts.
 */
export function Viewport({ id, filename, meshCols }: Props) {
  const nameRef = useRef(filename);
  nameRef.current = filename;
  const filenameOf = (): string => nameRef.current;
  const canvas = useRef<HTMLCanvasElement>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    let cancelled = false;
    let renderer: PreviewRenderer | null = null;
    let controller: CameraController | null = null;
    let logic: ViewerLogic | null = null;
    let keyCleanup: (() => void) | null = null;
    let detach: (() => void) | null = null;
    let observer: ResizeObserver | null = null;
    setReady(false);

    (async () => {
      try {
        renderer = new PreviewRenderer(el, { meshCols });
        const assets = await fetchAssets(assetUrl(id, "").replace(/\/$/, ""));
        if (cancelled) return;
        await renderer.loadAssets(assets);
        const r = renderer;
        const sync = (): void => {
          const fit = r.layout;
          if (fit && controller) {
            controller.setViewport(el.width, assets.meta.f_px_preview * fit.scale);
          }
        };
        controller = new CameraController({
          pivotZ: assets.meta.pivot_z,
          onChange: (orbit, camera) => {
            poseSignal.value = orbit;
            r.setCamera(camera); // requests one draw
            logic?.onChange(orbit, camera);
          },
          onCommit: (orbit) => void logic?.onCommit(orbit),
          onPickMode: (on) => {
            pickModeSignal.value = on;
            el.style.cursor = on ? "crosshair" : "grab";
          },
          depthAt: (x, y) => {
            const p = r.canvasToImage(x, y);
            return p
              ? medianDepthAround(
                  assets.depth,
                  assets.sky,
                  assets.meta.preview_width,
                  assets.meta.preview_height,
                  p[0],
                  p[1],
                )
              : null;
          },
        });
        logic = new ViewerLogic(id, controller);
        detach = attachControllerInput(controller, el);
        // Enter renders the current view; Space held compares with the last result (no scrolling).
        const c = controller;
        const onKeyDown = (e: KeyboardEvent): void => {
          if (e.ctrlKey || e.metaKey || e.altKey) return;
          if (e.key === "Enter" && !e.repeat) {
            e.preventDefault();
            void startRender(id, c.camera, c.orbit, filenameOf());
          } else if (e.key === " " && renderState.value.outcome) {
            e.preventDefault();
            holdCompare.value = true;
          }
        };
        const onKeyUp = (e: KeyboardEvent): void => {
          if (e.key === " ") holdCompare.value = false;
        };
        const onBlur = (): void => {
          holdCompare.value = false;
        };
        el.addEventListener("keydown", onKeyDown);
        el.addEventListener("keyup", onKeyUp);
        el.addEventListener("blur", onBlur);
        keyCleanup = () => {
          el.removeEventListener("keydown", onKeyDown);
          el.removeEventListener("keyup", onKeyUp);
          el.removeEventListener("blur", onBlur);
          holdCompare.value = false;
        };
        observer = new ResizeObserver(() => {
          r.resize();
          sync();
        });
        observer.observe(el);
        r.resize();
        sync();
        r.setCamera(controller.camera);
        if (new URLSearchParams(location.search).has("e2e")) {
          // Test hook (browser tests only, opt-in by URL): lets them drive and read the live viewer.
          (window as unknown as { __nvs?: unknown }).__nvs = {
            renderer: r,
            controller,
            canvas: el,
          };
        }
        controllerSignal.value = controller;
        poseSignal.value = controller.orbit;
        pickModeSignal.value = false;
        setReady(true);
      } catch (e) {
        if (cancelled) return;
        if (e instanceof RendererUnsupportedError) showError("webgl_unsupported", 0, e.message, id);
        else showError("preview_failed", 0, e instanceof Error ? e.message : String(e), id);
      }
    })();

    return () => {
      cancelled = true;
      detach?.();
      keyCleanup?.();
      delete (window as unknown as { __nvs?: unknown }).__nvs;
      logic?.dispose();
      observer?.disconnect();
      controller?.dispose();
      renderer?.dispose();
      controllerSignal.value = null;
      poseSignal.value = null;
    };
  }, [id, meshCols]);

  return (
    <canvas
      ref={canvas}
      class={`viewport-canvas${ready ? " ready" : ""}`}
      role="application"
      aria-roledescription="3D preview"
      aria-label="3D preview of your photo. Drag to orbit, scroll to zoom, double-click to reset, arrow keys to orbit."
    />
  );
}
