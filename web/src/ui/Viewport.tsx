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

/** The live controller and pose, for the panel (reset button, pose readout, pick toggle). */
export const controllerSignal = signal<CameraController | null>(null);
export const poseSignal = signal<OrbitParams | null>(null);
export const pickModeSignal = signal(false);

interface Props {
  id: string;
  /** Mesh columns: 512 = 1×, 1024 = 2×. The renderer is rebuilt when it changes. */
  meshCols: number;
}

/**
 * The 3D preview: renderer + controller on one canvas. Owns every resource it creates and frees them when the
 * session or the quality changes or the component unmounts.
 */
export function Viewport({ id, meshCols }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    let cancelled = false;
    let renderer: PreviewRenderer | null = null;
    let controller: CameraController | null = null;
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
          },
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
        detach = attachControllerInput(controller, el);
        observer = new ResizeObserver(() => {
          r.resize();
          sync();
        });
        observer.observe(el);
        r.resize();
        sync();
        r.setCamera(controller.camera);
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
