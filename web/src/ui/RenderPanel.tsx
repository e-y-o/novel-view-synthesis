import { poseKey, isStale, downloadName, type HoleMode } from "../api/render";
import type { OrbitParams } from "../camera";
import { announcement } from "../state/session";
import {
  cancelRender,
  elapsed,
  renderOptions,
  renderState,
  showResult,
  startRender,
} from "../state/render";
import type { CameraController } from "../camera";

interface Props {
  sessionId: string | null;
  filename: string;
  controller: CameraController | null;
  pose: OrbitParams | null;
}

const HOLES: { value: HoleMode; label: string }[] = [
  { value: "transparent", label: "Transparent" },
  { value: "magenta", label: "Magenta" },
  { value: "black", label: "Black" },
];

const pct = (f: number): string => `${(100 * f).toFixed(f < 0.1 ? 1 : 0)} %`;

/** Render the current view at the photo's resolution, compare with the preview, download or copy it. */
export function RenderPanel({ sessionId, filename, controller, pose }: Props) {
  const s = renderState.value;
  const options = renderOptions.value;
  const running = s.status === "running";
  const ready = sessionId !== null && controller !== null;
  const out = s.outcome;
  const stale = out && pose ? isStale(out.key, poseKey(pose, options)) : false;

  const render = (): void => {
    if (!sessionId || !controller) return;
    void startRender(sessionId, controller.camera, controller.orbit, filename);
  };
  const canCopy =
    typeof ClipboardItem !== "undefined" && typeof navigator.clipboard?.write === "function";
  const copy = async (): Promise<void> => {
    if (!out) return;
    try {
      const blob = await (await fetch(out.result.url)).blob();
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
      announcement.value = "Image copied to the clipboard.";
    } catch {
      announcement.value = "Could not copy the image. Use Download instead.";
    }
  };

  return (
    <section class="render" aria-label="Render">
      <h2>Render</h2>
      <div class="row">
        <label class="field compact">
          <span>Holes</span>
          <select
            aria-label="How holes are marked"
            value={options.hole}
            disabled={running}
            onChange={(e) =>
              (renderOptions.value = { ...options, hole: e.currentTarget.value as HoleMode })
            }
          >
            {HOLES.map((h) => (
              <option key={h.value} value={h.value}>
                {h.label}
              </option>
            ))}
          </select>
        </label>
        <label class="check">
          <input
            type="checkbox"
            checked={options.fillSpecks}
            disabled={running}
            onChange={(e) =>
              (renderOptions.value = { ...options, fillSpecks: e.currentTarget.checked })
            }
          />
          <span>Fill tiny specks</span>
        </label>
      </div>
      {running ? (
        <div class="row render-running" role="status">
          <span class="spinner" aria-hidden="true" />
          <span data-testid="render-elapsed">Rendering… {elapsed.value.toFixed(1)} s</span>
          <button type="button" class="btn" onClick={cancelRender}>
            Cancel
          </button>
        </div>
      ) : (
        <button
          type="button"
          class="btn primary wide"
          disabled={!ready}
          onClick={render}
          title="Renders the current view at the photo's full resolution (Enter while the preview is focused)"
        >
          {out ? "Render again" : "Render full resolution"}
        </button>
      )}
      <p class="muted small">
        About 1 to 3 s for a 12 MP photo. Holes are marked, not filled (infill is coming).
      </p>

      {s.error && (
        <div class="inline-error" role="alert">
          <strong>{s.error.title}</strong>
          <span>{s.error.message}</span>
          {s.error.action === "retry" && (
            <button type="button" class="btn" onClick={render}>
              Retry
            </button>
          )}
        </div>
      )}

      {out && (
        <div class="result" data-testid="render-result">
          <div class="row" role="group" aria-label="What to show">
            <button
              type="button"
              class="btn"
              aria-pressed={!showResult.value}
              onClick={() => (showResult.value = false)}
            >
              Preview
            </button>
            <button
              type="button"
              class="btn"
              aria-pressed={showResult.value}
              onClick={() => (showResult.value = true)}
            >
              Result
            </button>
            {stale && (
              <span class="badge warn" data-testid="stale-badge">
                outdated: view changed
              </span>
            )}
          </div>
          <p class="small muted" data-testid="render-info">
            {out.result.width} × {out.result.height} px · {pct(out.result.coverage.total)} holes
            {out.result.ms.total !== undefined && ` · ${Math.round(out.result.ms.total)} ms`}
          </p>
          <div class="row">
            <a
              class="btn"
              data-testid="download"
              href={`${out.result.url}?download=1&filename=${encodeURIComponent(downloadName(out.filename, out.orbit))}`}
              download={downloadName(out.filename, out.orbit)}
            >
              Download PNG
            </a>
            {canCopy && (
              <button type="button" class="btn" onClick={() => void copy()}>
                Copy image
              </button>
            )}
          </div>
          <p class="muted small">Hold Space on the preview to compare quickly.</p>
        </div>
      )}
    </section>
  );
}
