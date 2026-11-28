import type { OrbitParams } from "../camera";
import type { CameraController } from "../camera";

interface Props {
  controller: CameraController | null;
  pose: OrbitParams | null;
  pickMode: boolean;
  quality: 1 | 2;
  onQuality: (q: 1 | 2) => void;
}

const fmt = (v: number, unit: string): string =>
  `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}${unit}`;

/** Right-hand controls. T13 (coverage meter) and T14 (render) mount into the marked slots. */
export function Panel({ controller, pose, pickMode, quality, onQuality }: Props) {
  const disabled = controller === null;
  return (
    <aside class="panel" aria-label="Controls">
      <section>
        <h2>View</h2>
        <div class="row">
          <button type="button" class="btn" disabled={disabled} onClick={() => controller?.reset()}>
            Reset view
          </button>
          <button
            type="button"
            class="btn"
            disabled={disabled}
            aria-pressed={pickMode}
            onClick={() => controller?.togglePickMode()}
            title="Click a point to orbit around its depth (alt + double-click does the same)"
          >
            Pick pivot
          </button>
        </div>
        <label class="field">
          <span>Mesh quality</span>
          <select
            value={quality}
            onChange={(e) => onQuality(e.currentTarget.value === "1" ? 1 : 2)}
            aria-label="Mesh quality"
          >
            <option value="1">1× (fastest)</option>
            <option value="2">2× (sharper)</option>
          </select>
        </label>
        <p class="pose muted small" data-testid="pose" aria-label="Current view">
          {pose
            ? `yaw ${fmt(-pose.yawDeg, "°")} · pitch ${fmt(pose.pitchDeg, "°")} · dolly ${fmt(pose.dolly, " m")}`
            : "yaw — · pitch — · dolly —"}
        </p>
        <p class="muted small">
          Drag to orbit · right-drag or shift-drag to pan · scroll to zoom · double-click to reset
        </p>
      </section>
      {/* T13: coverage meter mounts here */}
      <section id="coverage-slot" aria-label="Coverage" />
      {/* T14: render button mounts here */}
      <section id="render-slot" aria-label="Render" />
    </aside>
  );
}
