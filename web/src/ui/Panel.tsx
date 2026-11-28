import type { CameraController, OrbitParams } from "../camera";
import {
  MAX_LIMIT,
  MIN_LIMIT,
  coverageView,
  limitSettings,
  updateLimitSettings,
} from "../state/viewer";
import { CoverageMeter } from "./CoverageMeter";

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
      <CoverageMeter
        coverage={coverageView.value.value}
        stale={coverageView.value.stale}
        countOutOfFrame={limitSettings.value.countOutOfFrame}
      />
      <LimitSettingsView />
      {/* T14: render button mounts here */}
      <section id="render-slot" aria-label="Render" />
    </aside>
  );
}

/** "Limit moves" toggle, the hole-fraction slider and the out-of-frame option. */
function LimitSettingsView() {
  const s = limitSettings.value;
  return (
    <section aria-label="Move limit" class="limit">
      <label
        class="check"
        title="The view resists moves that would leave more than this share of holes, and eases back when you let go."
      >
        <input
          type="checkbox"
          checked={s.enabled}
          onChange={(e) => updateLimitSettings({ enabled: e.currentTarget.checked })}
        />
        <span>Limit moves</span>
      </label>
      <label class="field slider">
        <span>Largest hole share</span>
        <input
          type="range"
          min={MIN_LIMIT * 100}
          max={MAX_LIMIT * 100}
          step={1}
          value={Math.round(s.maxFraction * 100)}
          disabled={!s.enabled}
          aria-label="Largest share of holes allowed"
          aria-valuetext={`${Math.round(s.maxFraction * 100)} percent`}
          onInput={(e) => updateLimitSettings({ maxFraction: Number(e.currentTarget.value) / 100 })}
        />
        <output>{Math.round(s.maxFraction * 100)} %</output>
      </label>
      <label
        class="check"
        title="Off by default: infill cannot invent a good border, and a telephoto photo would otherwise hit the limit on that area alone."
      >
        <input
          type="checkbox"
          checked={s.countOutOfFrame}
          disabled={!s.enabled}
          onChange={(e) => updateLimitSettings({ countOutOfFrame: e.currentTarget.checked })}
        />
        <span>Count area outside the photo</span>
      </label>
    </section>
  );
}
