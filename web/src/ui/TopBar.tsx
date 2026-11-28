import type { PreviewMeta } from "../gl/assets";
import type { WorkerState } from "../state/session";
import { exifChip } from "./helpers";

/** Focal length chip: calm when it comes from the file, amber when it is a guess. */
export function ExifChipView({ meta }: { meta: PreviewMeta }) {
  const chip = exifChip(meta);
  const extra = [...chip.details, ...(meta.warnings ?? [])];
  return (
    <span class={`chip ${chip.tone}`} title={extra.join("\n") || undefined} data-testid="exif-chip">
      {chip.tone === "warn" ? "⚠ " : ""}
      {chip.text}
      {extra.length > 0 && (
        <details class="chip-more">
          <summary aria-label="Show warnings">ⓘ</summary>
          <ul>
            {extra.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </details>
      )}
    </span>
  );
}

const WORKER_TEXT: Record<WorkerState["state"], string> = {
  starting: "Depth model loading",
  ready: "Depth model ready",
  error: "Depth model error",
  unknown: "Server status unknown",
};

export function WorkerDot({ worker }: { worker: WorkerState }) {
  const text = WORKER_TEXT[worker.state];
  return (
    <span class={`dot ${worker.state}`} role="img" aria-label={text} title={worker.detail || text}>
      <span class="dot-light" aria-hidden="true" />
      <span class="dot-text">{text}</span>
    </span>
  );
}

interface Props {
  filename: string;
  meta: PreviewMeta | null;
  worker: WorkerState;
  onHome: () => void;
}

export function TopBar({ filename, meta, worker, onHome }: Props) {
  return (
    <header class="topbar">
      <button
        type="button"
        class="brand"
        onClick={onHome}
        aria-label="Novel View Synthesis: back to start"
      >
        Novel View Synthesis
      </button>
      <span class="filename" title={filename}>
        {filename}
      </span>
      {meta && <ExifChipView meta={meta} />}
      <span class="spacer" />
      <WorkerDot worker={worker} />
    </header>
  );
}
