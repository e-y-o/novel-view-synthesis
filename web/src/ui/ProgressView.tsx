import { PHASES, type Phase } from "../state/session";

interface Props {
  phase: Phase;
  /** Progress inside the current phase, 0..100. */
  pct: number;
}

/** Named phases of upload and analysis; the current one has a bar, earlier ones a check mark. */
export function ProgressView({ phase, pct }: Props) {
  const current = PHASES.findIndex((p) => p.id === phase);
  const total = Math.min(100, Math.max(0, pct));
  return (
    <ol class="phases" aria-label="Progress">
      {PHASES.map((p, i) => {
        const status = i < current ? "done" : i === current ? "active" : "todo";
        return (
          <li
            key={p.id}
            class={`phase ${status}`}
            aria-current={status === "active" ? "step" : undefined}
          >
            <span class="phase-mark" aria-hidden="true">
              {status === "done" ? "✓" : status === "active" ? "•" : ""}
            </span>
            <span class="phase-label">{p.label}</span>
            {status === "active" && (
              <span
                class="bar"
                role="progressbar"
                aria-label={p.label}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(total)}
              >
                <span class="bar-fill" style={{ width: `${total}%` }} />
              </span>
            )}
          </li>
        );
      })}
    </ol>
  );
}
