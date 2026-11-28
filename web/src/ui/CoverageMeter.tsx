import type { Coverage } from "../api/coverage";
import { CALM_MAX, WARN_MAX, coverageLevel, countedFraction, type Level } from "../state/viewer";

interface Props {
  coverage: Coverage | null;
  /** The pose changed and the newest value is on its way: show the old one dimmed. */
  stale: boolean;
  /** Whether the area outside the photo counts toward the quality level (mirrors the limit setting). */
  countOutOfFrame: boolean;
}

/** The scale of the bar: it spans 0 to this fraction of the view (a bit beyond the "large" threshold). */
export const METER_MAX = 0.4;

const LEVEL_TEXT: Record<Level, string> = {
  calm: "Looks natural",
  warn: "Quality may drop",
  danger: "Large areas will be invented",
};

const pct = (f: number): string => `${(100 * f).toFixed(f < 0.1 ? 1 : 0)} %`;
const width = (f: number): string => `${Math.min(100, (100 * f) / METER_MAX)}%`;

/**
 * How much of the new view will be filled in. Segments are told apart by pattern and legend as well as by
 * colour: disocclusion (solid red), stretched surfaces (green stripes), outside the photo (blue dots).
 */
export function CoverageMeter({ coverage, stale, countOutOfFrame }: Props) {
  if (!coverage) {
    return (
      <section class="meter" aria-label="Hole coverage">
        <h2>Coverage</h2>
        <p class="muted small">Move the view to see how much will be filled in.</p>
        <div class="meter-track empty" aria-hidden="true" />
      </section>
    );
  }
  const counted = countedFraction(coverage, countOutOfFrame);
  const level = coverageLevel(counted);
  const segments = [
    { key: "disocclusion", label: "Hidden behind objects", value: coverage.disocclusion },
    { key: "stretch", label: "Stretched surfaces", value: coverage.stretch },
    { key: "out_of_frame", label: "Outside the photo", value: coverage.out_of_frame },
  ];
  return (
    <section class={`meter ${level}${stale ? " stale" : ""}`} aria-label="Hole coverage">
      <h2>Coverage</h2>
      <p class="meter-total" data-testid="meter-total">
        <strong>{pct(coverage.total)}</strong> will be filled
      </p>
      <p class={`meter-level ${level}`} data-testid="meter-level">
        {LEVEL_TEXT[level]}
        {!countOutOfFrame && coverage.out_of_frame > 0.0005 && (
          <span class="muted small"> (area outside the photo is not counted)</span>
        )}
      </p>
      <div
        class="meter-track"
        role="meter"
        aria-label="Share of the view that will be filled in"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(100 * coverage.total)}
        aria-valuetext={`${pct(coverage.total)}. ${LEVEL_TEXT[level]}`}
      >
        {segments.map((s) => (
          <span key={s.key} class={`seg ${s.key}`} style={{ width: width(s.value) }} />
        ))}
        <span class="tick" style={{ left: width(CALM_MAX) }} aria-hidden="true" />
        <span class="tick" style={{ left: width(WARN_MAX) }} aria-hidden="true" />
      </div>
      <ul class="legend small">
        {segments.map((s) => (
          <li key={s.key}>
            <span class={`swatch ${s.key}`} aria-hidden="true" />
            {s.label}: {pct(s.value)}
          </li>
        ))}
      </ul>
    </section>
  );
}
