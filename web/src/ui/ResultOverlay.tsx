import { holdCompare, renderState, showResult } from "../state/render";

/**
 * The finished render over the viewport. It ignores the pointer and takes no focus, so the camera controls under
 * it stay reachable by mouse and keyboard; moving the camera marks the result outdated (see the panel).
 */
export function ResultOverlay() {
  const out = renderState.value.outcome;
  if (!out || !(showResult.value || holdCompare.value)) return null;
  const transparent = out.options.hole === "transparent"; // the option the shown image was made with
  return (
    <div class={`result-overlay${transparent ? " checker" : ""}`} aria-hidden="false">
      <img
        src={out.result.url}
        alt={`Rendered view, ${out.result.width} by ${out.result.height} pixels`}
        width={out.result.width}
        height={out.result.height}
      />
    </div>
  );
}
