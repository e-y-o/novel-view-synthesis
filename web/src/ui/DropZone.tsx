import { useRef, useState } from "preact/hooks";
import { ACCEPT, FORMATS_TEXT, validateFile } from "./helpers";

interface Props {
  onFile: (file: File) => void;
  /** Called with a user-readable reason when a dropped or chosen file is not accepted. */
  onReject: (message: string) => void;
}

/** Empty state: drop a photo, choose one, or paste one (the paste handler lives in the app). */
export function DropZone({ onFile, onReject }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const take = (files: FileList | null | undefined): void => {
    const file = files?.[0];
    if (!file) return;
    const check = validateFile(file);
    if (check.ok) onFile(file);
    else onReject(check.message);
  };
  const choose = (): void => input.current?.click();

  return (
    <div
      class={`dropzone${over ? " over" : ""}`}
      role="button"
      tabIndex={0}
      aria-label={`Drop a photo here, or press Enter to choose one. ${FORMATS_TEXT}.`}
      onClick={choose}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          choose();
        }
      }}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        take(e.dataTransfer?.files);
      }}
    >
      <svg class="dz-icon" viewBox="0 0 48 48" width="48" height="48" aria-hidden="true">
        <rect
          x="6"
          y="10"
          width="36"
          height="28"
          rx="4"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
        />
        <circle cx="17" cy="20" r="3" fill="currentColor" />
        <path d="M8 36l11-10 7 6 6-5 8 9" fill="none" stroke="currentColor" stroke-width="2" />
      </svg>
      <p class="dz-title">Drop a photo here</p>
      <p class="muted">or</p>
      <button
        type="button"
        class="btn primary"
        onClick={(e) => {
          e.stopPropagation();
          choose();
        }}
      >
        Choose file
      </button>
      <p class="muted small">
        {FORMATS_TEXT}. You can also paste an image. Your photo never leaves this computer.
      </p>
      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        hidden
        data-testid="file-input"
        onChange={(e) => {
          take(e.currentTarget.files);
          e.currentTarget.value = ""; // allow choosing the same file again
        }}
      />
    </div>
  );
}
