import type { AppError } from "../state/session";

interface Props {
  error: AppError;
  onRetry?: () => void;
  onChoose?: () => void;
  onDismiss: () => void;
}

/** Inline error with a specific message, the next step, and "Copy details". Never a blank screen. */
export function Banner({ error, onRetry, onChoose, onDismiss }: Props) {
  const copy = (): void => {
    void navigator.clipboard?.writeText(`${error.title}\n${error.message}\n\n${error.details}`);
  };
  return (
    <div class="banner error" role="alert">
      <div class="banner-text">
        <strong>{error.title}</strong>
        <span>{error.message}</span>
      </div>
      <div class="banner-actions">
        {error.action === "retry" && onRetry && (
          <button type="button" class="btn primary" onClick={onRetry}>
            Retry
          </button>
        )}
        {error.action === "choose" && onChoose && (
          <button type="button" class="btn primary" onClick={onChoose}>
            Choose another photo
          </button>
        )}
        <button type="button" class="btn" onClick={copy}>
          Copy details
        </button>
        <button type="button" class="btn" onClick={onDismiss} aria-label="Dismiss">
          Dismiss
        </button>
      </div>
    </div>
  );
}
