import type { SessionSummary } from "../api/sessions";

interface Props {
  items: SessionSummary[];
  currentId: string | null;
  onOpen: (id: string) => void;
  onDelete: (id: string) => void;
}

/** Past photos: reopen or delete. Thumbnails exist for analysed sessions. */
export function RecentList({ items, currentId, onOpen, onDelete }: Props) {
  if (items.length === 0) return null;
  return (
    <section class="recent" aria-labelledby="recent-h">
      <h2 id="recent-h">Recent</h2>
      <ul>
        {items.map((s) => (
          <li key={s.id} class={s.id === currentId ? "current" : undefined}>
            <button
              type="button"
              class="recent-open"
              onClick={() => onOpen(s.id)}
              aria-label={`Open ${s.filename}`}
              aria-current={s.id === currentId ? "true" : undefined}
            >
              {s.thumbnail ? (
                <img src={s.thumbnail} alt="" width="56" height="42" loading="lazy" />
              ) : (
                <span class="thumb-empty" aria-hidden="true" />
              )}
              <span class="recent-name">{s.filename}</span>
              {s.state !== "ready" && <span class="muted small">{s.state}</span>}
            </button>
            <button
              type="button"
              class="btn icon"
              onClick={() => onDelete(s.id)}
              aria-label={`Delete ${s.filename}`}
              title="Delete"
            >
              ×
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
