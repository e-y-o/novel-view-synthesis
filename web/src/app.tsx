import { useEffect } from "preact/hooks";
import {
  announcement,
  appError,
  dismissError,
  goHome,
  loadRecent,
  meshQuality,
  openSession,
  recent,
  removeSession,
  retryUpload,
  showError,
  startGlobalEvents,
  startUpload,
  view,
  worker,
  clearBlur,
} from "./state/session";
import { Banner } from "./ui/Banner";
import { DropZone } from "./ui/DropZone";
import { Panel } from "./ui/Panel";
import { ProgressView } from "./ui/ProgressView";
import { RecentList } from "./ui/RecentList";
import { TopBar } from "./ui/TopBar";
import { ResultOverlay } from "./ui/ResultOverlay";
import { Viewport, controllerSignal, pickModeSignal, poseSignal } from "./ui/Viewport";
import { resetRender } from "./state/render";
import { firstImageFile, parseHash } from "./ui/helpers";

function applyHash(): void {
  const route = parseHash(location.hash);
  if (route.kind === "session") void openSession(route.id);
  else if (view.value.state !== "idle") goHome();
}

/** Upload, progress and viewer shell. Panel slots for T13 (coverage) and T14 (render) are in `Panel`. */
export function App() {
  useEffect(() => {
    const stopEvents = startGlobalEvents();
    void loadRecent();
    applyHash();
    const onHash = (): void => applyHash();
    const onPaste = (e: ClipboardEvent): void => {
      const file = firstImageFile(e.clipboardData?.files);
      if (file) {
        e.preventDefault();
        void startUpload(file);
      }
    };
    window.addEventListener("hashchange", onHash);
    window.addEventListener("paste", onPaste);
    return () => {
      stopEvents();
      window.removeEventListener("hashchange", onHash);
      window.removeEventListener("paste", onPaste);
    };
  }, []);

  const v = view.value;
  const error = appError.value;
  const w = worker.value;
  const ready = v.state === "ready" && v.id !== null;

  // The user's own photo stays blurred behind the viewer while it fades in, then is freed.
  useEffect(() => {
    if (!ready || !v.blurUrl) return;
    const t = window.setTimeout(clearBlur, 600);
    return () => window.clearTimeout(t);
  }, [ready, v.blurUrl]);

  // a result belongs to one session
  useEffect(() => {
    resetRender();
  }, [v.id]);

  const choose = (): void => {
    dismissError();
    goHome();
  };

  return (
    <div class="app">
      <TopBar filename={v.filename} meta={v.meta} worker={w} onHome={goHome} />
      <div>
        {error && (
          <Banner error={error} onRetry={retryUpload} onChoose={choose} onDismiss={dismissError} />
        )}
        {w.state === "starting" && (
          <div class="banner info" role="status">
            <span>
              Loading the depth model… the first start takes a few seconds. You can already add a
              photo.
            </span>
          </div>
        )}
      </div>
      <div class="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {announcement.value}
      </div>
      <main class="main">
        <div class="viewport-pane">
          {v.state === "idle" && (
            <div class="empty">
              <DropZone
                onFile={(f) => void startUpload(f)}
                onReject={(msg) => showError("unsupported_type", 0, msg)}
              />
            </div>
          )}
          {v.blurUrl && <img class="blur" src={v.blurUrl} alt="" />}
          {(v.state === "uploading" || v.state === "analyzing") && (
            <div class="progress">
              <div class="progress-card">
                <h2>{v.filename}</h2>
                <ProgressView phase={v.state === "uploading" ? "upload" : v.phase} pct={v.pct} />
              </div>
            </div>
          )}
          {ready && <ResultOverlay />}
          {ready && v.id && (
            <Viewport
              id={v.id}
              filename={v.filename}
              meshCols={meshQuality.value === 1 ? 512 : 1024}
            />
          )}
        </div>
        <div class="side">
          {ready ? (
            <Panel
              sessionId={v.id}
              filename={v.filename}
              controller={controllerSignal.value}
              pose={poseSignal.value}
              pickMode={pickModeSignal.value}
              quality={meshQuality.value}
              onQuality={(q) => (meshQuality.value = q)}
            />
          ) : (
            <aside class="panel" aria-label="Controls">
              <p class="muted small">Add a photo to start. The controls appear here.</p>
            </aside>
          )}
          <div class="panel recent-panel">
            <RecentList
              items={recent.value}
              currentId={v.id}
              onOpen={(id) => void openSession(id)}
              onDelete={(id) => void removeSession(id)}
            />
          </div>
        </div>
      </main>
    </div>
  );
}
