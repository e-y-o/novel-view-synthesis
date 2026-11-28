import { useEffect } from "preact/hooks";
import { health, refreshHealth } from "./state/app";

/** Placeholder shell: title and server health. The real UI arrives with T12. */
export function App() {
  useEffect(() => {
    void refreshHealth();
  }, []);

  const state = health.value;
  return (
    <main class="shell">
      <h1>Novel View Synthesis</h1>
      <p class="muted" data-testid="health" role="status">
        {state.kind === "loading" && "Connecting to the server…"}
        {state.kind === "ok" &&
          `Server OK · version ${state.health.version}${state.health.stub ? " · stub (no GPU)" : ""}`}
        {state.kind === "error" && `Server not reachable: ${state.message}`}
      </p>
    </main>
  );
}
