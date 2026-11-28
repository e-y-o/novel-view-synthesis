// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/preact";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  PHASES,
  announcement,
  appError,
  handleSessionEvent,
  showError,
  view,
  worker,
} from "../src/state/session";
import type { SessionState } from "../src/api/sessions";
import { Banner } from "../src/ui/Banner";
import { DropZone } from "../src/ui/DropZone";
import { ProgressView } from "../src/ui/ProgressView";
import { RecentList } from "../src/ui/RecentList";
import { WorkerDot } from "../src/ui/TopBar";

afterEach(cleanup);

const makeFile = (name: string, type: string): File => new File(["x"], name, { type });

describe("DropZone", () => {
  it("accepts a dropped image and rejects other types with a reason", () => {
    const onFile = vi.fn();
    const onReject = vi.fn();
    render(<DropZone onFile={onFile} onReject={onReject} />);
    const zone = screen.getByRole("group", { name: "Add a photo" });

    const good = makeFile("a.jpg", "image/jpeg");
    fireEvent.drop(zone, { dataTransfer: { files: [good] } });
    expect(onFile).toHaveBeenCalledWith(good);

    fireEvent.drop(zone, { dataTransfer: { files: [makeFile("notes.txt", "text/plain")] } });
    expect(onFile).toHaveBeenCalledTimes(1);
    expect(onReject).toHaveBeenCalledOnce();
    expect(onReject.mock.calls[0]?.[0]).toMatch(/notes\.txt.*JPEG, PNG, WebP or HEIC/);
  });

  it("highlights while dragging over and clears on leave or drop", () => {
    render(<DropZone onFile={vi.fn()} onReject={vi.fn()} />);
    const zone = screen.getByRole("group", { name: "Add a photo" });
    fireEvent.dragOver(zone);
    expect(zone.className).toContain("over");
    fireEvent.dragLeave(zone);
    expect(zone.className).not.toContain("over");
    fireEvent.dragOver(zone);
    fireEvent.drop(zone, { dataTransfer: { files: [] } });
    expect(zone.className).not.toContain("over");
  });

  it("has one control for the keyboard: the Choose file button opens the dialog", () => {
    render(<DropZone onFile={vi.fn()} onReject={vi.fn()} />);
    const group = screen.getByRole("group", { name: "Add a photo" });
    expect(group.getAttribute("tabindex")).toBeNull(); // no nested interactive elements
    const input = screen.getByTestId("file-input") as HTMLInputElement;
    const click = vi.spyOn(input, "click").mockImplementation(() => undefined);
    fireEvent.click(screen.getByRole("button", { name: "Choose file" }));
    expect(click).toHaveBeenCalledTimes(1); // the button's click does not also fire the container's
    fireEvent.click(group); // mouse users can click anywhere in the zone
    expect(click).toHaveBeenCalledTimes(2);
  });

  it("takes the file chosen in the dialog, lists the formats and the privacy line", () => {
    const onFile = vi.fn();
    render(<DropZone onFile={onFile} onReject={vi.fn()} />);
    const input = screen.getByTestId("file-input") as HTMLInputElement;
    expect(input.accept).toContain("image/heic");
    expect(input.accept).toContain(".heic");
    const f = makeFile("b.heic", "");
    fireEvent.change(input, { target: { files: [f] } });
    expect(onFile).toHaveBeenCalledWith(f);
    expect(screen.getByText(/never leaves this computer/i)).toBeTruthy();
    expect(screen.getByText(/JPEG, PNG, WebP or HEIC/)).toBeTruthy();
  });
});

describe("ProgressView", () => {
  it("shows every named phase, with earlier ones done and the current one active", () => {
    render(<ProgressView phase="depth" pct={40} />);
    for (const p of PHASES) expect(screen.getByText(p.label)).toBeTruthy();
    const items = screen.getAllByRole("listitem");
    expect(items.map((li) => li.className.replace("phase ", ""))).toEqual([
      "done",
      "done",
      "done",
      "active",
      "todo",
    ]);
    const bar = screen.getByRole("progressbar", { name: "Estimating depth" });
    expect(bar.getAttribute("aria-valuenow")).toBe("40");
    expect(items[3]?.getAttribute("aria-current")).toBe("step");
  });

  it("clamps the bar to 0..100", () => {
    const { rerender } = render(<ProgressView phase="upload" pct={250} />);
    expect(screen.getByRole("progressbar").getAttribute("aria-valuenow")).toBe("100");
    rerender(<ProgressView phase="upload" pct={-5} />);
    expect(screen.getByRole("progressbar").getAttribute("aria-valuenow")).toBe("0");
  });
});

describe("progress events from a fake stream", () => {
  const ID = "c".repeat(64);
  beforeEach(() => {
    view.value = {
      state: "analyzing",
      id: ID,
      filename: "a.jpg",
      phase: "decode",
      pct: 0,
      meta: null,
      blurUrl: null,
    };
    announcement.value = "";
    appError.value = null;
  });

  it("moves through the phases in order and announces each change once", () => {
    const seen: string[] = [];
    for (const [phase, pct] of [
      ["decode", 10],
      ["decode", 20],
      ["intrinsics", 30],
      ["depth", 50],
      ["depth", 70],
      ["package", 95],
    ] as const) {
      handleSessionEvent({ type: "analyze.progress", session: ID, phase, pct }, ID);
      seen.push(`${view.value.phase}:${view.value.pct}`);
    }
    expect(seen).toEqual([
      "decode:10",
      "decode:20",
      "intrinsics:30",
      "depth:50",
      "depth:70",
      "package:95",
    ]);
    expect(announcement.value).toBe("Preparing the preview");
  });

  it("ignores events for other sessions and unknown phases", () => {
    handleSessionEvent(
      { type: "analyze.progress", session: "d".repeat(64), phase: "depth", pct: 50 },
      "d".repeat(64),
    );
    expect(view.value.phase).toBe("decode");
    handleSessionEvent({ type: "analyze.progress", session: ID, phase: "nonsense", pct: 33 }, ID);
    expect(view.value.phase).toBe("decode");
    expect(view.value.pct).toBe(33);
  });

  it("turns an error event into a specific banner and returns to the start", () => {
    handleSessionEvent({ type: "error", session: ID, code: "gpu_oom", message: "raw" }, ID);
    expect(appError.value?.title).toBe("Not enough GPU memory");
    expect(appError.value?.action).toBe("retry");
    expect(appError.value?.details).toContain(`session: ${ID}`);
    expect(view.value.state).toBe("idle");
    expect(announcement.value).toContain("Not enough GPU memory");
  });

  it("updates the worker status from status events", () => {
    handleSessionEvent({ type: "status", worker: "starting", detail: "loading" }, ID);
    expect(worker.value).toEqual({ state: "starting", detail: "loading" });
  });
});

describe("Banner", () => {
  it("offers the next step for the error and copies the details", () => {
    showError("gpu_oom", 0, "boom", "s1");
    const error = appError.value!;
    const onRetry = vi.fn();
    const onDismiss = vi.fn();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    render(<Banner error={error} onRetry={onRetry} onDismiss={onDismiss} />);
    expect(screen.getByRole("alert").textContent).toContain("Not enough GPU memory");
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(onRetry).toHaveBeenCalledOnce();
    fireEvent.click(screen.getByRole("button", { name: "Copy details" }));
    expect(writeText).toHaveBeenCalledOnce();
    expect(writeText.mock.calls[0]?.[0]).toContain("code: gpu_oom");
    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(onDismiss).toHaveBeenCalledOnce();
  });

  it("shows 'choose another photo' for unreadable files and no retry for licence errors", () => {
    showError("unsupported_image", 415, "x");
    const onChoose = vi.fn();
    const { unmount } = render(
      <Banner error={appError.value!} onChoose={onChoose} onRetry={vi.fn()} onDismiss={vi.fn()} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Choose another photo" }));
    expect(onChoose).toHaveBeenCalledOnce();
    expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
    unmount();
    showError("licence_not_accepted", 0, "");
    render(
      <Banner error={appError.value!} onRetry={vi.fn()} onChoose={vi.fn()} onDismiss={vi.fn()} />,
    );
    expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
    expect(screen.getByRole("alert").textContent).toContain("--accept-license");
  });
});

describe("RecentList and WorkerDot", () => {
  const item = (id: string, filename: string, state: SessionState = "ready") => ({
    id,
    filename,
    state,
    created: 0,
    last_used: 0,
    error: null,
    width: 1,
    height: 1,
    bytes: 1,
    thumbnail: state === "ready" ? `/api/sessions/${id}/assets/color.jpg` : null,
  });

  it("opens and deletes, with labelled buttons", () => {
    const onOpen = vi.fn();
    const onDelete = vi.fn();
    render(
      <RecentList
        items={[item("a".repeat(64), "one.jpg"), item("b".repeat(64), "two.jpg", "analyzing")]}
        currentId={"a".repeat(64)}
        onOpen={onOpen}
        onDelete={onDelete}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Open two.jpg" }));
    expect(onOpen).toHaveBeenCalledWith("b".repeat(64));
    fireEvent.click(screen.getByRole("button", { name: "Delete one.jpg" }));
    expect(onDelete).toHaveBeenCalledWith("a".repeat(64));
    expect(screen.getByRole("button", { name: "Open one.jpg" }).getAttribute("aria-current")).toBe(
      "true",
    );
    expect(screen.getByText("analyzing")).toBeTruthy();
  });

  it("renders nothing for an empty list", () => {
    const { container } = render(
      <RecentList items={[]} currentId={null} onOpen={vi.fn()} onDelete={vi.fn()} />,
    );
    expect(container.textContent).toBe("");
  });

  it("exposes the worker state to assistive technology", () => {
    render(<WorkerDot worker={{ state: "starting", detail: "" }} />);
    expect(screen.getByRole("img", { name: "Depth model loading" })).toBeTruthy();
  });
});
