"""Analyze flow through the API with the stub backend: upload -> events -> assets (T05)."""

import io
import time

import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from nvs.api.app import create_app
from nvs.api.config import ServerConfig
from nvs.api.sessions import SessionStore

PORT = 8765
HOST = {"Host": f"127.0.0.1:{PORT}"}  # TestClient builds websocket URLs on "testserver"


def jpeg(size=(320, 240), seed=0) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    config = ServerConfig(port=PORT, data_dir=tmp_path, stub=True, web_dist=tmp_path / "nodist")
    app = create_app(config)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:
        c.app_ = app
        c.portal.call(app.state.worker.wait_ready, 60)
        yield c


def wait_state(client, sid, wanted, timeout=30.0):
    t = time.time()
    while time.time() - t < timeout:
        body = client.get(f"/api/sessions/{sid}").json()
        if body["state"] in wanted:
            return body
        time.sleep(0.05)
    raise AssertionError(f"session stayed {body['state']}")


def collect_until_end(ws):
    msgs = []
    while True:
        m = ws.receive_json()
        msgs.append(m)
        if m["type"] in ("ready", "error"):
            return msgs


def test_upload_events_phases_ready_and_assets(client):
    data = jpeg((400, 300))
    r = client.post("/api/sessions", files={"file": ("p.jpg", data)})
    assert r.status_code == 201
    sid = r.json()["id"]
    with client.websocket_connect(f"/api/sessions/{sid}/events", headers=HOST) as ws:
        msgs = collect_until_end(ws)  # replays the latest state if the job is already past some phases
    types = [m["type"] for m in msgs]
    assert types[0] == "status" and types[-1] == "ready"
    progress = [m for m in msgs if m["type"] == "analyze.progress"]
    assert all(m["session"] == sid for m in progress)
    phases = [m["phase"] for m in progress]
    order = ["upload", "decode", "intrinsics", "depth", "package"]
    assert phases == sorted(phases, key=order.index)  # never goes backwards
    pcts = [m["pct"] for m in progress]
    assert pcts == sorted(pcts)

    body = wait_state(client, sid, {"ready"})
    meta = body["meta"]
    assert meta["schema"] == 1 and (meta["width"], meta["height"]) == (400, 300)
    assert body["thumbnail"].endswith("/assets/color.jpg")
    pw, ph = meta["preview_width"], meta["preview_height"]
    depth = client.get(f"/api/sessions/{sid}/assets/depth.f16")
    assert depth.status_code == 200 and len(depth.content) == pw * ph * 2
    assert len(client.get(f"/api/sessions/{sid}/assets/sky.u8").content) == pw * ph
    img = Image.open(io.BytesIO(client.get(f"/api/sessions/{sid}/assets/color.jpg").content))
    assert img.size == (pw, ph)
    assert meta["depth"]["model"] == "stub" and meta["f_px_source"] > 0
    assert client.get(f"/api/sessions/{sid}").json()["width"] == 400


def test_live_phase_order_through_websocket(client):
    """Subscribe before the job runs: all phases arrive in order."""
    app = client.app_
    sid = client.post("/api/sessions", files={"file": ("p.jpg", jpeg(seed=1))}).json()["id"]
    wait_state(client, sid, {"ready"})
    # force a fresh analysis: remove the preview, subscribe, then re-upload
    import shutil

    shutil.rmtree(app.state.sessions.root / sid / "preview")
    app.state.sessions.update(sid, state="uploaded")
    client.portal.call(app.state.hub.forget, sid)  # drop the remembered `ready` of the first run
    with client.websocket_connect(f"/api/sessions/{sid}/events", headers=HOST) as ws:
        first = ws.receive_json()
        assert first["type"] == "status"
        client.post("/api/sessions", files={"file": ("p.jpg", jpeg(seed=1))})
        msgs = collect_until_end(ws)
    phases = []
    for m in msgs:
        if m["type"] == "analyze.progress" and (not phases or phases[-1] != m["phase"]):
            phases.append(m["phase"])
    assert phases == ["upload", "decode", "intrinsics", "depth", "package"]
    assert msgs[-1]["type"] == "ready"


def test_second_upload_is_ready_immediately(client):
    data = jpeg(seed=2)
    sid = client.post("/api/sessions", files={"file": ("p.jpg", data)}).json()["id"]
    wait_state(client, sid, {"ready"})
    t = time.perf_counter()
    r = client.post("/api/sessions", files={"file": ("again.jpg", data)})
    dt = time.perf_counter() - t
    print(f"second upload of the same photo: {dt * 1000:.0f} ms")
    assert r.status_code == 201
    body = r.json()
    assert body["existing"] is True and body["state"] == "ready" and body["id"] == sid
    assert dt < 0.3


def test_corrupt_image_is_415_and_truncated_jpeg_becomes_error_state(client):
    r = client.post("/api/sessions", files={"file": ("x.jpg", b"not an image")})
    assert r.status_code == 415  # sniffed at upload: never stored

    good = jpeg((300, 200), seed=3)
    cut = good[: len(good) // 2]  # header intact, data truncated: passes the upload sniff, fails decode
    r = client.post("/api/sessions", files={"file": ("cut.jpg", cut)})
    assert r.status_code == 201
    sid = r.json()["id"]
    body = wait_state(client, sid, {"error", "ready"})
    assert body["state"] == "error"
    assert body["error"]["code"] == "unsupported_image"
    assert "cut.jpg" in body["error"]["message"] or "unsupported" in body["error"]["message"]
    assert body["thumbnail"] is None and body["meta"] is None
    # the worker survived the bad image
    ok = client.post("/api/sessions", files={"file": ("ok.jpg", jpeg(seed=4))}).json()["id"]
    assert wait_state(client, ok, {"ready", "error"})["state"] == "ready"


def test_error_event_is_replayed_to_late_subscribers(client):
    cut = jpeg((300, 200), seed=5)
    cut = cut[: len(cut) // 2]
    sid = client.post("/api/sessions", files={"file": ("cut.jpg", cut)}).json()["id"]
    wait_state(client, sid, {"error"})
    with client.websocket_connect(f"/api/sessions/{sid}/events", headers=HOST) as ws:
        assert ws.receive_json()["type"] == "status"
        err = ws.receive_json()
        assert err["type"] == "error" and err["session"] == sid and err["code"] == "unsupported_image"


def test_reupload_after_error_retries(client):
    cut = jpeg((300, 200), seed=6)
    cut = cut[: len(cut) // 2]
    sid = client.post("/api/sessions", files={"file": ("cut.jpg", cut)}).json()["id"]
    wait_state(client, sid, {"error"})
    again = client.post("/api/sessions", files={"file": ("cut.jpg", cut)}).json()
    assert again["existing"] is True
    assert wait_state(client, sid, {"error"})["state"] == "error"  # analysed again, same verdict


def test_delete_while_queued_cancels_and_removes(client):
    app = client.app_
    # occupy the worker so the analyze job stays queued
    blocker = client.portal.call(
        lambda: app.state.scheduler.submit("echo_sleep", {"seconds": 1.5, "steps": 15})
    )
    sid = client.post("/api/sessions", files={"file": ("q.jpg", jpeg(seed=7))}).json()["id"]
    assert client.get(f"/api/sessions/{sid}").json()["state"] == "analyzing"
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    time.sleep(0.2)
    assert client.get(f"/api/sessions/{sid}").status_code == 404
    assert client.portal.call(lambda: app.state.scheduler.queue_length) == 0
    client.portal.call(blocker.result)
    assert not (app.state.sessions.root / sid).exists()


def test_preview_survives_restart_and_is_reused(tmp_path):
    config = ServerConfig(port=PORT, data_dir=tmp_path, stub=True, web_dist=tmp_path / "nodist")
    data = jpeg(seed=8)
    with TestClient(create_app(config), base_url=f"http://127.0.0.1:{PORT}") as c:
        sid = c.post("/api/sessions", files={"file": ("a.jpg", data)}).json()["id"]
        wait_state(c, sid, {"ready"})
    assert SessionStore(tmp_path / "sessions").get(sid).state == "ready"
    config2 = ServerConfig(
        port=PORT, data_dir=tmp_path, stub=True, web_dist=tmp_path / "nodist", start_worker=False
    )
    with TestClient(create_app(config2), base_url=f"http://127.0.0.1:{PORT}") as c:  # no worker at all
        body = c.post("/api/sessions", files={"file": ("a.jpg", data)}).json()
        assert body["state"] == "ready" and body["existing"] is True  # idempotent, model not needed
        assert c.get(f"/api/sessions/{sid}").json()["meta"]["schema"] == 1


def test_worker_unavailable_becomes_friendly_error(tmp_path):
    config = ServerConfig(
        port=PORT, data_dir=tmp_path, stub=True, web_dist=tmp_path / "nodist", start_worker=False
    )
    app = create_app(config)
    app.state.scheduler.worker_wait_s = 0.3  # never-started worker
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:
        sid = c.post("/api/sessions", files={"file": ("a.jpg", jpeg(seed=9))}).json()["id"]
        body = wait_state(c, sid, {"error"}, timeout=10)
        assert body["error"]["code"] == "worker_unavailable"
        assert "not ready" in body["error"]["message"]
