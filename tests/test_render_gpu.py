"""Real-model render path (GPU, slow): 12 MP render time, CLI parity, coverage latency (T06)."""

import io
import os
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from nvs.api.app import create_app
from nvs.api.config import ServerConfig
from nvs.cli.main import build_parser
from nvs.cli.render import run_render
from nvs.core.camera import orbit_pose
from nvs.licence import license_accepted

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

PORT = 8765
PHOTO = Path(__file__).resolve().parents[1] / "data" / "samples" / "landscape_02.HEIC"  # 12 MP, git-ignored


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    if not PHOTO.is_file():
        pytest.skip("sample photo not available")
    if not license_accepted() and os.environ.get("NVS_ACCEPT_DEPTHPRO2_LICENSE") != "1":
        pytest.skip("Depth Pro 2 licence not accepted on this machine")
    tmp = tmp_path_factory.mktemp("render_gpu")
    os.environ["NVS_ACCEPT_DEPTHPRO2_LICENSE"] = "1"
    config = ServerConfig(port=PORT, data_dir=tmp, stub=False, web_dist=tmp / "nodist")
    app = create_app(config)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:
        assert c.portal.call(app.state.worker.wait_ready, 120).state == "ready"
        c.tmp = tmp
        yield c


def pose(yaw, pitch, pivot):
    R, t = orbit_pose(yaw, pitch, pivot)
    orbit = {"yaw_deg": yaw, "pitch_deg": pitch, "dolly": 0.0, "pan": [0.0, 0.0], "pivot_z": pivot}
    return {"R": R.tolist(), "t": t.tolist(), "orbit": orbit}


@pytest.fixture(scope="module")
def session(server):
    sid = server.post("/api/sessions", files={"file": (PHOTO.name, PHOTO.read_bytes())}).json()["id"]
    t = time.time()
    while time.time() - t < 90:
        body = server.get(f"/api/sessions/{sid}").json()
        if body["state"] in ("ready", "error"):
            assert body["state"] == "ready", body
            return sid, body["meta"]
        time.sleep(0.05)
    raise AssertionError("analysis timeout")


def test_render_12mp_time_and_cli_parity(server, session, tmp_path):
    sid, meta = session
    p = pose(6.0, 1.0, meta["pivot_z"])
    runs = []
    for i in range(3):  # first call loads the source image into the worker cache
        q = pose(6.0 + 0.01 * i, 1.0, meta["pivot_z"])
        t = time.perf_counter()
        r = server.post(f"/api/sessions/{sid}/render", json=q)
        runs.append((time.perf_counter() - t, r.json()))
        assert r.status_code == 200, r.text
    for dt, out in runs:
        print(f"12 MP render: {dt:.2f} s round trip, stages {out['ms']}")
    assert runs[-1][0] < 4.0  # target ~3 s for the warm path including PNG encoding
    assert (runs[-1][1]["width"], runs[-1][1]["height"]) == (meta["width"], meta["height"])

    out = server.post(f"/api/sessions/{sid}/render", json=p).json()
    server_img = np.asarray(Image.open(io.BytesIO(server.get(out["url"]).content)))

    os.environ["NVS_CACHE_DIR"] = str(server.tmp / "cache")
    try:
        args = build_parser().parse_args(
            ["render", str(PHOTO), "--out", str(tmp_path / "cli.png"), "--yaw", "6", "--pitch", "1",
             "--pivot", repr(meta["pivot_z"])]
        )  # fmt: skip
        report = run_render(args)
    finally:
        del os.environ["NVS_CACHE_DIR"]
    assert report.depth_cached is True
    cli_img = np.asarray(Image.open(tmp_path / "cli.png"))
    assert cli_img.shape == server_img.shape
    diff = np.abs(cli_img.astype(int) - server_img.astype(int))
    # The splat accumulates with float atomics, so two runs may differ by one level in a few pixels.
    print(f"CLI vs server: max diff {diff.max()}, differing pixels {(diff.max(-1) > 0).sum()}")
    assert diff.max() <= 1 and (diff.max(-1) > 0).mean() < 1e-4


def test_coverage_round_trip_latency(server, session):
    sid, meta = session
    server.post(f"/api/sessions/{sid}/coverage", json=pose(5.0, 0.0, meta["pivot_z"]))  # warm the GPU cache
    times, worker_ms = [], []
    for i in range(100):
        q = pose(2.0 + 0.1 * i, 0.5, meta["pivot_z"])
        t = time.perf_counter()
        r = server.post(f"/api/sessions/{sid}/coverage", json=q)
        times.append((time.perf_counter() - t) * 1000)
        assert r.status_code == 200
        worker_ms.append(r.json()["ms"])
    times.sort()
    p95 = times[int(0.95 * len(times)) - 1]
    print(
        f"coverage round trip (in-process HTTP): median {statistics.median(times):.1f} ms, p95 {p95:.1f} ms, "
        f"worker median {statistics.median(worker_ms):.1f} ms"
    )
    assert p95 <= 25.0
