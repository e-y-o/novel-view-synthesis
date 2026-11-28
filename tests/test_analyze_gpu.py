"""Real-model analyze flow (GPU, slow): 12 MP upload -> preview; CLI and server share one depth cache."""

import io
import os
import time

import numpy as np
import pytest
import torch
from PIL import Image
from starlette.testclient import TestClient

from nvs.api.app import create_app
from nvs.api.config import ServerConfig
from nvs.cache import DepthCache, cache_key
from nvs.cli.main import build_parser
from nvs.cli.render import run_render
from nvs.core.imageio import load_image
from nvs.licence import license_accepted
from nvs.preview.assets import resample_depth

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

PORT = 8765


def synthetic_12mp() -> bytes:
    """4000x3000 image (no EXIF): smooth gradient, a few filled shapes and a little noise."""
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:3000, 0:4000]
    img = np.stack([x / 4000, y / 3000, (x + y) / 7000], -1) * 200 + 20
    img[900:2100, 1200:2200] = (200, 60, 50)
    img[1500:2800, 2600:3600] = (40, 120, 200)
    img += rng.normal(0, 3, img.shape)
    buf = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(buf, "JPEG", quality=92)
    return buf.getvalue()


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    if not license_accepted() and os.environ.get("NVS_ACCEPT_DEPTHPRO2_LICENSE") != "1":
        pytest.skip("Depth Pro 2 licence not accepted on this machine")
    tmp = tmp_path_factory.mktemp("srv")
    os.environ["NVS_ACCEPT_DEPTHPRO2_LICENSE"] = "1"
    config = ServerConfig(port=PORT, data_dir=tmp, stub=False, web_dist=tmp / "nodist")
    app = create_app(config)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:
        status = c.portal.call(app.state.worker.wait_ready, 120)
        assert status.state == "ready", status
        c.app_, c.tmp = app, tmp
        yield c


def wait_ready(client, sid, timeout=60):
    t = time.perf_counter()
    while time.perf_counter() - t < timeout:
        body = client.get(f"/api/sessions/{sid}").json()
        if body["state"] in ("ready", "error"):
            return body, time.perf_counter() - t
        time.sleep(0.01)
    raise AssertionError("timeout")


def test_12mp_upload_to_ready_then_cli_reads_same_depth(server, tmp_path):
    data = synthetic_12mp()
    path = tmp_path / "syn.jpg"
    path.write_bytes(data)

    t = time.perf_counter()
    sid = server.post("/api/sessions", files={"file": ("syn.jpg", data)}).json()["id"]
    body, _ = wait_ready(server, sid)
    cold = time.perf_counter() - t
    assert body["state"] == "ready", body
    meta = body["meta"]
    print(f"12 MP upload -> ready (warm model, uncached depth): {cold:.2f} s")
    assert cold < 8.0
    assert (meta["width"], meta["height"]) == (4000, 3000) and (
        meta["preview_width"],
        meta["preview_height"],
    ) == (
        2048,
        1536,
    )
    assert meta["depth"]["cached"] is False and meta["intrinsics"]["assumed"] is True  # no EXIF -> estimator

    # same file again with a cold session store entry removed: depth now comes from the cache
    server.delete(f"/api/sessions/{sid}")
    t = time.perf_counter()
    sid2 = server.post("/api/sessions", files={"file": ("syn.jpg", data)}).json()["id"]
    body2, _ = wait_ready(server, sid2)
    warm = time.perf_counter() - t
    print(f"12 MP upload -> ready (cached depth): {warm:.2f} s")
    assert body2["meta"]["depth"]["cached"] is True and warm < 3.0

    # CLI and server share one cache entry: the depth the server packaged is the cached depth
    img = load_image(path)
    key = cache_key(img.sha256, "depthpro2-h", "bf16", meta["f_px_source"])
    cached = DepthCache(server.tmp / "cache").get(key)
    assert cached is not None
    preview = np.frombuffer(
        server.get(f"/api/sessions/{sid2}/assets/depth.f16").content, dtype="<f2"
    ).reshape(meta["preview_height"], meta["preview_width"])
    expect = resample_depth(cached.depth, meta["preview_height"], meta["preview_width"]).numpy().astype("<f2")
    # (the server resamples on the GPU, this test on the CPU: equal up to float rounding and float16 storage)
    np.testing.assert_allclose(preview.astype(np.float32), expect.astype(np.float32), rtol=2e-3)

    os.environ["NVS_CACHE_DIR"] = str(server.tmp / "cache")
    try:
        args = build_parser().parse_args(
            ["render", str(path), "--out", str(tmp_path / "o.png"), "--yaw", "3"]
        )
        report = run_render(args)
    finally:
        del os.environ["NVS_CACHE_DIR"]
    assert report.depth_cached is True  # the CLI found the entry the server wrote
    assert report.intrinsics.f_px == pytest.approx(meta["f_px_source"])
    assert torch.cuda.is_available()
