"""Coverage, limit, render and result routes with the stub worker (CPU) (T06)."""

import asyncio
import io
import json
import shutil
import time

import httpx
import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from nvs.api.app import create_app
from nvs.api.config import ServerConfig
from nvs.cache import DepthCache
from nvs.core.camera import orbit_pose
from nvs.render.limits import estimate_coverage, limit_orbit

PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"


def make_jpeg(size=(320, 240), seed=0, orientation=None, make=None) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    exif = Image.Exif()
    if orientation:
        exif[0x0112] = orientation
    if make:
        exif[0x010F] = make
        exif[0xA405] = 30  # FocalLengthIn35mmFilm (Exif IFD in real files; enough for IFD0 here)
    buf = io.BytesIO()
    kwargs = {"exif": exif} if (orientation or make) else {}
    Image.fromarray(arr).save(buf, "JPEG", quality=90, **kwargs)
    return buf.getvalue()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("render_api")
    config = ServerConfig(port=PORT, data_dir=tmp, stub=True, web_dist=tmp / "nodist")
    app = create_app(config)
    with TestClient(app, base_url=BASE) as c:
        c.app_, c.tmp = app, tmp
        c.portal.call(app.state.worker.wait_ready, 60)
        yield c


def upload_ready(client, data, name="p.jpg") -> str:
    sid = client.post("/api/sessions", files={"file": (name, data)}).json()["id"]
    t = time.time()
    while time.time() - t < 60:
        body = client.get(f"/api/sessions/{sid}").json()
        if body["state"] in ("ready", "error"):
            assert body["state"] == "ready", body
            return sid
        time.sleep(0.05)
    raise AssertionError("analysis timeout")


@pytest.fixture(scope="module")
def sid(client):
    return upload_ready(client, make_jpeg((320, 240), seed=1))


def pose(yaw=0.0, pitch=0.0, pivot=3.0, dolly=0.0, pan=(0.0, 0.0)):
    R, t = orbit_pose(yaw, pitch, pivot, dolly, pan)
    orbit = {"yaw_deg": yaw, "pitch_deg": pitch, "dolly": dolly, "pan": list(pan), "pivot_z": pivot}
    return {"R": R.tolist(), "t": t.tolist(), "orbit": orbit}


def session_depth(client, sid):
    meta = client.get(f"/api/sessions/{sid}").json()["meta"]
    result = DepthCache(client.tmp / "cache").get(meta["depth"]["cache_key"])
    assert result is not None
    return result, meta


# ---------------------------------------------------------------------------------------- validation
IDENT = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
BAD_POSES = {
    "not orthonormal": {"R": [[2, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [0, 0, 0]},
    "reflection": {"R": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [0, 0, 0]},
    "huge t": {"R": IDENT, "t": [1000, 0, 0]},
    "huge t (norm)": {"R": IDENT, "t": [800, 800, 0]},
    "R wrong shape": {"R": [[1, 0], [0, 1]], "t": [0, 0, 0]},
    "R ragged": {"R": [[1, 0, 0], [0, 1], [0, 0, 1]], "t": [0, 0, 0]},
    "t wrong shape": {"R": IDENT, "t": [0, 0]},
    "R not numbers": {"R": [["a", 0, 0], [0, 1, 0], [0, 0, 1]], "t": [0, 0, 0]},
    "missing t": {"R": IDENT},
    "missing R": {"t": [0, 0, 0]},
}


@pytest.mark.parametrize("route", ["coverage", "render"])
@pytest.mark.parametrize("name", list(BAD_POSES))
def test_pose_validation_matrix(client, sid, route, name):
    r = client.post(f"/api/sessions/{sid}/{route}", json=BAD_POSES[name])
    assert r.status_code == 422, (name, r.text)
    err = r.json()["error"]
    assert err["code"] == "invalid_request" and err["message"]


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_nan_and_inf_rejected(client, sid, bad):
    for field in ("R", "t"):
        R = f"[[1,0,0],[0,1,0],[0,0,{bad}]]" if field == "R" else "[[1,0,0],[0,1,0],[0,0,1]]"
        t = f"[{bad},0,0]" if field == "t" else "[0,0,0]"
        r = client.post(
            f"/api/sessions/{sid}/coverage",
            content=f'{{"R": {R}, "t": {t}}}',
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


@pytest.mark.parametrize(
    "extra",
    [{"size": "huge"}, {"hole": "red"}, {"format": "gif"}, {"fill_specks": "maybe"}],
)
def test_unknown_enums_rejected(client, sid, extra):
    r = client.post(f"/api/sessions/{sid}/render", json={**pose(5), **extra})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


@pytest.mark.parametrize(
    "orbit",
    [
        {"yaw_deg": 5, "pitch_deg": 0, "pivot_z": 0},
        {"yaw_deg": 5, "pitch_deg": 0, "pivot_z": -1},
        {"yaw_deg": 500, "pitch_deg": 0, "pivot_z": 3},
        {"yaw_deg": 5, "pitch_deg": 90, "pivot_z": 3},
        {"yaw_deg": 5, "pitch_deg": 0, "pivot_z": 3, "pan": [1]},
        {"yaw_deg": 5, "pitch_deg": 0, "pivot_z": 3, "dolly": 1e9},
        {"yaw_deg": "x", "pivot_z": 3},
        {"yaw_deg": 5},
    ],
)
def test_limit_validation(client, sid, orbit):
    r = client.post(f"/api/sessions/{sid}/limit", json={"orbit": orbit})
    assert r.status_code == 422
    for frac in (0, -0.1, 1.5):
        r = client.post(
            f"/api/sessions/{sid}/limit",
            json={"orbit": {"yaw_deg": 5, "pivot_z": 3}, "max_hole_fraction": frac},
        )
        assert r.status_code == 422


def test_unknown_invalid_and_not_ready_sessions(client, sid):
    for route in ("coverage", "render"):
        assert client.post(f"/api/sessions/{'0' * 64}/{route}", json=pose()).status_code == 404
        assert client.post(f"/api/sessions/nothex/{route}", json=pose()).status_code == 404
    store = client.app_.state.sessions
    store.update(sid, state="uploaded")
    try:
        for route in ("coverage", "render"):
            r = client.post(f"/api/sessions/{sid}/{route}", json=pose())
            assert r.status_code == 409 and r.json()["error"]["code"] == "not_ready"
        r = client.post(f"/api/sessions/{sid}/limit", json={"orbit": {"yaw_deg": 5, "pivot_z": 3}})
        assert r.status_code == 409
    finally:
        store.update(sid, state="ready")


# -------------------------------------------------------------------------------------- coverage / limit
def test_coverage_matches_direct_call(client, sid):
    depth, meta = session_depth(client, sid)
    ident = client.post(f"/api/sessions/{sid}/coverage", json=pose(0)).json()
    assert ident["coverage"]["total"] == pytest.approx(0.0, abs=1e-6)
    p = pose(8.0, 2.0, meta["pivot_z"])
    got = client.post(f"/api/sessions/{sid}/coverage", json=p).json()
    assert got["coverage"]["total"] > 0 and got["ms"] > 0
    R, t = orbit_pose(8.0, 2.0, meta["pivot_z"])
    want = estimate_coverage(depth.depth, depth.f_px, R, t)
    assert set(got["coverage"]) == {"disocclusion", "stretch", "out_of_frame", "total"}
    for k, v in want.items():
        assert got["coverage"][k] == pytest.approx(v, abs=1e-6)


def test_limit_matches_direct_call(client, sid):
    depth, meta = session_depth(client, sid)
    orbit = {"yaw_deg": 25.0, "pitch_deg": 6.0, "dolly": 0.2, "pan": [0.05, 0.0], "pivot_z": meta["pivot_z"]}
    got = client.post(f"/api/sessions/{sid}/limit", json={"orbit": orbit, "max_hole_fraction": 0.1}).json()
    want = limit_orbit(
        depth.depth,
        depth.f_px,
        25.0,
        6.0,
        meta["pivot_z"],
        max_hole_fraction=0.1,
        dolly=0.2,
        pan=(0.05, 0.0),
        count_out_of_frame=False,  # the API default: the viewer does not count the area outside the photo
    )
    assert set(got) == {"scale", "yaw_deg", "pitch_deg", "dolly", "pan", "hole_fraction", "limited"}
    assert got["scale"] == pytest.approx(want.scale, abs=1e-9)
    assert got["yaw_deg"] == pytest.approx(want.yaw_deg) and got["pitch_deg"] == pytest.approx(want.pitch_deg)
    assert got["pan"] == pytest.approx(list(want.pan)) and got["limited"] == want.limited
    assert got["hole_fraction"] == pytest.approx(want.hole_fraction, abs=1e-6)
    # default max_hole_fraction is the library default
    assert client.post(f"/api/sessions/{sid}/limit", json={"orbit": orbit}).status_code == 200


def test_limit_count_out_of_frame_option(client, sid):
    """The viewer excludes the area outside the photo by default; including it can only limit more."""
    _, meta = session_depth(client, sid)
    orbit = {"yaw_deg": 30.0, "pitch_deg": 0.0, "dolly": 0.0, "pan": [0.0, 0.0], "pivot_z": meta["pivot_z"]}
    base = {"orbit": orbit, "max_hole_fraction": 0.1}
    excluded = client.post(f"/api/sessions/{sid}/limit", json=base).json()  # default: not counted
    counted = client.post(f"/api/sessions/{sid}/limit", json={**base, "count_out_of_frame": True}).json()
    assert excluded["scale"] >= counted["scale"]
    depth, _ = session_depth(client, sid)
    want = limit_orbit(
        depth.depth, depth.f_px, 30.0, 0.0, meta["pivot_z"], max_hole_fraction=0.1, count_out_of_frame=False
    )
    assert excluded["scale"] == pytest.approx(want.scale, abs=1e-9)
    assert excluded["hole_fraction"] == pytest.approx(want.hole_fraction, abs=1e-6)


# --------------------------------------------------------------------------------------------- render
def test_render_alpha_png_size_exif_and_idempotence(client):
    data = make_jpeg((320, 240), seed=2, orientation=6, make="TestCam")  # stored 320x240, shown 240x320
    sid = upload_ready(client, data, name="rot.jpg")
    meta = client.get(f"/api/sessions/{sid}").json()["meta"]
    assert (meta["width"], meta["height"]) == (240, 320)
    p = pose(6.0, 1.0, meta["pivot_z"])

    sched = client.app_.state.scheduler
    before = client.portal.call(lambda: sched.timings()["render"]["count"])
    r = client.post(f"/api/sessions/{sid}/render", json=p)
    assert r.status_code == 200, r.text
    out = r.json()
    assert (out["width"], out["height"]) == (240, 320)  # always the session's own size
    assert out["url"] == f"/api/sessions/{sid}/results/{out['name']}" and out["name"].endswith(".png")
    assert out["coverage"]["total"] > 0 and out["ms"]["total"] > 0
    assert {"load", "splat", "classify", "compose", "save"} <= set(out["ms"])

    img = client.get(out["url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    im = Image.open(io.BytesIO(img.content))
    assert im.size == (240, 320) and im.mode == "RGBA"  # holes marked through alpha
    alpha = np.asarray(im)[:, :, 3]
    assert (alpha < 255).any() and (alpha == 255).any()
    exif = im.getexif()
    assert exif.get(0x0112, 1) == 1  # orientation baked into the pixels
    assert exif.get(0x010F) == "TestCam"  # source EXIF carried over
    assert "yaw=6" in (exif.get(0x9286) or str(im.info))  # the CLI's note text

    # the identical request: same file, no new GPU job
    r2 = client.post(f"/api/sessions/{sid}/render", json={**p, "orbit": None})
    assert r2.json()["name"] == out["name"] and r2.json()["coverage"] == out["coverage"]
    after = client.portal.call(lambda: sched.timings()["render"]["count"])
    assert after == before + 1
    # a tiny pose change below the rounding step maps to the same file; a bigger one does not
    p_snap = json.loads(json.dumps(p))
    p_snap["t"][0] = 0.1234  # away from a rounding boundary
    named = client.post(f"/api/sessions/{sid}/render", json=p_snap).json()["name"]
    p_snap["t"][0] = 0.12342
    assert client.post(f"/api/sessions/{sid}/render", json=p_snap).json()["name"] == named
    p3 = pose(7.0, 1.0, meta["pivot_z"])
    assert client.post(f"/api/sessions/{sid}/render", json=p3).json()["name"] != out["name"]


def test_render_hole_modes_formats_and_specks(client, sid):
    p = pose(10.0, 0.0, 3.0)
    magenta = client.post(f"/api/sessions/{sid}/render", json={**p, "hole": "magenta"}).json()
    im = Image.open(io.BytesIO(client.get(magenta["url"]).content))
    assert im.mode == "RGB"
    arr = np.asarray(im)
    assert ((arr[:, :, 0] == 255) & (arr[:, :, 1] == 0) & (arr[:, :, 2] == 255)).any()
    black = client.post(f"/api/sessions/{sid}/render", json={**p, "hole": "black"}).json()
    assert black["name"] != magenta["name"]
    jpg = client.post(f"/api/sessions/{sid}/render", json={**p, "format": "jpg"}).json()  # alpha -> magenta
    r = client.get(jpg["url"])
    assert r.headers["content-type"] == "image/jpeg" and Image.open(io.BytesIO(r.content)).format == "JPEG"
    nospeck = client.post(f"/api/sessions/{sid}/render", json={**p, "fill_specks": False}).json()
    assert nospeck["name"] != client.post(f"/api/sessions/{sid}/render", json=p).json()["name"]


def test_render_preview_size_uses_scaled_intrinsics(client):
    sid = upload_ready(client, make_jpeg((2200, 1500), seed=3), name="big.jpg")
    meta = client.get(f"/api/sessions/{sid}").json()["meta"]
    assert meta["preview_width"] == 2048
    out = client.post(
        f"/api/sessions/{sid}/render",
        json={**pose(4, 0, meta["pivot_z"]), "size": "preview", "hole": "black"},
    ).json()
    assert (out["width"], out["height"]) == (meta["preview_width"], meta["preview_height"])
    source = client.post(
        f"/api/sessions/{sid}/render", json={**pose(4, 0, meta["pivot_z"]), "hole": "black"}
    ).json()
    assert (source["width"], source["height"]) == (2200, 1500) and source["name"] != out["name"]


def test_result_download_headers_etag_and_privacy(client, sid):
    out = client.post(f"/api/sessions/{sid}/render", json=pose(5, 0, 3.0)).json()
    plain = client.get(out["url"])
    assert "content-disposition" not in plain.headers
    assert plain.headers["cache-control"] == "private, max-age=31536000, immutable"
    dl = client.get(out["url"] + "?download=1")
    assert dl.headers["content-disposition"] == 'attachment; filename="p-view.png"'
    assert client.get(out["url"], headers={"If-None-Match": plain.headers["etag"]}).status_code == 304


@pytest.mark.parametrize(
    "name",
    ["..%2f..%2fsession.json", "..%5c..%5coriginal.jpg", "%2e%2e", "meta.json", "nope.png", "A.png", "x.gif"],
)
def test_result_name_traversal_and_foreign_files_blocked(client, sid, name):
    out = client.post(f"/api/sessions/{sid}/render", json=pose(5, 0, 3.0)).json()
    r = client.get(f"/api/sessions/{sid}/results/{name}")
    assert r.status_code in (404, 400)
    assert r.json()["error"]["code"] in ("not_found", "http_404")
    # the JSON sidecar next to a result is internal: not downloadable
    sidecar = out["name"].rsplit(".", 1)[0] + ".json"
    assert client.get(f"/api/sessions/{sid}/results/{sidecar}").status_code == 404


def test_result_of_other_session_not_reachable(client, sid):
    out = client.post(f"/api/sessions/{sid}/render", json=pose(5, 0, 3.0)).json()
    other = upload_ready(client, make_jpeg((200, 150), seed=9), name="other.jpg")
    assert client.get(f"/api/sessions/{other}/results/{out['name']}").status_code == 404


# ------------------------------------------------------------------------- coalescing and disconnects
def test_twenty_parallel_coverage_requests_only_the_last_completes(client, sid):
    app = client.app_

    async def scenario():
        sched = app.state.scheduler
        blocker = sched.submit("echo_sleep", {"seconds": 0.8, "steps": 8})  # keeps the worker busy
        await asyncio.sleep(0.2)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE) as http:
            tasks = []
            for i in range(20):
                body = pose(1.0 + i * 0.5, 0.0, 3.0)
                tasks.append(asyncio.ensure_future(http.post(f"/api/sessions/{sid}/coverage", json=body)))
                await asyncio.sleep(0)  # arrive in order
            responses = await asyncio.gather(*tasks)
        await blocker.result()
        return responses

    responses = client.portal.call(scenario)
    codes = [r.status_code for r in responses]
    assert codes == [409] * 19 + [200]  # the last one wins
    assert all(r.json()["error"]["code"] == "superseded" for r in responses[:19])
    assert responses[-1].json()["coverage"]["total"] > 0


def test_client_disconnect_cancels_the_queued_job(client, sid):
    app = client.app_

    async def scenario():
        sched = app.state.scheduler
        blocker = sched.submit("echo_sleep", {"seconds": 1.2, "steps": 12})
        await asyncio.sleep(0.2)
        count_before = sched.timings()["interactive"]["count"]
        body = json.dumps(pose(5.0, 0.0, 3.0)).encode()
        sent: list[dict] = []
        pending = [{"type": "http.request", "body": body, "more_body": False}]

        async def receive():
            if pending:
                return pending.pop(0)
            return {"type": "http.disconnect"}  # the client went away (Starlette polls without waiting)

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "path": f"/api/sessions/{sid}/coverage",
            "raw_path": f"/api/sessions/{sid}/coverage".encode(),
            "query_string": b"",
            "root_path": "",
            "scheme": "http",
            "headers": [
                (b"host", f"127.0.0.1:{PORT}".encode()),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
            "client": ("127.0.0.1", 1),
            "server": ("127.0.0.1", PORT),
        }
        t0 = time.perf_counter()
        await app(scope, receive, send)
        dt = time.perf_counter() - t0
        queue_after = sched.queue_length
        await blocker.result()
        return dt, queue_after, sched.timings()["interactive"]["count"] - count_before

    dt, queue_after, ran = client.portal.call(scenario)
    assert dt < 1.0  # returned long before the blocker (1.2 s) finished
    assert queue_after == 0 and ran == 0  # the job was removed and never ran


def test_depth_missing_gives_409_and_asks_for_reupload(client):
    sid2 = upload_ready(client, make_jpeg((260, 200), seed=11), name="gone.jpg")
    shutil.rmtree(client.tmp / "cache" / "depth")
    r = client.post(f"/api/sessions/{sid2}/coverage", json=pose(5, 0, 3.0))
    assert r.status_code == 409 and r.json()["error"]["code"] == "depth_missing"
    assert "upload" in r.json()["error"]["message"]


def test_cache_invalidated_by_reanalysis(client):
    sid2 = upload_ready(client, make_jpeg((240, 180), seed=12), name="re.jpg")
    first = client.post(f"/api/sessions/{sid2}/coverage", json=pose(8, 0, 3.0)).json()
    meta_path = client.tmp / "sessions" / sid2 / "preview" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["pivot_z"] = 9.9  # any rewrite of meta.json changes the cache stamp
    meta_path.write_text(json.dumps(meta, indent=2))
    again = client.post(f"/api/sessions/{sid2}/coverage", json=pose(8, 0, 3.0)).json()
    assert again["coverage"] == first["coverage"]  # reloaded from the depth cache: same numbers
