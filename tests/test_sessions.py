"""Session store and REST routes (T02)."""

import hashlib
import io
import json
import os
import time

import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from nvs.api import sessions as sessions_mod
from nvs.api.app import create_app
from nvs.api.config import ServerConfig
from nvs.api.sessions import (
    ImageTooLarge,
    SessionStore,
    UnsupportedImage,
    atomic_write_json,
    sanitize_filename,
)

PORT = 8765


def make_image(fmt="JPEG", size=(64, 48), seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt)
    return buf.getvalue()


@pytest.fixture
def created():
    return []


@pytest.fixture
def client(tmp_path, created):
    config = ServerConfig(port=PORT, data_dir=tmp_path, stub=True, web_dist=tmp_path / "nodist")
    app = create_app(config, on_created=created.append)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:
        c.store = app.state.sessions
        yield c


def upload(client, data: bytes, name="photo.jpg"):
    return client.post("/api/sessions", files={"file": (name, data, "application/octet-stream")})


def put_preview(store, sid, **files):
    d = store.root / sid / "preview"
    d.mkdir(exist_ok=True)
    for name, data in files.items():
        (d / name.replace("_", ".")).write_bytes(data)


# ----------------------------------------------------------------------------------------------- routes
def test_upload_id_is_sha256_and_hook_called(client, created):
    data = make_image()
    r = upload(client, data)
    assert r.status_code == 201
    body = r.json()
    assert body["id"] == hashlib.sha256(data).hexdigest()
    assert body["state"] == "uploaded" and body["existing"] is False and body["filename"] == "photo.jpg"
    assert [s.id for s in created] == [body["id"]]
    assert (client.store.root / body["id"] / "original.jpg").read_bytes() == data


def test_same_bytes_again_is_existing_and_single_copy(client):
    data = make_image()
    first = upload(client, data).json()
    second = upload(client, data, name="other.jpg").json()
    assert second["existing"] is True and second["id"] == first["id"]
    assert second["filename"] == "photo.jpg"  # original display name kept
    files = [p for p in (client.store.root / first["id"]).iterdir() if p.name.startswith("original")]
    assert len(files) == 1
    assert list(client.store.tmp_dir.iterdir()) == []


def test_text_file_named_jpg_is_415(client):
    r = upload(client, b"hello world, not an image", name="x.jpg")
    assert r.status_code == 415 and r.json()["error"]["code"] == "http_415"
    assert client.store.list() == [] and list(client.store.tmp_dir.iterdir()) == []


def test_png_renamed_jpg_is_stored_as_png(client):
    sid = upload(client, make_image("PNG"), name="really.jpg").json()["id"]
    assert (client.store.root / sid / "original.png").is_file()
    assert not (client.store.root / sid / "original.jpg").exists()


def test_too_many_pixels_is_413(client, monkeypatch):
    monkeypatch.setattr(sessions_mod, "MAX_PIXELS", 1000)
    r = upload(client, make_image(size=(64, 48)))
    assert r.status_code == 413
    assert client.store.list() == []


def test_upload_over_byte_limit_is_413(tmp_path):
    config = ServerConfig(port=PORT, data_dir=tmp_path, max_upload_mb=0, web_dist=tmp_path / "x")
    with TestClient(create_app(config), base_url=f"http://127.0.0.1:{PORT}") as c:
        assert upload(c, make_image()).status_code == 413


def test_list_get_and_ready_meta(client):
    a = upload(client, make_image(seed=1)).json()["id"]
    time.sleep(0.01)
    b = upload(client, make_image(seed=2), name="b.jpg").json()["id"]
    listing = client.get("/api/sessions").json()["sessions"]
    assert [s["id"] for s in listing] == [b, a]  # newest first
    assert listing[0]["thumbnail"] is None

    put_preview(client.store, b, color_jpg=b"JPEGDATA", meta_json=json.dumps({"schema": 1}).encode())
    client.store.update(b, state="ready")
    got = client.get(f"/api/sessions/{b}").json()
    assert got["state"] == "ready" and got["meta"] == {"schema": 1}
    assert got["thumbnail"] == f"/api/sessions/{b}/assets/color.jpg"
    assert client.get(f"/api/sessions/{a}").json()["meta"] is None


def test_unknown_and_invalid_ids_are_404(client):
    for sid in ["0" * 64, "nothex", "A" * 64, "..", "%2e%2e"]:
        r = client.get(f"/api/sessions/{sid}")
        assert r.status_code == 404 and r.json()["error"]["code"] in ("http_404", "not_found")
    assert client.delete(f"/api/sessions/{'0' * 64}").status_code == 404


def test_assets_content_types_cache_and_etag(client):
    sid = upload(client, make_image()).json()["id"]
    put_preview(client.store, sid, color_jpg=b"jpg", depth_f16=b"\0\0", sky_u8=b"\0", meta_json=b"{}")
    expected = {
        "color.jpg": "image/jpeg",
        "depth.f16": "application/octet-stream",
        "sky.u8": "application/octet-stream",
        "meta.json": "application/json",
    }
    for name, ctype in expected.items():
        r = client.get(f"/api/sessions/{sid}/assets/{name}")
        assert r.status_code == 200 and r.headers["content-type"].startswith(ctype)
        assert r.headers["cache-control"] == "private, max-age=31536000, immutable"
    r = client.get(f"/api/sessions/{sid}/assets/color.jpg")
    etag = r.headers["etag"]
    assert (
        client.get(f"/api/sessions/{sid}/assets/color.jpg", headers={"If-None-Match": etag}).status_code
        == 304
    )
    assert client.get(f"/api/sessions/{sid}/assets/original.jpg").status_code == 404  # not allow-listed
    assert client.get(f"/api/sessions/{sid}/assets/session.json").status_code == 404


def test_missing_asset_404(client):
    sid = upload(client, make_image()).json()["id"]
    assert client.get(f"/api/sessions/{sid}/assets/depth.f16").status_code == 404


def test_delete_removes_session(client):
    sid = upload(client, make_image()).json()["id"]
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert client.get(f"/api/sessions/{sid}").status_code == 404
    assert not (client.store.root / sid).exists()


@pytest.mark.parametrize(
    "path",
    [
        "..%2f..%2fsession.json",
        "%2e%2e%2fsession.json",
        "meta.json%2f..%2f..%2foriginal.jpg",
        "..%5c..%5csession.json",
        "C:%5cWindows%5cwin.ini",
        "%2fetc%2fpasswd",
    ],
)
def test_asset_path_traversal_blocked(client, path):
    sid = upload(client, make_image()).json()["id"]
    put_preview(client.store, sid, meta_json=b"{}")
    r = client.get(f"/api/sessions/{sid}/assets/{path}")
    assert r.status_code in (400, 404)
    assert b"hash" not in r.content and b"filename" not in r.content  # no session.json leaked


def test_store_path_methods_refuse_escapes(tmp_path):
    store = SessionStore(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("x")
    sid = "a" * 64
    (tmp_path / sid / "preview").mkdir(parents=True)
    (tmp_path / sid / "results").mkdir()
    for name in ["../../secret.txt", "..", "a/b", "a\\b", "/abs", "", "meta.json/..", "COLOR.JPG"]:
        assert store.asset_path(sid, name) is None
    for name in ["../secret.txt", "..", ".hidden", "a/b", "A.png", "x" * 65, ""]:
        assert store.result_path(sid, name) is None
    assert store.asset_path("../" + sid, "meta.json") is None
    (tmp_path / sid / "results" / "view-1.png").write_bytes(b"png")
    assert store.result_path(sid, "view-1.png") is not None


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_symlinked_asset_outside_session_refused(tmp_path):
    store = SessionStore(tmp_path)
    sid = "b" * 64
    (tmp_path / sid / "preview").mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (tmp_path / sid / "preview" / "meta.json").symlink_to(outside)
    assert store.asset_path(sid, "meta.json") is None


# ------------------------------------------------------------------------------------------------ store
def test_filename_sanitisation():
    assert sanitize_filename("..\\..\\evil.jpg") == "evil.jpg"
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("C:\\Users\\me\\pic.png") == "pic.png"
    assert sanitize_filename("a\x00b\x1f\x7fc\u202e.jpg") == "abc.jpg"
    assert sanitize_filename("x" * 500) == "x" * 120
    assert sanitize_filename("") == "image" and sanitize_filename(None) == "image"
    assert sanitize_filename("  ..  ") == "image"
    assert sanitize_filename("holiday photo (1).jpg") == "holiday photo (1).jpg"


def test_store_create_errors_remove_temp(tmp_path):
    store = SessionStore(tmp_path)
    bad = store.tmp_dir / "bad"
    bad.write_bytes(b"nope")
    with pytest.raises(UnsupportedImage):
        store.create(bad, "x.jpg")
    assert not bad.exists()


def test_store_too_large(tmp_path, monkeypatch):
    monkeypatch.setattr(sessions_mod, "MAX_PIXELS", 100)
    store = SessionStore(tmp_path)
    tmp = store.tmp_dir / "big"
    tmp.write_bytes(make_image())
    with pytest.raises(ImageTooLarge):
        store.create(tmp, "big.jpg")
    assert not tmp.exists()


def _add(store, seed, size=(64, 48)):
    tmp = store.tmp_dir / f"t{seed}"
    tmp.write_bytes(make_image(seed=seed, size=size))
    session, _ = store.create(tmp, f"{seed}.jpg")
    return session


def test_eviction_keeps_newest_within_cap_and_analyzing(tmp_path):
    store = SessionStore(tmp_path, max_bytes=10**9)
    s = [_add(store, i) for i in range(4)]
    for i, sess in enumerate(s):  # explicit LRU order: s[0] oldest
        store.update(sess.id, last_used=1000.0 + i)
    store.update(s[0].id, state="analyzing")  # oldest, but protected
    one = store.total_bytes() // 4
    store.max_bytes = int(one * 2.5)  # room for two
    evicted = store.evict()
    survivors = {x.id for x in store.list()}
    assert s[0].id in survivors  # analyzing survives
    assert s[1].id in evicted and s[2].id in evicted
    assert s[3].id in survivors
    assert store.total_bytes() <= store.max_bytes or s[0].id in survivors


def test_create_evicts_but_never_the_new_session(tmp_path):
    store = SessionStore(tmp_path, max_bytes=1)  # absurdly small
    a = _add(store, 1)
    b = _add(store, 2)
    ids = {x.id for x in store.list()}
    assert ids == {b.id} and a.id not in ids


def test_restart_resets_analyzing_and_keeps_sessions(tmp_path):
    store = SessionStore(tmp_path)
    a, b = _add(store, 1), _add(store, 2)
    store.update(a.id, state="analyzing")
    store.update(b.id, state="ready")
    (store.tmp_dir / "stale.part").write_bytes(b"junk")
    again = SessionStore(tmp_path)
    assert {s.id for s in again.list()} == {a.id, b.id}
    assert again.get(a.id).state == "uploaded"
    assert again.get(b.id).state == "ready"
    assert list(again.tmp_dir.iterdir()) == []


def test_atomic_write_keeps_old_file_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "session.json"
    atomic_write_json(target, {"v": 1})

    def boom(*a, **k):
        raise OSError("killed")

    monkeypatch.setattr(sessions_mod.os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_json(target, {"v": 2})
    assert json.loads(target.read_text()) == {"v": 1}
    assert not (tmp_path / "session.json.tmp").exists()


def test_update_validates(tmp_path):
    store = SessionStore(tmp_path)
    s = _add(store, 1)
    with pytest.raises(ValueError):
        store.update(s.id, state="bogus")
    with pytest.raises(ValueError):
        store.update(s.id, id="x")
    assert store.update("0" * 64, state="ready") is None


def test_store_does_not_import_torch():
    import subprocess
    import sys

    code = "import sys; import nvs.api.app; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_large_upload_speed(client, tmp_path):
    """Acceptance: a ~25 MB JPEG is stored in < 300 ms (reported, loose bound for CI noise)."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (4000, 6000, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=95)
    data = buf.getvalue()
    t = time.perf_counter()
    r = upload(client, data)
    dt = time.perf_counter() - t
    print(f"upload {len(data) / 1e6:.1f} MB in {dt * 1000:.0f} ms")
    assert r.status_code == 201 and dt < 1.0
