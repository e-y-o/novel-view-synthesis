"""Server scaffold: health, Host/Origin guards, headers, upload limit, lazy CLI import."""

import subprocess
import sys

import pytest
from fastapi import UploadFile, WebSocket
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from nvs.api.app import create_app
from nvs.api.config import ServerConfig
from nvs.api.uploads import read_upload_limited
from nvs.cli.main import main

PORT = 8765
WS_HOST = {"Host": f"127.0.0.1:{PORT}"}


def make_client(tmp_path, **cfg) -> TestClient:
    cfg.setdefault("web_dist", tmp_path / "no-dist")
    config = ServerConfig(port=PORT, data_dir=tmp_path, stub=True, **cfg)
    app = create_app(config)

    @app.post("/test/echo")
    async def echo() -> dict:
        return {"echo": True}

    @app.websocket("/test/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        await socket.send_json({"hello": 1})
        await socket.close()

    @app.post("/test/upload")
    async def upload(file: UploadFile) -> dict:
        stored = await read_upload_limited(file, 1024, dest_dir=tmp_path / "uploads")
        return {"size": stored.size, "sha256": stored.sha256}

    return TestClient(app, base_url=f"http://127.0.0.1:{PORT}")


@pytest.fixture
def client(tmp_path):
    with make_client(tmp_path) as c:
        yield c


def test_health_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["stub"] is True and "version" in body


@pytest.mark.parametrize("host", ["localhost", "[::1]", "127.0.0.1"])
def test_loopback_host_names_accepted(client, host):
    assert client.get("/api/health", headers={"Host": f"{host}:{PORT}"}).status_code == 200


@pytest.mark.parametrize(
    "host",
    ["evil.example", f"evil.example:{PORT}", "127.0.0.1", "127.0.0.1:1", f"127.0.0.1.evil.test:{PORT}"],
)
def test_bad_host_rejected_everywhere(client, host):
    for path in ["/api/health", "/", "/api/nope"]:
        r = client.get(path, headers={"Host": host})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "bad_host"


def test_bad_origin_post_rejected(client):
    r = client.post("/test/echo", headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    assert client.post("/test/echo", headers={"Origin": "null"}).status_code == 403


def test_good_or_missing_origin_post_passes(client):
    assert client.post("/test/echo").status_code == 200  # curl / tests: no Origin
    assert client.post("/test/echo", headers={"Origin": f"http://localhost:{PORT}"}).status_code == 200
    assert client.post("/test/echo", headers={"Origin": f"http://127.0.0.1:{PORT}"}).status_code == 200


def test_origin_not_checked_for_get(client):
    assert client.get("/api/health", headers={"Origin": "http://evil.example"}).status_code == 200


def test_dev_origin_only_with_dev_flag(tmp_path):
    dev = "http://localhost:5173"
    with make_client(tmp_path) as c:
        assert c.post("/test/echo", headers={"Origin": dev}).status_code == 403
    with make_client(tmp_path, dev=True) as c:
        assert c.post("/test/echo", headers={"Origin": dev}).status_code == 200


def test_websocket_bad_origin_refused(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/test/ws", headers={**WS_HOST, "Origin": "http://evil.example"}):
            pass


def test_websocket_good_origin_and_no_origin(client):
    # TestClient builds websocket URLs on "testserver", so send the real Host explicitly.
    with client.websocket_connect("/test/ws", headers=WS_HOST) as ws:
        assert ws.receive_json() == {"hello": 1}
    with client.websocket_connect(
        "/test/ws", headers={**WS_HOST, "Origin": f"http://localhost:{PORT}"}
    ) as ws:
        assert ws.receive_json() == {"hello": 1}


def test_websocket_bad_host_refused(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/test/ws", headers={"Host": "evil.example"}):
            pass


def test_no_cors_headers(client):
    r = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert not any(k.lower().startswith("access-control-") for k in r.headers)
    r = client.options(
        "/test/echo",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert not any(k.lower().startswith("access-control-") for k in r.headers)


def test_security_headers(client):
    r = client.get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert r.headers["cache-control"] == "no-store"
    page = client.get("/")
    assert page.status_code == 200 and "frontend is not built" in page.text
    assert "default-src 'self'" in page.headers["content-security-policy"]
    assert page.headers["x-content-type-options"] == "nosniff"


def test_unknown_api_route_is_json_404(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_static_dist_is_served_when_built(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>built</title>")
    with make_client(tmp_path, web_dist=dist) as c:
        r = c.get("/")
        assert "built" in r.text and "content-security-policy" in r.headers
        assert c.get("/api/health").status_code == 200


def test_upload_within_limit(client, tmp_path):
    r = client.post("/test/upload", files={"file": ("a.bin", b"x" * 1000)})
    assert r.status_code == 200 and r.json()["size"] == 1000
    assert len(r.json()["sha256"]) == 64


def test_oversized_upload_413_and_not_stored(client, tmp_path):
    r = client.post("/test/upload", files={"file": ("big.bin", b"x" * 5000)})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "http_413"
    assert list((tmp_path / "uploads").iterdir()) == []


def test_config_env_and_overrides():
    cfg = ServerConfig.from_env({"NVS_PORT": "9000", "NVS_STUB": "1", "NVS_PRELOAD_DEPTH": "0"})
    assert (cfg.port, cfg.stub, cfg.preload_depth) == (9000, True, False)
    cfg = cfg.with_overrides(port=9001, stub=None)
    assert (cfg.port, cfg.stub) == (9001, True)
    assert cfg.allowed_hosts == {"127.0.0.1:9001", "localhost:9001", "[::1]:9001"}


def test_serve_refuses_non_loopback_host(capsys):
    assert main(["serve", "--host", "0.0.0.0", "--stub"]) == 1
    assert "--i-know-this-is-unsafe" in capsys.readouterr().err


def test_serve_help_is_fast_and_lazy():
    code = (
        "import sys, time; t = time.perf_counter()\n"
        "from nvs.cli.main import main\n"
        "try:\n    main(['serve', '--help'])\nexcept SystemExit:\n    pass\n"
        "dt = time.perf_counter() - t\n"
        "heavy = [m for m in ('torch', 'fastapi', 'uvicorn') if m in sys.modules]\n"
        "print(dt, heavy)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    dt, heavy = out.strip().splitlines()[-1].split(" ", 1)
    assert float(dt) < 1.0, out
    assert heavy == "[]", out
