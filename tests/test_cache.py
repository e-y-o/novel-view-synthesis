import os
import threading
import time

import numpy as np
import pytest
import torch

import nvs.cache as cache_mod
from nvs.cache import DepthCache, cache_key
from nvs.core.types import DepthResult


def make_result(h=24, w=32, seed=0, f_px=1234.5):
    g = torch.Generator().manual_seed(seed)
    depth = 1 + 20 * torch.rand(h, w, generator=g)
    sky = torch.rand(h, w, generator=g) > 0.8
    return DepthResult(depth, f_px, sky, "depthpro2-h", (h // 2, w // 2), ["a note", "another"])


def test_round_trip_bit_exact(tmp_path):
    c = DepthCache(tmp_path)
    r = make_result()
    c.put("ab" * 32, r, image_sha256="deadbeef")
    got = c.get("ab" * 32)
    assert torch.equal(got.depth, r.depth) and got.depth.dtype == torch.float32
    assert torch.equal(got.sky_mask, r.sky_mask)
    assert (got.f_px, got.model, got.infer_size, got.notes) == (r.f_px, r.model, r.infer_size, r.notes)
    assert c.get("cd" * 32) is None
    assert c.size_bytes() > 0
    meta = (tmp_path / "depth" / "ab" / ("ab" * 32) / "meta.json").read_text()
    assert "deadbeef" in meta and '"schema_version": 1' in meta


def test_key_depends_on_f_px_model_precision_and_image():
    base = cache_key("s", "m", "bf16", 1000.0)
    assert cache_key("s", "m", "bf16", 1001.0) != base
    assert cache_key("s", "m", "bf16", 1000.04) == base
    assert cache_key("s", "m", "fp16", 1000.0) != base
    assert cache_key("s", "m2", "bf16", 1000.0) != base
    assert cache_key("t", "m", "bf16", 1000.0) != base
    assert len(base) == 64


def test_interrupted_write_leaves_nothing_visible(tmp_path, monkeypatch):
    c = DepthCache(tmp_path)
    calls = {"n": 0}
    real_save = np.save

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("disk on fire")
        return real_save(*a, **k)

    monkeypatch.setattr(cache_mod.np, "save", flaky)
    with pytest.raises(RuntimeError):
        c.put("ab" * 32, make_result())
    monkeypatch.undo()
    assert c.get("ab" * 32) is None
    assert c.size_bytes() == 0
    assert not [p for p in (tmp_path / "depth").rglob("*") if p.name.startswith(".tmp-")]


def test_corrupt_entry_is_a_miss_and_removed(tmp_path):
    c = DepthCache(tmp_path)
    key = "ab" * 32
    c.put(key, make_result())
    entry = tmp_path / "depth" / "ab" / key
    (entry / "depth.npy").write_bytes(b"garbage")
    assert c.get(key) is None
    assert not entry.exists()

    c.put(key, make_result())
    np.save(entry / "depth.npy", np.zeros((3, 3), np.float32))  # wrong shape
    assert c.get(key) is None and not entry.exists()

    c.put(key, make_result())
    (entry / "meta.json").write_text('{"schema_version": 99}')
    assert c.get(key) is None and not entry.exists()


def test_lru_eviction_keeps_newest_within_cap(tmp_path):
    probe = DepthCache(tmp_path / "probe")
    probe.put("00" * 32, make_result())
    one = probe.size_bytes()
    c = DepthCache(tmp_path / "c", max_bytes=int(one * 2.5))
    keys = [f"{i:02x}" * 32 for i in range(4)]
    for i, k in enumerate(keys):
        c.put(k, make_result(seed=i))
        e = tmp_path / "c" / "depth" / k[:2] / k
        os.utime(e, (1000 + i, 1000 + i))  # strictly increasing age
    assert c.size_bytes() <= int(one * 2.5)
    assert c.get(keys[0]) is None and c.get(keys[1]) is None
    assert c.get(keys[2]) is not None and c.get(keys[3]) is not None


def test_get_refreshes_lru_order(tmp_path):
    probe = DepthCache(tmp_path / "probe")
    probe.put("00" * 32, make_result())
    one = probe.size_bytes()
    c = DepthCache(tmp_path / "c", max_bytes=int(one * 2.5))
    k = [f"{i:02x}" * 32 for i in range(3)]
    for i in range(2):
        c.put(k[i], make_result(seed=i))
        os.utime(tmp_path / "c" / "depth" / k[i][:2] / k[i], (1000 + i, 1000 + i))
    assert c.get(k[0]) is not None  # touch the oldest: now k[1] is the least recently used
    c.put(k[2], make_result(seed=2))
    assert c.get(k[1]) is None and c.get(k[0]) is not None and c.get(k[2]) is not None


def test_disabled_cache(tmp_path):
    c = DepthCache(tmp_path, max_bytes=0)
    c.put("ab" * 32, make_result())
    assert c.get("ab" * 32) is None and not (tmp_path / "depth").exists()


def test_oversized_entry_not_stored(tmp_path):
    c = DepthCache(tmp_path, max_bytes=100)
    c.put("ab" * 32, make_result())
    assert c.get("ab" * 32) is None and c.size_bytes() == 0


def test_concurrent_puts_same_key(tmp_path):
    key = "ab" * 32
    results = [make_result(seed=i) for i in range(2)]
    caches = [DepthCache(tmp_path), DepthCache(tmp_path)]  # separate instances: no shared lock
    errors = []

    def worker(i):
        try:
            for _ in range(20):
                caches[i].put(key, results[i])
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    got = DepthCache(tmp_path).get(key)
    assert got is not None
    assert torch.equal(got.depth, results[0].depth) or torch.equal(got.depth, results[1].depth)


def test_clear_and_env_root(tmp_path, monkeypatch):
    monkeypatch.setenv("NVS_CACHE_DIR", str(tmp_path / "env"))
    c = DepthCache()
    assert c.root == tmp_path / "env"
    c.put("ab" * 32, make_result())
    c.clear()
    assert c.size_bytes() == 0 and c.get("ab" * 32) is None


def test_read_speed_12mp(tmp_path):
    c = DepthCache(tmp_path)
    h, w = 3000, 4000
    r = DepthResult(torch.rand(h, w) + 1, 3000.0, torch.zeros(h, w, dtype=torch.bool), "m", (1500, 2000), [])
    c.put("ab" * 32, r)
    c.get("ab" * 32)  # warm the OS file cache
    t0 = time.perf_counter()
    got = c.get("ab" * 32)
    dt = time.perf_counter() - t0
    print(f"\n12 MP depth cache read: {dt * 1000:.0f} ms")
    assert got.depth.shape == (h, w)
    assert dt < 0.1
