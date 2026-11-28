"""scripts/make_fixture_assets.py writes contract-exact assets for the frontend (T07)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from nvs.preview.layout import ASSET_NAMES, validate_dir

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_fixture_assets.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("make_fixture_assets", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("scene", ["two_layers", "room", "sky_tree"])
def test_fixture_matches_the_preview_contract(script, tmp_path, scene):
    out = tmp_path / scene
    info = script.make_fixture(scene, out, 160, 120)
    assert sorted(p.name for p in out.iterdir()) == sorted(ASSET_NAMES)
    assert validate_dir(out) == info and json.loads((out / "meta.json").read_text()) == info
    assert (info["preview_width"], info["preview_height"]) == (160, 120)
    depth = np.frombuffer((out / "depth.f16").read_bytes(), dtype="<f2")
    assert np.isfinite(depth).all() and depth.min() > 0


def test_sky_tree_has_sky_and_thin_poles(script, tmp_path):
    info = script.make_fixture("sky_tree", tmp_path / "t", 320, 240)
    sky = np.frombuffer((tmp_path / "t" / "sky.u8").read_bytes(), dtype=np.uint8).reshape(240, 320)
    assert 0.05 < (sky == 255).mean() < 0.95
    top_row = sky[5] == 255  # poles cross the sky: a few short non-sky runs on a sky row
    edges = np.flatnonzero(np.diff(top_row.astype(int)) != 0)
    assert len(edges) >= 4
    assert info["depth"]["sky_depth"] is not None


def test_main_cli(script, tmp_path, capsys):
    assert script.main(["--scenes", "two_layers", "--size", "96", "72", "--out", str(tmp_path)]) == 0
    assert "two_layers: 96x72" in capsys.readouterr().out
    assert validate_dir(tmp_path / "two_layers") is not None
