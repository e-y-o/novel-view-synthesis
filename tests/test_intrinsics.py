import math

import pytest

from nvs.core.exif import ImageMeta, f_px_from_35mm
from nvs.core.intrinsics import resolve_intrinsics

W, H = 4000, 3000


def meta(source="none", f_px=None, warnings=(), raw=None):
    return ImageMeta(W, H, f_px, source, None, list(warnings), dict(raw or {}))


F35 = f_px_from_35mm(26, W, H)
CROP = "cropped: EXIF size 4000x2250 vs file 3000x3000; 35mm-equivalent refers to the uncropped frame"


def test_override_f_px_wins():
    r = resolve_intrinsics(meta("35mm", F35), override_f_px=1234.0, estimator=lambda: 999.0)
    assert (r.f_px, r.source, r.assumed) == (1234.0, "override", False)


def test_override_35mm():
    r = resolve_intrinsics(meta("35mm", F35), override_35mm=50)
    assert r.f_px == pytest.approx(50 * 5000 / 43.267)
    assert (r.source, r.assumed) == ("override", False)


def test_invalid_override_rejected():
    with pytest.raises(ValueError):
        resolve_intrinsics(meta(), override_f_px=-1)
    with pytest.raises(ValueError):
        resolve_intrinsics(meta(), override_35mm=float("nan"))


def test_35mm_wins_on_normal_file():
    r = resolve_intrinsics(meta("35mm", F35, raw={"f_px_focal_plane": 2000.0}), estimator=lambda: 1.0)
    assert (r.f_px, r.source, r.assumed) == (F35, "35mm", False)


def test_crop_with_focal_plane_prefers_focal_plane():
    r = resolve_intrinsics(meta("35mm", F35, [CROP], {"f_px_focal_plane": 3500.0}))
    assert (r.f_px, r.source, r.assumed) == (3500.0, "focal_plane", False)
    assert any("focal-plane" in w for w in r.warnings) and CROP in r.warnings


def test_crop_without_focal_plane_keeps_35mm_with_warning():
    r = resolve_intrinsics(meta("35mm", F35, [CROP]))
    assert (r.f_px, r.source) == (F35, "35mm")
    assert CROP in r.warnings


def test_only_focal_plane():
    r = resolve_intrinsics(meta("focal_plane", 2800.0, raw={"f_px_focal_plane": 2800.0}))
    assert (r.f_px, r.source, r.assumed) == (2800.0, "focal_plane", False)


def test_estimator_when_nothing_else():
    r = resolve_intrinsics(meta(), estimator=lambda: 3100.0)
    assert (r.f_px, r.source, r.assumed) == (3100.0, "estimator", True)
    assert "focal length estimated from image" in r.warnings


def test_default_26mm():
    r = resolve_intrinsics(meta())
    assert r.f_px == pytest.approx(26 * 5000 / 43.267) and r.f_px == pytest.approx(3004.6, abs=0.1)
    assert (r.source, r.assumed) == ("default", True) and r.warnings


@pytest.mark.parametrize("bad", [float("nan"), 0.0, -5.0, math.inf, None])
def test_bad_estimator_ignored(bad):
    r = resolve_intrinsics(meta(), estimator=lambda: bad)
    assert r.source == "default"


def test_estimator_not_called_when_exif_present():
    def boom():
        raise AssertionError("estimator must not run")

    assert resolve_intrinsics(meta("35mm", F35), estimator=boom).source == "35mm"


def test_compare_estimator_is_informational_only():
    r = resolve_intrinsics(meta("35mm", F35), estimator=lambda: F35 * 0.6, compare_estimator=True)
    assert (r.f_px, r.source) == (F35, "35mm")
    assert any(w.startswith("info:") for w in r.warnings)
    r = resolve_intrinsics(meta("35mm", F35), estimator=lambda: F35 * 0.9, compare_estimator=True)
    assert not any(w.startswith("info:") for w in r.warnings)


def test_meta_warnings_not_mutated():
    m = meta("none")
    resolve_intrinsics(m)
    assert m.warnings == []
