"""Pixel I/O: load (oriented, sRGB, float tensor) and save (sRGB PNG/JPEG, EXIF carried over).

``load_image`` returns RGB float32 in [0, 1], shape [3, H, W], sRGB, with the EXIF Orientation applied, plus
the original EXIF/ICC bytes and a SHA-256 of the file (cache key). ``save_image`` writes PNG or JPEG by file
extension; given the ``LoadedImage`` it came from, it carries the EXIF over (Orientation reset, pixel
dimensions updated, thumbnail/MakerNote/GPS dropped).
"""

from __future__ import annotations

import hashlib
import struct
import warnings
import zlib
from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageCms, ImageOps

from nvs.core import _pillow  # noqa: F401  (registers HEIF support)
from nvs.core.exif import EXIF_IFD, ORIENTATION, PIXEL_X_DIM, PIXEL_Y_DIM, ImageMeta, extract_image_meta

MAX_PIXELS = 100_000_000
GPS_IFD = 0x8825
MAKER_NOTE = 0x927C  # holds absolute offsets that break when EXIF is rewritten
USER_COMMENT = 0x9286
IMAGE_DESCRIPTION = 0x010E

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


@dataclass
class LoadedImage:
    rgb: torch.Tensor  # [3, H, W] float32 in [0, 1], sRGB, oriented
    meta: ImageMeta  # oriented size and focal length (see exif.py)
    exif_bytes: bytes | None  # original EXIF block (as stored, Orientation unchanged)
    icc_profile: bytes | None  # original embedded ICC profile (pixels are already converted to sRGB)
    path: Path
    sha256: str  # hex digest of the file bytes
    notes: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _srgb_profile() -> ImageCms.ImageCmsProfile:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))


@lru_cache(maxsize=1)
def _srgb_icc_bytes() -> bytes:
    return _srgb_profile().tobytes()


def _is_srgb(profile: ImageCms.ImageCmsProfile) -> bool:
    return "srgb" in ImageCms.getProfileDescription(profile).lower()


def _orient_array(a: np.ndarray, orientation: int | None) -> np.ndarray:
    """EXIF orientation for an [H, W, C] array (same mapping as PIL's ImageOps.exif_transpose)."""
    if orientation == 2:
        return a[:, ::-1]
    if orientation == 3:
        return np.rot90(a, 2)
    if orientation == 4:
        return a[::-1]
    if orientation == 5:
        return a.transpose(1, 0, 2)
    if orientation == 6:
        return np.rot90(a, 3)
    if orientation == 7:
        return np.rot90(a.transpose(1, 0, 2), 2)
    if orientation == 8:
        return np.rot90(a, 1)
    return a


def _read_png16(data: bytes) -> tuple[np.ndarray, int] | None:
    """Decode a non-interlaced 16-bit grey/RGB(A) PNG to uint16 [H, W, C]; None if not applicable.

    Pillow returns 16-bit RGB PNGs as 8-bit, so this keeps the full precision. The Average/Paeth filters run
    in plain Python, so this path is slow for large images (it is rare in practice).
    """
    if data[:8] != _PNG_SIG:
        return None
    pos, idat, header = 8, [], None
    while pos + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[pos : pos + 8])
        body = data[pos + 8 : pos + 8 + length]
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
        pos += 12 + length
    if header is None:
        return None
    w, h, depth, ctype, _, _, interlace = header
    if depth != 16 or interlace or ctype not in (0, 2, 4, 6):
        return None
    ch = {0: 1, 2: 3, 4: 2, 6: 4}[ctype]
    bpp, stride = 2 * ch, w * 2 * ch
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), dtype=np.uint8).reshape(h, stride + 1)
    out = np.empty((h, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int64)
    for y in range(h):
        ftype, line = int(raw[y, 0]), raw[y, 1:].astype(np.int64)
        if ftype == 0:
            cur = line
        elif ftype == 1:  # Sub: running sum along each byte lane
            cur = line.reshape(w, bpp).cumsum(0).reshape(-1) & 255
        elif ftype == 2:  # Up
            cur = (line + prev) & 255
        else:  # Average (3) / Paeth (4): sequential in x
            cur_l, up, ln = [0] * stride, prev.tolist(), line.tolist()
            for i in range(stride):
                a = cur_l[i - bpp] if i >= bpp else 0
                b = up[i]
                if ftype == 3:
                    pred = (a + b) >> 1
                else:
                    c = up[i - bpp] if i >= bpp else 0
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                cur_l[i] = (ln[i] + pred) & 255
            cur = np.array(cur_l, dtype=np.int64)
        prev = cur
        out[y] = cur
    arr = np.frombuffer(out.tobytes(), dtype=">u2").reshape(h, w, ch).astype(np.uint16)
    return arr, ctype


def _flatten_float(arr: np.ndarray) -> np.ndarray:
    """float [H, W, C] in [0, 1] with C in (1, 2, 3, 4) -> RGB float [H, W, 3]; alpha flattened on white."""
    c = arr.shape[2]
    if c in (2, 4):
        color, alpha = arr[:, :, : c - 1], arr[:, :, c - 1 :]
        arr = color * alpha + (1 - alpha)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr


def _exif_bytes(im: Image.Image) -> bytes | None:
    raw = im.info.get("exif")
    if raw:
        return bytes(raw)
    exif = im.getexif()
    return exif.tobytes() if len(exif) else None


def _icc_to_srgb(im: Image.Image, icc: bytes, notes: list[str]) -> Image.Image:
    try:
        src = ImageCms.ImageCmsProfile(BytesIO(icc))
        if _is_srgb(src):
            return im
        desc = ImageCms.getProfileDescription(src).strip() or "embedded profile"
        out = ImageCms.profileToProfile(
            im, src, _srgb_profile(), renderingIntent=ImageCms.Intent.PERCEPTUAL, outputMode="RGB"
        )
        notes.append(f"converted from {desc}")
        return out
    except (ImageCms.PyCMSError, OSError) as e:
        notes.append(f"warning: could not apply embedded ICC profile ({e}); assuming sRGB")
        return im.convert("RGB")


def _pil_to_rgb_uint8(im: Image.Image, icc: bytes | None, notes: list[str]) -> np.ndarray:
    """Any 8-bit PIL image -> uint8 [H, W, 3] sRGB."""
    has_alpha = im.mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in im.info
    if has_alpha:
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        im = bg
        notes.append("warning: alpha channel flattened onto white")
    if im.mode == "CMYK":
        im = _icc_to_srgb(im, icc, notes) if icc else im.convert("RGB")
    elif im.mode == "RGB":
        if icc:
            im = _icc_to_srgb(im, icc, notes)
    else:
        im = im.convert("RGB")
    return np.array(im)


def load_image(path: str | Path, device: str | torch.device = "cpu") -> LoadedImage:
    """Load an image as an oriented sRGB float tensor. Refuses images over 100 MP (ValueError)."""
    path = Path(path)
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    notes: list[str] = []
    try:
        im = Image.open(BytesIO(data))
    except Image.DecompressionBombError as e:
        raise ValueError(f"{path.name}: image too large ({e})") from e
    with im:
        if im.width * im.height > MAX_PIXELS:
            raise ValueError(
                f"{path.name}: {im.width}x{im.height} = {im.width * im.height / 1e6:.0f} MP exceeds the "
                f"{MAX_PIXELS // 1_000_000} MP limit"
            )
        meta = extract_image_meta(path)
        icc = im.info.get("icc_profile")
        exif_bytes = _exif_bytes(im)
        orientation = im.getexif().get(ORIENTATION)
        im.load()

        png16 = _read_png16(data) if im.format == "PNG" else None
        if png16 is not None:
            arr16, _ = png16
            if icc and not _is_srgb(ImageCms.ImageCmsProfile(BytesIO(icc))):
                notes.append("warning: embedded ICC profile not applied to 16-bit PNG; assuming sRGB")
            arr = _flatten_float(arr16.astype(np.float32) / 65535.0)
            if arr16.shape[2] in (2, 4):
                notes.append("warning: alpha channel flattened onto white")
            arr = _orient_array(arr, orientation)
            rgb = torch.from_numpy(np.ascontiguousarray(arr)).to(device).permute(2, 0, 1)
        elif im.mode.startswith("I;16"):  # 16-bit greyscale via Pillow
            a = np.asarray(im).astype(np.float32) / 65535.0
            arr = _orient_array(_flatten_float(a[:, :, None]), orientation)
            rgb = torch.from_numpy(np.ascontiguousarray(arr)).to(device).permute(2, 0, 1)
        else:
            oriented = ImageOps.exif_transpose(im)
            arr8 = _pil_to_rgb_uint8(oriented, icc, notes)
            rgb = torch.from_numpy(arr8).to(device).permute(2, 0, 1).float().div_(255.0)
    rgb = rgb.float().contiguous()

    if (rgb.shape[2], rgb.shape[1]) != (meta.width, meta.height):
        raise ValueError(
            f"{path.name}: oriented size {rgb.shape[2]}x{rgb.shape[1]} disagrees with metadata "
            f"{meta.width}x{meta.height}"
        )
    return LoadedImage(
        rgb=rgb, meta=meta, exif_bytes=exif_bytes, icc_profile=icc, path=path, sha256=sha, notes=notes
    )


def _to_uint8(x: torch.Tensor) -> np.ndarray:
    return (x.detach().float().clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()


def _output_exif(
    source: LoadedImage | None, width: int, height: int, comment: str | None, keep_gps: bool
) -> Image.Exif | None:
    if source is None or not source.exif_bytes:
        if not comment:
            return None
        exif = Image.Exif()
    else:
        exif = Image.Exif()
        exif.load(source.exif_bytes)
        exif[ORIENTATION] = 1
        sub = exif.get_ifd(EXIF_IFD)
        sub[PIXEL_X_DIM] = width
        sub[PIXEL_Y_DIM] = height
        sub.pop(MAKER_NOTE, None)
        exif[EXIF_IFD] = sub
        if not keep_gps:
            exif.pop(GPS_IFD, None)
        # The embedded thumbnail lives in IFD1, which Pillow does not write back.
    if comment:
        sub = exif.get_ifd(EXIF_IFD)
        sub[USER_COMMENT] = b"ASCII\x00\x00\x00" + comment.encode("ascii", "replace")
        exif[EXIF_IFD] = sub
        exif[IMAGE_DESCRIPTION] = comment.encode("ascii", "replace").decode("ascii")
    return exif


def save_image(
    path: str | Path,
    rgb: torch.Tensor,
    *,
    alpha: torch.Tensor | None = None,
    source: LoadedImage | None = None,
    quality: int = 95,
    comment: str | None = None,
    keep_gps: bool = False,
    png_compress_level: int = 3,
) -> None:
    """Write rgb [3, H, W] float in [0, 1] (sRGB) as PNG or JPEG (by extension), full resolution.

    alpha [H, W] in [0, 1]: PNG becomes RGBA; JPEG ignores it with a warning. With ``source`` the original
    EXIF is carried over (Orientation=1, pixel dimensions updated, thumbnail and MakerNote dropped, GPS
    dropped unless ``keep_gps``); ``comment`` is stored as UserComment/ImageDescription. PNG uses zlib level
    ``png_compress_level`` (3: much faster than the default 6 at 12 MP, files a bit larger).
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg"):
        raise ValueError(f"unsupported output extension {ext!r} (use .png, .jpg or .jpeg)")
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"rgb must be [3, H, W], got {tuple(rgb.shape)}")
    arr = _to_uint8(rgb).transpose(1, 2, 0)
    height, width = arr.shape[:2]
    exif = _output_exif(source, width, height, comment, keep_gps)
    kwargs = {"icc_profile": _srgb_icc_bytes()}
    if exif is not None:
        kwargs["exif"] = exif
    if ext == ".png":
        if alpha is not None:
            arr = np.concatenate([arr, _to_uint8(alpha)[:, :, None]], axis=2)
        Image.fromarray(np.ascontiguousarray(arr)).save(
            path, format="PNG", compress_level=png_compress_level, **kwargs
        )
    else:
        if alpha is not None:
            warnings.warn("JPEG has no alpha channel; alpha ignored", UserWarning, stacklevel=2)
        Image.fromarray(np.ascontiguousarray(arr)).save(path, format="JPEG", quality=quality, **kwargs)
