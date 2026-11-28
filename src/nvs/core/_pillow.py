"""Registers optional Pillow plugins once (import for the side effect). Torch-free."""

MAX_PIXELS = 100_000_000  # largest accepted image, in pixels

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # HEIC then fails to open; everything else still works
    pass
