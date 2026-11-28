# Novel View Synthesis

Fast single-image novel view synthesis with a web GUI (the GUI is planned; today there is a command line tool).

Pipeline: depth estimation (EXIF-aware) → analytic reprojection → depth-weighted splatting → hole
classification. Later milestones add a classical background fill (interactive preview) and generative infill.

**Status: milestone 1 (core library and CLI).** `nvs render` turns one photo into a new view at the photo's full
resolution, with the holes (disocclusions, stretched surfaces, areas outside the original frame) **marked, not
filled**. There is no infill yet.

## Requirements

- Windows or Linux, an NVIDIA GPU with 16 GB of VRAM (developed on an RTX 5070 Ti, Blackwell `sm_120`) and a
  recent driver (`nvidia-smi` works). The depth model needs about 10.5 GiB at 4 MP.
- [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 and every dependency).

## Setup

```powershell
winget install --id astral-sh.uv   # if uv is missing
uv sync                            # creates .venv with PyTorch cu130, Depth Pro 2, Pillow, ...
```

PyTorch comes from the CUDA 13.0 wheel index (configured in `pyproject.toml`); the stable cu130 build includes
Blackwell kernels.

### Model weights and licence

Depth Pro 2 weights are **not** part of this repository. They are downloaded to your machine the first time the
depth model runs. Apple releases them under a **research-only licence** (`LICENSE_MODEL` in
<https://github.com/apple-aiml-research/ml-depth-pro-2>); by using them you agree to that licence. `nvs render`
shows this notice and asks you to accept it once (`--accept-license`, or `NVS_ACCEPT_DEPTHPRO2_LICENSE=1` for
scripts). The choice is remembered in `data/` (not tracked). This project is not endorsed by Apple.

## Usage

```powershell
uv run nvs render photo.jpg --yaw 5 --out out.png
uv run nvs render photo.heic --yaw -8 --pitch 3 --dolly 0.3 --hole magenta --out out.png
uv run nvs render photo.jpg --yaw 5 --out out.png --save-mask mask.png --save-depth-vis depth.png --timings
```

- `--yaw` / `--pitch` (degrees) orbit the camera around a pivot at the nearest part of the scene: positive yaw
  moves the camera to the right (the scene appears to shift left), positive pitch moves it up. `--dolly` moves
  toward the pivot (metres), `--pan-x` / `--pan-y` shift sideways, `--pivot` sets the pivot depth.
- Holes are written as transparent pixels in a PNG (`--hole alpha`, the default), or painted `black` /
  `magenta`. JPEG output cannot be transparent, so it falls back to magenta with a warning.
- `--clamp` shrinks the move so that at most `--max-hole-fraction` (default 12 %) of the view is holes.
  Without it a warning appears when more than 25 % is expected.
- `--f-px` / `--focal-35mm` override the focal length. By default it comes from EXIF (35 mm-equivalent, then the
  focal-plane tags), then from the depth model's estimate, then a 26 mm default; the summary says which one was
  used.
- `--save-mask`, `--save-classes`, `--save-depth-vis` write the feathered hole mask, a colour-coded hole class map
  (red disocclusion, green stretch, blue out of frame) and a depth visualisation. `--timings` prints a per-stage
  table. `uv run nvs render --help` lists everything.

Depth is cached in `data/cache` (up to 5 GiB, `NVS_CACHE_DIR` moves it, `--no-cache` skips it), so changing only
the view is fast: about 1 s for a 12 MP photo, against about 7 s the first time (model load + inference).

Supported inputs: JPEG, PNG, HEIC/HEIF, WebP (EXIF orientation and embedded colour profiles such as Display P3
are handled; output is sRGB, keeps the source EXIF without GPS). Outputs: PNG or JPEG.

## Web app (work in progress)

`nvs serve` starts a local server on `127.0.0.1:8765` (loopback only; it rejects requests that do not come from
its own page). The browser app lives in `web/` (TypeScript, Vite, Preact, raw WebGL2); Node 20+ is needed to
build it.

```powershell
cd web
npm ci
npm run build                 # writes web/dist, which `nvs serve` hosts at /
cd ..
uv run nvs serve              # real depth model (asks to accept the licence once)
uv run nvs serve --stub       # no GPU, fake depth: for UI work and tests
```

Frontend development (hot reload, `/api` proxied to the Python server):

```powershell
uv run nvs serve --stub --dev          # terminal 1: --dev also accepts the Vite page's origin
cd web; npm run dev                    # terminal 2: http://localhost:5173
npm run fixtures                       # synthetic preview assets in web/fixtures/ (no server or model needed)
                                       # renderer dev page: http://localhost:5173/dev/renderer.html?scene=two_layers
npm run typecheck; npm test; npm run lint; npm run format
```

`uv run nvs export-preview photo.jpg out_dir/` writes the browser preview assets (colour, depth, sky mask, meta)
for a real photo. `dist`, `fixtures` and `node_modules` are not tracked.

## Known limits

- Holes are only **marked**. Filling them (background fill, generative infill) is planned.
- Large moves create large holes; the quality of anything painted into them degrades quickly above ~25 % of
  the view. Telephoto photos have a very narrow field of view, so even a few degrees leave a wide
  out-of-frame strip.
- Depth comes from a single image: thin structures, reflections and the area right at depth edges can show
  halos or small unfilled specks.
- Depth Pro 2 H+ works at up to 4 MP (larger photos are downscaled for depth and the depth is guided-upsampled
  to full resolution); 24 MP photos render in about 9 s the first time.

## Troubleshooting

- *no CUDA GPU available*: install a current NVIDIA driver and check `nvidia-smi`. Without a GPU only
  `--device cpu --depth-file depth.npy` works (depth in metres at the image size).
- *GPU out of memory*: close other GPU applications (browsers and games use VRAM), or use `--depth-model s`
  (smaller, faster, lower quality). The process caps itself at 92 % of VRAM to keep the system responsive.
- *First run is slow*: the weights (about 3.6 GB) are downloaded once, and the first depth inference loads the
  model (about 5 s).
- *Wrong or missing focal length*: look at the `focal length:` line of the summary; pass `--focal-35mm` or
  `--f-px` if it says `default` or `estimator`.

## Development

```powershell
uv run ruff check           # lint
uv run ruff format --check  # formatting
uv run pytest -m "not slow" # fast tests (about 15 s; GPU tests skip without CUDA)
uv run pytest               # everything, including the depth model and the sample photos in data/samples
uv run python scripts/bench_render.py --with-depth   # stage timings at 2 / 12 / 24 MP
```

Code lives in `src/nvs/` (`core` camera, EXIF, I/O and shared types; `depth` model wrapper and refinement;
`render` splatting, holes and limits; `cli`). `tests/synth.py` is a small ray-caster that provides ground-truth
views for the tests.

## Third-party models and licences

- **Depth Pro 2** (<https://github.com/apple-aiml-research/ml-depth-pro-2>): the Python package is installed
  from Apple's repository (not copied here) under Apple's sample-code licence; the **model weights** are under
  Apple's research-only model licence (non-commercial scientific research and academic development). Apple
  Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License
  Agreement. The depth backend is an interface (`nvs.depth.base.DepthBackend`) so another model can be used
  where research-only terms do not fit. This project is not endorsed by Apple.
- The optional `infill` dependency group (for the later generative infill milestone) pulls in `diffusers` and
  friends; model weights for that stage are likewise never part of this repository and have their own terms
  (for example Stable Diffusion XL: CreativeML Open RAIL++-M).

## Licence of this repository

This project's own code is licensed under the [Apache License 2.0](LICENSE). That covers this code only: model
weights are separately licensed (see above) and are never distributed with it.
