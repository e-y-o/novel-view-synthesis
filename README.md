# Novel View Synthesis

Fast single-image novel view synthesis with a local web app and a command line tool.

Pipeline: depth estimation (EXIF-aware) → analytic reprojection → depth-weighted splatting → hole
classification. Later milestones add a classical background fill (interactive preview) and generative infill.

**Status: milestone 2 (local web app).** Drop a photo in the browser, wait a few seconds for depth, then drag to
orbit a live 3D preview; a coverage meter shows how much of the new view will be invented, and "Render full
resolution" produces the authoritative result (the same as `nvs render`) as a PNG that keeps the photo's EXIF. Holes
are **marked, not filled**: the background layer and generative infill come in later milestones.

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

## Web app

```powershell
uv run nvs serve            # http://127.0.0.1:8765, the real depth model
uv run nvs serve --open     # also opens your browser
uv run nvs serve --stub     # no GPU, fake depth: for trying the UI and for tests
```

The server listens on loopback only and rejects requests that do not come from its own page, so no firewall prompt
appears and other websites cannot drive it. On the first start with the real model the weights are downloaded
(about 3.6 GB) and the Depth Pro 2 research-only licence notice appears (accept it once with `--accept-license`,
as for `nvs render`). The model then loads in the background (about 7 s): the page says "Loading the depth model"
and you can already add a photo.

**Using it.** Drop or choose a photo (JPEG, PNG, WebP, HEIC) or paste one. Progress shows the phases; the 3D preview
opens when depth is ready (about 2.5 s for a 12 MP photo, under 1 s if it was analysed before). Your photo never
leaves the computer. The address (`#/s/...`) identifies the photo, so a reload keeps it; "Recent" reopens or deletes
past photos.

| Action | Mouse / touch | Keyboard (preview focused) |
|---|---|---|
| Orbit | drag | arrow keys (shift: 5 degrees) |
| Pan | right-drag, middle-drag or shift-drag; two fingers | |
| Dolly | wheel; pinch | `+` / `-` |
| Reset view | double-click, "Reset view" | `0` or Home |
| Pick the orbit pivot | alt + double-click, "Pick pivot" | `p`, then click |
| Render full resolution | "Render full resolution" | Enter |
| Compare with the result | Preview / Result buttons | hold Space |

The coverage meter shows how much of the view is hidden behind objects, stretched, or outside the photo. Above 12 %
the quality may drop; above 25 % large areas would be invented. "Limit moves" makes the view resist moves that would
leave more than the chosen share of holes and ease back when you let go (the area outside the photo is not counted
unless you ask).

**For contributors.** Node 20+ is needed to build the browser app in `web/` (TypeScript, Vite, Preact, raw WebGL2):

```powershell
cd web
npm ci
npm run build                 # writes web/dist, which `nvs serve` hosts at /
```

Frontend development (hot reload, `/api` proxied to the Python server):

```powershell
uv run nvs serve --stub --dev          # terminal 1: --dev also accepts the Vite page's origin
cd web; npm run dev                    # terminal 2: http://localhost:5173
npm run fixtures                       # synthetic preview assets in web/fixtures/ (no server or model needed)
                                       # renderer dev page: http://localhost:5173/dev/renderer.html?scene=two_layers
npm run typecheck; npm test; npm run lint; npm run format
```

Browser tests (Playwright, in `web/e2e/`) run the built app against `nvs serve --stub --dev` on port 8799 with a
fresh data directory, using generated images only:

```powershell
cd web
npm run e2e:chromium          # builds, then runs the Chromium suite (about 30 s); uses the installed Google Chrome
npm run e2e                   # all browsers; Firefox and WebKit first need: npx playwright install firefox webkit
$env:PLAYWRIGHT_REAL = "1"; npm run e2e:chromium -- real   # opt-in: real depth model, 12 MP upload timing
```

`npx playwright install` downloads the browsers (a few hundred MB, stored outside the repo); Chromium does not need it
because the config uses the installed Chrome (`PW_CHANNEL=` empty switches to Playwright's own Chromium). Reports and
artefacts go to `web/e2e-results/` (not tracked). `uv run nvs export-preview photo.jpg out_dir/` writes the browser
preview assets for a real photo; `uv run python scripts/bench_server.py` benchmarks the whole server with the real
model. `dist`, `fixtures`, `e2e-results` and `node_modules` are not tracked.

## Known limits

- Holes are only **marked**, in the web app as in the CLI. Filling them (background fill, generative infill) is
  planned. The 3D preview shows them as the background colour.
- The browser preview is a depth mesh: thin structures (wires, fences) and depth edges can tear or smear a little
  more than in the full-resolution render. Only Chromium has been tested; Firefox and Safari are untested.
- Large moves create large holes; the quality of anything painted into them degrades quickly above ~25 % of
  the view. Telephoto photos have a very narrow field of view, so even a few degrees leave a wide
  out-of-frame strip.
- Depth comes from a single image: thin structures, reflections and the area right at depth edges can show
  halos or small unfilled specks.
- Depth Pro 2 H+ works at up to 4 MP (larger photos are downscaled for depth and the depth is guided-upsampled
  to full resolution); 24 MP photos render in about 9 s the first time.

## Troubleshooting

- *Port already in use*: `uv run nvs serve --port 8766` (or set `NVS_PORT`).
- *The page says "Loading the depth model" for a while*: the worker starts in the background (about 7 s, longer on the
  very first run while the weights download). You can add a photo meanwhile; it waits for the model.
- *"3D preview not available"* or a slow, jerky preview: the browser has WebGL 2 turned off or uses software
  rendering. Enable hardware acceleration in the browser settings and update the graphics driver; try Chrome or Edge.
- *"Not enough GPU memory"* in the page: close other GPU applications and retry (the model needs about 10.5 GiB at
  its peak).
- *A firewall prompt*: none should appear; the server only listens on 127.0.0.1. Do not expose it (`--host` with a
  non-loopback address needs `--i-know-this-is-unsafe` and gives anyone on the network control of your GPU and disk).

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
