# SPHEREx Sky-Motion Viewer

Type a name or coordinates and get every NASA SPHEREx infrared image of that patch of
sky, aligned and played as a flipbook. Stars that move or change brightness stand out.
Click any star for its spectrum (0.75–5 µm) and light curve, computed on the fly.

## Run it

```bash
py -3.12 -m venv .venv
```
```bash
.venv\Scripts\python -m pip install -r backend/requirements.txt
```
```bash
.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --port 8000
```

Open http://localhost:8000 (the viewer) or http://localhost:8000/docs (the API).

## Features

| Challenge idea | What the tool does |
|---|---|
| Easy to update with new releases | Collections live in config: `SPHEREX_COLLECTIONS=["spherex_dr2","spherex_dr1","spherex_qr3"]` in `backend/.env`. Every query is live, so new data shows up within 24 h (metadata cache). |
| Use the 102 bands / per-detector stacking | Each visit is stacked for **all detectors** and for **each detector D1–D6** separately (6 broad colours, 0.75–5 µm), plus an **RGB false-colour** view (blue = D1+D2, green = D3+D4, red = D5+D6) where cold brown dwarfs look red. |
| Spectra on the fly | Aperture photometry of every exposure at its own wavelength → spectrum. **Click any star** in the frame: its spectrum and light curve are measured from the cached images in a few seconds, with no new downloads. |
| Add or subtract images taken at different times | **Stack per visit**, **Difference** (each visit minus the first: movers become black/white pairs), **A − B** (any two visits), **Blink A ↔ B**, **Deep** (all visits added), **Single exposures**. |
| Choose the field of view | 1′ to 30′. |
| Overlays from other telescopes | **Gaia DR3** (circles move with each visit's date by proper motion; fast movers get orange circles + 10-year arrows), **WISE** (CatWISE2020 / AllWISE), an **Aladin Lite** sky map (AllWISE, SPHEREx all-sky, 2MASS, DSS backgrounds) with the field box and both catalogues, and a **WiseView** link for 10 years of WISE motion. |
| Query by URL / embeddable | `?name=Barnard's Star` or `?ra=…&dec=…&fov=…`, plus `&band=`, `&mode=`. **Copy link** and **Copy embed code** buttons; `&embed=1` gives a compact view for an `<iframe>`. |
| Metadata and contamination flags | Per-visit and per-exposure tables: date, detector, wavelength at the target, flux, % masked, and each SPHEREx flag plane near the target (red = set inside the photometry aperture). Click a row to view that exposure. |
| Astropy / Aladin | Astropy, astroquery, reproject and photutils on the server; Aladin Lite v3 in the browser. |

| (extra) AI help understanding the output | An **AI guide** panel (Google Gemini): "Explain what I'm seeing" plus a chat. It gets a factual summary of what the tool measured (visits, per-detector spectrum, light-curve ranges, saturation and blend warnings, fast-moving Gaia stars, red WISE sources, alignment, contamination flags), your current view and clicked star, and the frame on screen, then streams a plain-language answer. |

Extras:
- **Animated GIF export** of any band/mode.
- **Blend warning**: a catalogued star close enough to fall in the 12.4″ aperture is
  named on the spectrum. WISE 0855 sits 7.9″ from a G = 16.3 star in 2025–26, which
  explains its short-wavelength flux.
- **Saturated-star light curves**: very bright stars (R Dra, Barnard's Star) saturate
  SPHEREx. Their cores are masked and the light curve shows relative brightness from
  the unsaturated PSF wings, labelled as such.
- **Gaia alignment check**: median offset between Gaia and our frames, reported in the
  UI (0.04–0.18 px, under 1″, on the test fields).
- A "How to use" panel and keyboard shortcuts (Space, ←/→, D, B).

## How the download is fast

A SPHEREx image file is 66 MB; a 5′ cutout needs ~25 KB of pixels. IRSA's server-side
cutout service was slow and overloaded during testing (19–25 s each; 17/20 parallel
requests got `503`), and every cutout carries a 4 MB PSF table. So the tool reads the
original files directly with **HTTP range requests**:

1. **Headers in one request.** One multi-range request returns the start of the file,
   the FLAGS header + tile index, and the tail (wavelength table). The reader knows each
   detector's layout (D1/D2 headers are 9 FITS blocks, D3–D6 8), learns new ones
   automatically, and caches each file's layout forever.
2. **Pixels in one request.** One multi-range request returns exactly the pixel columns
   of each needed row (not full 2040-pixel rows), plus the compressed FLAGS rows near the
   target, decompressed locally.
3. **Everything in parallel and cached.** 12 images in flight, 12 HTTP requests,
   keep-alive, retries with backoff. Cutouts cached forever, jobs 24 h, frames on disk.
   Frames stream to the browser as each visit finishes (Server-Sent Events).

| Reader | Bytes per image (5′) | Requests | 40 images, cold, home link |
|---|---|---|---|
| IRSA cutout service | ~4 MB | 1 (slow, often 503) | (failed) |
| Full-row range reads (S3) | ~1.1 MB | ~5 | 44 MB, 94 s |
| **Multi-range reads (IRSA, default)** | **~200 KB** | **2** | **6.9 MB, 19 s** |

S3 ignores multi-range requests (it sends the whole 66 MB file), so the reader refuses
any 200 response to a multi-range request. When the server runs **inside AWS
us-east-1**, set `DATA_SOURCE=s3`: bandwidth there is plentiful and S3 scales better.

Bad pixels: FLAGS are read only within ±4 rows of the target (`FLAGS_MODE=box`); that's
where they matter (photometry, contamination table). Elsewhere a despike filter removes
cosmic rays and hot pixels: a pixel > 5σ whose brightest neighbour has < 25% of its
excess. SPHEREx stars always spill light into neighbours, so stars are kept.
`FLAGS_MODE=full` reads all FLAGS rows (~2× the bytes).

Full, uncapped jobs on this home connection (2026-11-15, 5′ field, a few cutouts already cached):

| target | images | MB downloaded | first frame | cold total | failed | reload |
|---|---|---|---|---|---|---|
| WISE 0855 | 258 | 29 | 11 s | 61 s | 0 | < 1 ms |
| RW Aurigae | 335 | 40 | 17 s | 77 s | 0 | < 1 ms |

The old full-row reader would have moved ~280 MB for WISE 0855 (about 9 minutes here).
"Reload" is a repeat request (the finished job is cached for 24 h). Re-processing from
cached cutouts after that takes 11–15 s for 250–350 images.

## AI guide

Put the key in `backend/.env` (git-ignored) as `GEMINI_API_KEY=...`. The panel appears
only when a key is set.

- **The key stays on the server.** The browser posts its question to
  `POST /api/jobs/{id}/ask`; the backend adds the job summary and the current frame and
  calls Gemini (key in a header), then streams the answer back. `/api/health` reports
  `ai_enabled` but never the key.
- **Models:** `gemini-3.5-flash`, falling back to `gemini-2.5-flash` when busy, with low
  "thinking" so answers start in ~2 s. (On 2026-11-15 `gemini-flash-latest` returned 503,
  and with default thinking took over 60 s.)
- **What it knows:** the job summary includes the named target's catalogue proper motion
  and how many pixels it should shift between visits (WISE 0855 isn't in Gaia, so without
  this the guide missed its 8"/yr motion).
- **Grounded:** the instructions tell the model to use the tool's numbers, label general
  knowledge as such, mention caveats (saturation, blends, few images) and never claim a
  discovery.
- **Abuse limits** for public deployments: 20 questions per visitor per 10 minutes and 400
  per hour overall (`AI_PER_CLIENT_PER_10MIN`, `AI_GLOBAL_PER_HOUR`); questions up to 1000
  characters; the last 8 turns of history.

## Adding a new data release

Put the new collection first in `backend/.env`:

```
SPHEREX_COLLECTIONS=["spherex_dr2","spherex_dr1","spherex_qr3"]
```

Duplicates of the same exposure + detector across releases are merged; the first
listed release wins. Collections are part of every cache key, so old results aren't
reused. If a new release changes the file headers, the reader adapts (one extra request
per detector, once).

## Tests and tools

```bash
cd backend && ..\.venv\Scripts\python -m pytest
```
```bash
cd backend && ..\.venv\Scripts\python -m pytest -m network
```
```bash
cd backend && ..\.venv\Scripts\python -m bench.smoke --name "Barnard's Star"
```
```bash
cd backend && ..\.venv\Scripts\python -m bench.bench_targets --limit 24
```
```bash
cd backend && ..\.venv\Scripts\python -m bench.warm_cache --fovs 3 5 10
```

- `pytest`: 42 offline tests. These include the range reader against a synthetic file
  with the real SPHEREx layout (RICE-compressed FLAGS), multipart parsing, the
  refuse-the-whole-file guard, despiking, display filling, balanced sampling, Gaia
  alignment maths, NaN-safe JSON, and the AI guide (context, frame-path safety, rate
  limits, model fallback, key never leaked) with a mocked Gemini.
- `bench.smoke`: every API feature against a running server. All five challenge
  targets pass (2026-11-15).
- `bench.bench_targets`: stage timings. `bench.warm_cache`: warms the running server's
  cache for the challenge targets and checks reloads are < 2 s.

## Archive facts verified 2026-11-14/15

- SIA columns: `energy_bandpassname` (`SPHEREx-D1`…`D6`), `cloud_access` (S3 key),
  `t_min/t_max` (MJD), `em_min/em_max`.
- **`obs_id` is per exposure, shared by all 6 detectors**: de-duplicate on
  `(obs_id, detector)`.
- `spherex_dr1` has 258–539 images for the test targets; `spherex_qr3`/`qr2` returned 0
  there.
- Layout: `PRIMARY | IMAGE f32 2040² | FLAGS (RICE_1, one tile per row) | VARIANCE | ZODI | EPSF | WCS-WAVE`.
  WCS is `TAN-SIP`; wavelength from alternate WCS `W` (`WAVE-TAB`).
- **FLAGS bit 21 is `SOURCE`**: masking all non-zero flags would erase the stars; the
  mask uses named planes (`MASK_FLAGS` in config).
- IRSA's Apache serves multi-range requests; S3 does not.
- CDS services were down (SIMBAD TAP 404, VizieR 404, alasky HiPS 404/timeouts). The
  tool uses mirrors: `simbad.harvard.edu`, `vizier.cfa.harvard.edu`, and IRSA's HiPS
  copies (`irsa.ipac.caltech.edu/data/hips/…`, which also has a SPHEREx DR1 all-sky
  cube).
- Gaia: ESA 3–18 s, MAST 5–36 s; IRSA's Gaia/CatWISE TAP timed out or returned 502.
- Windows: worker processes need one BLAS thread each (`OPENBLAS_NUM_THREADS=1`), or
  OpenBLAS runs out of memory.

## Layout

```
backend/app/        config, resolver, metadata, fetcher (range reader), processing,
                    products (bands/RGB/diff/deep), render, jobs (SSE), main (routes),
                    assistant (AI guide)
backend/app/overlays/  gaia, wise
backend/tests/      offline unit tests + network tests (pytest -m network)
backend/bench/      smoke, bench_targets, warm_cache
frontend/index.html the viewer (single file: Plotly + Aladin Lite)
```

## Not built

NEOSSat (the plan's bonus item; NEOSSat pointings rarely overlap a random field).
