# ComfyUI-FAL

**FAL nodes ComfyUI doesn't have yet** — image→3D (Tripo / Hunyuan3D / TRELLIS, **no local GPU**),
Bria background removal, the **whole 2026 Topaz family in one gated node**, and an **Image Edit bar**
(object removal, erasers, mask inpaint, prompt edit, upscalers, outpaint — the newest FAL models,
prices right in the node names), plus a **model-catalog registry** and **new-model checker** that no
other FAL pack ships. Everything is browsable in the node tree under the **`FAL/`** category.

It also ships a small **folder-IO bar** under `image/folder`: upload a folder of photos straight from
the browser, run the graph once per photo, and download every result as a single ZIP. Those four
nodes make no API calls and need no key — they are here because batch photo work is what the FAL
upscalers are actually used for.

Built on [FAL](https://fal.ai); one `FAL_KEY`, pay-as-you-go, all heavy compute is in the cloud.

## Why this exists

Other FAL packs ([gokayfem/ComfyUI-fal-API](https://github.com/gokayfem/ComfyUI-fal-API) and
friends) cover Flux / image / video / LLM well but are **hand-written per model** and have
**no 3D, no background removal, and no catalog/automation**. The existing ComfyUI 3D nodes
(Tripo, Hunyuan3D, TripoSR, 3D-Pack) run **locally on a GPU** or hit a model's **native API** —
not FAL. This pack fills exactly those gaps and **coexists** with gokayfem's pack rather than
replacing it.

What's distinctive here:
- **Image→3D over FAL, no GPU** — Tripo / Hunyuan3D / TRELLIS on a single FAL key.
- **Bria background removal over FAL** — not in any other FAL pack.
- **Catalog registry + new-model checker** (`fal_registry.py`) — list/search FAL's 1400+ models,
  dump any model's input schema, and diff for newly-added endpoints. Foundation for the
  schema-driven node generator on the roadmap.

## Nodes

### `FAL/3D` — image → 3D mesh (`.glb`)
Output: `(glb_file, download_url, preview, info)`. The `.glb` lands in ComfyUI's `output/`;
`glb_file` is relative to it — wire it into the core **Preview3D** node for an interactive
in-graph 3D view, or open `download_url` in a browser to grab the file for Blender.

| Node | Endpoint | ~Cost |
|---|---|---|
| FAL 3D — Meshy v7 (rig + anim) | `meshy/v7/image-to-3d`, `meshy/v7/multi-image-to-3d` | $0.80 bare / $1.20 textured / $1.40 ultra, +$0.20 rig, +$0.12 anim |
| FAL 3D — Tripo v2.5 | `tripo3d/tripo/v2.5/image-to-3d` | $0.20 bare / $0.30 std / $0.40 HD texture, +$0.05 quad |
| FAL 3D — Tripo H3.1 (quality dial) | `tripo3d/h3.1/image-to-3d` | $0.20–0.40 base, +$0.20 detailed geometry, +$0.05 quad |
| FAL 3D — Hunyuan3D v2 | `fal-ai/hunyuan3d/v2` | $0.16 white / $0.48 textured |
| FAL 3D — Hunyuan3D v3.1 pro/rapid | `fal-ai/hunyuan-3d/v3.1/...` | pro $0.375 / rapid $0.225, +$0.15 each: PBR, multiview, custom face count |
| FAL 3D — Hunyuan Sketch→3D | `fal-ai/hunyuan3d-v3/sketch-to-3d` | $0.375, +$0.15 PBR |
| FAL 3D — TRELLIS | `fal-ai/trellis` | $0.02 |
| FAL 3D — Hi3D v3.0 2048³ / Hitem3D | `hitem3d/hi3d/v3.0/image-to-3d`, `…/multi-view-to-3d`, `hitem3d/hi3d/image-to-3d` | v3.0: $2.10 quality / $9.10 master (2048³ voxels, 8K textures); legacy v1.5–2.1: $0.30–0.90. Texture +$0.20, PBR +$0.10, triangle mesh only |
| FAL 3D — TripoSplat (Gaussian Splat) | `tripo3d/triposplat` | $0.05 |

Tripo endpoints bill in **Tripo credits** (1 credit = $0.01) — on the FAL usage page the
Quantity column is credits, not generations.

**Meshy v7** is the one call that does everything: geometry, PBR textures, game-ready quad
topology at a target polycount, humanoid auto-rigging and an animation clip. Note that GLB
cannot store quads — with `topology=quad` the `.glb` comes back triangulated and the real
quads live only in the `fbx`/`obj`/`blend` export, which is what the `extra_format` widget
fetches from the same billed job.

TripoSplat outputs a `FILE_3D` (`splat_3d`) that plugs straight into ComfyUI's core splat
nodes: **Get Splat → Transform / Render / Extract Mesh from Splat / Create 3D File →
Save 3D Model** (interactive viewer). The `.ply` also lands in `output/`.

#### Mesh in, mesh out
Wire a `glb_file` output into `mesh_file`.

| Node | Endpoint | ~Cost | What it's for |
|---|---|---|---|
| FAL 3D — Tripo Remesh | `tripo3d/tripo/remesh` | $0.01 | **The default retopo.** Real quads, bakes textures, returns a preview render. Caps at 20k faces (10k quad) — a game-asset/LOD decimator. |
| FAL 3D — Tripo Segment | `tripo3d/tripo/segment` | $0.01 | Split into named semantic parts. Feed `part_names` into Remesh to keep them separate — the pair costs $0.02. |
| FAL 3D — Meshy v5 Remesh | `fal-ai/meshy/v5/remesh` | $0.20 | The high-poly path: up to 300k faces, `.blend`/`.usdz` export, rescale + origin. |
| FAL 3D — Smart Topology | `fal-ai/hunyuan-3d/v3.1/smart-topology` | $0.75 | glb/obj in, mixed quad/tri out. |
| FAL 3D — Meshy Rigging | `fal-ai/meshy/rigging` | $0.20, +$0.12 anim | Rig **any** humanoid GLB, not only one Meshy just made — so a $0.20 Tripo mesh rigs for $0.40 total instead of $1.40 through a v7 generation. Under 300k faces, textured. |

### `FAL/Image/Material` — PATINA, PBR maps

Photo or prompt → `basecolor`, `normal`, `roughness`, `metalness`, `height` as separate
named IMAGE outputs.

| Node | Endpoint | ~Cost | Input |
|---|---|---|---|
| FAL Material — PATINA image→PBR maps | `fal-ai/patina` | $0.01 + $0.01/MP per map (1K, 5 maps = **$0.06**) | a photo or render → its five maps |
| FAL Material — PATINA prompt→tiling material | `fal-ai/patina/material` | $0.01 + $0.02/MP + $0.01/MP per map (1K, 5 maps = **$0.08**) | text → a seamlessly tiling material (optionally seeded from an image, or inpainted with a mask) |
| FAL Material — PATINA extract from photo | `fal-ai/patina/material/extract` | $0.10 + $0.02/MP + $0.01/MP per map (1K, 5 maps = **$0.17**) | photo + "the wall" → that material lifted out as a seamless tile |

Each map is billed separately, so the five booleans are a real cost dial — and on the two
tiling nodes turning all of them off is legal and means "texture only, skip the PBR pass".
Maps are matched by the response's `map_type` field rather than by position, because the
API's own examples return them in inconsistent orders. `upscale_factor` enlarges the maps
but **not** the base texture, which is why every map is its own output rather than a batch.

### `FAL/Image/Banana` — Nano Banana / Gemini

FAL lists this family under fifteen endpoint ids, but they are four models: the `gemini-*`
ids are identical aliases of the `nano-banana-*` ids, same fields and same price. So each
node takes a **tier** instead:

| Tier | Endpoint | Price | Extras |
|---|---|---|---|
| `pro` | `fal-ai/nano-banana-pro` | $0.15/img, 4K ×2 | 1K/2K/4K, system prompt, web search |
| `banana-2` | `fal-ai/nano-banana-2` | $0.08/img @1K | 0.5K ×0.75, 2K ×1.5, 4K ×2, thinking, system prompt, web search, extreme ratios |
| `lite` | `google/nano-banana-2-lite` | token-billed, fixed 1K | sub-2s, thinking, system prompt |
| `legacy` | `fal-ai/nano-banana` | $0.039/img | cheapest |

| Node | What it does |
|---|---|
| FAL Banana — Generate | text → image, any tier |
| FAL Banana — Edit | image(s) → image, any tier; batch + `image_2/3/4` all become references |
| FAL Banana — Context | `nano-banana-2/edit` with **`pdf_url` / `video_url` / `audio_url`** — generate from a document or a video (a plain YouTube link works), no local download |

All three also return the model's own `description`, which is where `thinking_level` output
surfaces. `use_gemini_ids` switches to the Gemini-branded id of the same model if a workflow
needs to pin it.

### `FAL/Image/Generate` — structure-controlled text-to-image

| Node | Endpoint | ~Cost |
|---|---|---|
| FAL Generate — Flux General | `fal-ai/flux-general` | $0.075/MP, **rounded up** |

FLUX.1 [dev] plus the whole conditioning stack — ControlNet, ControlNet Union, IP-Adapter
and up to two LoRAs. Feed a depth / normal / lineart map into `control_image` and the render
follows your geometry, which is what makes it the node to drive from Blender.

Billing rounds up per megapixel: 1536×864 is 1.33 MP and costs **$0.15**, not $0.075. Only a
render at or under 1 MP gets the single-megapixel rate.

### `FAL/Image/Restore` — recover what was photographed

| Node | Endpoint | ~Cost | |
|---|---|---|---|
| FAL Restore — NAFNet deblur / denoise | `fal-ai/nafnet/deblur`, `/denoise` | $0.0225/MP | **non-generative** |
| FAL Restore — DRCT 4x | `fal-ai/drct-super-resolution` | $0.0045/MP | **non-generative**, fixed 4x |
| FAL Restore — Bria FIBO | `bria/fibo-edit/restore` | $0.04 | one button, diffusion underneath |
| FAL Restore — DDColor | `fal-ai/ddcolor` | $0.001/MP | colourise B&W |

The distinction this shelf exists for: a **restoration** model recovers detail still physically
present in the file; a **generative** one invents plausible detail. For a reference photo heading
into a texture or a product shot you want the first — an invented leaf vein is worse than a soft
one, because it looks right and is wrong.

The catalogue makes that easy to check, and it is worth rechecking before adding anything here: a
genuine restoration endpoint takes one image and at most a scale or task selector. Anything
exposing `prompt`, `guidance_scale`, `num_inference_steps` or a `creativity` knob redraws. Two FAL
endpoints are literally named "photo-restoration" and fail that test — `image-editing/photo-restoration`
carries guidance 3.5 and 30 inference steps, and `image-apps-v2/photo-restoration` force-renders to
4K in one of five aspect ratios and cannot return your frame. Neither is wrapped here, on purpose.

**Order of operations** for cleaning a reference photo: denoise → deblur → enlarge → cut out.
Enlarging a soft photo just gives a large soft photo, and cutting out a soft edge gives a ragged
alpha. Try the $0.001 `FAL Finish — Sharpen` first if the photo is only nearly-good — it raises
edge contrast, which is not the same as undoing blur, but it is 20x cheaper.

### `FAL/Text` — vision and language

| Node | Endpoint | Billing |
|---|---|---|
| FAL Text — VLM | `openrouter/router/vision` | per token |
| FAL Text — LLM | `openrouter/router` | per token |

Token-billed rather than per call, so instead of a price in the title both return an `info`
output carrying the model, the token counts and the cost FAL reports for that request.

### `FAL/Background` — Bria
| Node | Endpoint | Output | ~Cost |
|---|---|---|---|
| FAL Background — Bria Remove | `fal-ai/bria/background/remove` | IMAGE + MASK | $0.018 |
| FAL Background — Bria Replace | `fal-ai/bria/background/replace` | IMAGE(s) | $0.04 |

### `FAL/Image` — remove / inpaint / edit / upscale / expand / vector / finish

One bar for everyday photo work, newest model per task. Masks follow ComfyUI convention
(MASK 1.0 = the area to remove/inpaint).

| Node | Endpoint | Input | ~Cost |
|---|---|---|---|
| FAL Remove — Object Removal | `fal-ai/object-removal[/mask]` | prompt **or** mask | $0.006–0.024 |
| FAL Remove — Bria Eraser | `fal-ai/bria/eraser` | mask | $0.04 |
| FAL Remove — Flux Pro v1 Erase | `fal-ai/flux-pro/v1/erase` | mask | ~$0.03/MP |
| FAL Remove — Finegrain Eraser | `fal-ai/finegrain-eraser` | prompt (kills shadows/reflections) | $0.18–0.36 |
| FAL Inpaint — Z-Image Turbo | `fal-ai/z-image/turbo/inpaint` | mask + prompt | $0.01/MP |
| FAL Inpaint — Qwen Image Edit v1 | `fal-ai/qwen-image-edit/inpaint` | mask + prompt (2511 has no mask endpoint) | ~$0.03/MP |
| FAL Inpaint — Bria GenFill v2 | `bria/genfill/v2` | mask + instruction | $0.04/MP |
| FAL Inpaint — Flux Pro v1 Fill | `fal-ai/flux-pro/v1/fill` | mask + prompt; BFL's quality bar | $0.05/MP |
| FAL Edit — Flux Kontext pro / max | `fal-ai/flux-pro/kontext[/max][/multi]` | instruction + up to 4 refs, auto single/multi routing | $0.04 / $0.08 per image |
| FAL Edit — Qwen Image Edit 2511 (newest) | `fal-ai/qwen-image-edit-2511` | prompt, multi-ref | $0.03/MP |
| FAL Edit — Seedream v5-pro / v5-lite / v4.5 | `bytedance/seedream/v5/...`, `fal-ai/bytedance/seedream/v4.5/edit` | prompt, up to 10 refs; v5-pro is region-precise + sketch completion | $0.04–0.14 |
| FAL Upscale — SeedVR v2 | `fal-ai/seedvr/upscale/image` | factor or target res | $0.001/MP |
| FAL Upscale — Topaz | `fal-ai/topaz/upscale/image` | 11 models, face enhance | $0.08 (≤24MP) – $1.36 |
| FAL Upscale — Recraft Crisp | `fal-ai/recraft/upscale/crisp` | image only | $0.004 |
| FAL Upscale — Clarity | `fal-ai/clarity-upscaler` | creativity/resemblance | $0.03/MP |
| FAL Expand — Bria Outpaint | `fal-ai/bria/expand` | canvas size (+prompt) | $0.04 |
| FAL Vector — Recraft Vectorize | `fal-ai/recraft/vectorize` | image → SVG (AI, zero-config) | $0.01 |
| FAL Vector — Image2SVG tracer | `fal-ai/image2svg` | image → SVG (vtracer knobs) | $0.005 |
| FAL Finish — Film Grain | `fal-ai/post-processing/grain` | 6 film stocks | $0.001 |
| FAL Finish — Vignette | `fal-ai/post-processing/vignette` | strength | $0.001 |
| FAL Finish — Color Correction | `fal-ai/post-processing/color-correction` | temp/contrast/sat/brightness/gamma | $0.001 |
| FAL Finish — Sharpen | `fal-ai/post-processing/sharpen` | basic/smart/CAS | $0.001 |

### `image/folder` — a folder of photos in, one ZIP out (no FAL key needed)

| Node | What it does |
|---|---|
| 📁 Load Images (upload folder) | *Upload folder… / Upload files…* buttons, or drop a folder onto the node: the files go to `input/<folder>/` through the stock `/upload/image` endpoint, one request per file (so proxies with small body caps are fine), and the folder is then selected on the node. Loads every photo as an IMAGE **list**, plus the original file stems and a count. EXIF orientation applied, embedded ICC (Display P3 phone shots) converted to sRGB, HEIC/HEIF/AVIF when `pillow-heif` is installed. |
| ✂️ Split by Short Side | Hands out only the photos whose short side is below the target, plus their indices. |
| 🔀 Merge Subset (by index) | Puts the processed photos back in place. `replacements` is a **lazy** input. |
| 💾 Save Images + ZIP | Writes `output/<folder>/<stem><suffix>.<jpg\|png\|webp>`, zips this run's files atomically, shows the whole batch as a gallery on the node and gives you a **⬇ Download ZIP** button that survives a page reload, plus a `download_url` output. |

**Why Split/Merge instead of a Switch.** In list mode ComfyUI resolves lazy inputs for the *whole
list* at once: one photo needing `on_true` makes the upstream node a strong link for every item, so
a paid upscaler behind a Switch runs on all of them and the Switch merely discards the extras.
Splitting the list before the paid node — and marking the way back lazy — means you pay for exactly
the photos that needed work, including none.

`workflows/Upscale to 1980 (folder + ZIP).json` is the reference graph: a folder of photos, only the
small ones sent to Topaz Wonder 3.5, short side normalised to 1980 px, JPEG + ZIP out.
Regenerate it with `python workflows/build_upscale_1980.py`.

### `FAL/Image/Upscale` — Topaz 2026

FAL retired `fal-ai/topaz/upscale/image` and split Topaz into `topaz/upscale/image/{precision,
generative,creative,transparent}` plus `topaz/restore/image`. **FAL Upscale — Topaz 2026** is one
node over all of it: pick from 19 models and the endpoint follows.

| Family | Models | For |
|---|---|---|
| generative | **Wonder 3.5**, Wonder 3/2/Wonder, Recover 3, Standard MAX, Redefine, Recovery V2/Recovery | small or compressed sources — rebuilds detail |
| precision | Standard V2, High Fidelity V3/V2, Low Resolution V2, CGI, Text Refine | clean, sharp sources — deterministic |
| creative | Bloom 2, Bloom, Bloom Realism | re-imagining rather than restoring |
| transparent | — | cutouts, keeps the alpha channel |

$0.01 per output megapixel across the family, so model choice is free; `upscale_factor` is 1–4 here,
not 1–8. Every optional dial is filtered against what the chosen model accepts, because Topaz scopes
almost all of them to a single model (`prompt`/`texture`/`sharpen`/`denoise`/`creativity` → Redefine,
`detail` → Recovery V2, `enhancement_strength` → Wonder 3/3.5, `fix_compression` → precision except
CGI, `strength` → Text Refine, `color_preservation` → Bloom 2) and FAL 422s on an unknown field.
Dropped dials are named in the console rather than silently ignored. `face_enhancement` defaults
**off**: on a small face its separate pass is what produces the waxy look.

**FAL Restore — Topaz** wraps `topaz/restore/image` (Recover 3 / Dust-Scratch V2 / Faces). It does
not change the image size, so chain it *before* an upscaler.

The old node is kept as *Topaz LEGACY endpoint* purely so saved graphs keep running.

## Catalog registry (`fal_registry.py`)

Run where `FAL_KEY` is set (e.g. inside the ComfyUI container):

```bash
python fal_registry.py fetch              # cache FAL's whole catalog -> fal_catalog.json
python fal_registry.py search upscale     # find models by name / category / description
python fal_registry.py category image-to-3d
python fal_registry.py schema fal-ai/bria/background/remove   # dump input params
python fal_registry.py diff               # new-model checker: live FAL vs cache
```

The catalog (1400+ models) and per-model parameter schemas are reachable with a normal
`FAL_KEY` — no admin key needed.

## Install

Clone into `ComfyUI/custom_nodes/` and restart ComfyUI:

```bash
git clone https://github.com/dufok/ComfyUI-FAL.git
```

Only dependency is `fal-client` (already present in most FAL-enabled ComfyUI setups).
For iPhone HEIC/HEIF input in the folder loader, add `pillow-heif` — everything else works without it:

```bash
pip install pillow-heif
```

## Auth

Reads `FAL_KEY` from the environment — no `config.ini`. Get a key at
[fal.ai/dashboard/keys](https://fal.ai/dashboard/keys).

## Layout

```
__init__.py        merges each module's NODE_CLASS_MAPPINGS, installs the retag hook
fal_common.py      shared helpers (upload, result parsing, file save, mesh runner)
fal_3d.py          FAL/3D nodes — generation, retopo, segmentation, rigging
fal_background.py  FAL/Background nodes (Bria)
fal_image_edit.py  FAL/Image/{Remove,Inpaint,Edit,Upscale,Expand,Vector,Finish}
fal_material.py    FAL/Image/Material — PATINA PBR maps
fal_banana.py      FAL/Image/Banana — Nano Banana / Gemini
fal_generate.py    FAL/Image/Generate — Flux General (ControlNet / LoRA / IP-Adapter)
fal_restore.py     FAL/Image/Restore — NAFNet, DRCT, Bria FIBO, DDColor
fal_text.py        FAL/Text — VLM and LLM over OpenRouter
fal_topaz.py       FAL/Image/Upscale + Restore — the 2026 Topaz family, one gated node
folderio_nodes.py  image/folder — folder upload, list loader, split/merge, save + ZIP
web/folderio.js    browser side of the folder nodes (upload buttons, drag-drop, ZIP button)
fal_retag.py       tidies the co-installed gokayfem pack's categories (see below)
fal_registry.py    catalog list / search / schema / diff
tests/             test_topaz_args.py (offline, no key), test_folderio_smoke.py (in-image),
                   test_folderio_e2e.py (against a running ComfyUI)
workflows/         example graphs, and the script that generates the 1980 px one
```

Adding a category = a new `fal_<x>.py` exposing `NODE_CLASS_MAPPINGS` + display names,
then import it in `__init__.py`.

### Coexisting with gokayfem's pack

If [gokayfem/ComfyUI-fal-API](https://github.com/gokayfem/ComfyUI-fal-API) is installed too
(it often is, and in a container image it may not be editable), its 87 nodes sit directly in
`FAL/Image` and `FAL/VideoGeneration`, interleaved with this pack's, and three of its helpers
escape into ComfyUI's stock `video` menu. `fal_retag.py` moves that whole pack under one root
at startup:

| from | to |
|---|---|
| `FAL/Image` | `FAL/zz-gokayfem/Image` |
| `FAL/VideoGeneration`, `.../DY` | `FAL/zz-gokayfem/Video` |
| its 4 video upscalers | `FAL/zz-gokayfem/Video Upscale` |
| its 4 Nano Banana nodes | `FAL/zz-gokayfem/Banana` |
| `FAL/LLM`, `FAL/VLM` | `FAL/zz-gokayfem/Text` |
| `FAL/Training` | `FAL/zz-gokayfem/Training` |
| stock `video` (upload helpers) | `FAL/zz-gokayfem/Utils` |

The `zz-` prefix sorts it to the bottom. It is deliberately *not* called "legacy" — that pack
owns video, LoRA training and most text-to-image here, and none of it is deprecated.

This mutates `cls.CATEGORY` from an `app.on_startup` hook. ComfyUI resolves a saved graph by
its `class_type` (the `NODE_CLASS_MAPPINGS` key) and re-reads `CATEGORY` off the class on
every `/object_info` request — category is presentation, `class_type` is the contract — so
retagging moves menu entries without touching a single saved workflow, and no fork is needed.

The rules are keyed on category rather than a list of node ids, so a node added upstream still
lands somewhere sensible. It only touches classes that prove they belong to that pack, skips
V3-schema nodes whose `CATEGORY` is a read-only classproperty, and is idempotent. Every path is
guarded: a failure here can never stop this pack's own nodes from loading. Delete the two
`fal_retag` lines from `__init__.py` to opt out.

## Roadmap

- **Schema-driven node generator** — given an allow-list of endpoint IDs, fetch each model's
  OpenAPI schema and generate a typed node automatically (so any FAL model becomes a node
  without hand-writing it).
- **Automated new-model checker** — `fal_registry.py diff` on a schedule, surfacing newly-added
  FAL endpoints.

## License

MIT — see [LICENSE](LICENSE).
