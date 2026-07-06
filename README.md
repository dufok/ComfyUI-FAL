# ComfyUI-FAL

**FAL nodes ComfyUI doesn't have yet** — image→3D (Tripo / Hunyuan3D / TRELLIS, **no local GPU**),
Bria background removal, and an **Image Edit bar** (object removal, erasers, mask inpaint, prompt
edit, upscalers, outpaint — the newest FAL models, prices right in the node names), plus a
**model-catalog registry** and **new-model checker** that no other FAL pack ships. Everything is
browsable in the node tree under the **`FAL/`** category.

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
Output: `(glb_path, download_url, preview, info)`. The `.glb` lands in ComfyUI's `output/`;
open `download_url` in a browser to grab it, then drop into Blender / any glTF viewer.

| Node | Endpoint | ~Cost |
|---|---|---|
| FAL 3D — Tripo v2.5 | `tripo3d/tripo/v2.5/image-to-3d` | ~$0.05 |
| FAL 3D — Hunyuan3D v2 | `fal-ai/hunyuan3d/v2` | ~$0.16 white / ~$0.48 textured |
| FAL 3D — TRELLIS | `fal-ai/trellis` | varies |

### `FAL/Background` — Bria
| Node | Endpoint | Output | ~Cost |
|---|---|---|---|
| FAL Background — Bria Remove | `fal-ai/bria/background/remove` | IMAGE + MASK | ~$0.018 |
| FAL Background — Bria Replace | `fal-ai/bria/background/replace` | IMAGE(s) | varies |

### `FAL/Image Edit` — remove / inpaint / edit / upscale / expand

One bar for everyday photo work, newest model per task. Masks follow ComfyUI convention
(MASK 1.0 = the area to remove/inpaint).

| Node | Endpoint | Input | ~Cost |
|---|---|---|---|
| FAL Remove — Object Removal | `fal-ai/object-removal[/mask]` | prompt **or** mask | $0.006–0.024 |
| FAL Remove — Bria Eraser | `fal-ai/bria/eraser` | mask | $0.04 |
| FAL Remove — Flux Pro Erase | `fal-ai/flux-pro/v1/erase` | mask | ~$0.03/MP |
| FAL Remove — Finegrain Eraser | `fal-ai/finegrain-eraser` | prompt (kills shadows/reflections) | $0.18–0.36 |
| FAL Inpaint — Z-Image Turbo | `fal-ai/z-image/turbo/inpaint` | mask + prompt | $0.01/MP |
| FAL Inpaint — Qwen Image Edit | `fal-ai/qwen-image-edit/inpaint` | mask + prompt | ~$0.03/MP |
| FAL Inpaint — Bria GenFill v2 | `bria/genfill/v2` | mask + instruction | $0.04/MP |
| FAL Edit — Qwen Image Edit 2511 | `fal-ai/qwen-image-edit-2511` | prompt, multi-ref | $0.03/MP |
| FAL Edit — Seedream v4.5 / v5-lite | `fal-ai/bytedance/seedream/...` | prompt, up to 10 refs | $0.04 |
| FAL Upscale — SeedVR2 | `fal-ai/seedvr/upscale/image` | factor or target res | varies |
| FAL Upscale — Topaz | `fal-ai/topaz/upscale/image` | 11 models, face enhance | varies |
| FAL Upscale — Recraft Crisp | `fal-ai/recraft/upscale/crisp` | image only | cheap |
| FAL Upscale — Clarity | `fal-ai/clarity-upscaler` | creativity/resemblance | varies |
| FAL Expand — Bria Outpaint | `fal-ai/bria/expand` | canvas size (+prompt) | varies |

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

## Auth

Reads `FAL_KEY` from the environment — no `config.ini`. Get a key at
[fal.ai/dashboard/keys](https://fal.ai/dashboard/keys).

## Layout

```
__init__.py        merges each module's NODE_CLASS_MAPPINGS
fal_common.py      shared helpers (upload, result parsing, file save, mesh runner)
fal_3d.py          FAL/3D nodes
fal_background.py  FAL/Background nodes (Bria)
fal_image_edit.py  FAL/Image Edit nodes (remove / inpaint / edit / upscale / expand)
fal_registry.py    catalog list / search / schema / diff
```

Adding a category = a new `fal_<x>.py` exposing `NODE_CLASS_MAPPINGS` + display names,
then import it in `__init__.py`.

## Roadmap

- **Schema-driven node generator** — given an allow-list of endpoint IDs, fetch each model's
  OpenAPI schema and generate a typed node automatically (so any FAL model becomes a node
  without hand-writing it).
- **Automated new-model checker** — `fal_registry.py diff` on a schedule, surfacing newly-added
  FAL endpoints.

## License

MIT — see [LICENSE](LICENSE).
