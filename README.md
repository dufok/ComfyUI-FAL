# ComfyUI-FAL

Custom ComfyUI nodes wrapping [FAL](https://fal.ai) endpoints, browsable directly in
the node tree under the **`FAL/`** category — no need to read docs to see what's available,
just open the menu.

Companion to [`ComfyUI-Marble`](https://github.com/dufok/ComfyUI-Marble) (Gaussian-splat
*worlds* via World Labs). This pack is FAL-only. It **coexists** with the
[`gokayfem/ComfyUI-fal-API`](https://github.com/gokayfem/ComfyUI-fal-API) pack — it fills
the gaps that pack doesn't cover (3D, background removal) rather than replacing it.

## Nodes

### `FAL/3D` — image → 3D mesh (`.glb`)
Output: `(glb_path, download_url, preview, info)`. The `.glb` lands in `/app/output/`;
open `download_url` in a browser to grab it.

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

## Auth

Reads `FAL_KEY` from the environment (passed via docker-compose from `~/comfyui-docker/.env`).
No `config.ini`. `fal_client` is already in the image.

## Layout

```
__init__.py        merges each module's NODE_CLASS_MAPPINGS
fal_common.py      shared helpers (upload, result parsing, file save, mesh runner)
fal_3d.py          FAL/3D nodes
fal_background.py  FAL/Background nodes (Bria)
```

Adding a category = a new `fal_<x>.py` exposing `NODE_CLASS_MAPPINGS` + display names,
then import it in `__init__.py`.

## Roadmap

- **Schema-driven generator** — auto-create a node per FAL endpoint from its OpenAPI
  schema (`expand=openapi` on FAL's Model Endpoints API), so the *whole* catalog shows
  up in the tree, auto-categorized.
- **New-model auto-checker** — periodically diff FAL's catalog against what's wrapped and
  surface new endpoints to add.

## Install (on proserver)

Bind-mounted in `docker-compose.yml` like the Marble node:

```yaml
- ./custom_nodes/ComfyUI-FAL:/app/custom_nodes/ComfyUI-FAL
```

Update flow: edit locally → `git push` → `git pull` on server → `docker compose restart comfyui`.
