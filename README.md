# ComfyUI-FAL-3D

Custom ComfyUI nodes for **image → 3D mesh** generation via [FAL](https://fal.ai).
Companion to `ComfyUI-Marble` (which does Gaussian-splat *worlds* via World Labs);
this pack does **textured `.glb` meshes** via FAL endpoints.

All nodes live under the **`FAL/3D`** category and share the same I/O:

| Output | Meaning |
|---|---|
| `glb_path` | absolute path of the saved `.glb` in the container (`/app/output/...`) |
| `download_url` | clickable `http://192.168.1.2:8188/view?...` link to grab the mesh |
| `preview` | rendered preview IMAGE (Tripo only; blank for the others) |
| `info` | one-line status (endpoint, file, size) |

## Nodes

| Node | FAL endpoint | Notes | ~Cost |
|---|---|---|---|
| **FAL 3D — Tripo v2.5** | `tripo3d/tripo/v2.5/image-to-3d` | cheapest/fastest, PBR, HD texture, quad/FBX | ~$0.05 (+$0.05 quad) |
| **FAL 3D — Hunyuan3D v2** | `fal-ai/hunyuan3d/v2` | high detail, octree-resolution control | ~$0.16 white / ~$0.48 textured |
| **FAL 3D — TRELLIS** | `fal-ai/trellis` | Microsoft TRELLIS, fine sparse/latent control | varies |

## Auth

Reads `FAL_KEY` from the environment (passed via `docker-compose` from
`~/comfyui-docker/.env`). No `config.ini` needed. `fal_client` is already in the image.

## Install (on proserver)

Bind-mounted in `docker-compose.yml` like the Marble node:

```yaml
- ./custom_nodes/ComfyUI-FAL-3D:/app/custom_nodes/ComfyUI-FAL-3D
```

Update flow: edit locally → push → `git pull` on server → `docker compose restart comfyui`.

## Output `.glb`

Lands in `/app/output/` (the `comfyui-output` volume). Open the printed `download_url`
in a browser to download it, then drop into Blender 4.x / any glTF viewer.
