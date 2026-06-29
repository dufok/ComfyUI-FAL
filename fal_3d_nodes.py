"""
ComfyUI nodes: FAL image-to-3D mesh generators.

Three nodes, one per FAL endpoint, all under the "FAL/3D" category:
  * FalTripoImageTo3D   -> tripo3d/tripo/v2.5/image-to-3d   (cheap, fast, PBR/quad/HD texture)
  * FalHunyuan3D        -> fal-ai/hunyuan3d/v2              (high-detail mesh, octree control)
  * FalTrellisImageTo3D -> fal-ai/trellis                  (Microsoft TRELLIS, fine control)

Each takes an IMAGE socket, uploads it to FAL, runs the model, downloads the resulting
.glb into ComfyUI's output dir, and returns:
  (glb_path, download_url, preview, info)

  glb_path     - absolute path of the saved .glb inside the container (/app/output/...)
  download_url - clickable http link (built from COMFYUI_PUBLIC_URL) to grab the .glb
  preview      - rendered preview IMAGE if the model returns one (Tripo), else a blank tensor
  info         - one-line status string (model, size, cost hint)

Auth: reads FAL_KEY from the environment (same as the ComfyUI-fal-API pack). fal_client is
already installed in the image. Shares no state with the Marble node.
"""
import io
import os
import tempfile
import urllib.request

import numpy as np
import torch
from PIL import Image

import fal_client
import folder_paths


# --------------------------------------------------------------------------- helpers

def _require_key():
    if not os.environ.get("FAL_KEY", "").strip():
        raise RuntimeError(
            "FAL_KEY is not set in the container environment. "
            "It is normally passed in via docker-compose from ~/comfyui-docker/.env."
        )


def _tensor_frame_to_png_path(tensor_frame):
    """First frame of an IMAGE tensor -> a temp PNG file, return its path."""
    arr = tensor_frame.detach().cpu().numpy()
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil = Image.fromarray(arr)
    fd, path = tempfile.mkstemp(suffix=".png", prefix="fal3d_")
    os.close(fd)
    pil.save(path, format="PNG")
    return path


def _upload_image(image):
    """IMAGE tensor (uses first frame) -> uploaded FAL URL."""
    if image is None:
        raise RuntimeError("no 'image' connected — connect a LoadImage (or any IMAGE) output")
    path = _tensor_frame_to_png_path(image[0])
    try:
        return fal_client.upload_file(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _deep_find(obj, key):
    if isinstance(obj, dict):
        if key in obj and obj[key] is not None:
            return obj[key]
        for v in obj.values():
            f = _deep_find(v, key)
            if f is not None:
                return f
    elif isinstance(obj, list):
        for item in obj:
            f = _deep_find(item, key)
            if f is not None:
                return f
    return None


def _mesh_url(result):
    """Locate the .glb / mesh URL across the various FAL output shapes."""
    for key in ("model_mesh", "model_glb_pbr", "model_glb"):
        node = result.get(key) if isinstance(result, dict) else None
        if isinstance(node, dict) and node.get("url"):
            return node["url"]
        if isinstance(node, str) and node.startswith("http"):
            return node
    # last resort: any nested "url" that looks like a 3D asset
    url = _deep_find(result, "url")
    if url and any(url.lower().split("?")[0].endswith(ext) for ext in (".glb", ".gltf", ".fbx", ".zip")):
        return url
    return url


def _url_to_image_tensor(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            raw = r.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        arr = np.asarray(pil).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except Exception:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


def _blank_image():
    return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


def _save_mesh(url, prefix):
    """Download the mesh URL into ComfyUI's output dir, return (path, download_url, size_mb)."""
    clean = url.split("?")[0]
    ext = clean.rsplit(".", 1)[-1].lower()
    if ext not in ("glb", "gltf", "fbx", "zip"):
        ext = "glb"
    base = os.path.basename(clean) or f"{prefix}.{ext}"
    fname = f"{prefix}_{base}"
    out_dir = folder_paths.get_output_directory()
    dest = os.path.join(out_dir, fname)
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / 1_000_000
    public = os.environ.get("COMFYUI_PUBLIC_URL", "").rstrip("/")
    tail = f"/view?filename={fname}&type=output"
    download_url = f"{public}{tail}" if public else tail
    return dest, download_url, size_mb


def _run(endpoint, arguments, prefix, want_preview=True):
    """Shared submit -> wait -> download -> package pipeline. Returns the node's 4-tuple."""
    _require_key()
    print(f"[FAL-3D] {endpoint} <- {arguments}")
    result = fal_client.subscribe(endpoint, arguments=arguments, with_logs=False)
    url = _mesh_url(result)
    if not url:
        raise RuntimeError(f"no mesh url in FAL response: {result}")
    dest, download_url, size_mb = _save_mesh(url, prefix)
    preview = _blank_image()
    if want_preview:
        rendered = _deep_find(result, "rendered_image")
        thumb = rendered.get("url") if isinstance(rendered, dict) else None
        if thumb:
            preview = _url_to_image_tensor(thumb)
    info = f"{endpoint} -> {os.path.basename(dest)} ({size_mb:.2f} MB)"
    print(f"[FAL-3D] DONE {info}")
    print(f"[FAL-3D] DOWNLOAD .glb: {download_url}")
    # OUTPUT_NODE: surface the link as UI text under the node, while the
    # downstream sockets still read from "result".
    return {
        "ui": {"text": [f"{info}\n⬇ {download_url}"]},
        "result": (dest, download_url, preview, info),
    }


_RET_TYPES = ("STRING", "STRING", "IMAGE", "STRING")
_RET_NAMES = ("glb_path", "download_url", "preview", "info")


# --------------------------------------------------------------------------- Tripo v2.5

class FalTripoImageTo3D:
    """tripo3d/tripo/v2.5/image-to-3d — cheapest/fastest, PBR + optional HD texture & quad mesh."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "texture": (["standard", "HD", "no"], {"default": "standard"}),
                "pbr": ("BOOLEAN", {"default": True, "tooltip": "Generate PBR materials."}),
            },
            "optional": {
                "quad": ("BOOLEAN", {"default": False, "tooltip": "Quad (FBX) mesh output — +$0.05."}),
                "auto_size": ("BOOLEAN", {"default": False, "tooltip": "Scale model to real-world meters."}),
                "face_limit": ("INT", {"default": 0, "min": 0, "max": 500000, "step": 1000,
                                       "tooltip": "0 = adaptive (model decides). Otherwise cap face count."}),
                "texture_alignment": (["original_image", "geometry"], {"default": "original_image"}),
                "orientation": (["default", "align_image"], {"default": "default"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
            },
        }

    RETURN_TYPES = _RET_TYPES
    RETURN_NAMES = _RET_NAMES
    FUNCTION = "generate"
    CATEGORY = "FAL/3D"
    OUTPUT_NODE = True

    def generate(self, image, texture, pbr, quad=False, auto_size=False, face_limit=0,
                 texture_alignment="original_image", orientation="default", seed=0):
        args = {
            "image_url": _upload_image(image),
            "texture": texture,
            "pbr": bool(pbr),
            "quad": bool(quad),
            "auto_size": bool(auto_size),
            "texture_alignment": texture_alignment,
            "orientation": orientation,
        }
        if face_limit and face_limit > 0:
            args["face_limit"] = int(face_limit)
        if seed:
            args["seed"] = int(seed)
        return _run("tripo3d/tripo/v2.5/image-to-3d", args, "tripo")


# --------------------------------------------------------------------------- Hunyuan3D v2

class FalHunyuan3D:
    """fal-ai/hunyuan3d/v2 — high-detail mesh with octree-resolution control."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "textured_mesh": ("BOOLEAN", {"default": True,
                                              "tooltip": "Generate textured mesh (pricier) vs white mesh."}),
            },
            "optional": {
                "octree_resolution": ("INT", {"default": 256, "min": 1, "max": 1024, "step": 16,
                                              "tooltip": "Higher = denser/more detailed mesh."}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 50}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
            },
        }

    RETURN_TYPES = _RET_TYPES
    RETURN_NAMES = _RET_NAMES
    FUNCTION = "generate"
    CATEGORY = "FAL/3D"
    OUTPUT_NODE = True

    def generate(self, image, textured_mesh, octree_resolution=256, num_inference_steps=50,
                 guidance_scale=7.5, seed=0):
        args = {
            "input_image_url": _upload_image(image),
            "textured_mesh": bool(textured_mesh),
            "octree_resolution": int(octree_resolution),
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
        }
        if seed:
            args["seed"] = int(seed)
        # Hunyuan3D returns model_mesh / model_glb(_pbr); no rendered preview image.
        return _run("fal-ai/hunyuan3d/v2", args, "hunyuan3d", want_preview=False)


# --------------------------------------------------------------------------- TRELLIS

class FalTrellisImageTo3D:
    """fal-ai/trellis — Microsoft TRELLIS, fine-grained sparse-structure / latent control."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "texture_size": (["512", "1024", "2048"], {"default": "1024"}),
            },
            "optional": {
                "ss_guidance_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "ss_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "slat_guidance_strength": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "slat_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "mesh_simplify": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01,
                                            "tooltip": "Higher = simpler mesh (fewer faces)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
            },
        }

    RETURN_TYPES = _RET_TYPES
    RETURN_NAMES = _RET_NAMES
    FUNCTION = "generate"
    CATEGORY = "FAL/3D"
    OUTPUT_NODE = True

    def generate(self, image, texture_size, ss_guidance_strength=7.5, ss_sampling_steps=12,
                 slat_guidance_strength=3.0, slat_sampling_steps=12, mesh_simplify=0.95, seed=0):
        args = {
            "image_url": _upload_image(image),
            "texture_size": int(texture_size),
            "ss_guidance_strength": float(ss_guidance_strength),
            "ss_sampling_steps": int(ss_sampling_steps),
            "slat_guidance_strength": float(slat_guidance_strength),
            "slat_sampling_steps": int(slat_sampling_steps),
            "mesh_simplify": float(mesh_simplify),
        }
        if seed:
            args["seed"] = int(seed)
        return _run("fal-ai/trellis", args, "trellis", want_preview=False)
