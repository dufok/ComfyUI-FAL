"""
Shared helpers for the ComfyUI-FAL node pack.

All FAL nodes follow the same shape: take an IMAGE (or params), upload any image to
FAL, call an endpoint via fal_client, then turn the result into ComfyUI types
(IMAGE / MASK / file path + download link / text).

Auth: reads FAL_KEY from the environment (passed via docker-compose, same as the
gokayfem ComfyUI-fal-API pack). fal_client is already installed in the image.
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


# --------------------------------------------------------------------------- auth

def require_key():
    if not os.environ.get("FAL_KEY", "").strip():
        raise RuntimeError(
            "FAL_KEY is not set in the container environment. "
            "It is normally passed in via docker-compose from ~/comfyui-docker/.env."
        )


# --------------------------------------------------------------------------- upload

def tensor_frame_to_png_path(tensor_frame):
    """First frame of an IMAGE tensor -> a temp PNG file, return its path."""
    arr = tensor_frame.detach().cpu().numpy()
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    fd, path = tempfile.mkstemp(suffix=".png", prefix="fal_")
    os.close(fd)
    Image.fromarray(arr).save(path, format="PNG")
    return path


def upload_image(image):
    """IMAGE tensor (uses first frame) -> uploaded FAL URL."""
    if image is None:
        raise RuntimeError("no 'image' connected — connect a LoadImage (or any IMAGE) output")
    path = tensor_frame_to_png_path(image[0])
    try:
        return fal_client.upload_file(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def upload_image_frames(image):
    """Every frame of an IMAGE tensor -> list of uploaded FAL URLs (batch = multi-ref input)."""
    if image is None:
        raise RuntimeError("no 'image' connected — connect a LoadImage (or any IMAGE) output")
    urls = []
    for frame in image:
        path = tensor_frame_to_png_path(frame)
        try:
            urls.append(fal_client.upload_file(path))
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    return urls


def upload_mask(mask):
    """MASK tensor [B,H,W] (1 = area to edit) -> uploaded grayscale PNG URL (white = edit)."""
    if mask is None:
        raise RuntimeError("no 'mask' connected — draw one in MaskEditor or connect a MASK output")
    arr = mask[0].detach().cpu().numpy()
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    fd, path = tempfile.mkstemp(suffix=".png", prefix="fal_mask_")
    os.close(fd)
    Image.fromarray(arr, mode="L").save(path, format="PNG")
    try:
        return fal_client.upload_file(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# --------------------------------------------------------------------------- result parsing

def deep_find(obj, key):
    if isinstance(obj, dict):
        if key in obj and obj[key] is not None:
            return obj[key]
        for v in obj.values():
            f = deep_find(v, key)
            if f is not None:
                return f
    elif isinstance(obj, list):
        for item in obj:
            f = deep_find(item, key)
            if f is not None:
                return f
    return None


def _fetch(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def blank_image():
    return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


def url_to_image_tensor(url):
    """Download a URL into a [1,H,W,3] RGB IMAGE tensor (blank on failure)."""
    try:
        pil = Image.open(io.BytesIO(_fetch(url))).convert("RGB")
        arr = np.asarray(pil).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except Exception:
        return blank_image()


def image_and_mask_from_url(url):
    """Download a (possibly RGBA) URL into ([1,H,W,3] IMAGE, [1,H,W] MASK).

    For background-removal output the alpha channel becomes the MASK (1 = subject).
    """
    pil = Image.open(io.BytesIO(_fetch(url)))
    if pil.mode == "RGBA":
        rgba = np.asarray(pil).astype(np.float32) / 255.0
        rgb = rgba[..., :3]
        alpha = rgba[..., 3]
    else:
        rgb = np.asarray(pil.convert("RGB")).astype(np.float32) / 255.0
        alpha = np.ones(rgb.shape[:2], dtype=np.float32)
    img = torch.from_numpy(np.ascontiguousarray(rgb)).unsqueeze(0)
    mask = torch.from_numpy(np.ascontiguousarray(alpha)).unsqueeze(0)
    return img, mask


def _collect_image_urls(result):
    urls = []
    imgs = result.get("images") if isinstance(result, dict) else None
    if isinstance(imgs, list):
        for it in imgs:
            u = it.get("url") if isinstance(it, dict) else (it if isinstance(it, str) else None)
            if u:
                urls.append(u)
    if not urls:
        single = deep_find(result, "image")
        u = single.get("url") if isinstance(single, dict) else (single if isinstance(single, str) else None)
        if u:
            urls.append(u)
    if not urls:
        u = deep_find(result, "url")
        if u:
            urls.append(u)
    return urls


def images_from_result(result):
    """Best-effort: turn any FAL image result into a batched [N,H,W,3] IMAGE tensor."""
    tensors = []
    for u in _collect_image_urls(result):
        try:
            arr = np.asarray(Image.open(io.BytesIO(_fetch(u))).convert("RGB")).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr))
        except Exception:
            pass
    if not tensors:
        return blank_image()
    # Assume FAL returns same-sized images in a batch; if not, fall back to the first.
    try:
        return torch.stack(tensors, 0)
    except Exception:
        return tensors[0].unsqueeze(0)


# --------------------------------------------------------------------------- image runner

def run_image(endpoint, arguments):
    """submit -> wait -> batched IMAGE tensor from any FAL image endpoint."""
    require_key()
    printable = {k: (f"<{len(v)} urls>" if k == "image_urls" else v) for k, v in arguments.items()}
    print(f"[FAL] {endpoint} <- {printable}")
    result = fal_client.subscribe(endpoint, arguments=arguments, with_logs=False)
    return images_from_result(result)


# --------------------------------------------------------------------------- files (meshes etc.)

def public_download_url(fname):
    public = os.environ.get("COMFYUI_PUBLIC_URL", "").rstrip("/")
    tail = f"/view?filename={fname}&type=output"
    return f"{public}{tail}" if public else tail


def mesh_url(result):
    """Locate the .glb / mesh URL across the various FAL 3D output shapes."""
    for key in ("model_mesh", "model_glb_pbr", "model_glb"):
        node = result.get(key) if isinstance(result, dict) else None
        if isinstance(node, dict) and node.get("url"):
            return node["url"]
        if isinstance(node, str) and node.startswith("http"):
            return node
    url = deep_find(result, "url")
    if url and any(url.lower().split("?")[0].endswith(ext) for ext in (".glb", ".gltf", ".fbx", ".zip")):
        return url
    return url


def save_file(url, prefix):
    """Download a URL into ComfyUI's output dir, return (fname, download_url, size_mb).

    `fname` is relative to the output dir — exactly what the core Preview3D node's
    model_file input expects.
    """
    clean = url.split("?")[0]
    ext = clean.rsplit(".", 1)[-1].lower()
    if ext not in ("glb", "gltf", "fbx", "zip", "png", "jpg", "jpeg", "webp"):
        ext = "glb"
    base = os.path.basename(clean) or f"{prefix}.{ext}"
    fname = f"{prefix}_{base}"
    out_dir = folder_paths.get_output_directory()
    dest = os.path.join(out_dir, fname)
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / 1_000_000
    return fname, public_download_url(fname), size_mb


# --------------------------------------------------------------------------- mesh runner (3D)

MESH_RET_TYPES = ("STRING", "STRING", "IMAGE", "STRING")
MESH_RET_NAMES = ("glb_file", "download_url", "preview", "info")


def run_mesh(endpoint, arguments, prefix, want_preview=True):
    """submit -> wait -> download mesh -> 4-tuple, with the link folded into `info`.

    `glb_file` is relative to ComfyUI's output dir — wire it straight into the core
    Preview3D node (model_file) for an interactive in-graph 3D view.
    """
    require_key()
    print(f"[FAL] {endpoint} <- {arguments}")
    result = fal_client.subscribe(endpoint, arguments=arguments, with_logs=False)
    url = mesh_url(result)
    if not url:
        raise RuntimeError(f"no mesh url in FAL response: {result}")
    fname, download_url, size_mb = save_file(url, prefix)
    preview = blank_image()
    if want_preview:
        rendered = deep_find(result, "rendered_image") or deep_find(result, "thumbnail")
        thumb = rendered.get("url") if isinstance(rendered, dict) else (
            rendered if isinstance(rendered, str) else None)
        if thumb:
            preview = url_to_image_tensor(thumb)
    info = f"{endpoint} -> {fname} ({size_mb:.2f} MB)  ⬇ {download_url}"
    print(f"[FAL] DONE {endpoint} -> {fname} ({size_mb:.2f} MB)")
    print(f"[FAL] DOWNLOAD: {download_url}")
    return (fname, download_url, preview, info)
