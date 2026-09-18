"""
Shared helpers for the ComfyUI-FAL node pack.

All FAL nodes follow the same shape: take an IMAGE (or params), upload any image to
FAL, call an endpoint via fal_client, then turn the result into ComfyUI types
(IMAGE / MASK / file path + download link / text).

Auth: reads FAL_KEY from the environment (passed via docker-compose, same as the
gokayfem ComfyUI-fal-API pack). fal_client is already installed in the image.
"""
import io
import json
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


# --------------------------------------------------------------------------- schema cache

SCHEMA_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fal_schema.json")
_SCHEMA = None


def _schema_for(endpoint):
    """Cached input schema for one endpoint, or None when the cache is absent.

    Written by `fal_registry.py limits` straight from FAL's public OpenAPI. Nothing in
    here is hand-typed on purpose: a hand-typed limit drifts, and a drifted limit is
    worse than no limit — it starts refusing calls FAL would have accepted.
    """
    global _SCHEMA
    if _SCHEMA is None:
        try:
            with open(SCHEMA_CACHE) as f:
                _SCHEMA = json.load(f).get("endpoints", {})
        except (OSError, ValueError):
            _SCHEMA = {}
    return _SCHEMA.get(endpoint)


# --------------------------------------------------------------------------- validation

_MEDIA_SUFFIXES = ("_url", "_urls", "_image", "_images", "_mask", "_video", "_audio")
_MEDIA_NAMES = {"image", "images", "mask", "video", "audio", "url", "urls", "medias"}


def _is_media_key(name):
    return name in _MEDIA_NAMES or name.endswith(_MEDIA_SUFFIXES)


def _media_problems(arguments):
    """Media keys that carry nothing.

    Nodes only add a media key once the upload returned something, so an empty one here
    means the upload produced nothing and nobody noticed. FAL answers that with a schema
    error naming the field but not the cause; this names the cause, before the call.
    """
    out = []
    for name, val in arguments.items():
        if not _is_media_key(name):
            continue
        if val is None or (isinstance(val, str) and not val.strip()):
            out.append(f"{name} is empty")
        elif isinstance(val, (list, tuple)):
            if not val:
                out.append(f"{name} is an empty list")
            else:
                blank = [i for i, v in enumerate(val)
                         if v is None or (isinstance(v, str) and not v.strip())]
                if blank:
                    out.append(f"{name} has empty entries at {blank}")
    return out


def _schema_problems(endpoint, arguments):
    """Required fields, list bounds and enum values, checked against FAL's own schema."""
    spec = _schema_for(endpoint)
    if not spec:
        return []
    out = []
    for name in spec.get("required", []):
        if name not in arguments:
            out.append(f"missing required '{name}'")
    fields = spec.get("fields") or {}
    for name, val in arguments.items():
        f = fields.get(name)
        if not f:
            continue
        if isinstance(val, (list, tuple)):
            hi, lo = f.get("max_items"), f.get("min_items")
            if hi is not None and len(val) > hi:
                out.append(f"{name}: {len(val)} items, model takes at most {hi}")
            if lo is not None and len(val) < lo:
                out.append(f"{name}: {len(val)} items, model needs at least {lo}")
        allowed = f.get("enum")
        if allowed and isinstance(val, str) and val not in allowed:
            out.append(f"{name}={val!r} not allowed (allowed: {', '.join(map(str, allowed))})")
    return out


def validate_arguments(endpoint, arguments):
    """Refuse locally what FAL would refuse remotely — before the money leaves.

    FAL_SKIP_VALIDATION=1 turns this off. That escape hatch is the point of the whole
    design: a stale cache must never be able to block a call FAL itself would take.
    """
    if os.environ.get("FAL_SKIP_VALIDATION", "").strip():
        return
    problems = _media_problems(arguments) + _schema_problems(endpoint, arguments)
    if problems:
        raise RuntimeError(
            f"{endpoint}: not sent — " + "; ".join(problems)
            + f"\n  check with `fal_registry.py schema {endpoint}`, "
              "or set FAL_SKIP_VALIDATION=1 to send anyway")


# --------------------------------------------------------------------------- the FAL call

# Older fal_client releases do not export these; an empty tuple in `except` is legal and
# simply never matches, so the pack keeps working against whatever version the image has.
_HTTP_ERROR = getattr(fal_client, "FalClientHTTPError", ())
_TIMEOUT_ERROR = getattr(fal_client, "FalClientTimeoutError", ())

_POLICY_HINTS = ("nsfw", "content_policy", "content-policy", "content policy",
                 "safety", "moderation", "prohibited", "ip_detected")


def _describe_http_error(endpoint, e):
    """Say which kind of refusal this is, because the three need opposite reactions.

    A content refusal needs a different prompt, a schema refusal needs different
    arguments, a 5xx needs nothing but patience. They used to read identically.

    Retrying stays a human decision on purpose: a job that failed after FAL accepted it
    has already been charged, so an automatic retry would quietly double the bill.
    """
    code = getattr(e, "status_code", None)
    etype = (getattr(e, "error_type", None) or "").lower()
    if not etype:
        headers = getattr(e, "response_headers", None) or {}
        try:
            etype = str(headers.get("x-fal-error-type", "") or "").lower()
        except Exception:
            etype = ""
    body = str(getattr(e, "message", "") or e)
    blob = f"{etype} {body}".lower()

    if any(h in blob for h in _POLICY_HINTS):
        return (f"{endpoint}: refused by the content filter"
                + (f" [{etype}]" if etype else "")
                + " — rephrase the prompt or swap the reference; the same input will refuse again."
                + f"\n  {body}")
    if "file_too_large" in blob or "exceeds the maximum allowed size" in blob:
        return (f"{endpoint}: an uploaded file is over this endpoint's size limit — scale the image "
                f"down before this node (ImageScale, long side ~2048). Resubmitting as is refuses "
                f"again.\n  {body}")
    if code == 422:
        return (f"{endpoint}: FAL rejected the arguments (422) — compare field names and values "
                f"with `fal_registry.py schema {endpoint}`.\n  {body}")
    if code == 429:
        return (f"{endpoint}: rate limited (429) — too many requests in flight, wait and resubmit."
                f"\n  {body}")
    if code in (401, 403):
        return (f"{endpoint}: FAL rejected the key ({code}) — check FAL_KEY in the container env."
                f"\n  {body}")
    if code and code >= 500:
        return f"{endpoint}: FAL server error ({code}) — their side, safe to resubmit.\n  {body}"
    return f"{endpoint}: FAL error{f' ({code})' if code else ''}\n  {body}"


def subscribe(endpoint, arguments):
    """The one door to FAL: validate, call, translate the failure.

    Every node in the pack goes through here, so a refusal reads the same wherever it
    happens and never arrives as a bare stack trace.
    """
    require_key()
    validate_arguments(endpoint, arguments)
    try:
        return fal_client.subscribe(endpoint, arguments=arguments, with_logs=False)
    except _HTTP_ERROR as e:
        raise RuntimeError(_describe_http_error(endpoint, e)) from e
    except _TIMEOUT_ERROR as e:
        raise RuntimeError(
            f"{endpoint}: timed out waiting for the result. The job may still be running on FAL "
            f"and is billable either way — check the dashboard before resubmitting.\n  {e}") from e


def check_content_filter(result, endpoint=""):
    """FAL's safety checker returns blank frames and a flag, never an error.

    Left unread, a flagged frame travels downstream as a finished render — exactly the
    silent-black-image failure images_from_result already refuses to allow.
    """
    flags = result.get("has_nsfw_concepts") if isinstance(result, dict) else None
    if not isinstance(flags, (list, tuple)) or not any(flags):
        return
    hit, total = sum(1 for f in flags if f), len(flags)
    where = f"{endpoint}: " if endpoint else ""
    if hit == total:
        raise RuntimeError(
            f"{where}the safety checker flagged every image ({hit}/{total}) — what came back is "
            f"blank frames, not a render. Rephrase, or set enable_safety_checker=false on models "
            f"that expose it.")
    print(f"[FAL] warning: safety checker flagged {hit} of {total} images — those come back blank.")


# --------------------------------------------------------------------------- upload limits

MB = 1_000_000  # decimal — the safe reading whenever a doc just says "MB"

# Input-image limits that these endpoints state in their own FAL input schema (the field
# descriptions, read 2026-09-17 with `fal_registry.py schema <endpoint>`). Only endpoints
# that document a limit are listed. Fitting an image "just in case" would shrink what an
# upscaler or an inpaint receives and quietly damage the result, so nothing else is touched.
#
#   recraft/vectorize    "less than 5 MB in size, have resolution less than 16 MP and max
#                        dimension less than 4096 pixels, min dimension more than 256".
#                        Its real cap is 5 MiB (5242880 B, from its own error message);
#                        decimal undershoots that on purpose.
#   hunyuan-3d v3.1      front view "128-5000px, max 8MB"; rapid adds "recommended ≤6MB for
#                        base64 encoding". The side views state nothing, but they reach the
#                        same model the same way, so they are fitted the same.
#   hunyuan3d-v3 sketch  "between 128x128 and 5000x5000 pixels".
#   hi3d                 "PNG, JPEG and WebP formats are supported, up to 20MB", every view.
UPLOAD_FIT = {
    "fal-ai/recraft/vectorize":
        {"max_bytes": 5 * MB, "max_pixels": 15_999_999, "max_side": 4095, "min_side": 257},
    "fal-ai/hunyuan-3d/v3.1/pro/image-to-3d": {"max_bytes": 8 * MB, "max_side": 5000, "min_side": 128},
    "fal-ai/hunyuan-3d/v3.1/rapid/image-to-3d": {"max_bytes": 6 * MB, "max_side": 5000, "min_side": 128},
    "fal-ai/hunyuan3d-v3/sketch-to-3d": {"max_side": 5000, "min_side": 128},
    "hitem3d/hi3d/image-to-3d": {"max_bytes": 20 * MB},
    "hitem3d/hi3d/v3.0/image-to-3d": {"max_bytes": 20 * MB},
    "hitem3d/hi3d/v3.0/multi-view-to-3d": {"max_bytes": 20 * MB},
}

# Pillow >= 9.1 keeps the filters under Image.Resampling; older ones on Image itself.
_LANCZOS = getattr(getattr(Image, "Resampling", None), "LANCZOS", None) or getattr(Image, "LANCZOS", None)

# upload URL -> "fitted WxH -> wxh" note, drained into the node's info by fit_notes()
_FIT_NOTES = {}


def _png_bytes(pil):
    """Same PNG settings upload_image has always used. optimize=True was measured on a real
    2764x1536 render on proserver: 0.8% smaller for twice the time (7.3 s vs 4.0 s), so a
    frame that already fits costs exactly what it did before fitting existed."""
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def fit_png(pil, spec, endpoint=""):
    """PIL image -> (PNG bytes, note) inside one endpoint's documented limits.

    Resolution first (longest side, then pixel count), then bytes. PNG size tracks area,
    so each byte pass shrinks the area by the ratio it is over, with a 5% margin. Every
    pass resizes from the original, so repeated passes never stack blur. It stays lossless
    PNG throughout: the endpoints here trace edges or read geometry, and JPEG ringing would
    hurt them more than a smaller frame does. The note is empty when nothing had to change.
    """
    w0, h0 = pil.size
    lo = spec.get("min_side")
    if lo and min(w0, h0) < lo:
        raise RuntimeError(
            f"{endpoint}: the image is {w0}x{h0}, this endpoint needs more than {lo - 1} px on the "
            f"short side — upscale it before this node")

    scale = 1.0
    if spec.get("max_side"):
        scale = min(scale, spec["max_side"] / max(w0, h0))
    if spec.get("max_pixels"):
        scale = min(scale, (spec["max_pixels"] / (w0 * h0)) ** 0.5)

    def at(s):
        return pil if s >= 1.0 else pil.resize((max(1, int(w0 * s)), max(1, int(h0 * s))), _LANCZOS)

    img = at(scale)
    data = _png_bytes(img)
    cap = spec.get("max_bytes")
    for _ in range(8):
        if not cap or len(data) <= cap:
            break
        scale *= (cap / len(data)) ** 0.5 * 0.95
        if lo and min(w0, h0) * scale < lo:
            raise RuntimeError(
                f"{endpoint}: cannot get this image under {cap / MB:.0f} MB without going below "
                f"{lo} px on the short side")
        img = at(scale)
        data = _png_bytes(img)
    else:
        raise RuntimeError(f"{endpoint}: could not fit the image under {cap / MB:.0f} MB")

    note = ""
    if img.size != (w0, h0):
        note = (f"fitted {w0}x{h0} -> {img.size[0]}x{img.size[1]} ({len(data) / MB:.1f} MB) "
                f"for {endpoint}'s upload limit")
    return data, note


def fit_notes(arguments):
    """The fitting notes for every upload in these arguments — for a node's info output.

    Printing alone is not enough: the next time the output looks soft, the reason should be
    on the node, not in a container log.
    """
    notes = []
    for v in arguments.values():
        for u in (v if isinstance(v, (list, tuple)) else (v,)):
            if isinstance(u, str) and u in _FIT_NOTES:
                notes.append(_FIT_NOTES.pop(u))
    return notes


def _upload_fitted(frame, spec, endpoint):
    """One IMAGE frame -> fitted PNG -> uploaded FAL URL."""
    arr = (np.clip(frame.detach().cpu().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8)
    data, note = fit_png(Image.fromarray(arr), spec, endpoint)
    fd, path = tempfile.mkstemp(suffix=".png", prefix="fal_fit_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        url = fal_client.upload_file(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if note:
        print(f"[FAL] {note}")
        _FIT_NOTES[url] = note
    return url


# --------------------------------------------------------------------------- upload

def tensor_frame_to_png_path(tensor_frame):
    """First frame of an IMAGE tensor -> a temp PNG file, return its path."""
    arr = tensor_frame.detach().cpu().numpy()
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    fd, path = tempfile.mkstemp(suffix=".png", prefix="fal_")
    os.close(fd)
    Image.fromarray(arr).save(path, format="PNG")
    return path


def upload_image(image, fit_for=None):
    """IMAGE tensor (uses first frame) -> uploaded FAL URL.

    fit_for=<endpoint> first fits the frame into the upload limits that endpoint documents
    (UPLOAD_FIT). An endpoint with no documented limit is sent exactly as before.
    """
    if image is None:
        raise RuntimeError("no 'image' connected — connect a LoadImage (or any IMAGE) output")
    spec = UPLOAD_FIT.get(fit_for) if fit_for else None
    if spec:
        return _upload_fitted(image[0], spec, fit_for)
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


def upload_image_rgba(image, mask):
    """IMAGE + MASK (1 = subject) -> uploaded RGBA PNG URL with the mask as alpha."""
    if image is None:
        raise RuntimeError("no 'image' connected — connect a LoadImage (or any IMAGE) output")
    rgb = (np.clip(image[0].detach().cpu().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8)
    a = (np.clip(mask[0].detach().cpu().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8)
    if a.shape != rgb.shape[:2]:
        a = np.asarray(Image.fromarray(a, mode="L").resize((rgb.shape[1], rgb.shape[0])))
    rgba = np.dstack([rgb, a])
    fd, path = tempfile.mkstemp(suffix=".png", prefix="fal_rgba_")
    os.close(fd)
    Image.fromarray(rgba, mode="RGBA").save(path, format="PNG")
    try:
        return fal_client.upload_file(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def grow_mask(mask, pixels):
    """Dilate a MASK tensor [B,H,W] by `pixels` (max-pool) — erasers need margin around
    the object or edge colors bleed back into the reconstruction."""
    if not pixels or pixels <= 0:
        return mask
    k = 2 * int(pixels) + 1
    return torch.nn.functional.max_pool2d(mask.unsqueeze(1), k, stride=1, padding=int(pixels)).squeeze(1)


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
    """Download a URL into a [1,H,W,3] RGB IMAGE tensor.

    Returns a blank frame on failure — this one is for decorative previews (mesh
    thumbnails, rendered turntables), where losing the picture should not fail a job
    that otherwise succeeded. It still says so out loud. For real image output use
    images_from_result, which raises.
    """
    try:
        pil = Image.open(io.BytesIO(_fetch(url))).convert("RGB")
        arr = np.asarray(pil).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except Exception as e:
        print(f"[FAL] warning: could not fetch preview {url}: {e}")
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


def images_from_result(result, endpoint=""):
    """Turn a FAL image result into a batched [N,H,W,3] IMAGE tensor.

    Raises rather than returning a blank frame. A silent black image is worse than an
    error: downstream nodes accept it, the prompt completes as a success, and anything
    driving ComfyUI headlessly (a Blender add-on, a script) treats the black frame as
    the render. Fail loudly instead.
    """
    check_content_filter(result, endpoint)
    urls = _collect_image_urls(result)
    tensors, failed = [], []
    for u in urls:
        try:
            arr = np.asarray(Image.open(io.BytesIO(_fetch(u))).convert("RGB")).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr))
        except Exception as e:
            failed.append(f"{u} ({e})")
    if not tensors:
        if urls:
            raise RuntimeError(
                "FAL returned image URLs but none could be downloaded/decoded: "
                + "; ".join(failed))
        raise RuntimeError(f"no image URL in the FAL response: {result}")
    if failed:
        print(f"[FAL] warning: {len(failed)} of {len(urls)} images failed to download: {failed}")
    # Assume FAL returns same-sized images in a batch; if not, fall back to the first.
    try:
        return torch.stack(tensors, 0)
    except Exception:
        return tensors[0].unsqueeze(0)


# --------------------------------------------------------------------------- image runner

def run_image(endpoint, arguments):
    """submit -> wait -> batched IMAGE tensor from any FAL image endpoint."""
    printable = {k: (f"<{len(v)} urls>" if k == "image_urls" else v) for k, v in arguments.items()}
    print(f"[FAL] {endpoint} <- {printable}")
    result = subscribe(endpoint, arguments)
    return images_from_result(result, endpoint)


def run_image_described(endpoint, arguments):
    """Like run_image, but also returns the model's own `description` string.

    The Gemini / Nano Banana family always returns `description` alongside `images` —
    it is where the model explains what it did, and where `thinking_level` output
    surfaces. Every other wrapper throws it away; here it becomes a STRING output.
    """
    printable = {k: (f"<{len(v)} urls>" if k == "image_urls" else v) for k, v in arguments.items()}
    print(f"[FAL] {endpoint} <- {printable}")
    result = subscribe(endpoint, arguments)
    description = result.get("description") if isinstance(result, dict) else None
    if description:
        print(f"[FAL] description: {description}")
    return images_from_result(result, endpoint), (description or "")


# --------------------------------------------------------------------------- text runner

def run_text(endpoint, arguments):
    """submit -> wait -> (text, reasoning, info) from an openrouter/router* endpoint.

    No empty-string fallback anywhere. An empty caption is invisible downstream — it just
    becomes an empty prompt on the next node and the graph completes green — so every
    failure mode raises instead.
    """
    printable = {k: (f"<{len(v)} urls>" if k.endswith("_urls") else v) for k, v in arguments.items()}
    print(f"[FAL] {endpoint} <- {printable}")
    result = subscribe(endpoint, arguments)
    if not isinstance(result, dict):
        raise RuntimeError(f"{endpoint}: unexpected response {result!r}")
    # openrouter/router can answer 200 with an `error` string and an empty output.
    err = result.get("error")
    if err:
        raise RuntimeError(f"{endpoint} returned an error: {err}")
    text = (result.get("output") or "").strip()
    if not text:
        raise RuntimeError(
            f"{endpoint} returned an empty output (model={arguments.get('model')}, "
            f"usage={result.get('usage')}) — raise max_tokens or try another model")
    if result.get("partial"):
        print(f"[FAL] warning: {endpoint} flagged the answer as partial — max_tokens hit?")
    u = result.get("usage") or {}
    info = (f"{arguments.get('model')} | in={u.get('prompt_tokens', '?')} "
            f"out={u.get('completion_tokens', '?')} | ${float(u.get('cost') or 0.0):.6f}")
    print(f"[FAL] DONE {endpoint} — {info}")
    return text, (result.get("reasoning") or ""), info


# --------------------------------------------------------------------------- files (meshes etc.)

def public_download_url(fname):
    public = os.environ.get("COMFYUI_PUBLIC_URL", "").rstrip("/")
    tail = f"/view?filename={fname}&type=output"
    return f"{public}{tail}" if public else tail


def file_url(node):
    """FAL returns files as {"url": ..., "content_type": ...} — or occasionally a bare
    string. Normalise either into a URL (None if there is nothing usable)."""
    if isinstance(node, dict):
        url = node.get("url")
        return url if isinstance(url, str) else None
    return node if isinstance(node, str) and node.startswith("http") else None


def mesh_url(result):
    """Locate the .glb / mesh URL across the various FAL 3D output shapes.

    `rigged_character_glb` is here for the standalone rigging endpoint, whose output
    carries none of the usual mesh keys — without it the generic deep_find fallback
    would return whichever "url" happened to serialize first (often a walk-cycle clip).
    """
    for key in ("model_mesh", "model_glb_pbr", "model_glb", "rigged_character_glb"):
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
    if ext not in ("glb", "gltf", "fbx", "obj", "mtl", "usdz", "blend", "stl", "zip",
                   "png", "jpg", "jpeg", "webp", "svg", "exr", "ply", "splat", "spz",
                   "mp4", "webm", "mov", "m4v", "mkv"):
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
    print(f"[FAL] {endpoint} <- {arguments}")
    result = subscribe(endpoint, arguments)
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
    for note in fit_notes(arguments):
        info += f"\n{note}"
    print(f"[FAL] DONE {endpoint} -> {fname} ({size_mb:.2f} MB)")
    print(f"[FAL] DOWNLOAD: {download_url}")
    return (fname, download_url, preview, info)
