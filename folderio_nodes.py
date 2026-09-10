"""ComfyUI-FolderIO — batch photos in, batch photos out, from the browser.

FolderIOLoadImages
    Pick a subfolder of ``input/`` and load every photo in it as an IMAGE *list* (one graph pass
    per photo), together with the original file stems and a count. The browser side
    (``web/folderio.js``) adds "Upload folder…" / "Upload files…" buttons and folder drag-and-drop
    that create that subfolder straight from your computer through the stock ``/upload/image``
    endpoint — no scripts, no scp. EXIF orientation is applied and embedded ICC profiles
    (Display P3 phone shots, CMYK scans) are converted to sRGB.

FolderIOLoadSequence
    The same folder, loaded as ONE batched IMAGE instead of a list — a numbered render sequence
    (Blender depth or beauty) on its way into a video. 16-bit PNGs keep their precision.

FolderIOSplitByShortSide / FolderIOMergeSubset
    Per-photo gating for paid per-image nodes. In list mode ComfyUI resolves lazy inputs for the
    whole list at once, so a Switch does NOT stop an upstream upscaler from running on every photo
    the moment one of them needs it. Split hands the upscaler only the photos whose short side is
    below the target (plus their indices); Merge puts the processed ones back in place and declares
    the processed list as a *lazy* input, so when nothing needs work the upscaler is never executed
    (and never receives an empty list, which per-item nodes cannot handle).

FolderIOSaveZip
    Collect the whole list back (``INPUT_IS_LIST``), write ``output/<folder>/<stem><suffix>.<ext>``,
    pack this run's files into one ZIP (written atomically), show all results as a gallery on the
    node and hand out a download link for the ZIP (button on the node + ``download_url`` output).
"""

import hashlib
import io
import json
import logging
import os
import re
import zipfile
from urllib.parse import quote

import numpy as np
import torch
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

import folder_paths
import node_helpers

log = logging.getLogger("ComfyUI-FolderIO")

PLAIN_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif")
HEIF_EXTS = (".heic", ".heif", ".hif", ".avif")
MAX_STEM_BYTES = 150  # keep stem + suffix + extension well under the 255-byte file-name limit

try:  # optional: iPhone HEIC. `pip install pillow-heif`
    import pillow_heif

    pillow_heif.register_heif_opener()
    try:
        pillow_heif.register_avif_opener()
    except Exception:  # noqa: BLE001 — older pillow-heif without AVIF
        pass
    HEIF_OK = True
except Exception:  # noqa: BLE001 — not installed: HEIC files are reported as skipped
    # Note for Docker setups: installing this with `pip install` *inside a running container* is
    # lost the next time the container is recreated (compose up after an edit). Put it in the image.
    HEIF_OK = False

try:  # littlecms ships with the standard Pillow wheels; guarded anyway
    from PIL import ImageCms

    _SRGB = ImageCms.createProfile("sRGB")
except Exception:  # noqa: BLE001
    ImageCms, _SRGB = None, None

_BAD_CHARS = re.compile(r'[\\:*?"<>|\x00-\x1f]')
_DOTS = re.compile(r"\.{2,}")
_NUM = re.compile(r"(\d+)")


def _first(value, default=None):
    """INPUT_IS_LIST hands every widget over as a 1-element list."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return default if value is None else value


def _natural(name):
    """Sort key so photo_2 comes before photo_10."""
    return [int(p) if p.isdigit() else p.lower() for p in _NUM.split(name)]


def _safe_stem(name):
    """A file-name stem safe to write under output/ AND to fetch back through /view:
    no path parts, no control chars, no '..' anywhere (the /view handler rejects it), bounded length."""
    stem = os.path.basename(str(name or "").replace("\\", "/"))
    stem = _DOTS.sub(".", _BAD_CHARS.sub("_", stem)).strip(" .")
    if len(stem.encode("utf-8")) > MAX_STEM_BYTES:
        stem = stem.encode("utf-8")[:MAX_STEM_BYTES].decode("utf-8", "ignore").rstrip(" .")
    return stem


def _safe_subfolder(folder):
    """Relative subfolder under output/ (nesting allowed) with no '..' or absolute parts."""
    parts = []
    for part in str(folder or "").replace("\\", "/").split("/"):
        part = _BAD_CHARS.sub("_", part).strip(" .")
        if part:
            parts.append(part)
    return "/".join(parts)


def _under(base, rel):
    """Absolute path of ``rel`` inside ``base``; refuses anything that escapes it."""
    base = os.path.abspath(base)
    path = os.path.abspath(os.path.join(base, rel))
    if os.path.commonpath((base, path)) != base:
        raise ValueError(f"path escapes {base}: {rel!r}")
    return path


def _to_srgb(img):
    """RGB pixels in sRGB. Honours an embedded ICC profile (Display P3 iPhone shots, Adobe RGB,
    CMYK scans) when littlecms is available; plain convert() otherwise."""
    icc = img.info.get("icc_profile") if _SRGB is not None else None
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            if "srgb" in (ImageCms.getProfileDescription(src) or "").lower():
                return img.convert("RGB")
            base = img if img.mode in ("RGB", "CMYK") else img.convert("RGB")
            return ImageCms.profileToProfile(base, src, _SRGB, outputMode="RGB")
        except Exception:  # noqa: BLE001 — odd/broken profile: fall back to the naive path
            log.debug("[FolderIO] ICC conversion failed, using plain RGB", exc_info=True)
    return img.convert("RGB")


def _present(value):
    """A lazy input that was not evaluated yet arrives as (None,) / [None] / None."""
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return any(v is not None for v in value)
    return True


# ------------------------------------------------------------------------------------------------
class FolderIOLoadImages:
    """Load every photo of an input/ subfolder as a list of IMAGE tensors."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": (
                    folder_paths.get_input_subfolders(),
                    {
                        "tooltip": "Subfolder of input/. Use the 📁 Upload folder… button, or drop a "
                        "folder onto this node, to create one from your computer. R refreshes the list.",
                    },
                ),
                "sort_by": (["name", "date", "date_desc"], {"default": "name"}),
                "start_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 99999, "tooltip": "Skip the first N files (after sorting)."},
                ),
                "max_images": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 99999,
                        "tooltip": "0 = all. Every image is held in RAM as float32 (~12 bytes per pixel, "
                        "a 24 MP photo ≈ 290 MB), so cap very big folders.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "filenames", "count")
    OUTPUT_IS_LIST = (True, True, False)
    OUTPUT_TOOLTIPS = (
        "List of images — every node downstream runs once per photo.",
        "Original file names without extension (list, same order). Wire into Save Images + ZIP.",
        "Number of images loaded.",
    )
    FUNCTION = "load"
    CATEGORY = "image/folder"
    DESCRIPTION = (
        "Loads all photos of an input/ subfolder as a list (EXIF orientation applied, ICC → sRGB, "
        "HEIC if pillow-heif is installed). The 📁 button / drag-and-drop upload a folder from your computer."
    )

    @classmethod
    def _dir(cls, folder):
        rel = os.path.normpath(str(folder or ""))
        if rel in ("", ".") or rel.startswith("..") or os.path.isabs(rel):
            raise ValueError("choose a subfolder of input/")
        return _under(folder_paths.get_input_directory(), rel)

    @classmethod
    def _files(cls, folder, sort_by, start_index, max_images):
        d = cls._dir(folder)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"input/{folder} does not exist")
        exts = PLAIN_EXTS + (HEIF_EXTS if HEIF_OK else ())
        rows, skipped = [], []
        with os.scandir(d) as it:
            for entry in it:
                if entry.name.startswith(".") or not entry.is_file():
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in exts:
                    st = entry.stat()
                    if st.st_size > 0:
                        rows.append((entry.name, st.st_mtime_ns, st.st_size))
                elif ext in HEIF_EXTS:
                    skipped.append(entry.name)
        if sort_by == "date":
            rows.sort(key=lambda r: (r[1], _natural(r[0])))
        elif sort_by == "date_desc":
            rows.sort(key=lambda r: (-r[1], _natural(r[0])))
        else:
            rows.sort(key=lambda r: _natural(r[0]))
        rows = rows[int(start_index or 0):]
        if max_images:
            rows = rows[: int(max_images)]
        return d, rows, skipped

    @classmethod
    def VALIDATE_INPUTS(cls, folder):
        # Naming `folder` here also skips the combo membership check, so a folder created after
        # the last /object_info fetch (i.e. just uploaded) validates fine.
        try:
            d = cls._dir(folder)
        except ValueError as exc:
            return str(exc)
        if not os.path.isdir(d):
            return (
                f"input/{folder} not found — upload a folder first "
                "(📁 button on the node, or drop a folder onto it)"
            )
        return True

    @classmethod
    def IS_CHANGED(cls, folder, sort_by, start_index, max_images):
        try:
            _, rows, _ = cls._files(folder, sort_by, start_index, max_images)
        except Exception:  # noqa: BLE001 — let execute() raise the readable error
            return float("nan")
        h = hashlib.sha256()
        for name, mtime, size in rows:
            h.update(f"{name}\0{mtime}\0{size}\n".encode("utf-8", "surrogateescape"))
        return h.hexdigest()

    def load(self, folder, sort_by, start_index, max_images):
        d, rows, skipped = self._files(folder, sort_by, start_index, max_images)
        if not rows:
            raise ValueError(
                f"input/{folder}: no images found (png/jpg/jpeg/webp/bmp/tif/gif"
                f"{'/heic' if HEIF_OK else ''})"
            )
        images, stems, pixels = [], [], 0
        for name, _, _ in rows:
            img = node_helpers.pillow(Image.open, os.path.join(d, name))
            img = node_helpers.pillow(ImageOps.exif_transpose, img)  # phone photos: honour orientation
            if img.mode == "I":
                img = img.point(lambda i: i * (1 / 255))
            img = _to_srgb(img)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            images.append(torch.from_numpy(arr)[None, ...])
            stems.append(os.path.splitext(name)[0])
            pixels += img.width * img.height
        log.info(
            "[FolderIO] input/%s: %d image(s), %.1f MP total (~%.1f GB as float32)",
            folder, len(images), pixels / 1e6, pixels * 12 / 1e9,
        )
        result = (images, stems, len(images))
        if skipped:
            msg = (f"{len(skipped)} HEIC/HEIF file(s) skipped in input/{folder} — "
                   f"`pip install pillow-heif` in the ComfyUI environment to load them: "
                   + ", ".join(skipped[:5]) + (" …" if len(skipped) > 5 else ""))
            log.warning("[FolderIO] %s", msg)
            return {"ui": {"folderio_warning": [msg]}, "result": result}
        return result


# ------------------------------------------------------------------------------------------------
class FolderIOSplitByShortSide:
    """Pick out the photos that still need enlarging, so a paid upscaler only sees those."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The full list from Load Images (folder)."}),
                "min_short_side": (
                    "INT",
                    {
                        "default": 1980, "min": 1, "max": 32768,
                        "tooltip": "Photos whose short side is smaller than this go to images_below "
                        "(to be upscaled); the others are left alone.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("images_below", "index_below", "count_below", "count_ok")
    OUTPUT_IS_LIST = (True, True, False, False)
    OUTPUT_TOOLTIPS = (
        "Only the photos that need work (list). Feed the upscaler with this.",
        "Their positions in the original list (list). Feed Merge Subset with this.",
        "How many photos need work.",
        "How many already meet the target.",
    )
    FUNCTION = "split"
    CATEGORY = "image/folder"
    DESCRIPTION = (
        "Splits a list of images by short side: those below the target go out as a shorter list (plus "
        "their indices) so a per-image paid node runs only on them. Pair with Merge Subset."
    )

    def split(self, images, min_short_side):
        target = int(_first(min_short_side, 1980))
        below, index = [], []
        for i, t in enumerate(images or []):
            if not torch.is_tensor(t) or t.ndim < 3:
                continue
            h, w = int(t.shape[-3]), int(t.shape[-2])
            if min(h, w) < target:
                below.append(t)
                index.append(i)
        total = len([t for t in (images or []) if torch.is_tensor(t)])
        log.info("[FolderIO] split: %d of %d image(s) below %d px short side", len(below), total, target)
        return (below, index, len(below), total - len(below))


class FolderIOMergeSubset:
    """Put processed items back into the full list by index; the processed list is lazy."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The full original list."}),
                "index": (
                    "INT",
                    {"forceInput": True, "tooltip": "Positions to replace — index_below from Split by Short Side."},
                ),
            },
            "optional": {
                "replacements": (
                    "IMAGE",
                    {
                        "lazy": True,
                        "tooltip": "Processed items, same order as index. Evaluated only when index is "
                        "non-empty, so the upscaler chain never runs (or gets an empty list) when "
                        "nothing needs work.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "merge"
    CATEGORY = "image/folder"
    DESCRIPTION = (
        "Replaces the items at the given indices with the processed ones and returns the full list in "
        "the original order. Pair with Split by Short Side."
    )

    def check_lazy_status(self, images, index, replacements=None):
        wanted = [i for i in (index or []) if isinstance(i, int)]
        if wanted and not _present(replacements):
            return ["replacements"]
        return []

    def merge(self, images, index, replacements=None):
        out = list(images or [])
        wanted = [i for i in (index or []) if isinstance(i, int)]
        if not wanted:
            return (out,)
        reps = [r for r in (replacements or []) if torch.is_tensor(r)]
        if len(reps) != len(wanted):
            raise ValueError(
                f"Merge Subset: {len(wanted)} index value(s) but {len(reps)} replacement image(s) — "
                "wire index_below and the processed images from the same Split."
            )
        for i, r in zip(wanted, reps):
            if not 0 <= i < len(out):
                raise ValueError(f"Merge Subset: index {i} is outside the list of {len(out)} images")
            out[i] = r
        return (out,)


# ------------------------------------------------------------------------------------------------
class FolderIOSaveZip:
    """Save a list/batch of images into output/<folder>/ and zip them; gallery + download link."""

    INPUT_IS_LIST = True
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "List or batch of images (all of them, in one go)."}),
                "folder": (
                    "STRING",
                    {"default": "upscaled_1980", "tooltip": "Subfolder of output/ to write into (created if missing)."},
                ),
                "format": (["jpg", "png", "webp"], {"default": "jpg"}),
                "quality": (
                    "INT",
                    {"default": 95, "min": 1, "max": 100, "tooltip": "jpg / webp quality. png ignores it."},
                ),
                "suffix": ("STRING", {"default": "", "tooltip": "Appended to every file name, e.g. _1980"}),
                "zip_name": ("STRING", {"default": "", "tooltip": "ZIP file name (no extension). Empty = folder name."}),
                "overwrite": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Replace files (and the ZIP) with the same name; off = add (1), (2)…"},
                ),
            },
            "optional": {
                "filenames": (
                    "STRING",
                    {"forceInput": True, "tooltip": "Original stems from Load Images (folder). Without it files are numbered."},
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("download_url", "saved_files")
    OUTPUT_TOOLTIPS = (
        "Link to the ZIP of this run (COMFYUI_PUBLIC_URL + /view?…). Wire into Preview as Text to keep it visible.",
        "Saved files, one per line, relative to output/.",
    )
    FUNCTION = "save"
    CATEGORY = "image/folder"
    DESCRIPTION = (
        "Writes every image to output/<folder>/, zips this run's files and shows them all as a gallery "
        "on the node with a ⬇ Download ZIP button."
    )

    def save(self, images, folder, format, quality, suffix, zip_name, overwrite,
             filenames=None, prompt=None, extra_pnginfo=None):
        fmt = _first(format, "jpg")
        quality = int(_first(quality, 95))
        suffix = _safe_stem(_first(suffix, "") or "")
        overwrite = bool(_first(overwrite, True))
        sub = _safe_subfolder(_first(folder, "")) or "folderio"
        out_dir = _under(folder_paths.get_output_directory(), sub)
        os.makedirs(out_dir, exist_ok=True)

        tensors = [t for t in (images or []) if torch.is_tensor(t)]
        n_frames = sum(int(t.shape[0]) for t in tensors)
        names = [str(n) for n in (filenames or []) if n is not None]
        if names and len(names) not in (len(tensors), n_frames):
            log.warning("[FolderIO] %d filenames for %d images — falling back to numbering", len(names), n_frames)
            names = []
        per_item = bool(names) and len(names) == len(tensors)

        frames = []  # (HWC tensor, stem)
        k = 0
        for i, t in enumerate(tensors):
            batch = int(t.shape[0])
            for b in range(batch):
                if not names:
                    stem = f"photo_{k + 1:05d}"
                elif per_item:
                    stem = names[i] + (f"_{b + 1:02d}" if batch > 1 else "")
                else:
                    stem = names[k]
                frames.append((t[b], stem))
                k += 1
        if not frames:
            raise ValueError("Save Images + ZIP: nothing to save (no images arrived)")

        ext = {"jpg": ".jpg", "png": ".png", "webp": ".webp"}[fmt]
        meta = self._png_meta(prompt, extra_pnginfo) if fmt == "png" else None
        saved, used = [], set()
        for tensor, stem in frames:
            base = (_safe_stem(stem) or "photo") + suffix
            fname, path, n = base + ext, os.path.join(out_dir, base + ext), 1
            while fname in used or (not overwrite and os.path.exists(path)):
                fname = f"{base} ({n}){ext}"
                path = os.path.join(out_dir, fname)
                n += 1
            used.add(fname)
            self._write(tensor, path, fmt, quality, meta)
            saved.append(fname)

        zstem = _safe_stem(_first(zip_name, "") or "") or _safe_stem(sub.rsplit("/", 1)[-1]) or "images"
        zname, zpath, n = zstem + ".zip", os.path.join(out_dir, zstem + ".zip"), 1
        while not overwrite and os.path.exists(zpath):
            zname = f"{zstem} ({n}).zip"
            zpath = os.path.join(out_dir, zname)
            n += 1
        tmp = zpath + ".part"  # build next to the target, swap in atomically: a failed run never leaves a half ZIP
        try:
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
                for fname in saved:
                    zf.write(os.path.join(out_dir, fname), arcname=fname)
            os.replace(tmp, zpath)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        size = os.path.getsize(zpath)

        rel = f"/view?filename={quote(zname)}&subfolder={quote(sub, safe='/')}&type=output"
        public = os.environ.get("COMFYUI_PUBLIC_URL", "").rstrip("/")
        url = f"{public}{rel}" if public else rel
        print(f"[FolderIO] {len(saved)} file(s) → output/{sub}/  |  ZIP {size / 1048576:.1f} MB: {url}")
        return {
            "ui": {
                "images": [{"filename": f, "subfolder": sub, "type": "output"} for f in saved],
                "zip": [{
                    "filename": zname, "subfolder": sub, "type": "output",
                    "url": rel, "bytes": size, "count": len(saved),
                }],
            },
            "result": (url, "\n".join(f"{sub}/{f}" for f in saved)),
        }

    @staticmethod
    def _png_meta(prompt, extra_pnginfo):
        try:
            from comfy.cli_args import args
            if args.disable_metadata:
                return None
        except Exception:  # noqa: BLE001
            pass
        p, x = _first(prompt), _first(extra_pnginfo)
        if not p and not x:
            return None
        meta = PngInfo()
        if p:
            meta.add_text("prompt", json.dumps(p))
        if x:
            for key, val in x.items():
                meta.add_text(key, json.dumps(val))
        return meta

    @staticmethod
    def _write(tensor, path, fmt, quality, meta):
        arr = np.clip(tensor.detach().cpu().float().numpy() * 255.0, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        img = Image.fromarray(arr)
        if fmt == "jpg":
            kw = {"quality": quality, "optimize": True}
            if quality >= 90:
                kw["subsampling"] = 0  # 4:4:4 — keep colour edges crisp on the deliverable
            img.convert("RGB").save(path, format="JPEG", **kw)
        elif fmt == "webp":
            img.save(path, format="WEBP", quality=quality, method=4)
        else:
            img.save(path, format="PNG", pnginfo=meta, compress_level=4)


# ------------------------------------------------------------------------------------------------
class FolderIOLoadSequence:
    """Load a numbered frame sequence from an input/ subfolder as ONE batched IMAGE.

    The sibling loader above hands out a *list* — one graph pass per photo, which is what you
    want for per-photo work and exactly what you must not have for a video: a depth sequence
    has to arrive as a single [N,H,W,3] tensor, in frame order, to be encoded as one clip.

    16-bit PNGs (what Blender writes for a depth pass) are read at their real precision. Loading
    them through the 8-bit path would divide 0..65535 by 255 and clip every frame to white.
    """

    EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": (
                    folder_paths.get_input_subfolders(),
                    {
                        "tooltip": "Subfolder of input/ holding the frames, numbered in name order. "
                        "Use the 📁 Upload folder… button or drop a folder onto this node. R refreshes.",
                    },
                ),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 99999,
                                        "tooltip": "Skip the first N frames."}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 999,
                                       "tooltip": "0 = all. Wan takes 4n+1 frames (…49, 57, 61, 65, 81…), "
                                                  "so this is where you cut the sequence to length."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "frame_count")
    OUTPUT_TOOLTIPS = ("All frames as one batch, in name order.", "How many frames were loaded.")
    FUNCTION = "load"
    CATEGORY = "image/folder"
    DESCRIPTION = ("Loads a numbered frame sequence (a Blender depth or beauty render) from an "
                   "input/ subfolder as one batched IMAGE — ready for FAL Video — Wan VACE, or "
                   "for the core Create Video node.")

    @classmethod
    def _files(cls, folder, start_index, max_frames):
        d = FolderIOLoadImages._dir(folder)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"input/{folder} does not exist")
        rows = []
        with os.scandir(d) as it:
            for entry in it:
                if entry.name.startswith(".") or not entry.is_file():
                    continue
                if os.path.splitext(entry.name)[1].lower() in cls.EXTS:
                    st = entry.stat()
                    if st.st_size > 0:
                        rows.append((entry.name, st.st_mtime_ns, st.st_size))
        rows.sort(key=lambda r: _natural(r[0]))
        rows = rows[int(start_index or 0):]
        if max_frames:
            rows = rows[: int(max_frames)]
        return d, rows

    @classmethod
    def VALIDATE_INPUTS(cls, folder):
        try:
            d = FolderIOLoadImages._dir(folder)
        except ValueError as exc:
            return str(exc)
        if not os.path.isdir(d):
            return (f"input/{folder} not found — upload a folder first "
                    "(📁 button on the node, or drop a folder onto it)")
        return True

    @classmethod
    def IS_CHANGED(cls, folder, start_index, max_frames):
        try:
            _, rows = cls._files(folder, start_index, max_frames)
        except Exception:  # noqa: BLE001 — let load() raise the readable error
            return float("nan")
        h = hashlib.sha256()
        for name, mtime, size in rows:
            h.update(f"{name}\0{mtime}\0{size}\n".encode("utf-8", "surrogateescape"))
        return h.hexdigest()

    @staticmethod
    def _frame(path):
        """One file -> ([H,W,3] float32 0..1 array, a label for the bit depth)."""
        img = node_helpers.pillow(Image.open, path)
        mode = img.mode
        if mode in ("I", "I;16", "I;16B", "I;16L", "I;16N"):
            # 16-bit grey (Blender depth). PIL hands these over as uint16, or int32 for "I".
            arr = np.asarray(img).astype(np.float32) / 65535.0
            return np.repeat(arr[..., None], 3, axis=2), "16-bit"
        if mode == "F":
            arr = np.asarray(img, dtype=np.float32)
            lo, hi = float(arr.min()), float(arr.max())
            if hi > 1.001 or lo < -0.001:
                log.warning("[FolderIO] %s: float values %.3f..%.3f outside 0..1 — clipped. "
                            "Normalise the depth pass on export.", os.path.basename(path), lo, hi)
                arr = np.clip(arr, 0.0, 1.0)
            return np.repeat(arr[..., None], 3, axis=2), "float"
        arr = np.asarray(_to_srgb(img), dtype=np.float32) / 255.0
        return arr, "8-bit"

    def load(self, folder, start_index, max_frames):
        d, rows = self._files(folder, start_index, max_frames)
        if not rows:
            raise ValueError(f"input/{folder}: no frames found ({', '.join(self.EXTS)})")
        frames, depths, shape = [], set(), None
        for name, _, _ in rows:
            arr, depth = self._frame(os.path.join(d, name))
            if shape is None:
                shape = arr.shape
            elif arr.shape != shape:
                raise ValueError(
                    f"input/{folder}/{name} is {arr.shape[1]}x{arr.shape[0]}, but the sequence "
                    f"starts at {shape[1]}x{shape[0]} — every frame must be the same size")
            depths.add(depth)
            frames.append(torch.from_numpy(np.ascontiguousarray(arr)))
        images = torch.stack(frames, 0)
        h, w = shape[0], shape[1]
        log.info("[FolderIO] input/%s: %d frame(s) %dx%d, %s (~%.2f GB as float32)",
                 folder, len(frames), w, h, "/".join(sorted(depths)), images.numel() * 4 / 1e9)
        result = (images, len(frames))
        if len(depths) > 1:
            msg = (f"input/{folder} mixes {', '.join(sorted(depths))} frames — a depth sequence "
                   f"should be one format throughout, or the normalisation jumps mid-flight.")
            log.warning("[FolderIO] %s", msg)
            return {"ui": {"folderio_warning": [msg]}, "result": result}
        if h % 2 or w % 2:
            msg = (f"input/{folder} is {w}x{h} — h264 cannot encode odd dimensions, so this "
                   f"sequence will not go into a video. Re-render at an even size.")
            log.warning("[FolderIO] %s", msg)
            return {"ui": {"folderio_warning": [msg]}, "result": result}
        return result


NODE_CLASS_MAPPINGS = {
    "FolderIOLoadImages": FolderIOLoadImages,
    "FolderIOLoadSequence": FolderIOLoadSequence,
    "FolderIOSplitByShortSide": FolderIOSplitByShortSide,
    "FolderIOMergeSubset": FolderIOMergeSubset,
    "FolderIOSaveZip": FolderIOSaveZip,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FolderIOLoadImages": "📁 Load Images (upload folder)",
    "FolderIOLoadSequence": "🎞 Load Frame Sequence (one batch)",
    "FolderIOSplitByShortSide": "✂️ Split by Short Side",
    "FolderIOMergeSubset": "🔀 Merge Subset (by index)",
    "FolderIOSaveZip": "💾 Save Images + ZIP",
}
