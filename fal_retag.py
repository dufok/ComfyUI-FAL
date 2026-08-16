"""Retag another pack's node categories once every pack has loaded.

The container also ships gokayfem's ComfyUI-fal-API, baked into the docker image (root-owned,
not a bind mount), so we cannot edit its files — they would come back on the next rebuild.
But its categories are wrong in ways that make the FAL menu hard to use: three video nodes
sit under FAL/Image, and its four Nano Banana nodes are scattered among 30 unrelated ones.

ComfyUI reads `cls.CATEGORY` off the class every time /object_info is served, so mutating the
attribute late is enough — no fork, no rebuild. The catch is ordering: custom_nodes are walked
with an unsorted os.listdir, so gokayfem's classes may not exist when this module is imported.
We therefore defer to aiohttp's on_startup, which fires after init_extra_nodes() and before the
first request.

Two failure modes make the paranoia here load-bearing rather than decorative:
  * an exception escaping install() makes ComfyUI drop OUR ENTIRE PACK (load_custom_node
    swallows the traceback and returns False), and
  * an exception escaping the on_startup handler stops the server booting at all
    (AppRunner.setup propagates it).
Hence `except BaseException` in both, and sys.modules lookups instead of `import server`,
which outside ComfyUI's exact import order drags in torch and can hard-fail.
"""

import logging
import sys

log = logging.getLogger(__name__)

OWNER_MODULE = "custom_nodes.ComfyUI-fal-API"

# node id -> (category we expect to find, category we want)
# The expected value is a guard: if gokayfem fixes these upstream, or renames something, we
# leave it alone rather than fighting over it.
RETAG = {
    # video work filed under an image category
    "Bria_Video_Increase_Resolution_fal": ("FAL/Image", "FAL/VideoGeneration/Upscale"),
    "Seedvr_Upscale_Video_fal":           ("FAL/Image", "FAL/VideoGeneration/Upscale"),
    "Topaz_Upscale_Video_fal":            ("FAL/Image", "FAL/VideoGeneration/Upscale"),
    "VideoUpscaler_fal":                  ("FAL/VideoGeneration", "FAL/VideoGeneration/Upscale"),
    # helpers that escaped into ComfyUI's stock "video" menu
    "UploadVideo_fal":                    ("video", "FAL/Utils"),
    "UploadFile_fal":                     ("video", "FAL/Utils"),
    "LoadVideoURL":                       ("video", "FAL/Utils"),
    # the Banana family, gathered next to ours
    "NanoBanana2_fal":                    ("FAL/Image", "FAL/Image/Banana"),
    "NanoBananaPro_fal":                  ("FAL/Image", "FAL/Image/Banana"),
    "NanoBananaEdit_fal":                 ("FAL/Image", "FAL/Image/Banana"),
    "NanoBananaTextToImage_fal":          ("FAL/Image", "FAL/Image/Banana"),
}


def apply_retag(mappings):
    changed, skipped = [], []
    for name, (expected, new_cat) in RETAG.items():
        try:
            cls = mappings.get(name)
            if cls is None:
                skipped.append((name, "absent"))
                continue
            owner = getattr(cls, "RELATIVE_PYTHON_MODULE", None)
            if owner != OWNER_MODULE:
                skipped.append((name, f"owned by {owner!r}, not ours to touch"))
                continue
            # V3-schema nodes expose CATEGORY as a @final classproperty and /object_info
            # short-circuits to GET_NODE_INFO_V1(), so the write would succeed and be
            # silently ignored. Skip with a reason instead of pretending it worked.
            if hasattr(cls, "GET_NODE_INFO_V1"):
                skipped.append((name, "V3 schema node, CATEGORY write is a no-op"))
                continue
            cur = getattr(cls, "CATEGORY", None)
            if cur != expected:
                skipped.append((name, f"category is {cur!r}, expected {expected!r}"))
                continue
            cls.CATEGORY = new_cat
            changed.append((name, cur, new_cat))
        except BaseException as e:  # noqa: BLE001 — must never escape
            skipped.append((name, f"error: {e!r}"))
    return changed, skipped


def install():
    """Register the retag to run once, after every pack has loaded. Never raises."""
    try:
        srv = sys.modules.get("server")      # deliberately NOT `import server`
        nodes_mod = sys.modules.get("nodes")
        if srv is None or nodes_mod is None:
            log.debug("[ComfyUI-FAL] retag: not running under ComfyUI, skipped")
            return False
        app = getattr(getattr(srv, "PromptServer", None), "instance", None)
        app = getattr(app, "app", None)
        if app is None or not hasattr(app, "on_startup"):
            log.debug("[ComfyUI-FAL] retag: PromptServer.instance.app unavailable, skipped")
            return False

        async def _retag(_app):
            try:
                changed, skipped = apply_retag(nodes_mod.NODE_CLASS_MAPPINGS)
                if changed:
                    log.info("[ComfyUI-FAL] retagged %d foreign node(s)", len(changed))
                for n, old, new in changed:
                    log.info("[ComfyUI-FAL]   %s: %s -> %s", n, old, new)
                for n, why in skipped:
                    log.debug("[ComfyUI-FAL] retag skip %s (%s)", n, why)
            except BaseException:  # noqa: BLE001 — raising here stops the server booting
                log.warning("[ComfyUI-FAL] retag failed", exc_info=True)

        app.on_startup.append(_retag)        # RuntimeError if the app is already frozen
        return True
    except BaseException as e:  # noqa: BLE001
        log.debug("[ComfyUI-FAL] retag hook not installed: %r", e)
        return False
