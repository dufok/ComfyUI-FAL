"""Retag another pack's node categories once every pack has loaded.

The container also ships gokayfem's ComfyUI-fal-API, baked into the docker image (root-owned,
not a bind mount), so we cannot edit its files — they would come back on the next rebuild. Its
87 nodes are useful (they own video, LoRA training and most text-to-image) but they sit directly
in `FAL/Image` and `FAL/VideoGeneration`, interleaved with ours, and three helpers escape into
ComfyUI's stock `video` menu. That makes the FAL tree hard to read.

So we move the whole pack under a single `FAL/zz-gokayfem/` root: it sorts to the bottom, it
groups, and — unlike a name such as "legacy" — it does not claim the nodes are deprecated. They
are not: that pack is an active complement, and it is the only source of video in this install.

Why this is safe: ComfyUI resolves a saved graph by the NODE_CLASS_MAPPINGS key (`class_type`),
and re-reads `cls.CATEGORY` off the class on every /object_info request. Category is presentation;
class_type is the contract. So retagging moves menu entries without touching a single saved
workflow — including the graphs the Comfyder Blender add-ons POST to /prompt with pack-B
class names hardcoded.

Ordering: custom_nodes are walked with an unsorted os.listdir, so gokayfem's classes may not
exist yet when this module is imported. We defer to aiohttp's on_startup, which fires after
init_extra_nodes() and before the first request.

Two failure modes make the paranoia here load-bearing rather than decorative:
  * an exception escaping install() makes ComfyUI drop OUR ENTIRE PACK (load_custom_node
    swallows the traceback and returns False), and
  * an exception escaping the on_startup handler stops the server booting at all
    (AppRunner.setup propagates it).
Hence `except BaseException` in both, and sys.modules lookups instead of `import server`,
which outside ComfyUI's exact import order drags in torch and can hard-fail.

To opt out entirely, delete the two `fal_retag` lines from __init__.py.
"""

import logging
import sys

log = logging.getLogger(__name__)

OWNER_MODULE = "custom_nodes.ComfyUI-fal-API"
ROOT = "FAL/zz-gokayfem"

# Source category -> where it goes. Keyed on the category the class declares in gokayfem's own
# source, which is what we see at hook time. Rule-based rather than a list of 87 node ids, so a
# node added upstream lands somewhere sensible instead of being silently left behind.
CATEGORY_MAP = {
    "FAL/Image": f"{ROOT}/Image",
    "FAL/VideoGeneration": f"{ROOT}/Video",
    "FAL/VideoGeneration/DY": f"{ROOT}/Video",
    "FAL/Training": f"{ROOT}/Training",
    "FAL/LLM": f"{ROOT}/Text",
    "FAL/VLM": f"{ROOT}/Text",
    "video": f"{ROOT}/Utils",   # upload/download helpers that escaped into ComfyUI's own menu
}

# Per-node exceptions, applied before the category map.
NODE_OVERRIDES = {
    # Video upscalers gokayfem files under an image category.
    "Bria_Video_Increase_Resolution_fal": f"{ROOT}/Video Upscale",
    "Seedvr_Upscale_Video_fal": f"{ROOT}/Video Upscale",
    "Topaz_Upscale_Video_fal": f"{ROOT}/Video Upscale",
    "VideoUpscaler_fal": f"{ROOT}/Video Upscale",
    # The Nano Banana family. Ours supersede these on every axis (tier routing, seed,
    # system_prompt, thinking_level, safety_tolerance, web search, and the model's own
    # description as a second output), so they belong at the bottom rather than sitting
    # in FAL/Image/Banana next to ours, where they would read as equal alternatives.
    "NanoBanana2_fal": f"{ROOT}/Banana",
    "NanoBananaPro_fal": f"{ROOT}/Banana",
    "NanoBananaEdit_fal": f"{ROOT}/Banana",
    "NanoBananaTextToImage_fal": f"{ROOT}/Banana",
}


def _target(name, current):
    """Where this node should end up, or None to leave it alone."""
    if isinstance(current, str) and current.startswith(ROOT):
        return None                      # already ours to begin with — idempotent
    if name in NODE_OVERRIDES:
        return NODE_OVERRIDES[name]
    return CATEGORY_MAP.get(current)


def apply_retag(mappings):
    changed, skipped = [], []
    for name, cls in list(mappings.items()):
        try:
            # Only ever touch nodes that prove they belong to that pack.
            if getattr(cls, "RELATIVE_PYTHON_MODULE", None) != OWNER_MODULE:
                continue
            # V3-schema nodes expose CATEGORY as a @final classproperty and /object_info
            # short-circuits to GET_NODE_INFO_V1(), so the write would succeed and be silently
            # ignored. Skip with a reason instead of pretending it worked.
            if hasattr(cls, "GET_NODE_INFO_V1"):
                skipped.append((name, "V3 schema node, CATEGORY write is a no-op"))
                continue
            current = getattr(cls, "CATEGORY", None)
            new_cat = _target(name, current)
            if new_cat is None:
                skipped.append((name, f"no rule for category {current!r}"))
                continue
            cls.CATEGORY = new_cat
            changed.append((name, current, new_cat))
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
                buckets = {}
                for _, _, new in changed:
                    buckets[new] = buckets.get(new, 0) + 1
                if changed:
                    log.info("[ComfyUI-FAL] retagged %d node(s) of %s into %s/",
                             len(changed), OWNER_MODULE, ROOT)
                    for cat in sorted(buckets):
                        log.info("[ComfyUI-FAL]   %-32s %d", cat, buckets[cat])
                for n, why in skipped:
                    log.debug("[ComfyUI-FAL] retag skip %s (%s)", n, why)
            except BaseException:  # noqa: BLE001 — raising here stops the server booting
                log.warning("[ComfyUI-FAL] retag failed", exc_info=True)

        app.on_startup.append(_retag)        # RuntimeError if the app is already frozen
        return True
    except BaseException as e:  # noqa: BLE001
        log.debug("[ComfyUI-FAL] retag hook not installed: %r", e)
        return False
