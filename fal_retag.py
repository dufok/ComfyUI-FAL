"""Retag another pack's node categories once every pack has loaded.

gokayfem's ComfyUI-fal-API is installed next to this pack. Since its v2 it is two things at once:
an auto-generated catalog of every live FAL model (~1,500 nodes under `FAL/Models/<category>`,
a hand-picked `FAL/Featured`, plus `FAL/Platform` and `FAL/Utils`), and the ~90 hand-written
"curated" nodes it has always had. The catalog files itself sensibly and is left where it is.
The curated nodes are the problem: they sit directly in `FAL/Image` and `FAL/VideoGeneration`,
interleaved with ours, so the FAL tree is hard to read.

So two groups move to the bottom of the tree, each under its own root:
  * `FAL/zz-curated/`  the hand-written nodes. Not "legacy": upstream still adds to them, they
                       are just the older, narrower way to reach models the catalog now covers.
  * `FAL/zz-removed/`  upstream's `FAL/Compatibility`: endpoints FAL no longer lists, kept only
                       so saved graphs still load. Nothing to pick from when building a new one.
With the v1 pack (curated nodes only, no catalog) the first rule is all that applies.

We cannot edit that pack's files instead: in the docker image it is root-owned, and wherever it
is a git clone an edit would block the next pull.

Why this is safe: ComfyUI resolves a saved graph by the NODE_CLASS_MAPPINGS key (`class_type`),
and re-reads `cls.CATEGORY` off the class on every /object_info request. Category is presentation;
class_type is the contract. So retagging moves menu entries without touching a single saved
workflow.

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
ROOT = "FAL/zz-curated"
REMOVED = "FAL/zz-removed"
ROOTS = (ROOT, REMOVED)

# Source category -> where it goes. Keyed on the category the class declares in gokayfem's own
# source, which is what we see at hook time. Rule-based rather than a list of node ids, so a
# node added upstream lands somewhere sensible instead of being silently left behind.
CATEGORY_MAP = {
    "FAL/Image": f"{ROOT}/Image",
    "FAL/VideoGeneration": f"{ROOT}/Video",
    "FAL/VideoGeneration/DY": f"{ROOT}/Video",
    "FAL/VideoUpscaling": f"{ROOT}/Video Upscale",
    "FAL/Training": f"{ROOT}/Training",
    "FAL/LLM": f"{ROOT}/Text",
    "FAL/VLM": f"{ROOT}/Text",
    # Upload/download helpers. v1 let them escape into ComfyUI's own `video` menu; v2 files them
    # in `FAL/Video`, which is where our video nodes live. Its own utilities shelf fits better.
    "video": "FAL/Utils/Video",
    "FAL/Video": "FAL/Utils/Video",
}

# Whole subtrees, sub-category kept: FAL/Compatibility/image-to-video -> FAL/zz-removed/image-to-video
PREFIX_MAP = {
    "FAL/Compatibility": REMOVED,
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
    # description as a second output), so they get their own shelf rather than hiding
    # among the other image nodes.
    "NanoBanana2_fal": f"{ROOT}/Banana",
    "NanoBananaPro_fal": f"{ROOT}/Banana",
    "NanoBananaEdit_fal": f"{ROOT}/Banana",
    "NanoBananaTextToImage_fal": f"{ROOT}/Banana",
}


def _target(name, current):
    """Where this node should end up, or None to leave it alone."""
    if not isinstance(current, str) or current.startswith(ROOTS):
        return None                      # already moved — idempotent
    if name in NODE_OVERRIDES:
        return NODE_OVERRIDES[name]
    if current in CATEGORY_MAP:
        return CATEGORY_MAP[current]
    for prefix, dest in PREFIX_MAP.items():
        if current == prefix or current.startswith(prefix + "/"):
            return dest + current[len(prefix):]
    return None


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
                mappings = nodes_mod.NODE_CLASS_MAPPINGS
                changed, skipped = apply_retag(mappings)
                buckets = {}
                for _, _, new in changed:
                    buckets[new] = buckets.get(new, 0) + 1
                if changed:
                    log.info("[ComfyUI-FAL] retagged %d node(s) of %s", len(changed), OWNER_MODULE)
                    for cat in sorted(buckets):
                        log.info("[ComfyUI-FAL]   %-32s %d", cat, buckets[cat])
                elif not any(getattr(c, "RELATIVE_PYTHON_MODULE", None) == OWNER_MODULE
                             for c in mappings.values()):
                    # Not an error on its own — the pack may simply not be installed. But when
                    # it IS expected, this is the only early warning that it failed to import:
                    # ComfyUI logs that as a WARNING and carries on, so its nodes vanish
                    # silently and anything driving them by class_type breaks at request time.
                    log.warning("[ComfyUI-FAL] %s registered NO nodes — if it is installed, it "
                                "failed to import; check the startup log above for its traceback",
                                OWNER_MODULE)
                for n, why in skipped:
                    log.debug("[ComfyUI-FAL] retag skip %s (%s)", n, why)
            except BaseException:  # noqa: BLE001 — raising here stops the server booting
                log.warning("[ComfyUI-FAL] retag failed", exc_info=True)

        app.on_startup.append(_retag)        # RuntimeError if the app is already frozen
        return True
    except BaseException as e:  # noqa: BLE001
        log.debug("[ComfyUI-FAL] retag hook not installed: %r", e)
        return False
