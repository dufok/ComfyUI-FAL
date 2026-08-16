"""
Google Nano Banana / Gemini image nodes (category: FAL/Image/Banana).

FAL ships this family under fifteen endpoint ids, but they are only **four models** —
the "gemini-*" ids are byte-identical aliases of the "nano-banana-*" ids, same fields,
same defaults, same price. Building a node per id would ship eleven duplicates, so each
node here takes a `tier` and routes to the right endpoint:

  tier      endpoint (t2i / edit)                                            price
  ────────  ──────────────────────────────────────────────────────────────   ─────────────
  pro       fal-ai/nano-banana-pro            == gemini-3-pro-image-preview  $0.15/img, 4K x2
  banana-2  fal-ai/nano-banana-2              == gemini-3.1-flash-image-...  $0.08/img @1K
  lite      google/nano-banana-2-lite                                        token-billed
  legacy    fal-ai/nano-banana                == gemini-25-flash-image       $0.039/img

Tiers differ in what they *accept*, which is why the nodes gate per tier rather than
sending everything and hoping:

  resolution        pro [1K,2K,4K] · banana-2 [0.5K,1K,2K,4K] · lite/legacy: none (fixed)
  thinking_level    banana-2 and lite only — the key must be OMITTED to disable, not null
  system_prompt     everything except legacy
  enable_web_search pro and banana-2 only (+$0.015)
  4:1 / 8:1 ratios  banana-2 and lite only
  aspect_ratio      legacy text-to-image has no "auto" — it is a plain enum there

Nodes:
  * FalBananaGenerate -> text -> image, all four tiers
  * FalBananaEdit     -> image(s) -> image, all four tiers, multi-ref via batch + sockets
  * FalBananaContext  -> banana-2 edit with audio / video / PDF context (see below)

FalBananaContext is the one thing no other pack exposes: `fal-ai/nano-banana-2/edit`
accepts `audio_url`, `video_url` and `pdf_url` (15MB inline, and video_url takes a plain
YouTube link), so you can generate an image from a document or a video without downloading
anything locally.

All three return the model's own `description` alongside the image — that is where the
model explains what it did, and where `thinking_level` output surfaces.
"""
from .fal_common import (
    run_image_described,
    upload_image_frames,
)


# tier -> (text-to-image endpoint, edit endpoint)
TIERS = {
    "pro": ("fal-ai/nano-banana-pro", "fal-ai/nano-banana-pro/edit"),
    "banana-2": ("fal-ai/nano-banana-2", "fal-ai/nano-banana-2/edit"),
    "lite": ("google/nano-banana-2-lite", "google/nano-banana-2-lite/edit"),
    "legacy": ("fal-ai/nano-banana", "fal-ai/nano-banana/edit"),
}

# Same models under their Gemini branding — offered so a workflow can pin the id it wants.
ALIASES = {
    "pro": ("fal-ai/gemini-3-pro-image-preview", "fal-ai/gemini-3-pro-image-preview/edit"),
    "banana-2": ("fal-ai/gemini-3.1-flash-image-preview", "fal-ai/gemini-3.1-flash-image-preview/edit"),
    "legacy": ("fal-ai/gemini-25-flash-image", "fal-ai/gemini-25-flash-image/edit"),
}

RESOLUTIONS = {
    "pro": ["1K", "2K", "4K"],
    "banana-2": ["0.5K", "1K", "2K", "4K"],
    "lite": [],
    "legacy": [],
}

HAS_THINKING = ("banana-2", "lite")
HAS_SYSTEM_PROMPT = ("pro", "banana-2", "lite")
HAS_WEB_SEARCH = ("pro", "banana-2")
EXTREME_RATIOS = ("banana-2", "lite")

BASE_RATIOS = ["auto", "21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16"]
WIDE_RATIOS = BASE_RATIOS + ["4:1", "1:4", "8:1", "1:8"]

_RES_TOOLTIP = ("pro: 1K/2K/4K (4K costs double). banana-2: 0.5K x0.75, 1K, 2K x1.5, 4K x2. "
                "lite and legacy ignore this — they are fixed resolution.")
_THINK_TOOLTIP = ("banana-2 and lite only. 'off' omits the key entirely; 'high' makes the model "
                  "reason before drawing and report it on the description output.")


def _endpoint(tier, mode, use_gemini_ids):
    ids = ALIASES.get(tier) if use_gemini_ids and tier in ALIASES else TIERS[tier]
    return ids[0 if mode == "t2i" else 1]


def _common_args(tier, prompt, aspect_ratio, resolution, num_images, output_format,
                 thinking_level, system_prompt, enable_web_search, safety_tolerance, seed,
                 is_t2i):
    """Build the argument dict, sending only what this tier actually accepts."""
    if not prompt.strip():
        raise RuntimeError("prompt is required")
    args = {
        "prompt": prompt.strip(),
        "num_images": int(num_images),
        "output_format": output_format,
        "safety_tolerance": str(safety_tolerance),   # a STRING everywhere, never an int
    }

    # legacy text-to-image is the one endpoint whose aspect_ratio enum has no "auto" member,
    # so there we omit the key and let the API pick rather than sending a value it rejects.
    if not (is_t2i and tier == "legacy" and aspect_ratio == "auto"):
        if aspect_ratio in ("4:1", "1:4", "8:1", "1:8") and tier not in EXTREME_RATIOS:
            raise RuntimeError(
                f"aspect_ratio {aspect_ratio} exists only on the banana-2 and lite tiers")
        args["aspect_ratio"] = aspect_ratio

    if RESOLUTIONS[tier]:
        if resolution not in RESOLUTIONS[tier]:
            raise RuntimeError(
                f"tier '{tier}' supports resolution {RESOLUTIONS[tier]}, got '{resolution}'")
        args["resolution"] = resolution

    if thinking_level != "off":
        if tier not in HAS_THINKING:
            raise RuntimeError(f"thinking_level is only available on tiers {list(HAS_THINKING)}")
        args["thinking_level"] = thinking_level      # omitted entirely when off

    if system_prompt.strip() and tier in HAS_SYSTEM_PROMPT:
        args["system_prompt"] = system_prompt.strip()

    if enable_web_search:
        if tier not in HAS_WEB_SEARCH:
            raise RuntimeError(f"enable_web_search is only available on tiers {list(HAS_WEB_SEARCH)}")
        args["enable_web_search"] = True             # +$0.015

    if seed:
        args["seed"] = int(seed)
    return args


def _shared_optional():
    return {
        "resolution": (["0.5K", "1K", "2K", "4K"], {"default": "1K", "tooltip": _RES_TOOLTIP}),
        "aspect_ratio": (WIDE_RATIOS, {"default": "auto"}),
        "thinking_level": (["off", "minimal", "high"], {"default": "off", "tooltip": _THINK_TOOLTIP}),
        "system_prompt": ("STRING", {"default": "", "multiline": True,
                          "tooltip": "Persona / style steering. Not on the legacy tier. Gemini has no negative prompt — phrase exclusions here ('no text, no watermark')."}),
        "enable_web_search": ("BOOLEAN", {"default": False,
                              "tooltip": "pro and banana-2 only. Lets the model pull live info from the web. +$0.015."}),
        "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "4",
                             "tooltip": "1 strictest, 6 loosest."}),
        "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
        "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
        "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
        "use_gemini_ids": ("BOOLEAN", {"default": False,
                           "tooltip": "Call the identical Gemini-branded endpoint id instead (same model, same price). No effect on the lite tier."}),
    }


class FalBananaGenerate:
    """Nano Banana / Gemini text-to-image across all four tiers.

    pro $0.15 · banana-2 $0.08 · legacy $0.039 · lite token-billed (fixed 1K out)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "tier": (list(TIERS), {"default": "banana-2"}),
            },
            "optional": _shared_optional(),
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "description")
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Banana"

    def run(self, prompt, tier, resolution="1K", aspect_ratio="auto", thinking_level="off",
            system_prompt="", enable_web_search=False, safety_tolerance="4",
            output_format="png", num_images=1, seed=0, use_gemini_ids=False):
        args = _common_args(tier, prompt, aspect_ratio, resolution, num_images, output_format,
                            thinking_level, system_prompt, enable_web_search, safety_tolerance,
                            seed, is_t2i=True)
        return run_image_described(_endpoint(tier, "t2i", use_gemini_ids), args)


class FalBananaEdit:
    """Nano Banana / Gemini image editing across all four tiers.

    Every connected image becomes a reference and the prompt can address them by number
    ("replace the jacket in image 1 with the one from image 2"). A batched IMAGE counts as
    several references on its own.

    pro $0.15 · banana-2 $0.08 · legacy $0.039 · lite token-billed."""

    @classmethod
    def INPUT_TYPES(cls):
        opt = _shared_optional()
        opt.update({
            "image_2": ("IMAGE",),
            "image_3": ("IMAGE",),
            "image_4": ("IMAGE",),
        })
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "tier": (list(TIERS), {"default": "banana-2"}),
            },
            "optional": opt,
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "description")
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Banana"

    def run(self, image, prompt, tier, resolution="1K", aspect_ratio="auto", thinking_level="off",
            system_prompt="", enable_web_search=False, safety_tolerance="4",
            output_format="png", num_images=1, seed=0, use_gemini_ids=False,
            image_2=None, image_3=None, image_4=None):
        args = _common_args(tier, prompt, aspect_ratio, resolution, num_images, output_format,
                            thinking_level, system_prompt, enable_web_search, safety_tolerance,
                            seed, is_t2i=False)
        urls = upload_image_frames(image)
        for extra in (image_2, image_3, image_4):
            if extra is not None:
                urls.extend(upload_image_frames(extra))
        args["image_urls"] = urls
        return run_image_described(_endpoint(tier, "edit", use_gemini_ids), args)


class FalBananaContext:
    """fal-ai/nano-banana-2/edit with document, video and audio context.

    The banana-2 edit endpoint accepts far more than images: a PDF, a video (including a
    plain YouTube URL — nothing is downloaded locally), and an audio track, each up to 15MB
    inline. The model reads them and draws from them, so you can go straight from a spec
    sheet or a clip to an image.

    Images are optional here — this endpoint will happily run on document or video context
    alone. $0.08 per image at 1K (2K x1.5, 4K x2)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True,
                                      "tooltip": "What to make out of the attached context."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional — this endpoint can run on document/video context alone."}),
                "pdf_url": ("STRING", {"default": "", "tooltip": "URL of a PDF to read (15MB inline cap)."}),
                "video_url": ("STRING", {"default": "", "tooltip": "URL of a video — a plain YouTube link works."}),
                "audio_url": ("STRING", {"default": "", "tooltip": "URL of an audio file."}),
                "resolution": (["0.5K", "1K", "2K", "4K"], {"default": "1K"}),
                "aspect_ratio": (WIDE_RATIOS, {"default": "auto"}),
                "thinking_level": (["off", "minimal", "high"], {"default": "off", "tooltip": _THINK_TOOLTIP}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "enable_web_search": ("BOOLEAN", {"default": False, "tooltip": "+$0.015"}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "4"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
                "use_gemini_ids": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "description")
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Banana"

    def run(self, prompt, image=None, pdf_url="", video_url="", audio_url="",
            resolution="1K", aspect_ratio="auto", thinking_level="off", system_prompt="",
            enable_web_search=False, safety_tolerance="4", output_format="png",
            num_images=1, seed=0, use_gemini_ids=False):
        args = _common_args("banana-2", prompt, aspect_ratio, resolution, num_images,
                            output_format, thinking_level, system_prompt, enable_web_search,
                            safety_tolerance, seed, is_t2i=False)
        if image is not None:
            args["image_urls"] = upload_image_frames(image)
        attached = []
        for key, val in (("pdf_url", pdf_url), ("video_url", video_url), ("audio_url", audio_url)):
            if val and val.strip():
                args[key] = val.strip()
                attached.append(key)
        if image is None and not attached:
            raise RuntimeError(
                "connect an image or give at least one of pdf_url / video_url / audio_url — "
                "otherwise use the plain Banana Generate node")
        print(f"[FAL] banana context: {', '.join(attached) or 'images only'}")
        return run_image_described(_endpoint("banana-2", "edit", use_gemini_ids), args)


NODE_CLASS_MAPPINGS = {
    "FalBananaGenerate": FalBananaGenerate,
    "FalBananaEdit": FalBananaEdit,
    "FalBananaContext": FalBananaContext,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalBananaGenerate": "FAL Banana — Generate, 4 tiers ($0.039–0.30)",
    "FalBananaEdit": "FAL Banana — Edit, 4 tiers ($0.039–0.30)",
    "FalBananaContext": "FAL Banana — Context: PDF / video / audio ($0.08+)",
}
