"""
PATINA — PBR material nodes (category: FAL/Image/Material).

Turn a photo, a render or a text prompt into a set of physically-based texture maps:
basecolor, normal, roughness, metalness, height.

  * FalPatinaMaps     -> fal-ai/patina                  (image -> 5 PBR maps, $0.01 + $0.01/MP/map)
  * FalPatinaMaterial -> fal-ai/patina/material         (prompt -> seamless tiling material + maps,
                                                         $0.01 + $0.02/MP + $0.01/MP/map)
  * FalPatinaExtract  -> fal-ai/patina/material/extract (photo + prompt -> seamless tiling texture
                                                         lifted out of the photo + maps,
                                                         $0.10 + $0.02/MP + $0.01/MP/map)

Two things drive the design here:

1. **Maps are self-labelling, order is not contractual.** Every PBR entry in the returned
   `images` list carries a required `map_type` field (basecolor|normal|roughness|metalness|
   height). The official examples for `patina` and for `material` list the same five maps in
   *different* orders, so indexing by position is a silent-corruption bug waiting to happen.
   We key by `map_type` and never by index. On material/extract the *texture* is simply the
   one entry with no `map_type` at all.

2. **The maps and the texture can differ in size**, because `upscale_factor` upscales only the
   PBR maps — "the base texture image is not upscaled". A single batched IMAGE output would
   therefore throw on torch.cat, so every map gets its own named output.

Billing is per requested map, so switching maps off is a real saving — hence five booleans
rather than one all-or-nothing toggle. Deselecting every map is legal on material/extract and
means "texture only, skip the PBR pass".

Note on colour: basecolor is albedo, but normal/roughness/metalness/height are DATA. They are
decoded straight to tensor with no sRGB/gamma transform (the pack's url_to_image_tensor already
does exactly that) — applying one would quietly corrupt roughness and displacement.
"""
import torch

from .fal_common import (
    require_key,
    upload_image,
    upload_mask,
    url_to_image_tensor,
)

import fal_client


MAP_TYPES = ("basecolor", "normal", "roughness", "metalness", "height")

# Neutral stand-ins for maps that were not requested — returning None would crash any
# downstream node, and a flat mid-grey/flat-normal is the conventional "no effect" value.
_PLACEHOLDER_COLOR = {
    "basecolor": (0.5, 0.5, 0.5),
    "normal": (0.5, 0.5, 1.0),   # flat tangent-space normal
    "roughness": (0.5, 0.5, 0.5),
    "metalness": (0.0, 0.0, 0.0),
    "height": (0.5, 0.5, 0.5),
}


def _placeholder(map_type):
    rgb = _PLACEHOLDER_COLOR.get(map_type, (0.5, 0.5, 0.5))
    t = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    for c, v in enumerate(rgb):
        t[..., c] = v
    return t


def _maps_arg(basecolor, normal, roughness, metalness, height):
    wanted = []
    for name, on in zip(MAP_TYPES, (basecolor, normal, roughness, metalness, height)):
        if on:
            wanted.append(name)
    return wanted


def _split_images(result):
    """-> (texture_entries, {map_type: entry})

    The discriminator is the presence of the `map_type` key, nothing else: entries that
    have it are PBR maps, the entry that lacks it is the generated/extracted texture.
    """
    images = result.get("images") if isinstance(result, dict) else None
    if not isinstance(images, list):
        return [], {}
    textures = [e for e in images if isinstance(e, dict) and "map_type" not in e]
    by_type = {e["map_type"]: e for e in images
               if isinstance(e, dict) and isinstance(e.get("map_type"), str)}
    return textures, by_type


def _maps_tuple(by_type, wanted):
    out = []
    for m in MAP_TYPES:
        entry = by_type.get(m)
        url = entry.get("url") if isinstance(entry, dict) else None
        if url:
            out.append(url_to_image_tensor(url))
        else:
            if m in wanted:
                print(f"[FAL] warning: map '{m}' was requested but not returned")
            out.append(_placeholder(m))
    return tuple(out)


def _report(endpoint, by_type, textures):
    got = ", ".join(sorted(by_type)) or "none"
    print(f"[FAL] DONE {endpoint} -> {len(textures)} texture(s), maps: {got}")


# ============================================================================ image -> maps

class FalPatinaMaps:
    """fal-ai/patina — read PBR material properties straight off a photo or render.

    No prompt, no resizing: map resolution follows the input image. This is the one to use
    when you already have the texture and just need it decomposed into shader inputs.
    $0.01 + $0.01 per megapixel per map (1024x1024, all five maps = $0.06)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "basecolor": ("BOOLEAN", {"default": True, "tooltip": "Albedo without lighting. Billed per map."}),
                "normal": ("BOOLEAN", {"default": True, "tooltip": "Per-pixel surface orientation. Billed per map."}),
                "roughness": ("BOOLEAN", {"default": True, "tooltip": "Reflection sharpness. Billed per map."}),
                "metalness": ("BOOLEAN", {"default": True, "tooltip": "Metal vs dielectric. Billed per map."}),
                "height": ("BOOLEAN", {"default": True, "tooltip": "Displacement / parallax. Billed per map."}),
            },
            "optional": {
                "output_format": (["png", "webp", "jpeg"], {"default": "png",
                                  "tooltip": "Keep png — jpeg artefacts are visibly destructive in normal and height maps."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("basecolor", "normal", "roughness", "metalness", "height", "seed")
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Material"

    def run(self, image, basecolor, normal, roughness, metalness, height,
            output_format="png", seed=0):
        wanted = _maps_arg(basecolor, normal, roughness, metalness, height)
        if not wanted:
            raise RuntimeError("select at least one map — this endpoint only produces PBR maps")
        require_key()
        args = {
            "image_url": upload_image(image),
            "maps": wanted,
            "output_format": output_format,
        }
        if seed:
            args["seed"] = int(seed)
        print(f"[FAL] fal-ai/patina <- maps={wanted}")
        result = fal_client.subscribe("fal-ai/patina", arguments=args, with_logs=False)
        textures, by_type = _split_images(result)
        _report("fal-ai/patina", by_type, textures)
        return _maps_tuple(by_type, wanted) + (int(result.get("seed") or 0),)


# ============================================================================ prompt -> material

_TILING = (["both", "horizontal", "vertical"], {"default": "both",
           "tooltip": "Which axes wrap seamlessly. 'both' for surfaces, single-axis for trim sheets."})
_SIZES = ["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9", "custom"]


def _image_size_arg(image_size, custom_width, custom_height):
    if image_size != "custom":
        return image_size
    return {"width": int(custom_width), "height": int(custom_height)}


def _estimate(width, height, n_maps, base, upscale_factor):
    """Rough cost preview printed before the call — the megapixel terms are what actually
    bite here (image_size accepts up to 14142x14142)."""
    mp = (width * height) / 1_000_000.0
    cost = base + mp * 0.02 + mp * 0.01 * n_maps
    if upscale_factor == 2:
        cost += mp * 0.004 * n_maps
    elif upscale_factor == 4:
        cost += mp * 0.016 * n_maps
    return cost


_SIZE_PX = {
    "square_hd": (1024, 1024), "square": (512, 512),
    "portrait_4_3": (768, 1024), "portrait_16_9": (576, 1024),
    "landscape_4_3": (1024, 768), "landscape_16_9": (1024, 576),
}


class _PatinaTextureBase:
    """Shared body for material (prompt -> texture) and extract (photo -> texture)."""

    ENDPOINT = ""
    BASE_FEE = 0.01

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "INT")
    RETURN_NAMES = ("texture", "basecolor", "normal", "roughness", "metalness", "height",
                    "prompt", "seed")
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Material"

    def _run(self, args, image_size, custom_width, custom_height, wanted, upscale_factor):
        require_key()
        if image_size == "custom":
            w, h = int(custom_width), int(custom_height)
        else:
            w, h = _SIZE_PX.get(image_size, (1024, 1024))
        est = _estimate(w, h, len(wanted), self.BASE_FEE, upscale_factor)
        print(f"[FAL] {self.ENDPOINT} <- {w}x{h}, maps={wanted or 'none (texture only)'}, "
              f"upscale={upscale_factor}x  ~${est:.3f}")
        result = fal_client.subscribe(self.ENDPOINT, arguments=args, with_logs=False)
        textures, by_type = _split_images(result)
        _report(self.ENDPOINT, by_type, textures)
        texture = url_to_image_tensor(textures[0]["url"]) if textures else _placeholder("basecolor")
        return ((texture,) + _maps_tuple(by_type, wanted)
                + (result.get("prompt") or "", int(result.get("seed") or 0)))


class FalPatinaMaterial(_PatinaTextureBase):
    """fal-ai/patina/material — describe a material, get a seamlessly tiling texture plus its
    full PBR set. Optionally seed it from an image (image-to-image) or repaint part of one
    (connect a mask: white = regenerated, black = preserved).

    $0.01 + $0.02/MP + $0.01/MP per map — 1024x1024 with all five maps = $0.08."""

    ENDPOINT = "fal-ai/patina/material"
    BASE_FEE = 0.01

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "mossy stone wall", "multiline": True,
                                      "tooltip": "The material to generate."}),
                "image_size": (_SIZES, {"default": "square_hd"}),
                "tiling_mode": _TILING,
                "basecolor": ("BOOLEAN", {"default": True}),
                "normal": ("BOOLEAN", {"default": True}),
                "roughness": ("BOOLEAN", {"default": True}),
                "metalness": ("BOOLEAN", {"default": True}),
                "height": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "custom_width": ("INT", {"default": 1024, "min": 64, "max": 14142, "step": 64,
                                         "tooltip": "Used when image_size is 'custom'. Cost scales with megapixels."}),
                "custom_height": ("INT", {"default": 1024, "min": 64, "max": 14142, "step": 64}),
                "upscale_factor": ([0, 2, 4], {"default": 0,
                                   "tooltip": "SeedVR seamless upscale of the PBR maps only — the base texture is NOT upscaled, so texture and maps come back at different sizes."}),
                "image": ("IMAGE", {"tooltip": "Optional starting image (image-to-image)."}),
                "mask": ("MASK", {"tooltip": "Inpaint the image — white = regenerated, black = preserved. Requires image."}),
                "strength": ("FLOAT", {"default": 0.6, "min": 0.01, "max": 1.0, "step": 0.01,
                                       "tooltip": "How far to move from the input image. Only used when image is connected."}),
                "num_inference_steps": ("INT", {"default": 8, "min": 1, "max": 8,
                                                "tooltip": "Capped at 8 — z-image turbo is distilled, higher is not better."}),
                "tile_size": ("INT", {"default": 128, "min": 32, "max": 256,
                                      "tooltip": "Latent tile size (128 = 1024px)."}),
                "tile_stride": ("INT", {"default": 64, "min": 16, "max": 128,
                                        "tooltip": "Step between tiles. Must be <= tile_size; smaller = more overlap = better hidden seams."}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": True,
                                            "tooltip": "An LLM enriches the prompt. The expanded text comes back on the `prompt` output — feed it back in to reproduce a result, seed alone is not enough."}),
                "output_format": (["png", "webp", "jpeg"], {"default": "png"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
            },
        }

    def run(self, prompt, image_size, tiling_mode, basecolor, normal, roughness, metalness, height,
            custom_width=1024, custom_height=1024, upscale_factor=0, image=None, mask=None,
            strength=0.6, num_inference_steps=8, tile_size=128, tile_stride=64,
            enable_prompt_expansion=True, output_format="png", seed=0):
        if not prompt.strip():
            raise RuntimeError("prompt is required — describe the material")
        if mask is not None and image is None:
            raise RuntimeError("mask needs an image to inpaint — connect one, or drop the mask")
        if int(tile_stride) > int(tile_size):
            raise RuntimeError(f"tile_stride ({tile_stride}) must be <= tile_size ({tile_size})")
        wanted = _maps_arg(basecolor, normal, roughness, metalness, height)
        args = {
            "prompt": prompt.strip(),
            "image_size": _image_size_arg(image_size, custom_width, custom_height),
            "tiling_mode": tiling_mode,
            "maps": wanted,
            "num_inference_steps": int(num_inference_steps),
            "tile_size": int(tile_size),
            "tile_stride": int(tile_stride),
            "enable_prompt_expansion": bool(enable_prompt_expansion),
            "upscale_factor": int(upscale_factor),
            "output_format": output_format,
            "num_images": 1,
        }
        if image is not None:
            args["image_url"] = upload_image(image)
            args["strength"] = float(strength)
            if mask is not None:
                args["mask_url"] = upload_mask(mask)
        if seed:
            args["seed"] = int(seed)
        return self._run(args, image_size, custom_width, custom_height, wanted, int(upscale_factor))


class FalPatinaExtract(_PatinaTextureBase):
    """fal-ai/patina/material/extract — point at a material inside a photo ("the wall",
    "the fabric of the chair") and lift it out as a seamlessly tiling texture with PBR maps.

    `strength` is the fidelity dial and is always active here: low stays faithful to the
    photograph, high drifts toward the prompt.

    $0.10 + $0.02/MP + $0.01/MP per map — 1024x1024 with all five maps = $0.17. The $0.10
    base fee makes prompt iteration on this endpoint roughly 10x pricier than on Material."""

    ENDPOINT = "fal-ai/patina/material/extract"
    BASE_FEE = 0.10

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "the wall", "multiline": True,
                                      "tooltip": "Which material in the photo to extract — this selects, it does not invent."}),
                "strength": ("FLOAT", {"default": 0.6, "min": 0.01, "max": 1.0, "step": 0.01,
                                       "tooltip": "Low = faithful to the photo, high = drifts toward the prompt."}),
                "image_size": (_SIZES, {"default": "square_hd"}),
                "tiling_mode": _TILING,
                "basecolor": ("BOOLEAN", {"default": True}),
                "normal": ("BOOLEAN", {"default": True}),
                "roughness": ("BOOLEAN", {"default": True}),
                "metalness": ("BOOLEAN", {"default": True}),
                "height": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "custom_width": ("INT", {"default": 1024, "min": 64, "max": 14142, "step": 64}),
                "custom_height": ("INT", {"default": 1024, "min": 64, "max": 14142, "step": 64}),
                "upscale_factor": ([0, 2, 4], {"default": 0,
                                   "tooltip": "Upscales the PBR maps only — the extracted texture is NOT upscaled."}),
                "num_inference_steps": ("INT", {"default": 8, "min": 1, "max": 8}),
                "tile_size": ("INT", {"default": 128, "min": 32, "max": 256}),
                "tile_stride": ("INT", {"default": 64, "min": 16, "max": 128}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": True}),
                "output_format": (["png", "webp", "jpeg"], {"default": "png"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
            },
        }

    def run(self, image, prompt, strength, image_size, tiling_mode,
            basecolor, normal, roughness, metalness, height,
            custom_width=1024, custom_height=1024, upscale_factor=0, num_inference_steps=8,
            tile_size=128, tile_stride=64, enable_prompt_expansion=True,
            output_format="png", seed=0):
        if not prompt.strip():
            raise RuntimeError("prompt is required — say which material to extract, e.g. 'the wall'")
        if int(tile_stride) > int(tile_size):
            raise RuntimeError(f"tile_stride ({tile_stride}) must be <= tile_size ({tile_size})")
        wanted = _maps_arg(basecolor, normal, roughness, metalness, height)
        args = {
            "image_url": upload_image(image),
            "prompt": prompt.strip(),
            "strength": float(strength),
            "image_size": _image_size_arg(image_size, custom_width, custom_height),
            "tiling_mode": tiling_mode,
            "maps": wanted,
            "num_inference_steps": int(num_inference_steps),
            "tile_size": int(tile_size),
            "tile_stride": int(tile_stride),
            "enable_prompt_expansion": bool(enable_prompt_expansion),
            "upscale_factor": int(upscale_factor),
            "output_format": output_format,
            "num_images": 1,
        }
        if seed:
            args["seed"] = int(seed)
        return self._run(args, image_size, custom_width, custom_height, wanted, int(upscale_factor))


# ============================================================================ registry

NODE_CLASS_MAPPINGS = {
    "FalPatinaMaps": FalPatinaMaps,
    "FalPatinaMaterial": FalPatinaMaterial,
    "FalPatinaExtract": FalPatinaExtract,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalPatinaMaps": "FAL Material — PATINA image→PBR maps ($0.06 @1K/5 maps)",
    "FalPatinaMaterial": "FAL Material — PATINA prompt→tiling material ($0.08 @1K/5 maps)",
    "FalPatinaExtract": "FAL Material — PATINA extract from photo ($0.17 @1K/5 maps)",
}
