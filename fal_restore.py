"""
Photo restoration — recover what was photographed (category: FAL/Image/Restore).

  * FalNafnetRestore -> fal-ai/nafnet/deblur | /denoise   ($0.0225/MP, non-generative)
  * FalDrctUpscale   -> fal-ai/drct-super-resolution      ($0.0045/MP, non-generative, fixed 4x)
  * FalFiboRestore   -> bria/fibo-edit/restore            ($0.04, one button, diffusion underneath)
  * FalDDColor       -> fal-ai/ddcolor                    ($0.001/MP, colourise black & white)

The distinction this whole shelf exists for: a **restoration** model recovers detail that is
still physically present in the file, a **generative** one invents plausible detail. For a
reference photo heading into a texture or a product shot you want the first — an invented leaf
vein is worse than a soft one, because it looks right and is wrong.

The catalogue makes that easy to check, and the check is worth repeating before adding anything
here: a genuine restoration endpoint takes one image and at most a scale or task selector.
Anything exposing `prompt`, `guidance_scale`, `num_inference_steps` or a `creativity` knob
redraws. Two endpoints on FAL are literally named "photo-restoration" and fail that test
(fal-ai/image-editing/photo-restoration carries guidance_scale 3.5 and 30 inference steps;
fal-ai/image-apps-v2/photo-restoration force-renders to 4K in one of five aspect ratios and
cannot return your frame) — they are deliberately NOT wrapped here.

Where this sits next to what the pack already has:
  * FAL Finish — Sharpen ($0.001) raises edge contrast. It cannot undo defocus or motion blur,
    but it is 20x cheaper, so try it first on a nearly-good photo.
  * FAL Upscale — Topaz / SeedVR / Clarity all enlarge, and the latter two invent detail while
    doing it. Deblur BEFORE upscaling: enlarging a soft photo just produces a large soft photo.
  * FAL Background — Bria Remove AFTER restoring, not before: cutting out a blurry edge gives a
    ragged alpha.

Size ceiling, measured rather than documented: FAL rejects an input with either side above
3981px (`image_too_large`). 3840x2160 passes, 4096x4096 does not — so "4K" is fine in the video
sense and one pixel too wide in the texture sense. It is not a megapixel or file-size limit; the
4096 square that failed was only a 7 MB PNG. The NAFNet node checks this before uploading.
"""
import json

import fal_client

from .fal_common import (
    images_from_result,
    require_key,
    run_image,
    upload_image,
)


# FAL's input validator rejects either side above this, with image_too_large. It is not a
# megapixel or file-size cap: 3840x2160 (8.3 MP) passes, 4096x4096 (16.8 MP, a 7 MB PNG)
# does not. Measured, because neither the OpenAPI schema nor the model page documents it.
MAX_DIMENSION = 3981


def _check_max_dimension(w, h):
    if max(w, h) <= MAX_DIMENSION:
        return
    scale = MAX_DIMENSION / max(w, h)
    raise RuntimeError(
        f"image is {w}x{h}; FAL rejects either side above {MAX_DIMENSION}px. "
        f"Resize to about {int(w * scale)}x{int(h * scale)} first — or work at half size and "
        f"enlarge afterwards with FAL Restore — DRCT 4x, which is both cheaper and sharper "
        f"than restoring at full resolution.")


class FalNafnetRestore:
    """fal-ai/nafnet/deblur | /denoise — NAFNet, a plain restoration CNN. $0.0225/MP.

    Not a diffusion model: no prompt, no reference, no sampling step. It regresses a clean
    image from the degraded one, which means it physically cannot invent a leaf vein that was
    not photographed. This is the node for anything heading into a bake.

    The trade is a ceiling: it recovers detail still present under the blur or grain, and
    nothing beyond it. Detail that is genuinely gone stays gone.

    One caveat worth knowing before you spend: the deblur weights are trained on motion blur.
    If the photo is soft because the camera moved it helps a lot; if focus was simply missed,
    it will do much less, and nothing in the API can tell the two apart in advance.

    Billed per megapixel, unlike the flat-priced nodes around it — a 24 MP camera plate is
    about $0.54 a pass. Crop first."""

    ENDPOINTS = {"deblur": "fal-ai/nafnet/deblur", "denoise": "fal-ai/nafnet/denoise"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (list(cls.ENDPOINTS), {"default": "deblur",
                         "tooltip": "deblur for camera shake; denoise for high-ISO grain. Same network, two sets of weights. On a noisy AND soft photo run denoise first — deblur amplifies grain."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Restore"

    def run(self, image, mode):
        # The schema carries a `seed`, but the network is a deterministic regression — a seed
        # cannot change its output. It is template boilerplate from a relighting endpoint (the
        # same template also stamped "URL of image to be used for relighting" onto image_url
        # here). Not exposed, and not sent.
        h, w = int(image.shape[1]), int(image.shape[2])
        _check_max_dimension(w, h)
        out = run_image(self.ENDPOINTS[mode], {"image_url": upload_image(image)})
        oh, ow = int(out.shape[1]), int(out.shape[2])
        if (oh, ow) != (h, w):
            print(f"[FAL] note: NAFNet returned {ow}x{oh} for a {w}x{h} input — "
                  "resample it back before using it as a texture plate")
        return (out,)


class FalDrctUpscale:
    """fal-ai/drct-super-resolution — 4x upscale with a regression transformer, $0.0045/MP.

    The honest partner to NAFNet: no sampler anywhere in the loop, so it multiplies the pixels
    you have instead of re-imagining them. That makes it the right enlarger for a texture,
    where SeedVR and Clarity would invent structure.

    Fixed at 4x — the API declares the factor as a constant, so there is no dial to expose."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Restore"

    def run(self, image):
        # upscale_factor is declared const:4 — it is both the default and the only legal value,
        # so sending it can only ever be redundant or wrong.
        return (run_image("fal-ai/drct-super-resolution", {"image_url": upload_image(image)}),)


class FalFiboRestore:
    """bria/fibo-edit/restore — Bria FIBO's restoration preset: denoise, deblur and damage
    cleanup in one call, $0.04 flat. No knobs at all.

    Its entire input schema is a single image URL, so nothing in the API is capable of cropping
    or resampling your frame — which is what makes it safe ahead of a bake, unlike the two
    endpoints elsewhere on FAL that share the "photo restoration" name.

    It is still diffusion underneath, with the sampler fixed server-side. It cannot be steered
    into inventing, but heavy damage comes back as a plausible reconstruction rather than a
    measurement. Use it when the photo is damaged rather than merely soft, and compare the
    result against the source before it goes anywhere permanent. For a hard fidelity guarantee
    use NAFNet, which is a CNN and cannot hallucinate at all.

    `structured_instruction` is the VGL description of what restore did — feed it to
    bria/fibo-edit/edit as original_vgl to keep editing from where this left off."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "structured_instruction")
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Restore"

    def run(self, image):
        require_key()
        args = {"image_url": upload_image(image)}
        endpoint = "bria/fibo-edit/restore"
        print(f"[FAL] {endpoint} <- {args}")
        # A dedicated runner because run_image would drop structured_instruction. The response
        # carries BOTH `image` and an empty `images: []`; images_from_result falls through the
        # empty list to the singular key, so the parsing is unchanged.
        result = fal_client.subscribe(endpoint, arguments=args, with_logs=False)
        instruction = result.get("structured_instruction") if isinstance(result, dict) else None
        return (images_from_result(result),
                json.dumps(instruction, ensure_ascii=False, indent=2) if instruction else "")


class FalDDColor:
    """fal-ai/ddcolor — colourise a black & white photograph, $0.001/MP.

    The cheapest node in the pack. Structure is preserved exactly — it only predicts chroma —
    but the colours themselves are an educated guess, not a recovery: nothing in a greyscale
    file records that the leaf was green rather than red. Treat the output as a plausible
    colourisation, and correct it downstream if the real colour matters."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Restore"

    def run(self, image):
        return (run_image("fal-ai/ddcolor", {"image_url": upload_image(image)}),)


NODE_CLASS_MAPPINGS = {
    "FalNafnetRestore": FalNafnetRestore,
    "FalDrctUpscale": FalDrctUpscale,
    "FalFiboRestore": FalFiboRestore,
    "FalDDColor": FalDDColor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalNafnetRestore": "FAL Restore — NAFNet deblur / denoise, non-generative ($0.0225/MP)",
    "FalDrctUpscale": "FAL Restore — DRCT 4x, non-generative ($0.0045/MP)",
    "FalFiboRestore": "FAL Restore — Bria FIBO, one button ($0.04)",
    "FalDDColor": "FAL Restore — DDColor, colourise B&W ($0.001/MP)",
}
