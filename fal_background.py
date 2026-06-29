"""
FAL background nodes via Bria (category: FAL/Background).

  * FalBriaBackgroundRemove  -> fal-ai/bria/background/remove   (RMBG 2.0, ~$0.018)
  * FalBriaBackgroundReplace -> fal-ai/bria/background/replace  (generative bg from a prompt or ref)

Remove returns the cut-out as IMAGE + MASK (alpha = subject). Replace returns IMAGE(s).
"""
import fal_client

from .fal_common import (
    require_key,
    upload_image,
    deep_find,
    image_and_mask_from_url,
    images_from_result,
    blank_image,
)


class FalBriaBackgroundRemove:
    """fal-ai/bria/background/remove — Bria RMBG 2.0 background removal."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "run"
    CATEGORY = "FAL/Background"

    def run(self, image):
        require_key()
        url = upload_image(image)
        print("[FAL] fal-ai/bria/background/remove")
        result = fal_client.subscribe(
            "fal-ai/bria/background/remove",
            arguments={"image_url": url},
            with_logs=False,
        )
        out = deep_find(result, "image")
        img_url = out.get("url") if isinstance(out, dict) else (out if isinstance(out, str) else None)
        if not img_url:
            img_url = deep_find(result, "url")
        if not img_url:
            raise RuntimeError(f"no image url in Bria response: {result}")
        img, mask = image_and_mask_from_url(img_url)
        return (img, mask)


class FalBriaBackgroundReplace:
    """fal-ai/bria/background/replace — replace the background from a text prompt or reference image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bg_prompt": ("STRING", {"default": "", "multiline": True,
                                         "tooltip": "Describe the new background. Ignored if a ref_image is connected."}),
            },
            "optional": {
                "ref_image": ("IMAGE", {"tooltip": "Optional reference image for the new background."}),
                "num_results": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "FAL/Background"

    def run(self, image, bg_prompt, ref_image=None, num_results=1, seed=0):
        require_key()
        args = {"image_url": upload_image(image), "num_results": int(num_results)}
        if ref_image is not None:
            args["ref_image_url"] = upload_image(ref_image)
        elif bg_prompt.strip():
            args["bg_prompt"] = bg_prompt.strip()
        else:
            raise RuntimeError("provide either a bg_prompt or connect a ref_image")
        if seed:
            args["seed"] = int(seed)
        print(f"[FAL] fal-ai/bria/background/replace <- {args}")
        result = fal_client.subscribe(
            "fal-ai/bria/background/replace",
            arguments=args,
            with_logs=False,
        )
        return (images_from_result(result),)


NODE_CLASS_MAPPINGS = {
    "FalBriaBackgroundRemove": FalBriaBackgroundRemove,
    "FalBriaBackgroundReplace": FalBriaBackgroundReplace,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FalBriaBackgroundRemove": "FAL Background — Bria Remove (RMBG 2.0)",
    "FalBriaBackgroundReplace": "FAL Background — Bria Replace",
}
