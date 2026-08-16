"""
FAL image-editing nodes (category: FAL/Image/<shelf>).

One bar for everyday photo work — the newest FAL models per task, cheapest-first:

  Remove   * FalObjectRemoval      -> fal-ai/object-removal[/mask]     ($0.006-0.024, prompt or mask)
           * FalBriaEraser         -> fal-ai/bria/eraser               ($0.04, mask)
           * FalFluxProErase       -> fal-ai/flux-pro/v1/erase         (~$0.03/MP, mask, BFL)
           * FalFinegrainEraser    -> fal-ai/finegrain-eraser          ($0.18-0.36, prompt, kills shadows/reflections)
  Inpaint  * FalZImageTurboInpaint -> fal-ai/z-image/turbo/inpaint     ($0.01/MP, mask+prompt)
           * FalQwenImageEditInpaint -> fal-ai/qwen-image-edit/inpaint (mask+prompt)
           * FalBriaGenFill        -> bria/genfill/v2                  ($0.04/MP, generate object in mask)
  Edit     * FalQwenImageEdit2511  -> fal-ai/qwen-image-edit-2511      ($0.03/MP, prompt, multi-ref)
           * FalSeedreamEdit       -> bytedance/seedream v4.5 | v5-lite ($0.04/img, up to 10 refs)
           * FalGeminiFlashEdit    -> gemini 3.1-flash-preview | 2.5-flash ($0.04-0.08, Google, multi-ref)
  Upscale  * FalSeedVRUpscale      -> fal-ai/seedvr/upscale/image      (SeedVR2)
           * FalTopazUpscale       -> fal-ai/topaz/upscale/image       (photo standard)
           * FalRecraftCrispUpscale-> fal-ai/recraft/upscale/crisp     (cheap utility)
           * FalClarityUpscaler    -> fal-ai/clarity-upscaler          (creative detail)
  Expand   * FalBriaExpand         -> fal-ai/bria/expand               (outpaint to a bigger canvas)

Masks follow ComfyUI convention: MASK 1.0 = area to remove/inpaint (uploaded as white).
"""
import fal_client

from .fal_common import (
    run_image,
    upload_image,
    upload_image_frames,
    upload_mask,
    require_key,
    deep_find,
    save_file,
)


def _seed_arg(args, seed):
    if seed:
        args["seed"] = int(seed)
    return args


# ============================================================================ Remove

class FalObjectRemoval:
    """fal-ai/object-removal — describe the object in the prompt, or connect a MASK
    (switches to the /mask endpoint). Cheapest remover: $0.006-0.024 by quality."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True,
                                      "tooltip": "What to remove, e.g. 'the trash can on the left'. Ignored if a mask is connected."}),
                "quality": (["low_quality", "medium_quality", "high_quality", "best_quality"],
                            {"default": "best_quality"}),
                "mask_expansion": ("INT", {"default": 15, "min": 0, "max": 100}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional. If connected, the mask defines the removal area and the prompt is ignored."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Remove"

    def run(self, image, prompt, quality, mask_expansion, mask=None):
        args = {
            "image_url": upload_image(image),
            "model": quality,
            "mask_expansion": int(mask_expansion),
        }
        if mask is not None:
            args["mask_url"] = upload_mask(mask)
            return (run_image("fal-ai/object-removal/mask", args),)
        if not prompt.strip():
            raise RuntimeError("describe what to remove in the prompt, or connect a mask")
        args["prompt"] = prompt.strip()
        return (run_image("fal-ai/object-removal", args),)


class FalBriaEraser:
    """fal-ai/bria/eraser — precise mask-based removal, commercially licensed data. $0.04.
    grow_mask dilates the painted mask before upload: erasers need margin around the
    object (edge pixels, color bounce, shadow) or the border color bleeds back in."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "grow_mask": ("INT", {"default": 15, "min": 0, "max": 100,
                                      "tooltip": "Dilate the mask by N px — cover edge pixels and color bounce."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Remove"

    def run(self, image, mask, grow_mask=15):
        from .fal_common import grow_mask as _grow
        args = {"image_url": upload_image(image), "mask_url": upload_mask(_grow(mask, grow_mask))}
        return (run_image("fal-ai/bria/eraser", args),)


class FalFluxProErase:
    """fal-ai/flux-pro/v1/erase — BFL's dedicated eraser (newer than Fill for removal). ~$0.03/MP."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "dilate_pixels": ("INT", {"default": 10, "min": 0, "max": 100}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Remove"

    def run(self, image, mask, dilate_pixels):
        args = {
            "image_url": upload_image(image),
            "mask_url": upload_mask(mask),
            "dilate_pixels": int(dilate_pixels),
            "output_format": "png",
        }
        return (run_image("fal-ai/flux-pro/v1/erase", args),)


class FalFinegrainEraser:
    """fal-ai/finegrain-eraser — prompt-based removal that also erases the object's
    shadows and reflections. Premium option: $0.18 express / $0.27 standard / $0.36 premium."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True,
                                      "tooltip": "What to remove; shadows and reflections go with it."}),
                "mode": (["express", "standard", "premium"], {"default": "standard"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Remove"

    def run(self, image, prompt, mode, seed=0):
        if not prompt.strip():
            raise RuntimeError("describe what to remove in the prompt")
        args = {"image_url": upload_image(image), "prompt": prompt.strip(), "mode": mode}
        return (run_image("fal-ai/finegrain-eraser", _seed_arg(args, seed)),)


# ============================================================================ Inpaint

class FalZImageTurboInpaint:
    """fal-ai/z-image/turbo/inpaint — fast 6B inpaint, $0.01/MP (5x cheaper than Flux Pro Fill).
    Good default for draft iterations."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "num_inference_steps": ("INT", {"default": 8, "min": 1, "max": 50}),
                "acceleration": (["none", "regular", "high"], {"default": "regular"}),
            },
            "optional": {
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Inpaint"

    def run(self, image, mask, prompt, strength, num_inference_steps, acceleration,
            num_images=1, seed=0):
        args = {
            "image_url": upload_image(image),
            "mask_image_url": upload_mask(mask),
            "prompt": prompt.strip(),
            "strength": float(strength),
            "num_inference_steps": int(num_inference_steps),
            "acceleration": acceleration,
            "num_images": int(num_images),
            "output_format": "png",
        }
        return (run_image("fal-ai/z-image/turbo/inpaint", _seed_arg(args, seed)),)


class FalQwenImageEditInpaint:
    """fal-ai/qwen-image-edit/inpaint — Qwen Image Edit v1 (the original Aug-2025 model)
    constrained to a mask. The newer 2511 has no mask endpoint on FAL."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "strength": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Inpaint"

    def run(self, image, mask, prompt, strength, guidance_scale, num_inference_steps,
            negative_prompt="", num_images=1, seed=0):
        args = {
            "image_url": upload_image(image),
            "mask_url": upload_mask(mask),
            "prompt": prompt.strip(),
            "strength": float(strength),
            "guidance_scale": float(guidance_scale),
            "num_inference_steps": int(num_inference_steps),
            "num_images": int(num_images),
            "output_format": "png",
        }
        if negative_prompt.strip():
            args["negative_prompt"] = negative_prompt.strip()
        return (run_image("fal-ai/qwen-image-edit/inpaint", _seed_arg(args, seed)),)


class FalBriaGenFill:
    """bria/genfill/v2 — generate a new object inside the mask from an instruction. $0.04/MP."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "instruction": ("STRING", {"default": "", "multiline": True,
                                           "tooltip": "What to generate inside the masked region."}),
                "steps_num": ("INT", {"default": 30, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 5555, "min": 0, "max": 2_147_483_647}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Inpaint"

    def run(self, image, mask, instruction, steps_num, seed):
        if not instruction.strip():
            raise RuntimeError("describe what to generate inside the mask")
        args = {
            "image_url": upload_image(image),
            "mask_url": upload_mask(mask),
            "instruction": instruction.strip(),
            "steps_num": int(steps_num),
            "seed": int(seed),
        }
        return (run_image("bria/genfill/v2", args),)


class FalFluxProFill:
    """fal-ai/flux-pro/v1/fill — BFL's FLUX.1 Fill [pro], the quality bar for masked
    inpainting. $0.05/MP, billed rounded up per megapixel.

    Fill has no server-side dilate (unlike /erase), so grow_mask is the only way to give
    the model margin — but it defaults to 0 here: an inpaint mask is a composition the
    user painted, and silently growing it moves the object.

    Set finetune_id to route to fal-ai/flux-pro/v1/fill-finetuned instead ($0.06/MP)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "prompt": ("STRING", {"default": "", "multiline": True,
                           "tooltip": "What to paint inside the mask. For pure removal the cheaper FAL Remove — Flux Pro v1 Erase is the better node."}),
                "grow_mask": ("INT", {"default": 0, "min": 0, "max": 100,
                              "tooltip": "Dilate the mask by N px before upload. 4–12 px hides the seam when the painted edge hugs the object."}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2",
                                     "tooltip": "BFL moderation: 1 strictest, 6 most permissive."}),
            },
            "optional": {
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4,
                               "tooltip": "API hard cap is 4."}),
                "enhance_prompt": ("BOOLEAN", {"default": False,
                                   "tooltip": "Server-side LLM rewrite of your prompt. Off by default: it makes the same prompt non-reproducible."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
                "finetune_id": ("STRING", {"default": "",
                                "tooltip": "Your BFL finetune id. Non-empty switches to fal-ai/flux-pro/v1/fill-finetuned ($0.06/MP)."}),
                "finetune_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                       "tooltip": "Only used with finetune_id. Raise if the concept is not coming through."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Inpaint"

    def run(self, image, mask, prompt, grow_mask, safety_tolerance,
            num_images=1, enhance_prompt=False, seed=0, finetune_id="", finetune_strength=1.0):
        from .fal_common import grow_mask as _grow
        # Fill requires the mask to match the image exactly — cheaper to catch here than to
        # pay for the 422.
        ih, iw = int(image.shape[1]), int(image.shape[2])
        mh, mw = int(mask.shape[-2]), int(mask.shape[-1])
        if (mh, mw) != (ih, iw):
            raise RuntimeError(
                f"mask is {mw}x{mh} but the image is {iw}x{ih} — flux-pro/v1/fill needs them to "
                "match exactly; mask the same image you feed in, or resize the mask first")
        args = {
            "image_url": upload_image(image),
            "mask_url": upload_mask(_grow(mask, grow_mask)),
            "prompt": prompt.strip(),
            "num_images": int(num_images),
            "output_format": "png",     # deliberate non-default: lossless for compositing back
        }
        if safety_tolerance != "2":
            args["safety_tolerance"] = safety_tolerance
        if enhance_prompt:
            args["enhance_prompt"] = True
        endpoint = "fal-ai/flux-pro/v1/fill"
        if finetune_id.strip():
            endpoint = "fal-ai/flux-pro/v1/fill-finetuned"
            args["finetune_id"] = finetune_id.strip()
            args["finetune_strength"] = float(finetune_strength)
        return (run_image(endpoint, _seed_arg(args, seed)),)


# ============================================================================ Edit

class FalFluxKontextEdit:
    """FLUX.1 Kontext (Black Forest Labs) — instruction editing that leaves the rest of the
    frame intact. Every connected image becomes a reference and the prompt can address them
    by number ("put the logo from image 2 on the mug in image 1"); a batched IMAGE counts as
    several references on its own.

    Routing is automatic: one reference uses the single-image endpoint, two or more use
    /multi. Both cost the same, so there is nothing to choose.
    pro $0.04/image · max $0.08/image — per image, so num_images=4 on max is $0.32."""

    ENDPOINTS = {
        ("pro", False): "fal-ai/flux-pro/kontext",
        ("pro", True): "fal-ai/flux-pro/kontext/multi",
        ("max", False): "fal-ai/flux-pro/kontext/max",
        ("max", True): "fal-ai/flux-pro/kontext/max/multi",
    }
    RATIOS = ["auto", "21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Reference 1. A batched IMAGE counts as several references and routes to /multi."}),
                "prompt": ("STRING", {"default": "", "multiline": True,
                           "tooltip": "An instruction, not a caption — 'change the jacket to red, keep everything else'."}),
                "tier": (["pro", "max"], {"default": "pro",
                         "tooltip": "pro $0.04/image. max $0.08/image — better prompt adherence and typography, same inputs."}),
            },
            "optional": {
                "image_2": ("IMAGE", {"tooltip": "Extra references. Any second image routes the call to the /multi endpoint — same price."}),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "aspect_ratio": (cls.RATIOS, {"default": "auto",
                                 "tooltip": "'auto' omits the key so the result keeps the input's shape."}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.1,
                                   "tooltip": "How literally the instruction is followed. Above ~5 it starts damaging the parts you did not ask it to change."}),
                "enhance_prompt": ("BOOLEAN", {"default": False,
                                   "tooltip": "Let BFL rewrite your prompt server-side. Helps short prompts, overrides precise ones."}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "2"}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Edit"

    def run(self, image, prompt, tier, image_2=None, image_3=None, image_4=None,
            aspect_ratio="auto", guidance_scale=3.5, enhance_prompt=False, num_images=1,
            safety_tolerance="2", output_format="png", seed=0):
        if not prompt.strip():
            raise RuntimeError("prompt is required — Kontext is instruction editing, "
                               "there is no unconditional mode")
        urls = upload_image_frames(image)
        for extra in (image_2, image_3, image_4):
            if extra is not None:
                urls.extend(upload_image_frames(extra))
        if not urls:
            raise RuntimeError("no reference image — connect a LoadImage (or any IMAGE) output")

        multi = len(urls) > 1
        args = {"prompt": prompt.strip(), "output_format": output_format}
        args["image_urls" if multi else "image_url"] = urls if multi else urls[0]
        if aspect_ratio != "auto":
            # nullable with no default on the edit routes: omitting it keeps the input shape
            args["aspect_ratio"] = aspect_ratio
        if abs(float(guidance_scale) - 3.5) > 1e-6:
            args["guidance_scale"] = float(guidance_scale)
        if int(num_images) != 1:
            args["num_images"] = int(num_images)
        if str(safety_tolerance) != "2":
            args["safety_tolerance"] = str(safety_tolerance)   # a STRING, never an int
        if enhance_prompt:
            args["enhance_prompt"] = True
        return (run_image(self.ENDPOINTS[(tier, multi)], _seed_arg(args, seed)),)


class FalQwenImageEdit2511:
    """fal-ai/qwen-image-edit-2511 — newest Qwen edit (Nov 2025), $0.03/MP.
    A batched IMAGE input and the optional sockets all become references."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "guidance_scale": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "acceleration": (["none", "regular", "high"], {"default": "regular"}),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Edit"

    def run(self, image, prompt, guidance_scale, num_inference_steps, acceleration,
            image_2=None, image_3=None, negative_prompt="", num_images=1, seed=0):
        urls = upload_image_frames(image)
        for extra in (image_2, image_3):
            if extra is not None:
                urls.extend(upload_image_frames(extra))
        args = {
            "image_urls": urls,
            "prompt": prompt.strip(),
            "guidance_scale": float(guidance_scale),
            "num_inference_steps": int(num_inference_steps),
            "acceleration": acceleration,
            "num_images": int(num_images),
            "output_format": "png",
        }
        if negative_prompt.strip():
            args["negative_prompt"] = negative_prompt.strip()
        return (run_image("fal-ai/qwen-image-edit-2511", _seed_arg(args, seed)),)


class FalSeedreamEdit:
    """Bytedance Seedream edit — natural-language multi-ref editing ('replace the
    product in image 1 with the one from image 2'), up to 10 refs.
    v5-pro: region-precise, layer separation, sketch completion ($0.0675/1.5K, $0.135/2K).
    v5-lite / v4.5: ~$0.04. Seed input works on v4.5 only."""

    ENDPOINTS = {
        "v5-pro": "bytedance/seedream/v5/pro/edit",
        "v5-lite": "bytedance/seedream/v5/lite/edit",
        "v4.5": "fal-ai/bytedance/seedream/v4.5/edit",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "version": (list(cls.ENDPOINTS), {"default": "v5-pro"}),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647,
                                 "tooltip": "v4.5 only; v5-lite has no seed input."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Edit"

    def run(self, image, prompt, version, image_2=None, image_3=None, image_4=None,
            num_images=1, seed=0):
        urls = upload_image_frames(image)
        for extra in (image_2, image_3, image_4):
            if extra is not None:
                urls.extend(upload_image_frames(extra))
        args = {
            "image_urls": urls,
            "prompt": prompt.strip(),
            "num_images": int(num_images),
        }
        if version == "v4.5":
            _seed_arg(args, seed)
        return (run_image(self.ENDPOINTS[version], args),)


class FalGeminiFlashEdit:
    """Google Gemini Flash Image edit — 3.1 preview (newest, $0.08/1K, up to 4K) or
    2.5 ($0.039). Multi-ref like Nano Banana (same family), prompt references images
    by number. Nano Banana Pro = Gemini 3 Pro, this is its cheaper/fresher Flash tier."""

    ENDPOINTS = {
        "3.1-flash-preview": "fal-ai/gemini-3.1-flash-image-preview/edit",
        "2.5-flash": "fal-ai/gemini-25-flash-image/edit",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "version": (list(cls.ENDPOINTS), {"default": "3.1-flash-preview"}),
                "resolution": (["0.5K", "1K", "2K", "4K"],
                               {"default": "1K", "tooltip": "3.1 only; 2.5 ignores it."}),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "system_prompt": ("STRING", {"default": "", "multiline": True,
                                             "tooltip": "3.1 only. Gemini has NO negative prompt — phrase exclusions here or in the prompt ('no text, no watermark')."}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Banana"

    def run(self, image, prompt, version, resolution, image_2=None, image_3=None,
            system_prompt="", num_images=1, seed=0):
        if not prompt.strip():
            raise RuntimeError("prompt is required")
        urls = upload_image_frames(image)
        for extra in (image_2, image_3):
            if extra is not None:
                urls.extend(upload_image_frames(extra))
        args = {
            "image_urls": urls,
            "prompt": prompt.strip(),
            "num_images": int(num_images),
            "output_format": "png",
        }
        if version == "3.1-flash-preview":
            args["resolution"] = resolution
            if system_prompt.strip():
                args["system_prompt"] = system_prompt.strip()
        return (run_image(self.ENDPOINTS[version], _seed_arg(args, seed)),)


# ============================================================================ Upscale

class FalSeedVRUpscale:
    """fal-ai/seedvr/upscale/image — SeedVR2, strong generative photo upscaler.

    `seamless` switches to the /seamless variant, the only endpoint on FAL that upscales a
    tiling texture and keeps the tile edges matching afterwards. It is still generative — it
    invents detail on the way up — so for a reference photo that must stay faithful use
    FAL Restore — DRCT instead."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_mode": (["factor", "target"], {"default": "factor"}),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.5}),
                "target_resolution": (["720p", "1080p", "1440p", "2160p"], {"default": "1080p"}),
                "noise_scale": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "seamless": ("BOOLEAN", {"default": False,
                             "tooltip": "Keep a tiling texture tiling after the upscale ($0.0025/MP)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Upscale"

    def run(self, image, upscale_mode, upscale_factor, target_resolution, noise_scale,
            seamless=False, seed=0):
        args = {
            "image_url": upload_image(image),
            "upscale_mode": upscale_mode,
            "noise_scale": float(noise_scale),
            # "png" is the one format valid on BOTH variants — the plain endpoint's enum has
            # "jpg" where the seamless one has "jpeg", so never promote this to a widget.
            "output_format": "png",
        }
        if upscale_mode == "factor":
            args["upscale_factor"] = float(upscale_factor)
        else:
            args["target_resolution"] = target_resolution
        endpoint = ("fal-ai/seedvr/upscale/image/seamless" if seamless
                    else "fal-ai/seedvr/upscale/image")
        return (run_image(endpoint, _seed_arg(args, seed)),)


class FalTopazUpscale:
    """fal-ai/topaz/upscale/image — Topaz, the photo-restoration standard (faces, denoise)."""

    MODELS = ["Low Resolution V2", "Standard V2", "CGI", "High Fidelity V2", "Text Refine",
              "Recovery", "Redefine", "Recovery V2", "Standard MAX", "Wonder", "Wonder 3"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (cls.MODELS, {"default": "Standard V2"}),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.5}),
                "face_enhancement": ("BOOLEAN", {"default": True}),
                "subject_detection": (["All", "Foreground", "Background"], {"default": "All"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Upscale"

    def run(self, image, model, upscale_factor, face_enhancement, subject_detection):
        args = {
            "image_url": upload_image(image),
            "model": model,
            "upscale_factor": float(upscale_factor),
            "face_enhancement": bool(face_enhancement),
            "subject_detection": subject_detection,
            "output_format": "png",
        }
        return (run_image("fal-ai/topaz/upscale/image", args),)


class FalRecraftCrispUpscale:
    """fal-ai/recraft/upscale/crisp — cheap, fast, non-generative sharpening upscale."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Upscale"

    def run(self, image):
        return (run_image("fal-ai/recraft/upscale/crisp", {"image_url": upload_image(image)}),)


class FalClarityUpscaler:
    """fal-ai/clarity-upscaler — creative upscaler; `creativity` re-imagines detail,
    `resemblance` pins it to the source."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5}),
                "creativity": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "resemblance": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "prompt": ("STRING", {"default": "masterpiece, best quality, highres", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Upscale"

    def run(self, image, upscale_factor, creativity, resemblance,
            prompt="masterpiece, best quality, highres", seed=0):
        args = {
            "image_url": upload_image(image),
            "upscale_factor": float(upscale_factor),
            "creativity": float(creativity),
            "resemblance": float(resemblance),
            "prompt": prompt.strip() or "masterpiece, best quality, highres",
        }
        return (run_image("fal-ai/clarity-upscaler", _seed_arg(args, seed)),)


# ============================================================================ Expand

class FalBriaExpand:
    """fal-ai/bria/expand — outpaint onto a larger canvas. The source keeps its size;
    offset 0/0 centers it (default), otherwise it is placed at (x, y) on the new canvas."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "canvas_width": ("INT", {"default": 1920, "min": 64, "max": 5000}),
                "canvas_height": ("INT", {"default": 1080, "min": 64, "max": 5000}),
            },
            "optional": {
                "prompt": ("STRING", {"default": "", "multiline": True,
                                      "tooltip": "Optional description of what appears in the expanded area."}),
                "offset_x": ("INT", {"default": 0, "min": 0, "max": 5000}),
                "offset_y": ("INT", {"default": 0, "min": 0, "max": 5000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Expand"

    def run(self, image, canvas_width, canvas_height, prompt="", offset_x=0, offset_y=0, seed=0):
        h, w = int(image.shape[1]), int(image.shape[2])
        cw, ch = int(canvas_width), int(canvas_height)
        if w > cw or h > ch:
            raise RuntimeError(f"canvas {cw}x{ch} is smaller than the source image {w}x{h}")
        x = int(offset_x) if offset_x else (cw - w) // 2
        y = int(offset_y) if offset_y else (ch - h) // 2
        args = {
            "image_url": upload_image(image),
            "canvas_size": [cw, ch],
            "original_image_size": [w, h],
            "original_image_location": [x, y],
        }
        if prompt.strip():
            args["prompt"] = prompt.strip()
        return (run_image("fal-ai/bria/expand", _seed_arg(args, seed)),)


# ============================================================================ Finish
# fal-ai/post-processing/* — $0.001/image finishing touches. Typical series-unification
# chain: ColorMatch (KJNodes, local, free) -> FalGrain (same style over every frame).
# These nodes are batch-aware: a batch in = one $0.001 call per frame, batch out.

import torch


def _run_finish(endpoint, image, extra):
    outs = [run_image(endpoint, {"image_url": url, **extra}) for url in upload_image_frames(image)]
    if len(outs) == 1:
        return (outs[0],)
    try:
        return (torch.cat(outs, 0),)
    except Exception:
        return (outs[0],)

class FalGrain:
    """fal-ai/post-processing/grain — film grain, 6 stocks. $0.001."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "grain_style": (["modern", "analog", "kodak", "fuji", "cinematic", "newspaper"],
                                {"default": "cinematic"}),
                "grain_intensity": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "grain_scale": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 50.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Finish"

    def run(self, image, grain_style, grain_intensity, grain_scale):
        extra = {
            "grain_style": grain_style,
            "grain_intensity": float(grain_intensity),
            "grain_scale": float(grain_scale),
        }
        return _run_finish("fal-ai/post-processing/grain", image, extra)


class FalVignette:
    """fal-ai/post-processing/vignette — corner darkening. $0.001."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "vignette_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Finish"

    def run(self, image, vignette_strength):
        return _run_finish("fal-ai/post-processing/vignette", image,
                           {"vignette_strength": float(vignette_strength)})


class FalColorCorrection:
    """fal-ai/post-processing/color-correction — global grade: temperature, contrast,
    saturation, brightness, gamma. $0.001. A poor man's LUT for locking a series look."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "temperature": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
                "contrast": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
                "saturation": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Finish"

    def run(self, image, temperature, contrast, saturation, brightness, gamma):
        extra = {
            "temperature": float(temperature),
            "contrast": float(contrast),
            "saturation": float(saturation),
            "brightness": float(brightness),
            "gamma": float(gamma),
        }
        return _run_finish("fal-ai/post-processing/color-correction", image, extra)


class FalSharpen:
    """fal-ai/post-processing/sharpen — basic / smart / CAS sharpening. $0.001."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sharpen_mode": (["basic", "smart", "cas"], {"default": "cas"}),
                "sharpen_radius": ("INT", {"default": 1, "min": 1, "max": 10}),
                "sharpen_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "cas_amount": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Finish"

    def run(self, image, sharpen_mode, sharpen_radius, sharpen_alpha, cas_amount):
        extra = {
            "sharpen_mode": sharpen_mode,
            "sharpen_radius": int(sharpen_radius),
            "sharpen_alpha": float(sharpen_alpha),
            "cas_amount": float(cas_amount),
        }
        return _run_finish("fal-ai/post-processing/sharpen", image, extra)


# ============================================================================ Vector

def _run_svg(endpoint, args, prefix):
    """Call an image->SVG endpoint, save the .svg into output/, return (file, url, info)."""
    require_key()
    print(f"[FAL] {endpoint}")
    result = fal_client.subscribe(endpoint, arguments=args, with_logs=False)
    node = deep_find(result, "image") or deep_find(result, "images")
    if isinstance(node, list):
        node = node[0] if node else None
    url = node.get("url") if isinstance(node, dict) else (node if isinstance(node, str) else None)
    if not url:
        url = deep_find(result, "url")
    if not url:
        raise RuntimeError(f"no svg url in FAL response: {result}")
    fname, download_url, size_mb = save_file(url, prefix)
    info = f"{endpoint} -> {fname} ({size_mb:.2f} MB)  ⬇ {download_url}"
    print(f"[FAL] DONE {info}")
    return (fname, download_url, info)


class FalRecraftVectorize:
    """fal-ai/recraft/vectorize — AI raster->SVG, zero knobs, $0.01. Best for logos,
    illustrations, flat graphics. The .svg lands in output/ (download link in info)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("svg_file", "download_url", "info")
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Vector"
    OUTPUT_NODE = True

    def run(self, image):
        return _run_svg("fal-ai/recraft/vectorize",
                        {"image_url": upload_image(image)}, "recraft_vec")


class FalImage2SVG:
    """fal-ai/image2svg — vtracer-style tracer with full control, $0.005. Good for
    patterns and mechanical tracing; color_precision/filter_speckle are the main dials."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "colormode": (["color", "binary"], {"default": "color"}),
                "color_precision": ("INT", {"default": 6, "min": 1, "max": 8,
                                            "tooltip": "Color quantization bits — higher = more colors/layers."}),
                "filter_speckle": ("INT", {"default": 4, "min": 0, "max": 16,
                                           "tooltip": "Drop specks smaller than this (px)."}),
                "mode": (["spline", "polygon"], {"default": "spline"}),
            },
            "optional": {
                "hierarchical": (["stacked", "cutout"], {"default": "stacked"}),
                "corner_threshold": ("INT", {"default": 60, "min": 0, "max": 180}),
                "layer_difference": ("INT", {"default": 16, "min": 0, "max": 128}),
                "path_precision": ("INT", {"default": 3, "min": 0, "max": 8}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("svg_file", "download_url", "info")
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Vector"
    OUTPUT_NODE = True

    def run(self, image, colormode, color_precision, filter_speckle, mode,
            hierarchical="stacked", corner_threshold=60, layer_difference=16, path_precision=3):
        args = {
            "image_url": upload_image(image),
            "colormode": colormode,
            "color_precision": int(color_precision),
            "filter_speckle": int(filter_speckle),
            "mode": mode,
            "hierarchical": hierarchical,
            "corner_threshold": int(corner_threshold),
            "layer_difference": int(layer_difference),
            "path_precision": int(path_precision),
        }
        return _run_svg("fal-ai/image2svg", args, "image2svg")


# ============================================================================ registry

NODE_CLASS_MAPPINGS = {
    "FalObjectRemoval": FalObjectRemoval,
    "FalBriaEraser": FalBriaEraser,
    "FalFluxProErase": FalFluxProErase,
    "FalFinegrainEraser": FalFinegrainEraser,
    "FalZImageTurboInpaint": FalZImageTurboInpaint,
    "FalQwenImageEditInpaint": FalQwenImageEditInpaint,
    "FalBriaGenFill": FalBriaGenFill,
    "FalFluxProFill": FalFluxProFill,
    "FalFluxKontextEdit": FalFluxKontextEdit,
    "FalQwenImageEdit2511": FalQwenImageEdit2511,
    "FalSeedreamEdit": FalSeedreamEdit,
    "FalGeminiFlashEdit": FalGeminiFlashEdit,
    "FalSeedVRUpscale": FalSeedVRUpscale,
    "FalTopazUpscale": FalTopazUpscale,
    "FalRecraftCrispUpscale": FalRecraftCrispUpscale,
    "FalClarityUpscaler": FalClarityUpscaler,
    "FalBriaExpand": FalBriaExpand,
    "FalRecraftVectorize": FalRecraftVectorize,
    "FalImage2SVG": FalImage2SVG,
    "FalGrain": FalGrain,
    "FalVignette": FalVignette,
    "FalColorCorrection": FalColorCorrection,
    "FalSharpen": FalSharpen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalObjectRemoval": "FAL Remove — Object Removal (prompt/mask, $0.006+)",
    "FalBriaEraser": "FAL Remove — Bria Eraser (mask, $0.04)",
    "FalFluxProErase": "FAL Remove — Flux Pro v1 Erase (mask, ~$0.03/MP)",
    "FalFinegrainEraser": "FAL Remove — Finegrain Eraser (prompt+shadows, $0.18+)",
    "FalZImageTurboInpaint": "FAL Inpaint — Z-Image Turbo ($0.01/MP)",
    "FalQwenImageEditInpaint": "FAL Inpaint — Qwen Image Edit v1 (mask, $0.03/MP)",
    "FalBriaGenFill": "FAL Inpaint — Bria GenFill v2 ($0.04/MP)",
    "FalFluxProFill": "FAL Inpaint — Flux Pro v1 Fill, BFL ($0.05/MP)",
    "FalFluxKontextEdit": "FAL Edit — Flux Kontext pro / max, BFL ($0.04 / $0.08)",
    "FalQwenImageEdit2511": "FAL Edit — Qwen Image Edit 2511, newest ($0.03/MP)",
    "FalSeedreamEdit": "FAL Edit — Seedream v5-pro / v5-lite / v4.5 ($0.04–0.14)",
    "FalGeminiFlashEdit": "FAL Banana — Gemini Flash 3.1 / 2.5, older node ($0.039–0.08)",
    "FalSeedVRUpscale": "FAL Upscale — SeedVR v2, opt. seamless ($0.001–0.0025/MP)",
    "FalTopazUpscale": "FAL Upscale — Topaz, model in dropdown ($0.08–1.36)",
    "FalRecraftCrispUpscale": "FAL Upscale — Recraft Crisp ($0.004)",
    "FalClarityUpscaler": "FAL Upscale — Clarity, creative ($0.03/MP)",
    "FalBriaExpand": "FAL Expand — Bria Outpaint ($0.04)",
    "FalRecraftVectorize": "FAL Vector — Recraft Vectorize ($0.01)",
    "FalImage2SVG": "FAL Vector — Image2SVG tracer ($0.005)",
    "FalGrain": "FAL Finish — Film Grain ($0.001)",
    "FalVignette": "FAL Finish — Vignette ($0.001)",
    "FalColorCorrection": "FAL Finish — Color Correction ($0.001)",
    "FalSharpen": "FAL Finish — Sharpen ($0.001)",
}
