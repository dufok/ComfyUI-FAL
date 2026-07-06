"""
FAL image-editing nodes (category: FAL/Image Edit).

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
from .fal_common import (
    run_image,
    upload_image,
    upload_image_frames,
    upload_mask,
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
    CATEGORY = "FAL/Image Edit/Remove"

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
    """fal-ai/bria/eraser — precise mask-based removal, commercially licensed data. $0.04."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image Edit/Remove"

    def run(self, image, mask):
        args = {"image_url": upload_image(image), "mask_url": upload_mask(mask)}
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
    CATEGORY = "FAL/Image Edit/Remove"

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
    CATEGORY = "FAL/Image Edit/Remove"

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
    CATEGORY = "FAL/Image Edit/Inpaint"

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
    CATEGORY = "FAL/Image Edit/Inpaint"

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
    CATEGORY = "FAL/Image Edit/Inpaint"

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


# ============================================================================ Edit

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
    CATEGORY = "FAL/Image Edit/Edit"

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
    """Bytedance Seedream edit, $0.04/image — natural-language multi-ref editing
    ('replace the product in image 1 with the one from image 2'). v4.5 or v5 lite."""

    ENDPOINTS = {
        "v4.5": "fal-ai/bytedance/seedream/v4.5/edit",
        "v5-lite": "fal-ai/bytedance/seedream/v5/lite/edit",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "version": (list(cls.ENDPOINTS), {"default": "v4.5"}),
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
    CATEGORY = "FAL/Image Edit/Edit"

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
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image Edit/Edit"

    def run(self, image, prompt, version, resolution, image_2=None, image_3=None,
            num_images=1, seed=0):
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
        return (run_image(self.ENDPOINTS[version], _seed_arg(args, seed)),)


# ============================================================================ Upscale

class FalSeedVRUpscale:
    """fal-ai/seedvr/upscale/image — SeedVR2, strong generative photo upscaler."""

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
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image Edit/Upscale"

    def run(self, image, upscale_mode, upscale_factor, target_resolution, noise_scale, seed=0):
        args = {
            "image_url": upload_image(image),
            "upscale_mode": upscale_mode,
            "noise_scale": float(noise_scale),
            "output_format": "png",
        }
        if upscale_mode == "factor":
            args["upscale_factor"] = float(upscale_factor)
        else:
            args["target_resolution"] = target_resolution
        return (run_image("fal-ai/seedvr/upscale/image", _seed_arg(args, seed)),)


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
    CATEGORY = "FAL/Image Edit/Upscale"

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
    CATEGORY = "FAL/Image Edit/Upscale"

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
    CATEGORY = "FAL/Image Edit/Upscale"

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
    CATEGORY = "FAL/Image Edit/Expand"

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


# ============================================================================ registry

NODE_CLASS_MAPPINGS = {
    "FalObjectRemoval": FalObjectRemoval,
    "FalBriaEraser": FalBriaEraser,
    "FalFluxProErase": FalFluxProErase,
    "FalFinegrainEraser": FalFinegrainEraser,
    "FalZImageTurboInpaint": FalZImageTurboInpaint,
    "FalQwenImageEditInpaint": FalQwenImageEditInpaint,
    "FalBriaGenFill": FalBriaGenFill,
    "FalQwenImageEdit2511": FalQwenImageEdit2511,
    "FalSeedreamEdit": FalSeedreamEdit,
    "FalGeminiFlashEdit": FalGeminiFlashEdit,
    "FalSeedVRUpscale": FalSeedVRUpscale,
    "FalTopazUpscale": FalTopazUpscale,
    "FalRecraftCrispUpscale": FalRecraftCrispUpscale,
    "FalClarityUpscaler": FalClarityUpscaler,
    "FalBriaExpand": FalBriaExpand,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalObjectRemoval": "FAL Remove — Object Removal (prompt/mask, $0.006+)",
    "FalBriaEraser": "FAL Remove — Bria Eraser (mask, $0.04)",
    "FalFluxProErase": "FAL Remove — Flux Pro v1 Erase (mask, ~$0.03/MP)",
    "FalFinegrainEraser": "FAL Remove — Finegrain Eraser (prompt+shadows, $0.18+)",
    "FalZImageTurboInpaint": "FAL Inpaint — Z-Image Turbo ($0.01/MP)",
    "FalQwenImageEditInpaint": "FAL Inpaint — Qwen Image Edit v1 (mask, 2511 has none)",
    "FalBriaGenFill": "FAL Inpaint — Bria GenFill v2 ($0.04/MP)",
    "FalQwenImageEdit2511": "FAL Edit — Qwen Image Edit 2511, newest ($0.03/MP)",
    "FalSeedreamEdit": "FAL Edit — Seedream v4.5 / v5-lite ($0.04)",
    "FalGeminiFlashEdit": "FAL Edit — Gemini Flash 3.1 / 2.5 (Google, $0.04–0.08)",
    "FalSeedVRUpscale": "FAL Upscale — SeedVR v2",
    "FalTopazUpscale": "FAL Upscale — Topaz (model in dropdown)",
    "FalRecraftCrispUpscale": "FAL Upscale — Recraft Crisp",
    "FalClarityUpscaler": "FAL Upscale — Clarity (creative)",
    "FalBriaExpand": "FAL Expand — Bria Outpaint",
}
