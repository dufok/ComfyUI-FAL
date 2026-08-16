"""
Structure-controlled text-to-image (category: FAL/Image/Generate).

  * FalFluxGeneral -> fal-ai/flux-general   ($0.075/MP, rounded UP to the next megapixel)

FLUX.1 [dev] plus the whole conditioning stack — ControlNet, ControlNet Union, IP-Adapter
and up to two LoRAs. This is the pack's first text-to-image node, and the only place in it
where a render can be made to follow geometry: feed a depth / normal / lineart map into
`control_image` and the output obeys it.

Billing rounds UP per megapixel, which surprises people: 1536x864 is 1.33 MP and bills as
2 MP = $0.15, not $0.075. Only a render at or under 1 MP gets the single-megapixel rate.

The API's conditioning parameters are arrays of nested objects — `controlnet_unions` is two
levels deep (a union holds a `controls` list) — so the widgets here are flat and the nesting
is built in the helpers below. That is also where the weight URLs live: for ControlNets the
bare HuggingFace repo id does not resolve, only the full resolve/main/*.safetensors URL does.

Deferred to a later pass (not oversights): control_loras, easycontrols, Redux fill_image,
reference-only, the nag_* knobs, scheduler / shift controls, per-layer LoRA scale dicts,
more than one ControlNet or IP-Adapter (the API only supports one anyway), and the
image-to-image / inpainting / differential-diffusion / rf-inversion sub-endpoints, which
want their own node rather than three more inert widgets on this one.
"""
from .fal_common import (
    run_image,
    upload_image,
    upload_mask,
)


# The bare repo id does not resolve for these — the full weight URL is required.
CONTROLNET_WEIGHTS = {
    "InstantX/FLUX.1-dev-Controlnet-Canny":
        "https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny/resolve/main/diffusion_pytorch_model.safetensors",
    "InstantX/FLUX.1-dev-Controlnet-Union":
        "https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union/resolve/main/diffusion_pytorch_model.safetensors",
    "jasperai/Flux.1-dev-Controlnet-Depth":
        "https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Depth/resolve/main/diffusion_pytorch_model.safetensors",
    "jasperai/Flux.1-dev-Controlnet-Surface-Normals":
        "https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Surface-Normals/resolve/main/diffusion_pytorch_model.safetensors",
    "jasperai/Flux.1-dev-Controlnet-Upscaler":
        "https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler/resolve/main/diffusion_pytorch_model.safetensors",
    "promeai/FLUX.1-controlnet-lineart-promeai":
        "https://huggingface.co/promeai/FLUX.1-controlnet-lineart-promeai/resolve/main/diffusion_pytorch_model.safetensors",
    "Shakker-Labs/FLUX.1-dev-ControlNet-Depth":
        "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Depth/resolve/main/diffusion_pytorch_model.safetensors",
    "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro":
        "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/diffusion_pytorch_model.safetensors",
}

# Unions are the opposite — the bare repo id is what the API wants.
UNION_PATHS = [
    "None",
    "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
    "InstantX/FLUX.1-dev-Controlnet-Union",
]

IP_ADAPTER_WEIGHTS = {
    # the ?download=true is part of the working URL — do not tidy it away
    "XLabs-AI/flux-ip-adapter":
        "https://huggingface.co/XLabs-AI/flux-ip-adapter/resolve/main/flux-ip-adapter.safetensors?download=true",
}
IP_ADAPTER_ENCODER = "openai/clip-vit-large-patch14"   # required on every IPAdapter, no default

# The API enum spells this with a hyphen. The older node elsewhere in the ecosystem ships the
# underscore, which is a live 422 — accept both spellings and normalise on send.
UNION_MODES = ["canny", "tile", "depth", "blur", "pose", "gray", "low-quality", "low_quality"]

SIZES = ["square_hd", "square", "portrait_4_3", "portrait_16_9",
         "landscape_4_3", "landscape_16_9", "custom"]


def _size_arg(image_size, width, height):
    """'custom' is a widget convenience, not an API enum member — map it to {width,height}."""
    if image_size != "custom":
        return image_size
    w, h = int(width) // 16 * 16, int(height) // 16 * 16
    if (w, h) != (int(width), int(height)):
        print(f"[FAL] flux-general wants multiples of 16 — rounding {width}x{height} to {w}x{h}")
    return {"width": w, "height": h}


def _controlnet_args(path, control_url, conditioning_scale, mask_url, start, end):
    cn = {
        "path": CONTROLNET_WEIGHTS.get(path, path),
        "control_image_url": control_url,
        "conditioning_scale": float(conditioning_scale),
        "start_percentage": float(start),
        "end_percentage": float(end),
    }
    if mask_url:
        cn["mask_image_url"] = mask_url
    return [cn]


def _union_args(path, control_mode, control_url, conditioning_scale, mask_url, start, end):
    """A union nests one more level: the union carries a list of `controls`."""
    control = {
        "control_image_url": control_url,
        "control_mode": "low-quality" if control_mode == "low_quality" else control_mode,
        "conditioning_scale": float(conditioning_scale),
        "start_percentage": float(start),
        "end_percentage": float(end),
    }
    if mask_url:
        control["mask_image_url"] = mask_url
    return [{"path": path, "controls": [control]}]


def _ip_adapter_args(path, image_url, scale, mask_url):
    ip = {
        "path": IP_ADAPTER_WEIGHTS.get(path, path),
        "image_encoder_path": IP_ADAPTER_ENCODER,
        "image_url": image_url,
        "scale": float(scale),
    }
    if mask_url:
        ip["mask_image_url"] = mask_url
    return [ip]


def _lora_args(pairs):
    loras = [{"path": p.strip(), "scale": float(s)} for p, s in pairs if p and p.strip()]
    return loras or None


class FalFluxGeneral:
    """fal-ai/flux-general — FLUX.1 [dev] with ControlNet, ControlNet Union, IP-Adapter and
    LoRAs. Feed a depth or normal or lineart map into `control_image` and the render follows
    your geometry; that is what makes this the node to drive from Blender.

    $0.075 per megapixel, billed rounded UP — a 1536x864 frame is 1.33 MP and costs $0.15.

    Only one ControlNet and one IP-Adapter are supported by the API at a time; the node
    refuses the combinations it would otherwise pay to have rejected."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (SIZES, {"default": "landscape_4_3",
                               "tooltip": "'custom' uses width/height below; every other value ignores them."}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16,
                          "tooltip": "'custom' only. Flux wants multiples of 16 — anything else is rounded down."}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 50}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1,
                                   "tooltip": "Flux's distilled guidance. 3–4 is the useful band."}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True,
                                    "tooltip": "Flux has no CFG negative, so FAL routes this through NAG. Only sent when non-empty."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2_147_483_647,
                         "tooltip": "-1 = random. 0 is a real, reproducible seed here — unlike elsewhere in this pack."}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 10}),
                "real_cfg_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 5.0, "step": 0.1,
                                    "tooltip": "Classical CFG — only does anything when use_real_cfg is on."}),
                "use_real_cfg": ("BOOLEAN", {"default": False,
                                 "tooltip": "Classical CFG: slower and more expensive."}),
                "enable_safety_checker": ("BOOLEAN", {"default": True,
                                          "tooltip": "Turning it off needs an authorised FAL account. Unauthorised requests are checked anyway, and flagged images come back BLACK — a second source of black frames unrelated to any error."}),
                # --- ControlNet
                "controlnets": (["None"] + sorted(CONTROLNET_WEIGHTS), {"default": "None",
                                "tooltip": "A single-purpose ControlNet. Needs control_image."}),
                "controlnet_path": ("STRING", {"default": "",
                                    "tooltip": "Escape hatch: any .safetensors URL, overrides the dropdown."}),
                # --- ControlNet Union
                "controlnet_unions": (UNION_PATHS, {"default": "None",
                                      "tooltip": "One model, many map types — pick which with controlnet_union_control_mode."}),
                "controlnet_union_control_mode": (UNION_MODES, {"default": "depth",
                                                  "tooltip": "What kind of map control_image is. 'low_quality' is a legacy misspelling kept so old graphs load; both send 'low-quality'."}),
                "controlnet_conditioning_scale": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.05,
                                                  "tooltip": "How hard the control map is held. API max is 2."}),
                "control_image": ("IMAGE", {"tooltip": "The depth / normal / lineart / pose map."}),
                "control_mask": ("MASK", {"tooltip": "White = where the control map is obeyed."}),
                "control_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                  "tooltip": "Fraction of the steps before the control kicks in."}),
                "control_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                "tooltip": "Release the control after this fraction — structure early, freedom late."}),
                # --- IP-Adapter
                "ip_adapters": (["None"] + sorted(IP_ADAPTER_WEIGHTS), {"default": "None"}),
                "ip_adapter_image": ("IMAGE", {"tooltip": "Style / subject reference."}),
                "ip_adapter_scale": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "ip_adapter_mask": ("MASK",),
                # --- LoRAs
                "lora_path_1": ("STRING", {"default": "", "tooltip": "HF repo id or .safetensors URL."}),
                "lora_scale_1": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "lora_path_2": ("STRING", {"default": ""}),
                "lora_scale_2": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Generate"

    def run(self, prompt, image_size, width, height, num_inference_steps, guidance_scale,
            negative_prompt="", seed=-1, num_images=1, real_cfg_scale=3.5, use_real_cfg=False,
            enable_safety_checker=True,
            controlnets="None", controlnet_path="", controlnet_unions="None",
            controlnet_union_control_mode="depth", controlnet_conditioning_scale=0.75,
            control_image=None, control_mask=None, control_start=0.0, control_end=1.0,
            ip_adapters="None", ip_adapter_image=None, ip_adapter_scale=0.6, ip_adapter_mask=None,
            lora_path_1="", lora_scale_1=1.0, lora_path_2="", lora_scale_2=1.0):

        # --- guards, all before anything is uploaded or paid for
        if not prompt.strip():
            raise RuntimeError("prompt is required")
        want_cn = controlnets != "None" or bool(controlnet_path.strip())
        want_union = controlnet_unions != "None"
        if want_cn and want_union:
            raise RuntimeError("pick a ControlNet OR a ControlNet Union, not both — "
                               "flux-general accepts one at a time")
        if (want_cn or want_union) and control_image is None:
            raise RuntimeError("a ControlNet needs control_image connected — the API requires "
                               "control_image_url and rejects the request without it")
        if control_mask is not None and control_image is None:
            raise RuntimeError("control_mask without control_image")
        if ip_adapters != "None" and ip_adapter_image is None:
            raise RuntimeError("the IP-Adapter needs ip_adapter_image — the API requires image_url")
        if float(control_start) >= float(control_end):
            raise RuntimeError(f"control_start ({control_start}) must be below control_end ({control_end})")

        args = {
            "prompt": prompt.strip(),
            "image_size": _size_arg(image_size, width, height),
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "num_images": int(num_images),
            "enable_safety_checker": bool(enable_safety_checker),
            "output_format": "png",
        }
        if negative_prompt.strip():
            args["negative_prompt"] = negative_prompt.strip()
        # NOT the pack's usual truthiness test: 0 is a legitimate, reproducible seed here.
        if int(seed) >= 0:
            args["seed"] = int(seed)
        if use_real_cfg:
            args["use_real_cfg"] = True
            args["real_cfg_scale"] = float(real_cfg_scale)

        control_url = upload_image(control_image) if control_image is not None else None
        mask_url = upload_mask(control_mask) if control_mask is not None else None

        if want_cn:
            path = controlnet_path.strip() or controlnets
            args["controlnets"] = _controlnet_args(
                path, control_url, controlnet_conditioning_scale, mask_url,
                control_start, control_end)
        elif want_union:
            args["controlnet_unions"] = _union_args(
                controlnet_unions, controlnet_union_control_mode, control_url,
                controlnet_conditioning_scale, mask_url, control_start, control_end)

        if ip_adapters != "None":
            args["ip_adapters"] = _ip_adapter_args(
                ip_adapters, upload_image(ip_adapter_image), ip_adapter_scale,
                upload_mask(ip_adapter_mask) if ip_adapter_mask is not None else None)

        loras = _lora_args(((lora_path_1, lora_scale_1), (lora_path_2, lora_scale_2)))
        if loras:
            args["loras"] = loras

        return (run_image("fal-ai/flux-general", args),)


NODE_CLASS_MAPPINGS = {
    "FalFluxGeneral": FalFluxGeneral,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalFluxGeneral": "FAL Generate — Flux General, ControlNet/LoRA/IP-Adapter ($0.075/MP)",
}
