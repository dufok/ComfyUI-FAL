"""
Topaz on FAL — the 2026 endpoint family (category: FAL/Image/Upscale, FAL/Image/Restore).

FAL retired the single `fal-ai/topaz/upscale/image` endpoint and split Topaz into a family:

    topaz/upscale/image/precision     deterministic enhancers   (Standard V2, High Fidelity V3/V2,
                                                                 Low Resolution V2, CGI, Text Refine)
    topaz/upscale/image/generative    diffusion-based rebuilds  (Wonder 3.5/3/2/Wonder, Recover 3,
                                                                 Standard MAX, Redefine, Recovery V2/Recovery)
    topaz/upscale/image/creative      Bloom, free-hand detail   (Bloom 2, Bloom, Bloom Realism)
    topaz/upscale/image/transparent   keeps the alpha channel
    topaz/restore/image               Recover 3 / Dust-Scratch V2 / Faces, no upscale

All of them bill at $0.01 per output megapixel, so a 2x pass on a 12 MP photo is ~$0.12 and the
choice of model costs nothing extra — pick on looks, not on price.

Which one to use:
  * small or compressed sources (phone/messenger photos, web JPEGs) -> generative "Wonder 3.5".
    The precision models are deterministic: on a compressed source they sharpen the JPEG blocks
    themselves, which shows up as a crosshatch/"fabric" texture in flat areas and a waxy face.
  * already-clean, sharp, high-resolution sources -> precision "High Fidelity V3".
  * renders and CG -> precision "CGI"; screenshots and text -> "Text Refine".
  * a picture you want re-imagined rather than restored -> creative "Bloom 2".
  * cutouts with alpha -> "Transparent".

Every dial below is gated per model, because Topaz applies most of them to one model only and FAL
rejects (422) unknown fields. Dials that do not apply to the chosen model are dropped and named in
the console, so nothing is silently ignored.
"""

from .fal_common import run_image, upload_image

PRECISION = "topaz/upscale/image/precision"
GENERATIVE = "topaz/upscale/image/generative"
CREATIVE = "topaz/upscale/image/creative"
TRANSPARENT = "topaz/upscale/image/transparent"
RESTORE = "topaz/restore/image"

FACE = ("face_enhancement", "face_enhancement_strength", "face_enhancement_creativity")

# label -> (endpoint, API model name, extra params this model actually accepts)
# "upscale_factor" is accepted by everything except /transparent, which takes the image alone.
MODELS = {
    # ---- generative: rebuilds detail, the right family for weak sources
    "gen · Wonder 3.5 (best all-round)":   (GENERATIVE, "Wonder 3.5", {*FACE, "enhancement_strength"}),
    "gen · Wonder 3":                      (GENERATIVE, "Wonder 3", {*FACE, "enhancement_strength", "subject_detection"}),
    "gen · Wonder 2":                      (GENERATIVE, "Wonder 2", {*FACE}),
    "gen · Wonder":                        (GENERATIVE, "Wonder", {*FACE}),
    "gen · Recover 3":                     (GENERATIVE, "Recover 3", {*FACE}),
    "gen · Standard MAX":                  (GENERATIVE, "Standard MAX", {*FACE}),
    "gen · Redefine (prompt)":             (GENERATIVE, "Redefine",
                                            {*FACE, "prompt", "autoprompt", "creativity", "texture", "sharpen", "denoise"}),
    "gen · Recovery V2 (very low-res)":    (GENERATIVE, "Recovery V2", {*FACE, "subject_detection", "detail"}),
    "gen · Recovery (very low-res)":       (GENERATIVE, "Recovery", {*FACE, "subject_detection"}),
    # ---- precision: deterministic, for sources that are already clean
    "precision · Standard V2":             (PRECISION, "Standard V2",
                                            {*FACE, "subject_detection", "sharpen", "denoise", "fix_compression"}),
    "precision · High Fidelity V3":        (PRECISION, "High Fidelity V3",
                                            {*FACE, "subject_detection", "sharpen", "denoise", "fix_compression"}),
    "precision · High Fidelity V2":        (PRECISION, "High Fidelity V2",
                                            {*FACE, "subject_detection", "sharpen", "denoise", "fix_compression"}),
    "precision · Low Resolution V2":       (PRECISION, "Low Resolution V2",
                                            {*FACE, "subject_detection", "sharpen", "denoise", "fix_compression"}),
    "precision · CGI (renders)":           (PRECISION, "CGI",  # fix_compression is unsupported here
                                            {*FACE, "subject_detection", "sharpen", "denoise"}),
    "precision · Text Refine":             (PRECISION, "Text Refine",
                                            {*FACE, "subject_detection", "sharpen", "denoise", "fix_compression", "strength"}),
    # ---- creative: Bloom re-imagines the picture
    "creative · Bloom 2":                  (CREATIVE, "Bloom 2", {"creativity", "color_preservation", "autoprompt"}),
    "creative · Bloom":                    (CREATIVE, "Bloom", set()),
    "creative · Bloom Realism":            (CREATIVE, "Bloom Realism", set()),
    # ---- alpha
    "alpha · Transparent (keeps alpha)":   (TRANSPARENT, None, set()),
}

DEFAULT_MODEL = "gen · Wonder 3.5 (best all-round)"

# Labels that shipped in an earlier version. A saved graph stores the label as its widget value, so
# renaming one would fail validation with "value not in list"; accept the old spelling and map it.
LEGACY_LABELS = {
    "gen · Wonder 3.5 (лучший универсал)": DEFAULT_MODEL,
}

AUTO = "auto"
TRI = [AUTO, "on", "off"]


def _tri(value):
    """auto/on/off tri-state -> True/False/None (None = leave the field out)."""
    return {"on": True, "off": False}.get(value, None)


class FalTopazUpscale2026:
    """topaz/upscale/image/{precision,generative,creative,transparent} — one node, all Topaz models.

    The endpoint is chosen from the model name, and every optional dial is filtered against what
    that model accepts, so switching models can never produce a 422."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (list(MODELS), {"default": DEFAULT_MODEL,
                                         "tooltip": "gen = rebuilds detail (weak/compressed sources), "
                                                    "precision = deterministic (clean sources), "
                                                    "creative = Bloom re-imagines, alpha = keeps transparency."}),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.25,
                                             "tooltip": "1–4 on this family (the retired endpoint allowed 8). "
                                                        "Ignored by the Transparent model."}),
                "face_enhancement": ("BOOLEAN", {"default": False,
                                                 "tooltip": "Topaz's separate face pass. On small faces it tends to "
                                                            "look waxy — leave it off unless the face is the subject."}),
            },
            "optional": {
                "face_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "face_creativity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "enhancement_strength": ([AUTO, "low", "medium", "high"], {"default": AUTO,
                                          "tooltip": "Wonder 3 / 3.5 only. auto = let Topaz decide."}),
                "subject_detection": ([AUTO, "All", "Foreground", "Background"], {"default": AUTO,
                                       "tooltip": "Precision models, Wonder 3, Recovery, Recovery V2."}),
                "creativity": ("INT", {"default": 0, "min": 0, "max": 9,
                                       "tooltip": "0 = auto. Redefine 1–6, Bloom 2 1–9."}),
                "texture": ("INT", {"default": 0, "min": 0, "max": 5, "tooltip": "0 = auto. Redefine only."}),
                "detail": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05,
                                     "tooltip": "-1 = auto. Recovery V2 only."}),
                "denoise": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "-1 = auto. Precision models and Redefine."}),
                "sharpen": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "-1 = auto. Precision models and Redefine."}),
                "fix_compression": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05,
                                              "tooltip": "-1 = auto. Precision models except CGI — removes JPEG artefacts."}),
                "strength": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05,
                                       "tooltip": "-1 = auto. Text Refine only."}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Redefine only (max 1024 chars)."}),
                "autoprompt": (TRI, {"default": AUTO, "tooltip": "Redefine and Bloom 2."}),
                "color_preservation": (TRI, {"default": AUTO, "tooltip": "Bloom 2 only."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Upscale"

    @classmethod
    def VALIDATE_INPUTS(cls, model):
        # Naming `model` here replaces ComfyUI's combo-membership check, so a graph saved with a
        # renamed label still validates and is mapped in run().
        if model in MODELS or model in LEGACY_LABELS:
            return True
        return f"Unknown Topaz model {model!r} — pick one from the dropdown"

    def run(self, image, model, upscale_factor, face_enhancement,
            face_strength=0.8, face_creativity=0.0, enhancement_strength=AUTO, subject_detection=AUTO,
            creativity=0, texture=0, detail=-1.0, denoise=-1.0, sharpen=-1.0, fix_compression=-1.0,
            strength=-1.0, prompt="", autoprompt=AUTO, color_preservation=AUTO):
        model = LEGACY_LABELS.get(model, model)
        try:
            endpoint, api_model, allowed = MODELS[model]
        except KeyError:
            raise ValueError(
                f"Unknown Topaz model {model!r}. Pick one of: {', '.join(MODELS)}"
            ) from None

        args = {"image_url": upload_image(image), "output_format": "png"}
        if api_model:
            args["model"] = api_model
        if endpoint != TRANSPARENT:
            args["upscale_factor"] = round(float(upscale_factor), 2)

        # Everything the user set away from its "auto" sentinel, before gating.
        wanted = {}
        if face_enhancement:
            wanted["face_enhancement"] = True
            wanted["face_enhancement_strength"] = float(face_strength)
            wanted["face_enhancement_creativity"] = float(face_creativity)
        elif endpoint in (PRECISION, GENERATIVE):
            wanted["face_enhancement"] = False  # the API default is True, so "off" must be sent
        if enhancement_strength != AUTO:
            wanted["enhancement_strength"] = enhancement_strength
        if subject_detection != AUTO:
            wanted["subject_detection"] = subject_detection
        if creativity:
            wanted["creativity"] = int(creativity)
        if texture:
            wanted["texture"] = int(texture)
        for name, value in (("detail", detail), ("denoise", denoise), ("sharpen", sharpen),
                            ("fix_compression", fix_compression), ("strength", strength)):
            if value is not None and value >= 0:
                wanted[name] = float(value)
        if prompt.strip():
            wanted["prompt"] = prompt.strip()[:1024]
        for name, tri in (("autoprompt", autoprompt), ("color_preservation", color_preservation)):
            flag = _tri(tri)
            if flag is not None:
                wanted[name] = flag

        dropped = sorted(k for k in wanted if k not in allowed)
        if dropped:
            # Loud on purpose: Topaz scopes most dials to a single model, and a silently ignored
            # dial reads as "the model has no effect".
            print(f"[FAL] Topaz {model}: ignoring {', '.join(dropped)} — not accepted by this model")
        args.update({k: v for k, v in wanted.items() if k in allowed})

        if creativity and endpoint == GENERATIVE and creativity > 6:
            raise ValueError("Redefine takes creativity 1–6 (1–9 is the Bloom 2 range)")
        return (run_image(endpoint, args),)


class FalTopazRestore:
    """topaz/restore/image — repair pass with no upscale ($0.01/MP).

    Recover 3 rebuilds a degraded photo, Dust-Scratch V2 removes film dust and scratches, Faces
    repairs faces only. Output size equals input size, so chain it before an upscale node."""

    MODELS = ["Recover 3", "Dust-Scratch V2", "Faces"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (cls.MODELS, {"default": "Recover 3"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "FAL/Image/Restore"

    def run(self, image, model):
        args = {"image_url": upload_image(image), "model": model, "output_format": "png"}
        return (run_image(RESTORE, args),)


NODE_CLASS_MAPPINGS = {
    "FalTopazUpscale2026": FalTopazUpscale2026,
    "FalTopazRestore": FalTopazRestore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalTopazUpscale2026": "FAL Upscale — Topaz 2026 (Wonder/Bloom/precision, $0.01/MP)",
    "FalTopazRestore": "FAL Restore — Topaz (Recover 3 / dust / faces, $0.01/MP)",
}
