"""
Vision and language nodes over FAL's OpenRouter router (category: FAL/Text).

  * FalVLM -> openrouter/router/vision   (image(s) + question -> text)
  * FalLLM -> openrouter/router          (prompt -> text)

Both are token-billed rather than per-call, so instead of a fake price in the node title
they return an `info` string carrying the model, the token counts and the actual cost FAL
reports for that request — the honest answer to "what did this run cost".

Ported out of gokayfem's pack for one reason: its shared helper swallows every exception
and hands the graph a placeholder, so a bad key or a rate-limit produces an empty caption
that quietly becomes an empty prompt on the next node. `run_text` raises on an error field,
on an empty output, and on a malformed response. A headless run should fail, not drift.

The model dropdowns are FAL's own published examples for each endpoint. The field takes any
OpenRouter slug, so pick "Custom" and type one if the list is behind.
"""
from .fal_common import (
    run_text,
    upload_image,
    upload_image_frames,
)


VISION_MODELS = [
    "google/gemini-2.5-flash",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-4o",
    "moonshotai/kimi-k2.5",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "x-ai/grok-4-fast",
    "Custom",
]

CHAT_MODELS = [
    "google/gemini-2.5-flash",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-4.6",
    "openai/gpt-4.1",
    "openai/gpt-oss-120b",
    "meta-llama/llama-4-maverick",
    "moonshotai/kimi-k2.5",
    "Custom",
]

_MODEL_TIP = ("Any OpenRouter slug. Billed per token, so a large model on a large image costs "
              "real money — gemini-2.5-flash is roughly $0.0006 a caption. Pick 'Custom' to "
              "type a slug this list does not have.")
_TEMP_TIP = ("0 gives the same answer every time for the same input — use it when the text feeds "
             "a deterministic pipeline.")
_MAXTOK_TIP = ("Cap on the ANSWER length. 0 = leave it to the model; the API minimum is 1, so 0 is "
               "omitted rather than sent.")
_SEED_TIP = ("Not sent to the API. Bumping it busts ComfyUI's cache so the same input can be "
             "re-run without editing the prompt.")


def _resolve_model(model, custom_model_name):
    if model != "Custom":
        return model
    if not custom_model_name.strip():
        raise RuntimeError("model is 'Custom' but custom_model_name is empty — "
                           "type an OpenRouter slug, e.g. 'google/gemini-3-pro'")
    return custom_model_name.strip()


def _common_args(model, prompt, system_prompt, temperature, max_tokens, reasoning,
                 enable_web_search):
    if not prompt.strip():
        raise RuntimeError("prompt is required")
    args = {"model": model, "prompt": prompt.strip()}
    if system_prompt.strip():
        args["system_prompt"] = system_prompt.strip()
    if abs(float(temperature) - 1.0) > 1e-9:
        args["temperature"] = float(temperature)
    if int(max_tokens) > 0:              # API minimum is 1; 0 means "unset", so omit it
        args["max_tokens"] = int(max_tokens)
    if reasoning:
        args["reasoning"] = True
    if enable_web_search:
        args["enable_web_search"] = True
    return args


class FalVLM:
    """openrouter/router/vision — ask a vision model about an image and get text back.

    Token-billed; the `info` output reports the real cost of the call. The classic use is
    captioning a reference photo into a prompt for an image model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "Describe this image.", "multiline": True,
                           "tooltip": "The question asked about the image."}),
                "model": (VISION_MODELS, {"default": "google/gemini-2.5-flash", "tooltip": _MODEL_TIP}),
            },
            "optional": {
                "system_prompt": ("STRING", {"default": "", "multiline": True,
                                  "tooltip": "Style / format instructions. Not sent when blank."}),
                "custom_model_name": ("STRING", {"default": "",
                                      "tooltip": "OpenRouter slug, used only when model = Custom."}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                "tooltip": _TEMP_TIP}),
                "max_tokens": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": _MAXTOK_TIP}),
                "reasoning": ("BOOLEAN", {"default": False,
                              "tooltip": "Let the model think out loud. The vision endpoint has no "
                                         "separate reasoning field, so this lands inside `text`."}),
                "all_frames": ("BOOLEAN", {"default": True,
                               "tooltip": "Send every frame of an IMAGE batch as a separate reference "
                                          "(and pay for all of them). Off = first frame only."}),
                "enable_web_search": ("BOOLEAN", {"default": False,
                                      "tooltip": "Let the model search the web. Charged on top of tokens."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647, "tooltip": _SEED_TIP}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "info")
    FUNCTION = "run"
    CATEGORY = "FAL/Text"

    def run(self, image, prompt, model, system_prompt="", custom_model_name="", temperature=1.0,
            max_tokens=0, reasoning=False, all_frames=True, enable_web_search=False, seed=0):
        args = _common_args(_resolve_model(model, custom_model_name), prompt, system_prompt,
                            temperature, max_tokens, reasoning, enable_web_search)
        args["image_urls"] = upload_image_frames(image) if all_frames else [upload_image(image)]
        text, _reasoning, info = run_text("openrouter/router/vision", args)
        return (text, info)


class FalLLM:
    """openrouter/router — plain text in, text out, across the OpenRouter roster.

    Token-billed; `info` reports the real cost. Unlike the vision endpoint this one has a
    separate `reasoning` field, so thinking output comes back on its own socket instead of
    polluting the answer."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (CHAT_MODELS, {"default": "google/gemini-2.5-flash", "tooltip": _MODEL_TIP}),
            },
            "optional": {
                "system_prompt": ("STRING", {"default": "", "multiline": True,
                                  "tooltip": "Persona / format instructions. Not sent when blank."}),
                "custom_model_name": ("STRING", {"default": "",
                                      "tooltip": "OpenRouter slug, used only when model = Custom."}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                "tooltip": _TEMP_TIP}),
                "max_tokens": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": _MAXTOK_TIP}),
                "reasoning": ("BOOLEAN", {"default": False,
                              "tooltip": "Ask the model to reason first. Comes back on the `reasoning` "
                                         "output, separate from the answer."}),
                "enable_web_search": ("BOOLEAN", {"default": False,
                                      "tooltip": "Let the model search the web. Charged on top of tokens."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647, "tooltip": _SEED_TIP}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "reasoning", "info")
    FUNCTION = "run"
    CATEGORY = "FAL/Text"

    def run(self, prompt, model, system_prompt="", custom_model_name="", temperature=1.0,
            max_tokens=0, reasoning=False, enable_web_search=False, seed=0):
        args = _common_args(_resolve_model(model, custom_model_name), prompt, system_prompt,
                            temperature, max_tokens, reasoning, enable_web_search)
        text, reasoning_text, info = run_text("openrouter/router", args)
        return (text, reasoning_text, info)


NODE_CLASS_MAPPINGS = {
    "FalVLM": FalVLM,
    "FalLLM": FalLLM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalVLM": "FAL Text — VLM, image→text (per token)",
    "FalLLM": "FAL Text — LLM, text→text (per token)",
}
