"""ComfyUI-FAL — custom nodes wrapping FAL endpoints, browsable under the FAL/ category.

Each node module exposes its own NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS;
this file merges them. New modules just need to be imported and merged here.
"""
from . import (
    fal_3d,
    fal_background,
    fal_banana,
    fal_generate,
    fal_image_edit,
    fal_material,
    fal_text,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _mod in (fal_3d, fal_background, fal_banana, fal_generate, fal_image_edit,
             fal_material, fal_text):
    NODE_CLASS_MAPPINGS.update(_mod.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_mod.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Tidy up the categories of the other FAL pack in the image (see fal_retag). Cosmetic, and
# guarded so a failure can never take this pack's own nodes down with it.
try:
    from . import fal_retag
    fal_retag.install()
except BaseException:  # noqa: BLE001
    import logging
    logging.getLogger(__name__).debug("[ComfyUI-FAL] retag unavailable", exc_info=True)
