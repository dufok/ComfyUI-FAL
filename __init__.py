"""ComfyUI-FAL — custom nodes wrapping FAL endpoints, browsable under the FAL/ category,
plus the folder-IO bar under image/folder (a folder of photos in, one ZIP out — no FAL key needed).

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
    fal_restore,
    fal_text,
    fal_topaz,
    fal_video,
    folderio_nodes,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _mod in (fal_3d, fal_background, fal_banana, fal_generate, fal_image_edit,
             fal_material, fal_restore, fal_text, fal_topaz, fal_video, folderio_nodes):
    NODE_CLASS_MAPPINGS.update(_mod.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_mod.NODE_DISPLAY_NAME_MAPPINGS)

# Browser-side extension for the folder nodes (upload buttons, folder drag-and-drop,
# Download-ZIP button). ComfyUI serves this directory at /extensions/ComfyUI-FAL/.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Tidy up the categories of the other FAL pack in the image (see fal_retag). Cosmetic, and
# guarded so a failure can never take this pack's own nodes down with it.
try:
    from . import fal_retag
    fal_retag.install()
except BaseException:  # noqa: BLE001
    import logging
    logging.getLogger(__name__).debug("[ComfyUI-FAL] retag unavailable", exc_info=True)
