"""ComfyUI-FAL — custom nodes wrapping FAL endpoints, browsable under the FAL/ category.

Each node module exposes its own NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS;
this file merges them. New modules just need to be imported and merged here.
"""
from . import fal_3d, fal_background, fal_image_edit

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _mod in (fal_3d, fal_background, fal_image_edit):
    NODE_CLASS_MAPPINGS.update(_mod.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_mod.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
