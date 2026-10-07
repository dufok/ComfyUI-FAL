"""image/save — saving that the stock SaveImage cannot do. No API calls, no FAL key.

SaveImage16Gray
    Writes a grayscale 16-bit PNG (65536 levels instead of 256). Built for height maps that
    drive real displacement (Cycles): an 8-bit height gives 256 visible terraces once the
    surface is actually moved, 16-bit is smooth. Used by the Texturizeme Blender add-on for
    the CHORD / PATINA height output.

    Why each step exists:
    * Accepts both IMAGE layouts it meets in practice: the regular [B, H, W, C] and the
      [B, H, W] that ChordNormalToHeight / Chord roughness outputs return. Multi-channel input
      is reduced to its mean - a height map is one number per pixel.
    * Values are clamped to 0..1 before scaling: the CHORD height is already normalised, and a
      stray overshoot must not wrap around in uint16.
    * Pillow writes mode "I;16" PNGs natively, so there is no extra dependency. Pillow cannot
      write 16-bit RGB, which is why the node is grayscale only (normals are fine in 8 bit).
    * Returns the same `ui.images` entries as SaveImage, so /history and /view serve the file
      exactly like any other output - clients need no special case.
"""

import os

import numpy as np
from PIL import Image

import folder_paths


class SaveImage16Gray:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "height16",
                                               "tooltip": "Same rules as SaveImage, subfolders allowed."}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "image/save"

    def save(self, images, filename_prefix="height16"):
        arr = images.detach().float().cpu().numpy()
        if arr.ndim == 4:
            arr = arr.mean(axis=-1)
        elif arr.ndim == 2:
            arr = arr[None]
        b, h, w = arr.shape
        out_dir = folder_paths.get_output_directory()
        full_dir, fname, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, out_dir, w, h)
        results = []
        for i in range(b):
            px = np.round(np.clip(arr[i], 0.0, 1.0) * 65535.0).astype(np.uint16)
            name = f"{fname}_{counter:05}_.png"
            Image.fromarray(px).save(os.path.join(full_dir, name), compress_level=4)
            results.append({"filename": name, "subfolder": subfolder, "type": "output"})
            counter += 1
        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {"SaveImage16Gray": SaveImage16Gray}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveImage16Gray": "💾 Save Image 16-bit (grayscale PNG)"}
