"""
FAL image-to-3D mesh nodes (category: FAL/3D).

  * FalTripoImageTo3D   -> tripo3d/tripo/v2.5/image-to-3d   (cheap/fast, PBR/quad/HD texture)
  * FalHunyuan3D        -> fal-ai/hunyuan3d/v2              (high-detail mesh, octree control)
  * FalTrellisImageTo3D -> fal-ai/trellis                  (Microsoft TRELLIS, fine control)

Each: IMAGE in -> (glb_file, download_url, preview, info). The .glb lands in ComfyUI's
output dir; glb_file is relative to it — wire into the core Preview3D node to orbit the
mesh right in the graph.
"""
from .fal_common import upload_image, run_mesh, MESH_RET_TYPES, MESH_RET_NAMES


class FalTripoImageTo3D:
    """tripo3d/tripo/v2.5/image-to-3d — cheapest/fastest, PBR + optional HD texture & quad mesh."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "texture": (["standard", "HD", "no"], {"default": "standard"}),
                "pbr": ("BOOLEAN", {"default": True, "tooltip": "Generate PBR materials."}),
            },
            "optional": {
                "quad": ("BOOLEAN", {"default": False, "tooltip": "Quad (FBX) mesh output — +$0.05."}),
                "auto_size": ("BOOLEAN", {"default": False, "tooltip": "Scale model to real-world meters."}),
                "face_limit": ("INT", {"default": 0, "min": 0, "max": 500000, "step": 1000,
                                       "tooltip": "0 = adaptive (model decides). Otherwise cap face count."}),
                "texture_alignment": (["original_image", "geometry"], {"default": "original_image"}),
                "orientation": (["default", "align_image"], {"default": "default"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
            },
        }

    RETURN_TYPES = MESH_RET_TYPES
    RETURN_NAMES = MESH_RET_NAMES
    FUNCTION = "generate"
    CATEGORY = "FAL/3D"
    OUTPUT_NODE = True

    def generate(self, image, texture, pbr, quad=False, auto_size=False, face_limit=0,
                 texture_alignment="original_image", orientation="default", seed=0):
        args = {
            "image_url": upload_image(image),
            "texture": texture,
            "pbr": bool(pbr),
            "quad": bool(quad),
            "auto_size": bool(auto_size),
            "texture_alignment": texture_alignment,
            "orientation": orientation,
        }
        if face_limit and face_limit > 0:
            args["face_limit"] = int(face_limit)
        if seed:
            args["seed"] = int(seed)
        return run_mesh("tripo3d/tripo/v2.5/image-to-3d", args, "tripo")


class FalHunyuan3D:
    """fal-ai/hunyuan3d/v2 — high-detail mesh with octree-resolution control."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "textured_mesh": ("BOOLEAN", {"default": True,
                                              "tooltip": "Generate textured mesh (pricier) vs white mesh."}),
            },
            "optional": {
                "octree_resolution": ("INT", {"default": 256, "min": 1, "max": 1024, "step": 16,
                                              "tooltip": "Higher = denser/more detailed mesh."}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 50}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
            },
        }

    RETURN_TYPES = MESH_RET_TYPES
    RETURN_NAMES = MESH_RET_NAMES
    FUNCTION = "generate"
    CATEGORY = "FAL/3D"
    OUTPUT_NODE = True

    def generate(self, image, textured_mesh, octree_resolution=256, num_inference_steps=50,
                 guidance_scale=7.5, seed=0):
        args = {
            "input_image_url": upload_image(image),
            "textured_mesh": bool(textured_mesh),
            "octree_resolution": int(octree_resolution),
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
        }
        if seed:
            args["seed"] = int(seed)
        return run_mesh("fal-ai/hunyuan3d/v2", args, "hunyuan3d", want_preview=False)


class FalTrellisImageTo3D:
    """fal-ai/trellis — Microsoft TRELLIS, fine-grained sparse-structure / latent control."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "texture_size": (["512", "1024", "2048"], {"default": "1024"}),
            },
            "optional": {
                "ss_guidance_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "ss_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "slat_guidance_strength": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "slat_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "mesh_simplify": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01,
                                            "tooltip": "Higher = simpler mesh (fewer faces)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
            },
        }

    RETURN_TYPES = MESH_RET_TYPES
    RETURN_NAMES = MESH_RET_NAMES
    FUNCTION = "generate"
    CATEGORY = "FAL/3D"
    OUTPUT_NODE = True

    def generate(self, image, texture_size, ss_guidance_strength=7.5, ss_sampling_steps=12,
                 slat_guidance_strength=3.0, slat_sampling_steps=12, mesh_simplify=0.95, seed=0):
        args = {
            "image_url": upload_image(image),
            "texture_size": int(texture_size),
            "ss_guidance_strength": float(ss_guidance_strength),
            "ss_sampling_steps": int(ss_sampling_steps),
            "slat_guidance_strength": float(slat_guidance_strength),
            "slat_sampling_steps": int(slat_sampling_steps),
            "mesh_simplify": float(mesh_simplify),
        }
        if seed:
            args["seed"] = int(seed)
        return run_mesh("fal-ai/trellis", args, "trellis", want_preview=False)


NODE_CLASS_MAPPINGS = {
    "FalTripoImageTo3D": FalTripoImageTo3D,
    "FalHunyuan3D": FalHunyuan3D,
    "FalTrellisImageTo3D": FalTrellisImageTo3D,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FalTripoImageTo3D": "FAL 3D — Tripo v2.5 (image→3D)",
    "FalHunyuan3D": "FAL 3D — Hunyuan3D v2 (image→3D)",
    "FalTrellisImageTo3D": "FAL 3D — TRELLIS (image→3D)",
}
