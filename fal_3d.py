"""
FAL image-to-3D mesh nodes (category: FAL/3D).

  * FalTripoImageTo3D    -> tripo3d/tripo/v2.5/image-to-3d       (cheap/fast draft, ~$0.05)
  * FalTripoH31          -> tripo3d/h3.1/image-to-3d             (quality dial: geometry/texture
                             standard|detailed; $0.20-0.40 base, +$0.20 detailed geometry)
  * FalHunyuan3D         -> fal-ai/hunyuan3d/v2                  (octree control)
  * FalHunyuan3DV31      -> fal-ai/hunyuan-3d/v3.1 pro|rapid     (pro $0.375 + $0.15 each for
                             PBR / multiview / custom face count; rapid $0.225)
  * FalHunyuanSketchTo3D -> fal-ai/hunyuan3d-v3/sketch-to-3d     (sketch + prompt -> 3D, $0.375+)
  * FalTrellisImageTo3D  -> fal-ai/trellis                      (Microsoft TRELLIS, fine control)

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


class FalTripoH31:
    """tripo3d/h3.1/image-to-3d — Tripo's newest generation with an actual quality dial:
    geometry_quality and texture_quality standard|detailed. $0.20 no texture / $0.30
    standard / $0.40 HD-detailed texture, +$0.20 detailed geometry, +$0.05 quad."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "geometry_quality": (["standard", "detailed"], {"default": "detailed",
                                     "tooltip": "detailed = +$0.20, noticeably denser mesh."}),
                "texture_quality": (["standard", "detailed"], {"default": "standard"}),
                "texture": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "pbr": ("BOOLEAN", {"default": True}),
                "quad": ("BOOLEAN", {"default": False, "tooltip": "Quad mesh — +$0.05."}),
                "face_limit": ("INT", {"default": 0, "min": 0, "max": 500000, "step": 1000,
                                       "tooltip": "0 = adaptive (model decides)."}),
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

    def generate(self, image, geometry_quality, texture_quality, texture, pbr=True, quad=False,
                 face_limit=0, texture_alignment="original_image", orientation="default", seed=0):
        args = {
            "image_url": upload_image(image),
            "geometry_quality": geometry_quality,
            "texture_quality": texture_quality,
            "texture": bool(texture),
            "pbr": bool(pbr),
            "quad": bool(quad),
            "texture_alignment": texture_alignment,
            "orientation": orientation,
        }
        if face_limit and face_limit > 0:
            args["face_limit"] = int(face_limit)
        if seed:
            args["model_seed"] = int(seed)
        return run_mesh("tripo3d/h3.1/image-to-3d", args, "tripo_h31")


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


class FalHunyuan3DV31:
    """fal-ai/hunyuan-3d/v3.1 — newest Hunyuan. pro: $0.375 (+$0.15 each: PBR, multiview,
    custom face_count); rapid: $0.225 (+$0.15 PBR). Optional side views (pro only) pin
    the geometry from more angles."""

    ENDPOINTS = {
        "pro": "fal-ai/hunyuan-3d/v3.1/pro/image-to-3d",
        "rapid": "fal-ai/hunyuan-3d/v3.1/rapid/image-to-3d",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "version": (list(cls.ENDPOINTS), {"default": "pro"}),
                "enable_pbr": ("BOOLEAN", {"default": False, "tooltip": "+$0.15"}),
                "geometry_only": ("BOOLEAN", {"default": False,
                                              "tooltip": "White mesh without texture."}),
            },
            "optional": {
                "face_count": ("INT", {"default": 0, "min": 0, "max": 500000, "step": 10000,
                                       "tooltip": "pro only; 0 = model default (500k). Setting a custom value costs +$0.15."}),
                "back_image": ("IMAGE", {"tooltip": "pro only — extra view."}),
                "left_image": ("IMAGE", {"tooltip": "pro only — extra view. Multiview costs +$0.15."}),
                "right_image": ("IMAGE", {"tooltip": "pro only — extra view."}),
            },
        }

    RETURN_TYPES = MESH_RET_TYPES
    RETURN_NAMES = MESH_RET_NAMES
    FUNCTION = "generate"
    CATEGORY = "FAL/3D"
    OUTPUT_NODE = True

    def generate(self, image, version, enable_pbr, geometry_only, face_count=0,
                 back_image=None, left_image=None, right_image=None):
        args = {
            "input_image_url": upload_image(image),
            "enable_pbr": bool(enable_pbr),
        }
        if version == "pro":
            args["generate_type"] = "Geometry" if geometry_only else "Normal"
            if face_count and face_count > 0:
                args["face_count"] = int(face_count)
            for key, img in (("back_image_url", back_image),
                             ("left_image_url", left_image),
                             ("right_image_url", right_image)):
                if img is not None:
                    args[key] = upload_image(img)
        else:
            args["enable_geometry"] = bool(geometry_only)
        return run_mesh(self.ENDPOINTS[version], args, f"hunyuan31_{version}")


class FalHunyuanSketchTo3D:
    """fal-ai/hunyuan3d-v3/sketch-to-3d — sketch + text prompt straight to a 3D mesh,
    $0.375 (+$0.15 PBR, +$0.15 custom face_count). The prompt tells the model what the
    sketch depicts ('orange cat', 'wooden chair')."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True,
                                      "tooltip": "What the sketch depicts — object, material, style."}),
                "enable_pbr": ("BOOLEAN", {"default": False, "tooltip": "+$0.15"}),
            },
            "optional": {
                "face_count": ("INT", {"default": 0, "min": 0, "max": 500000, "step": 10000,
                                       "tooltip": "0 = model default (500k). Custom value costs +$0.15."}),
            },
        }

    RETURN_TYPES = MESH_RET_TYPES
    RETURN_NAMES = MESH_RET_NAMES
    FUNCTION = "generate"
    CATEGORY = "FAL/3D"
    OUTPUT_NODE = True

    def generate(self, image, prompt, enable_pbr, face_count=0):
        if not prompt.strip():
            raise RuntimeError("prompt is required — say what the sketch depicts")
        args = {
            "input_image_url": upload_image(image),
            "prompt": prompt.strip(),
            "enable_pbr": bool(enable_pbr),
        }
        if face_count and face_count > 0:
            args["face_count"] = int(face_count)
        return run_mesh("fal-ai/hunyuan3d-v3/sketch-to-3d", args, "hunyuan_sketch")


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
    "FalTripoH31": FalTripoH31,
    "FalHunyuan3D": FalHunyuan3D,
    "FalHunyuan3DV31": FalHunyuan3DV31,
    "FalHunyuanSketchTo3D": FalHunyuanSketchTo3D,
    "FalTrellisImageTo3D": FalTrellisImageTo3D,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FalTripoImageTo3D": "FAL 3D — Tripo v2.5, draft (~$0.05)",
    "FalTripoH31": "FAL 3D — Tripo H3.1, quality dial ($0.30–0.65)",
    "FalHunyuan3D": "FAL 3D — Hunyuan3D v2 (octree)",
    "FalHunyuan3DV31": "FAL 3D — Hunyuan3D v3.1 pro/rapid ($0.225–0.525)",
    "FalHunyuanSketchTo3D": "FAL 3D — Hunyuan Sketch→3D (prompt, $0.375+)",
    "FalTrellisImageTo3D": "FAL 3D — TRELLIS (fine control)",
}
