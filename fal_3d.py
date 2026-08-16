"""
FAL image-to-3D mesh nodes (category: FAL/3D).

  * FalTripoImageTo3D    -> tripo3d/tripo/v2.5/image-to-3d       ($0.20 bare / $0.30 standard /
                             $0.40 HD texture, +$0.05 quad — Tripo bills in credits, 1 cr = $0.01)
  * FalTripoH31          -> tripo3d/h3.1/image-to-3d             (quality dial: geometry/texture
                             standard|detailed; $0.20-0.40 base, +$0.20 detailed geometry)
  * FalHunyuan3D         -> fal-ai/hunyuan3d/v2                  (octree control)
  * FalHunyuan3DV31      -> fal-ai/hunyuan-3d/v3.1 pro|rapid     (pro $0.375 + $0.15 each for
                             PBR / multiview / custom face count; rapid $0.225)
  * FalHunyuanSketchTo3D -> fal-ai/hunyuan3d-v3/sketch-to-3d     (sketch + prompt -> 3D, $0.375+)
  * FalTrellisImageTo3D  -> fal-ai/trellis                      (Microsoft TRELLIS, fine control)

  * FalMeshyV7           -> meshy/v7 image|multi-image        (geometry + PBR + quad topology +
                             rigging + animation in one call; $0.80 bare / $1.20 textured /
                             $1.40 ultra, +$0.20 rig, +$0.12 anim)

  * FalTripoSplat        -> tripo3d/triposplat               (image -> 3D Gaussian Splat, $0.05;
                             FILE_3D output plugs into the core splat nodes)

Mesh-in, mesh-out utilities (wire a glb_file output into mesh_file):
  * FalTripoRemesh       -> tripo3d/tripo/remesh              ($0.01 — the default retopo: real
                             quads, bakes textures, caps at 20k faces / 10k quad)
  * FalTripoSegment      -> tripo3d/tripo/segment             ($0.01 — split into named parts;
                             feed part_names into Remesh to preserve them)
  * FalMeshyRemesh       -> fal-ai/meshy/v5/remesh            ($0.20 — the high-poly path: up to
                             300k faces, .blend/.usdz, rescale + origin)
  * FalSmartTopology     -> fal-ai/hunyuan-3d/v3.1/…          ($0.75 — glb/obj in, mixed quad/tri)
  * FalMeshyRigging      -> fal-ai/meshy/rigging              ($0.20 +$0.12 anim — rig ANY
                             humanoid GLB, not just one Meshy just made)

Each mesh node: IMAGE in -> (glb_file, download_url, preview, info). The .glb lands in
ComfyUI's output dir; glb_file is relative to it — wire into the core Preview3D node to
orbit the mesh right in the graph.
"""
import io
import os
import urllib.request

import fal_client
import folder_paths

from .fal_common import (
    upload_image,
    upload_image_frames,
    upload_image_rgba,
    run_mesh,
    require_key,
    deep_find,
    save_file,
    url_to_image_tensor,
    blank_image,
    public_download_url,
    MESH_RET_TYPES,
    MESH_RET_NAMES,
)

try:
    from comfy_api.latest import Types as _comfy_types  # File3D for the core splat nodes
except ImportError:  # pre-2026 ComfyUI without comfy_api geometry types
    _comfy_types = None


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


def _upload_mesh_file(mesh_file):
    """Resolve a mesh path (relative to ComfyUI's output dir, e.g. a glb_file output)
    and upload it to FAL, returning the URL."""
    path = (mesh_file or "").strip()
    if not path:
        raise RuntimeError("mesh_file is empty — wire a glb_file output or type a filename from output/")
    if not os.path.isabs(path):
        path = os.path.join(folder_paths.get_output_directory(), path)
    if not os.path.isfile(path):
        raise RuntimeError(f"mesh file not found: {path}")
    return fal_client.upload_file(path)


class FalSmartTopology:
    """fal-ai/hunyuan-3d/v3.1/smart-topology — AI retopology, $0.75. Feed a glb/obj
    (wire a glb_file output or type a filename from output/); polygon_type
    'quadrilateral' gives clean quads for Blender/DCC work."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_file": ("STRING", {"default": "", "tooltip": "glb/obj path relative to output/ — wire glb_file from a FAL 3D node. FBX is NOT accepted here."}),
                "face_level": (["high", "medium", "low"], {"default": "medium"}),
                "polygon_type": (["quadrilateral", "triangle"], {"default": "quadrilateral"}),
            },
        }

    RETURN_TYPES = MESH_RET_TYPES
    RETURN_NAMES = MESH_RET_NAMES
    FUNCTION = "generate"
    CATEGORY = "FAL/3D"
    OUTPUT_NODE = True

    def generate(self, mesh_file, face_level, polygon_type):
        ext = mesh_file.strip().split("?")[0].rsplit(".", 1)[-1].lower()
        if ext not in ("glb", "obj"):
            raise RuntimeError(f"smart-topology accepts glb/obj only, got .{ext} — "
                               "generate with quad=false (glb) or convert via Meshy Remesh first")
        args = {
            "input_file_url": _upload_mesh_file(mesh_file),
            "input_file_type": ext,
            "face_level": face_level,
            "polygon_type": polygon_type,
        }
        return run_mesh("fal-ai/hunyuan-3d/v3.1/smart-topology", args, "smart_topo", want_preview=False)


class FalMeshyRemesh:
    """fal-ai/meshy/v5/remesh — remesh + polycount + format conversion, $0.20.
    Wire a glb_file output (or type a filename from output/); outputs fbx/obj/usdz
    for DCC or glb for the viewer."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_file": ("STRING", {"default": "", "tooltip": "Mesh path relative to output/ — wire glb_file from a FAL 3D node."}),
                "target_polycount": ("INT", {"default": 30000, "min": 100, "max": 300000, "step": 1000}),
                "topology": (["quad", "triangle"], {"default": "quad"}),
                "output_format": (["glb", "fbx", "obj", "usdz", "stl"], {"default": "fbx"}),
            },
        }

    RETURN_TYPES = MESH_RET_TYPES
    RETURN_NAMES = MESH_RET_NAMES
    FUNCTION = "generate"
    CATEGORY = "FAL/3D"
    OUTPUT_NODE = True

    def generate(self, mesh_file, target_polycount, topology, output_format):
        args = {
            "model_url": _upload_mesh_file(mesh_file),
            "target_polycount": int(target_polycount),
            "topology": topology,
            "target_formats": [output_format],
        }
        return run_mesh("fal-ai/meshy/v5/remesh", args, "meshy_remesh", want_preview=False)


class FalTripoSplat:
    """tripo3d/triposplat — one photo -> 3D Gaussian Splat, $0.05. The splat_3d output
    plugs straight into the core splat nodes: Get Splat -> Transform / Render / Extract
    Mesh from Splat / Create 3D File -> Save 3D Model (interactive viewer). The raw file
    also lands in output/ (splat_file + download_url)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_gaussians": ("INT", {"default": 262144, "min": 32768, "max": 262144, "step": 32,
                                          "tooltip": "FAL caps this at 262144 (the model's native density) — higher values are rejected with a 422."}),
                "num_inference_steps": ("INT", {"default": 20, "min": 1, "max": 50}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional subject mask (1 = object). Baked into the alpha channel before upload, like the local TripoSplat preprocess. Without it FAL removes the background itself."}),
                "output_format": (["ply", "splat"], {"default": "ply",
                                  "tooltip": "ply carries full spherical harmonics — best for Get Splat / editing."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
            },
        }

    RETURN_TYPES = ("FILE_3D", "STRING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("splat_3d", "splat_file", "download_url", "preview", "info")
    FUNCTION = "generate"
    CATEGORY = "FAL/3D"
    OUTPUT_NODE = True

    def generate(self, image, num_gaussians, num_inference_steps, guidance_scale,
                 mask=None, output_format="ply", seed=0):
        require_key()
        args = {
            "image_url": upload_image_rgba(image, mask) if mask is not None else upload_image(image),
            "num_gaussians": min(int(num_gaussians), 262144),  # FAL rejects higher with 422
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "output_format": output_format,
        }
        if seed:
            args["seed"] = int(seed)
        print(f"[FAL] tripo3d/triposplat <- {args}")
        result = fal_client.subscribe("tripo3d/triposplat", arguments=args, with_logs=False)
        node = deep_find(result, "model_mesh")
        url = node.get("url") if isinstance(node, dict) else (node if isinstance(node, str) else None)
        if not url:
            raise RuntimeError(f"no splat url in FAL response: {result}")
        with urllib.request.urlopen(url, timeout=300) as r:
            data = r.read()

        base = os.path.basename(url.split("?")[0]) or f"triposplat.{output_format}"
        if "." not in base:
            base = f"{base}.{output_format}"
        fname = f"triposplat_{base}"
        with open(os.path.join(folder_paths.get_output_directory(), fname), "wb") as f:
            f.write(data)
        download_url = public_download_url(fname)

        splat_3d = None
        if _comfy_types is not None:
            splat_3d = _comfy_types.File3D(io.BytesIO(data), file_format=output_format)
        else:
            print("[FAL] warning: this ComfyUI has no comfy_api File3D — splat_3d output is empty, "
                  "load the saved file from output/ instead")

        preview = blank_image()
        pre = deep_find(result, "preprocessed_image")
        thumb = pre.get("url") if isinstance(pre, dict) else (pre if isinstance(pre, str) else None)
        if thumb:
            preview = url_to_image_tensor(thumb)

        info = f"tripo3d/triposplat -> {fname} ({len(data) / 1_000_000:.2f} MB)  ⬇ {download_url}"
        print(f"[FAL] DONE {info}")
        return (splat_3d, fname, download_url, preview, info)


# ============================================================================ Meshy v7

MESHY_V7_ENDPOINTS = {
    "single image": "meshy/v7/image-to-3d",
    "multi image": "meshy/v7/multi-image-to-3d",
}


class FalMeshyV7:
    """meshy/v7 image→3D — the most complete single call in the pack: geometry, PBR textures,
    game-ready quad topology at a target polycount, humanoid auto-rigging and an animation
    clip, all billed as one job.

    $0.80 bare mesh / $1.20 textured / $1.40 ultra, +$0.20 rigging, +$0.12 animation.
    PBR, quad topology and polycount are free here (unlike Tripo and Hunyuan).

    Two things to know:
      * GLB cannot store quads. With topology=quad the returned .glb is triangulated and the
        real quads live only in the fbx/obj/blend export — that is what `extra_format` fetches,
        and it defaults to fbx as soon as you pick quad.
      * `multi image` takes up to 4 views of the SAME object; extra frames are dropped. It has
        no ultra mode (the price copy on fal.ai claiming otherwise is a page error)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (list(MESHY_V7_ENDPOINTS), {"default": "single image",
                         "tooltip": "'multi image' reconstructs from up to 4 views of one object — a batched IMAGE, or the image_2/3/4 sockets."}),
                "should_texture": ("BOOLEAN", {"default": True, "tooltip": "Off = bare mesh, $0.80 instead of $1.20."}),
                "enable_pbr": ("BOOLEAN", {"default": True,
                               "tooltip": "Metallic / roughness / normal maps on top of base colour. Free. Needs textures on."}),
                "topology": (["triangle", "quad"], {"default": "triangle",
                             "tooltip": "quad = clean edge loops for DCC work. Only visible in the fbx/obj/blend export — GLB is always triangulated."}),
                "target_polycount": ("INT", {"default": 30000, "min": 100, "max": 300000, "step": 1000}),
            },
            "optional": {
                "image_2": ("IMAGE", {"tooltip": "'multi image' mode — another view of the same object."}),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "ultra_mode": ("BOOLEAN", {"default": False,
                               "tooltip": "Finer surface detail, $1.40. 'single image' mode only — the multi-image endpoint has no such field."}),
                "should_remesh": ("BOOLEAN", {"default": True,
                                  "tooltip": "Off = raw triangular mesh, ignoring topology and polycount. Meshy recommends off for maximum geometric quality."}),
                "extra_format": (["none", "fbx", "obj", "usdz", "blend", "stl"], {"default": "none",
                                 "tooltip": "Also download this format from the same billed call. Required to get real quads."}),
                "texture_prompt": ("STRING", {"default": "", "multiline": True,
                                   "tooltip": "Steer the texturing in words. Max 600 chars."}),
                "texture_image": ("IMAGE", {"tooltip": "Steer the texturing from a reference image."}),
                "enable_rigging": ("BOOLEAN", {"default": False,
                                   "tooltip": "+$0.20. Humanoid characters with clear limbs only, and they must be textured."}),
                "pose_mode": (["", "a-pose", "t-pose"], {"default": "",
                              "tooltip": "Strongly recommended when rigging."}),
                "rigging_height_meters": ("FLOAT", {"default": 1.7, "min": 0.01, "max": 100.0, "step": 0.1}),
                "enable_animation": ("BOOLEAN", {"default": False,
                                     "tooltip": "+$0.12. Requires rigging."}),
                "animation_action_id": ("INT", {"default": 0, "min": 0, "max": 696,
                                        "tooltip": "0 = Idle. Full preset list: https://docs.meshy.ai/en/api/animation-library (ids documented up to 489)."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("glb_file", "download_url", "preview", "info",
                    "rigged_glb_file", "animation_glb_file", "base_color")
    FUNCTION = "generate"
    CATEGORY = "FAL/3D"
    OUTPUT_NODE = True

    def generate(self, image, mode, should_texture, enable_pbr, topology, target_polycount,
                 image_2=None, image_3=None, image_4=None, ultra_mode=False, should_remesh=True,
                 extra_format="none", texture_prompt="", texture_image=None,
                 enable_rigging=False, pose_mode="", rigging_height_meters=1.7,
                 enable_animation=False, animation_action_id=0):
        # Pre-flight: these combinations are rejected server-side, and a $1.20 round trip is
        # an expensive way to find that out.
        if enable_animation and not enable_rigging:
            raise RuntimeError("enable_animation requires enable_rigging — turn rigging on (+$0.20)")
        if not should_texture and (enable_pbr or texture_prompt.strip() or texture_image is not None):
            raise RuntimeError("PBR / texture_prompt / texture_image all need should_texture on")
        if enable_rigging and not should_texture:
            raise RuntimeError("Meshy cannot rig untextured meshes — turn should_texture on")

        endpoint = MESHY_V7_ENDPOINTS[mode]
        args = {
            "should_texture": bool(should_texture),
            "enable_pbr": bool(enable_pbr),
            "topology": topology,
            "target_polycount": int(target_polycount),
            "should_remesh": bool(should_remesh),
        }

        if mode == "multi image":
            urls = upload_image_frames(image)
            for extra in (image_2, image_3, image_4):
                if extra is not None:
                    urls.extend(upload_image_frames(extra))
            if len(urls) > 4:
                print(f"[FAL] warning: {len(urls)} views given, Meshy uses only the first 4")
                urls = urls[:4]
            args["image_urls"] = urls
            if ultra_mode:
                print("[FAL] warning: multi-image has no ultra mode — ignoring ultra_mode")
        else:
            args["image_url"] = upload_image(image)
            args["ultra_mode"] = bool(ultra_mode)

        if texture_prompt.strip():
            args["texture_prompt"] = texture_prompt.strip()[:600]
        if texture_image is not None:
            args["texture_image_url"] = upload_image(texture_image)
        if enable_rigging:
            args["enable_rigging"] = True
            args["rigging_height_meters"] = float(rigging_height_meters)
            if pose_mode:
                args["pose_mode"] = pose_mode
            if enable_animation:
                args["enable_animation"] = True
                args["animation_action_id"] = int(animation_action_id)

        if topology == "quad" and extra_format == "none":
            extra_format = "fbx"
            print("[FAL] topology=quad: GLB is always triangulated, also fetching fbx for the real quads")

        require_key()
        print(f"[FAL] {endpoint} <- {args}")
        result = fal_client.subscribe(endpoint, arguments=args, with_logs=False)

        url = deep_find(result, "model_glb")
        url = url.get("url") if isinstance(url, dict) else url
        if not url:
            raise RuntimeError(f"no mesh url in FAL response: {result}")
        fname, download_url, size_mb = save_file(url, "meshy_v7")
        extras = [f"{endpoint} -> {fname} ({size_mb:.2f} MB)  ⬇ {download_url}"]

        if extra_format != "none":
            node = (result.get("model_urls") or {}).get(extra_format)
            xurl = node.get("url") if isinstance(node, dict) else None
            if xurl:
                xname, xdl, xmb = save_file(xurl, f"meshy_v7_{extra_format}")
                extras.append(f"{extra_format}: {xname} ({xmb:.2f} MB)  ⬇ {xdl}")
            else:
                extras.append(f"{extra_format}: not returned by Meshy for this job")

        def _grab(key, prefix):
            node = result.get(key)
            u = node.get("url") if isinstance(node, dict) else None
            if not u:
                return ""
            name, dl, mb = save_file(u, prefix)
            extras.append(f"{key}: {name} ({mb:.2f} MB)  ⬇ {dl}")
            return name

        rigged = _grab("rigged_character_glb", "meshy_v7_rig")
        animated = _grab("animation_glb", "meshy_v7_anim")

        preview = blank_image()
        thumb = result.get("thumbnail")
        thumb = thumb.get("url") if isinstance(thumb, dict) else None
        if thumb:
            preview = url_to_image_tensor(thumb)

        base_color = blank_image()
        tex = result.get("texture_urls")
        if isinstance(tex, list) and tex:
            bc = tex[0].get("base_color") if isinstance(tex[0], dict) else None
            bc_url = bc.get("url") if isinstance(bc, dict) else None
            if bc_url:
                base_color = url_to_image_tensor(bc_url)

        info = "\n".join(extras)
        for line in extras:
            print(f"[FAL] {line}")
        return (fname, download_url, preview, info, rigged, animated, base_color)


# ============================================================================ Tripo utilities

class FalTripoRemesh:
    """tripo3d/tripo/remesh — retopologise a high-poly mesh into a clean low-poly one, $0.01.

    Twenty times cheaper than Meshy v5 Remesh and seventy-five times cheaper than Smart
    Topology, it produces real quads, can bake the existing textures onto the result, and is
    the only one of the three that returns a preview render.

    Its limit is the ceiling: 20000 faces (10000 with quad), so it is a game-asset / LOD
    decimator — it cannot remesh upward. For >20k faces, .blend/.usdz export or rescaling,
    use Meshy v5 Remesh instead.

    Feed `part_names` from Tripo Segment to keep segmented parts separate through the retopo."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_file": ("STRING", {"default": "", "tooltip": "glb/obj/fbx/stl path relative to output/ — wire a glb_file output."}),
                "quad": ("BOOLEAN", {"default": False,
                         "tooltip": "Real quad topology. Forces FBX output and caps face_limit at 10000."}),
                "face_limit": ("INT", {"default": 0, "min": 0, "max": 20000, "step": 500,
                               "tooltip": "0 = adaptive (Tripo decides). Otherwise 500–20000, or 500–10000 with quad."}),
                "bake": ("BOOLEAN", {"default": True, "tooltip": "Bake the existing textures onto the retopo'd mesh."}),
            },
            "optional": {
                "part_names": ("STRING", {"default": "", "multiline": True,
                               "tooltip": "Paste the part_names output of Tripo Segment (one per line) to preserve parts."}),
            },
        }

    RETURN_TYPES = MESH_RET_TYPES
    RETURN_NAMES = MESH_RET_NAMES
    FUNCTION = "generate"
    CATEGORY = "FAL/3D"
    OUTPUT_NODE = True

    def generate(self, mesh_file, quad, face_limit, bake, part_names=""):
        ext = mesh_file.strip().split("?")[0].rsplit(".", 1)[-1].lower()
        if ext not in ("glb", "obj", "fbx", "stl"):
            raise RuntimeError(f"Tripo remesh accepts glb/obj/fbx/stl, got .{ext}")
        limit = int(face_limit)
        if quad and limit > 10000:
            print(f"[FAL] quad caps face_limit at 10000 — clamping {limit}")
            limit = 10000
        args = {"mesh_url": _upload_mesh_file(mesh_file), "quad": bool(quad), "bake": bool(bake)}
        if limit > 0:
            args["face_limit"] = limit
        names = [n.strip() for n in part_names.replace(",", "\n").splitlines() if n.strip()]
        if names:
            args["part_names"] = names
        return run_mesh("tripo3d/tripo/remesh", args, "tripo_remesh")


class FalTripoSegment:
    """tripo3d/tripo/segment — split a mesh into semantic parts, $0.01.

    You get one mesh with the parts separated inside it plus the list of part names — there
    are no per-part files. Wire `part_names` straight into Tripo Remesh to keep those parts
    intact through retopology; the pair costs $0.02 and is the cheapest thing in FAL/3D.

    The preview render usually colour-codes the parts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_file": ("STRING", {"default": "", "tooltip": "glb/obj/fbx/stl path relative to output/ — wire a glb_file output."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("glb_file", "part_names", "download_url", "preview", "info")
    FUNCTION = "generate"
    CATEGORY = "FAL/3D"
    OUTPUT_NODE = True

    def generate(self, mesh_file):
        ext = mesh_file.strip().split("?")[0].rsplit(".", 1)[-1].lower()
        if ext not in ("glb", "obj", "fbx", "stl"):
            raise RuntimeError(f"Tripo segment accepts glb/obj/fbx/stl, got .{ext}")
        require_key()
        args = {"mesh_url": _upload_mesh_file(mesh_file)}
        print(f"[FAL] tripo3d/tripo/segment <- {args}")
        result = fal_client.subscribe("tripo3d/tripo/segment", arguments=args, with_logs=False)

        node = deep_find(result, "model_mesh")
        url = node.get("url") if isinstance(node, dict) else (node if isinstance(node, str) else None)
        if not url:
            raise RuntimeError(f"no mesh url in FAL response: {result}")
        fname, download_url, size_mb = save_file(url, "tripo_seg")

        names = [n for n in (result.get("part_names") or []) if isinstance(n, str)]
        preview = blank_image()
        rendered = deep_find(result, "rendered_image")
        thumb = rendered.get("url") if isinstance(rendered, dict) else None
        if thumb:
            preview = url_to_image_tensor(thumb)

        if names:
            print(f"[FAL] {len(names)} parts: {', '.join(names)}")
            parts_note = f"{len(names)} parts"
        else:
            print("[FAL] Tripo reported no part names for this mesh")
            parts_note = "no part names reported"
        info = (f"tripo3d/tripo/segment -> {fname} ({size_mb:.2f} MB), {parts_note}"
                f"  ⬇ {download_url}")
        print(f"[FAL] DONE {info}")
        return (fname, "\n".join(names), download_url, preview, info)


# ============================================================================ rigging

class FalMeshyRigging:
    """fal-ai/meshy/rigging — auto-rig any humanoid GLB, $0.20 (+$0.12 with an animation).

    Meshy v7 can rig what it just generated, but this rigs a mesh from anywhere: Tripo,
    Hunyuan, TRELLIS, or a file you dropped in output/. Rigging a $0.20 Tripo mesh here costs
    $0.40 all-in instead of $1.40 through a v7 generation.

    Humanoids with clearly defined limbs only, textured, and under 300k faces — run Tripo
    Remesh ($0.01) first if the mesh is denser than that. Basic walk and run cycles come
    included with every rig."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_file": ("STRING", {"default": "", "tooltip": "GLB path relative to output/ — wire a glb_file output."}),
                "height_meters": ("FLOAT", {"default": 1.7, "min": 0.01, "max": 100.0, "step": 0.1,
                                  "tooltip": "Roughly how tall the character is — helps scaling and rig accuracy."}),
            },
            "optional": {
                "enable_animation": ("BOOLEAN", {"default": False, "tooltip": "+$0.12"}),
                "animation_action_id": ("INT", {"default": 0, "min": 0, "max": 696,
                                        "tooltip": "0 = Idle. Preset list: https://docs.meshy.ai/en/api/animation-library"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("rigged_glb_file", "animation_glb_file", "walking_glb_file",
                    "download_url", "info")
    FUNCTION = "generate"
    CATEGORY = "FAL/3D"
    OUTPUT_NODE = True

    def generate(self, mesh_file, height_meters, enable_animation=False, animation_action_id=0):
        ext = mesh_file.strip().split("?")[0].rsplit(".", 1)[-1].lower()
        if ext != "glb":
            raise RuntimeError(
                f"Meshy rigging takes GLB only, got .{ext} — convert with Tripo Remesh ($0.01)")
        require_key()
        args = {"model_url": _upload_mesh_file(mesh_file), "height_meters": float(height_meters)}
        if enable_animation:
            args["enable_animation"] = True
            args["animation_action_id"] = int(animation_action_id)
        print(f"[FAL] fal-ai/meshy/rigging <- {args}")
        result = fal_client.subscribe("fal-ai/meshy/rigging", arguments=args, with_logs=False)

        lines = []

        def _grab(node, prefix, label):
            u = node.get("url") if isinstance(node, dict) else None
            if not u:
                return "", ""
            name, dl, mb = save_file(u, prefix)
            lines.append(f"{label}: {name} ({mb:.2f} MB)  ⬇ {dl}")
            return name, dl

        rigged, rigged_dl = _grab(result.get("rigged_character_glb"), "meshy_rig", "rigged")
        if not rigged:
            raise RuntimeError(f"no rigged character in FAL response: {result}")
        animated, _ = _grab(result.get("animation_glb"), "meshy_anim", "animation")
        walking, _ = _grab((result.get("basic_animations") or {}).get("walking_glb"),
                           "meshy_walk", "walking")

        info = "\n".join(lines)
        for line in lines:
            print(f"[FAL] {line}")
        return (rigged, animated, walking, rigged_dl, info)


NODE_CLASS_MAPPINGS = {
    "FalTripoImageTo3D": FalTripoImageTo3D,
    "FalTripoH31": FalTripoH31,
    "FalHunyuan3D": FalHunyuan3D,
    "FalHunyuan3DV31": FalHunyuan3DV31,
    "FalHunyuanSketchTo3D": FalHunyuanSketchTo3D,
    "FalTrellisImageTo3D": FalTrellisImageTo3D,
    "FalTripoSplat": FalTripoSplat,
    "FalMeshyV7": FalMeshyV7,
    "FalSmartTopology": FalSmartTopology,
    "FalMeshyRemesh": FalMeshyRemesh,
    "FalTripoRemesh": FalTripoRemesh,
    "FalTripoSegment": FalTripoSegment,
    "FalMeshyRigging": FalMeshyRigging,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FalMeshyV7": "FAL 3D — Meshy v7, rig + anim ($0.80–1.72)",
    "FalTripoImageTo3D": "FAL 3D — Tripo v2.5 ($0.20–0.45)",
    "FalTripoH31": "FAL 3D — Tripo H3.1, quality dial ($0.20–0.65)",
    "FalHunyuan3D": "FAL 3D — Hunyuan3D v2, octree ($0.16–0.48)",
    "FalHunyuan3DV31": "FAL 3D — Hunyuan3D v3.1 pro/rapid ($0.225–0.525)",
    "FalHunyuanSketchTo3D": "FAL 3D — Hunyuan Sketch→3D (prompt, $0.375+)",
    "FalTrellisImageTo3D": "FAL 3D — TRELLIS, fine control ($0.02)",
    "FalTripoSplat": "FAL 3D — TripoSplat, Gaussian Splat ($0.05)",
    "FalTripoRemesh": "FAL 3D — Tripo Remesh, smart low-poly ($0.01)",
    "FalTripoSegment": "FAL 3D — Tripo Segment, parts ($0.01)",
    "FalSmartTopology": "FAL 3D — Smart Topology retopo (Hunyuan, $0.75)",
    "FalMeshyRemesh": "FAL 3D — Meshy v5 Remesh, high-poly + formats ($0.20)",
    "FalMeshyRigging": "FAL 3D — Meshy Rigging, any humanoid GLB ($0.20–0.32)",
}
