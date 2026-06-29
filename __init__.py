from .fal_3d_nodes import (
    FalTripoImageTo3D,
    FalHunyuan3D,
    FalTrellisImageTo3D,
)

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

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
