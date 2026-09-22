"""mask/D5 — masks from D5 Render channel maps. No API calls, no FAL key.

D5MaterialSelect
    A few brush strokes on the objects you want to keep (MaskEditor on the beauty render) plus
    D5's Material ID channel (+ the Transparent channel) -> an exact mask of every material the
    strokes touched. Built for the "AI re-renders the space, the render keeps the objects" loop:
    a Flux img2img pass rewrites the room, and the brand objects are composited back from D5
    pixel for pixel.

    Why each step exists:
    * Transparent is laid over Material ID as its own layer (4th channel). Acrylic and glass are
      invisible in Material ID — D5 shows what is behind them — so a stroke on an acrylic cube
      would otherwise select the brick wall behind it. D5 paints every transparent material in
      its own colour (cubes and windows differ), so they stay separable.
    * min_share: a material counts only if it covers that share of the strokes — anti-aliased
      edge colours the brush grazes do not drag in a neighbour. The dominant one always counts.
    * only_touched_parts: Material ID is per material, not per object. Only the connected pieces
      the strokes hit are kept, not every object sharing that material across the frame.
    * bridge: connectivity and closing across hairline gaps — the grid of a disco ball, seams.
    * fill_holes_upto: fills only SMALL holes (logos stuck on acrylic), as % of the frame, so a
      neighbouring object enclosed by the selection is never filled in wholesale.

    Needs numpy + scipy (both ship with ComfyUI).
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

log = logging.getLogger(__name__)

_SQ = np.ones((3, 3), bool)


def _img_np(t, hw=None):
    """IMAGE [B,H,W,C] float -> HxWx3 uint8 (first frame), optional nearest resize to (H, W)."""
    x = t[:1, ..., :3]
    if hw is not None and tuple(x.shape[1:3]) != tuple(hw):
        x = F.interpolate(x.movedim(-1, 1), size=hw, mode="nearest").movedim(1, -1)
    return (x[0].detach().cpu().float().clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)


def _mask_np(m, hw):
    """MASK [B,H,W] or [H,W] -> HxW bool, nearest-resized to (H, W). Load Image hands out a
    64x64 zero mask for images without alpha — that resizes to 'no strokes'."""
    x = m if m.ndim == 3 else m[None]
    x = x[:1, None].float()
    if tuple(x.shape[-2:]) != tuple(hw):
        x = F.interpolate(x, size=hw, mode="nearest")
    return x[0, 0].detach().cpu().numpy() > 0.5


def build_ids(material_id, transparent=None, trans_threshold=40):
    """HxWx3 uint8 Material ID (+ optional Transparent) -> HxWx4 int32 ID map."""
    H, W, _ = material_id.shape
    ids = np.zeros((H, W, 4), np.int32)
    ids[..., :3] = material_id
    if transparent is not None:
        tm = transparent.max(-1) > trans_threshold
        ids[tm, :3] = transparent[tm]
        ids[tm, 3] = 255
    return ids


def select_np(ids, strokes, tolerance=12, min_share=0.02, only_touched_parts=True, bridge=4,
              fill_holes_upto=0.3, grow=3, feather=2):
    """ids: HxWx4 int ID map, strokes: HxW bool. Returns (mask float32 HxW in 0..1, info str)."""
    H, W, _ = ids.shape
    n_s = int(strokes.sum())
    if n_s == 0:
        return (np.zeros((H, W), np.float32),
                "no strokes: right-click the beauty Load Image -> Open in MaskEditor -> paint the objects -> Save")
    flat = ids.reshape(-1, 4)
    codes = ((flat[:, 0].astype(np.int64) << 24) | (flat[:, 1] << 16) | (flat[:, 2] << 8)
             | flat[:, 3]).reshape(H, W)
    sel, cnt = np.unique(codes[strokes], return_counts=True)
    keep = cnt >= max(1, int(min_share * n_s))
    keep[np.argmax(cnt)] = True
    sel = sel[keep]
    cols = np.stack([(sel >> 24) & 255, (sel >> 16) & 255, (sel >> 8) & 255, sel & 255], 1)

    m = np.zeros(H * W, bool)
    for c in cols:                                   # Chebyshev distance over the 4 channels
        m |= np.abs(flat - c).max(1) <= tolerance
    m = m.reshape(H, W)

    if only_touched_parts:
        mb = ndimage.binary_dilation(m, _SQ, iterations=bridge) if bridge > 0 else m
        lab, _ = ndimage.label(mb, structure=_SQ)
        hit = np.unique(lab[strokes & mb])
        hit = hit[hit > 0]
        m = m & np.isin(lab, hit)

    if fill_holes_upto > 0:
        if bridge > 0:
            m = ndimage.binary_closing(m, _SQ, iterations=bridge)
        holes = ndimage.binary_fill_holes(m) & ~m
        hl, n = ndimage.label(holes)
        if n:
            sizes = ndimage.sum(holes, hl, index=np.arange(1, n + 1))
            small = np.zeros(n + 1, bool)
            small[1:] = sizes <= fill_holes_upto / 100.0 * H * W
            m = m | small[hl]

    if grow > 0:
        m = ndimage.binary_dilation(m, _SQ, iterations=grow)
    mask = m.astype(np.float32)
    if feather > 0:
        mask = np.clip(ndimage.gaussian_filter(mask, sigma=feather), 0.0, 1.0).astype(np.float32)

    n_tr = int((cols[:, 3] > 0).sum())
    info = (f"materials: {len(cols)} ({n_tr} transparent) | "
            f"mask {int(m.sum())} px = {m.mean() * 100:.1f}% of frame")
    return mask, info


class D5MaterialSelect:
    DESCRIPTION = ("Brush strokes on objects + D5 Material ID (+ Transparent) -> a mask of the whole "
                   "materials the strokes touched. Paint the strokes in MaskEditor on the beauty render.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "material_id": ("IMAGE", {"tooltip": "D5 Material ID channel"}),
                "strokes": ("MASK", {"tooltip": "Strokes from MaskEditor — the MASK output of Load Image"}),
                "tolerance": ("INT", {"default": 12, "min": 0, "max": 128,
                                      "tooltip": "Colour tolerance on the ID map. Raise it if an object comes out patchy"}),
                "min_share": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.005,
                                        "tooltip": "A material counts only if it covers at least this share of the strokes (drops edges the brush grazed)"}),
                "only_touched_parts": ("BOOLEAN", {"default": True,
                                                   "tooltip": "Only the pieces the strokes hit, not every object with that material in the frame"}),
                "bridge": ("INT", {"default": 4, "min": 0, "max": 32,
                                   "tooltip": "Bridge hairline gaps, px (disco-ball grid, seams)"}),
                "fill_holes_upto": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 10.0, "step": 0.05,
                                              "tooltip": "Fill holes up to this size, % of frame (logos on acrylic). 0 = no filling"}),
                "grow": ("INT", {"default": 3, "min": 0, "max": 64, "tooltip": "Grow the mask, px"}),
                "feather": ("INT", {"default": 2, "min": 0, "max": 32, "tooltip": "Feather the edge, px"}),
            },
            "optional": {
                "transparent": ("IMAGE", {"tooltip": "D5 Transparent channel — needed for acrylic and glass"}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("mask", "preview", "info")
    OUTPUT_TOOLTIPS = ("The object mask",
                       "Material ID dimmed, selection in pink, strokes in green",
                       "How many materials were taken and how much of the frame")
    FUNCTION = "run"
    CATEGORY = "mask/D5"

    def run(self, material_id, strokes, tolerance, min_share, only_touched_parts, bridge,
            fill_holes_upto, grow, feather, transparent=None):
        mid = _img_np(material_id)
        hw = mid.shape[:2]
        tr = _img_np(transparent, hw) if transparent is not None else None
        st = _mask_np(strokes, hw)
        mask, info = select_np(build_ids(mid, tr), st, tolerance, min_share, only_touched_parts,
                               bridge, fill_holes_upto, grow, feather)
        log.info("[D5MaterialSelect] %s", info)
        base = mid.astype(np.float32) / 255.0
        hl = np.array([1.0, 0.24, 0.9], np.float32)
        prev = base * 0.25 * (1 - mask[..., None]) + (base * 0.4 + hl * 0.6) * mask[..., None]
        prev = np.where(st[..., None], np.array([0.1, 1.0, 0.3], np.float32), prev)
        return (torch.from_numpy(mask)[None], torch.from_numpy(prev.astype(np.float32))[None], info)


NODE_CLASS_MAPPINGS = {"D5MaterialSelect": D5MaterialSelect}
NODE_DISPLAY_NAME_MAPPINGS = {"D5MaterialSelect": "🎯 D5 Material Select (strokes → mask)"}
