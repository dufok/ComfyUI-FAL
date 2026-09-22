"""Offline test for D5MaterialSelect (d5_nodes.py) — no ComfyUI, no key, no network.

    python tests/test_d5_select.py

Synthetic D5 channels: a wall, two objects sharing one material, a disco ball whose tiles are split
by a 1-px grid of another material, an acrylic cube that exists only in the Transparent channel
(Material ID shows the wall behind it) with a logo stuck on it, and a ring around a big patch.
Needs numpy + scipy. With torch installed it also runs the node end to end; without it, torch is
stubbed and only the numpy core is checked.
"""
import importlib.util
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    t = types.ModuleType("torch"); nn = types.ModuleType("torch.nn"); f = types.ModuleType("torch.nn.functional")
    t.nn = nn; nn.functional = f
    sys.modules.update({"torch": t, "torch.nn": nn, "torch.nn.functional": f})

spec = importlib.util.spec_from_file_location("d5_nodes", os.path.join(HERE, "..", "d5_nodes.py"))
d5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d5)

WALL, SHARED, TILE, GRID, RING, INSIDE = (102, 0, 0), (255, 0, 51), (255, 204, 102), (138, 204, 24), (0, 80, 255), (40, 40, 40)
CUBE = (255, 153, 153)
H = W = 120
mid = np.zeros((H, W, 3), np.uint8); mid[:] = WALL
mid[10:30, 10:30] = SHARED                       # object A
mid[10:30, 90:110] = SHARED                      # object B — same material, elsewhere
mid[40:70, 40:70] = TILE; mid[40:70, 49] = GRID; mid[40:70, 59] = GRID; mid[49, 40:70] = GRID; mid[59, 40:70] = GRID
mid[80:115, 80:115] = RING; mid[84:111, 84:111] = INSIDE   # ring with a large patch of another material inside
tr = np.zeros((H, W, 3), np.uint8)
tr[75:115, 10:40] = CUBE; tr[90:96, 20:26] = 0   # acrylic cube, logo sticker = opaque hole

def dot(y, x, r=2):
    yy, xx = np.ogrid[:H, :W]
    return (yy - y) ** 2 + (xx - x) ** 2 <= r * r

fails = 0
def check(name, cond):
    global fails
    print(("ok   " if cond else "FAIL ") + name)
    fails += 0 if cond else 1

ids = d5.build_ids(mid, tr)

m, info = d5.select_np(ids, dot(20, 20), grow=0, feather=0)
check("touched object only, not its twin with the same material", m[20, 20] > .5 and m[20, 100] < .5)
m, _ = d5.select_np(ids, dot(20, 20), only_touched_parts=False, grow=0, feather=0)
check("only_touched_parts=False takes every object of the material", m[20, 20] > .5 and m[20, 100] > .5)

m, _ = d5.select_np(ids, dot(44, 44), grow=0, feather=0)
check("disco ball: whole ball across the 1-px grid", m[44, 44] > .5 and m[65, 65] > .5 and m[49, 55] > .5 and m[59, 45] > .5)
m, _ = d5.select_np(ids, dot(44, 44), bridge=0, grow=0, feather=0)
check("disco ball without bridge: only the touched tile", m[44, 44] > .5 and m[65, 65] < .5)

m, info = d5.select_np(ids, dot(80, 15), grow=0, feather=0)
check("acrylic cube via Transparent, logo hole filled", m[80, 15] > .5 and m[93, 23] > .5 and m[110, 35] > .5)
check("  ...and the wall behind it NOT selected", m[5, 60] < .5 and "1 transparent" in info)
m_no_tr, _ = d5.select_np(d5.build_ids(mid, None), dot(80, 15), grow=0, feather=0)
check("without Transparent the stroke falls through to the wall (why the layer exists)", m_no_tr[5, 60] > .5)

m, _ = d5.select_np(ids, dot(82, 82), grow=0, feather=0)
check("big enclosed patch is not filled (fill_holes_upto)", m[82, 82] > .5 and m[97, 97] < .5)
m, _ = d5.select_np(ids, dot(82, 82), fill_holes_upto=10.0, grow=0, feather=0)
check("  ...unless the limit allows it", m[97, 97] > .5)

stroke = dot(20, 20, r=6); stroke[20, 14:31] = True   # brush grazing the wall around object A
m, _ = d5.select_np(ids, stroke, min_share=0.5, grow=0, feather=0)
check("min_share drops a material the brush only grazed", m[20, 20] > .5 and m[5, 60] < .5)

m, info = d5.select_np(ids, np.zeros((H, W), bool))
check("no strokes -> empty mask + hint", m.max() == 0 and info.startswith("no strokes"))

m, _ = d5.select_np(ids, dot(20, 20), grow=3, feather=2)
check("grow + feather give a soft 0..1 edge", 0 < m[7, 20] < 1 and m[20, 20] == 1.0 and m.dtype == np.float32)

if HAVE_TORCH:
    node = d5.D5MaterialSelect()
    img = torch.from_numpy(mid.astype(np.float32) / 255)[None]
    trn = torch.from_numpy(tr.astype(np.float32) / 255)[None]
    strokes = torch.from_numpy(dot(80, 15).astype(np.float32))[None]
    mask, prev, info = node.run(img, strokes, 12, 0.02, True, 4, 0.3, 3, 2, transparent=trn)
    check("node: MASK [1,H,W], preview [1,H,W,3]", tuple(mask.shape) == (1, H, W) and tuple(prev.shape) == (1, H, W, 3))
    empty = torch.zeros((64, 64))              # Load Image without alpha hands out 64x64 zeros
    mask, _, info = node.run(img, empty, 12, 0.02, True, 4, 0.3, 3, 2)
    check("node: 64x64 empty mask from Load Image -> no strokes", float(mask.max()) == 0 and info.startswith("no strokes"))
else:
    print("skip  node end-to-end (no torch here)")

print(f"\n{'ALL PASSED' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
