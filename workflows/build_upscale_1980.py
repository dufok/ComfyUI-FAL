#!/usr/bin/env python3
"""Generates "Upscale to 1980 (folder + ZIP).json" — the example graph for the folder nodes.

    photos (folder upload) ─┬─► Split by Short Side ─► GetImageSize ─► factor ─► Topaz (FAL) ─► resize ─┐
                            │          │ index_below                                                     │
                            └──────────┴──────────────────────────► Merge Subset (lazy replacements) ◄───┘
                                                                            │
                                                                            ▼
                                                             Save Images + ZIP ─► Preview as Text (link)

Only photos whose short side is below the target reach the paid upscaler; the merge's lazy input
keeps the whole Topaz branch unexecuted when nothing needs work.
"""
import json
import os
import uuid

CORE = {"cnr_id": "comfy-core", "ver": "0.23.0"}
FAL = {"aux_id": "dufok/ComfyUI-FAL", "ver": "229c61a6cf8bc8e00cbfc6b7e81235be5cf58fbf"}
OURS = {"aux_id": "dufok/ComfyUI-FAL", "ver": "0.1.0"}

NOTE = (
    "HOW TO USE\n"
    "1. On the first node press \u201c\U0001F4C1 Upload folder\u2026\u201d and pick a folder of photos on your own "
    "computer, or just drag the folder onto the node. The files go to input/<folder>/ on the server and the "
    "folder is selected for you.\n"
    "2. 1980 is the target SHORT side. \u201c\u2702\ufe0f Split by Short Side\u201d sends ONLY the photos below it "
    "to Topaz, so FAL is not paid a cent for the ones that are already big enough, even in a mixed folder. "
    "A plain Switch cannot do this: in list mode it would run Topaz on every photo and throw the extras away.\n"
    "3. Run. Results land in output/upscaled_1980/<name>_1980.jpg, re-encoded to JPEG q95 in sRGB without EXIF. "
    "The last node shows the whole batch as a gallery and gives you a \u201c\u2b07 Download ZIP\u201d button; the "
    "same link appears in Preview as Text and stays in the queue history.\n"
    "MODEL: gen \u00b7 Wonder 3.5 is generative and rebuilds detail on small or compressed sources. The older "
    "precision \u00b7 Standard V2 sharpens the JPEG blocks themselves on such photos: a crosshatch pattern on "
    "walls and waxy skin. face_enhancement is OFF by default because on small faces that pass is what makes "
    "them waxy; turn it on only for close-ups. For clean, already large photos prefer precision \u00b7 High "
    "Fidelity V3, for renders precision \u00b7 CGI, for screenshots with text Text Refine.\n"
    "The Topaz factor is computed automatically in steps of 0.5, from 1x to 4x, which is the new API ceiling; "
    "the short side is then matched exactly with Lanczos. If a source is tiny, under 495 px on the short side, "
    "4x is not enough and the rest is covered by the resize. EXIF rotation is honoured, iPhone HEIC is read, "
    "Display P3 is converted to sRGB.\n"
    "Cost: $0.01 per output megapixel, so a 1980x2640 photo is about $0.07."
)


def w(name, typ):
    return {"localized_name": name, "name": name, "type": typ, "widget": {"name": name}, "link": None}


def sock(name, typ, optional=False, label=None, localized=None):
    d = {"localized_name": localized or name, "name": name, "type": typ, "link": None}
    if optional:
        d["shape"] = 7
    if label:
        d["label"] = label
    return d


def out(name, typ, is_list=False):
    d = {"localized_name": name, "name": name, "type": typ, "links": []}
    if is_list:
        d["shape"] = 6
    return d


def node(nid, typ, title, pos, size, inputs, outputs, widgets, props, **extra):
    d = {"id": nid, "type": typ, "pos": list(pos), "size": list(size), "flags": {}, "order": 0, "mode": 0,
         "inputs": inputs, "outputs": outputs, "title": title,
         "properties": {**props, "Node name for S&R": typ}, "widgets_values": widgets}
    d.update(extra)
    return d


nodes = [
    node("T", "PrimitiveInt", "Target short side (px)", (80, 80), (270, 82),
         [w("value", "INT")], [out("INT", "INT")], [1980, "fixed"], CORE),
    node("L", "FolderIOLoadImages", "Photo folder — 📁 Upload folder… or drop a folder here", (80, 250), (440, 200),
         [w("folder", "COMBO"), w("sort_by", "COMBO"), w("start_index", "INT"), w("max_images", "INT")],
         [out("images", "IMAGE", True), out("filenames", "STRING", True), out("count", "INT")],
         ["photos", "name", 0, 0], OURS),
    node("SP", "FolderIOSplitByShortSide", "Which photos need upscaling? (short side < target)", (620, 250), (380, 120),
         [sock("images", "IMAGE"), w("min_short_side", "INT")],
         [out("images_below", "IMAGE", True), out("index_below", "INT", True), out("count_below", "INT"), out("count_ok", "INT")],
         [1980], OURS),
    node("S", "GetImageSize", "Source size", (1100, 80), (140, 66),
         [sock("image", "IMAGE")], [out("width", "INT"), out("height", "INT"), out("batch_size", "INT")], [], CORE),
    node("F", "ComfyMathExpression", "Topaz factor (steps of 0.5, 1x–4x)", (1340, 80), (400, 200),
         [sock("values.a", "FLOAT,INT,BOOLEAN", label="a"), sock("values.b", "FLOAT,INT,BOOLEAN", True, "b"),
          sock("values.c", "FLOAT,INT,BOOLEAN", True, "c"), sock("values.d", "FLOAT,INT,BOOLEAN", True, "d"),
          w("expression", "STRING")],
         [out("FLOAT", "FLOAT"), out("INT", "INT"), {"localized_name": "BOOL", "name": "BOOL", "type": "BOOLEAN", "links": []}],
         ["max(1, min(4, ceil(c / min(a, b) * 2) / 2))"], CORE),
    node("TP", "FalTopazUpscale2026", "Topaz Wonder 3.5 (FAL, $0.01/MP — images_below only)", (1840, 80), (470, 480),
         [sock("image", "IMAGE"), w("model", "COMBO"), w("upscale_factor", "FLOAT"), w("face_enhancement", "BOOLEAN"),
          w("face_strength", "FLOAT"), w("face_creativity", "FLOAT"), w("enhancement_strength", "COMBO"),
          w("subject_detection", "COMBO"), w("creativity", "INT"), w("texture", "INT"), w("detail", "FLOAT"),
          w("denoise", "FLOAT"), w("sharpen", "FLOAT"), w("fix_compression", "FLOAT"), w("strength", "FLOAT"),
          w("prompt", "STRING"), w("autoprompt", "COMBO"), w("color_preservation", "COMBO")],
         [out("image", "IMAGE")],
         ["gen · Wonder 3.5 (best all-round)", 2, False, 0.8, 0.0, "auto", "auto", 0, 0,
          -1.0, -1.0, -1.0, -1.0, -1.0, "", "auto", "auto"], FAL),
    node("U", "ResizeImagesByShorterEdge", "Match the short side to the target", (2410, 80), (310, 58),
         [sock("images", "IMAGE"), w("shorter_edge", "INT")], [out("images", "IMAGE")], [512], CORE),
    node("M", "FolderIOMergeSubset", "Put the upscaled ones back in place", (2820, 250), (330, 90),
         [sock("images", "IMAGE"), sock("index", "INT"), sock("replacements", "IMAGE", True)],
         [out("images", "IMAGE", True)], [], OURS),
    node("O", "FolderIOSaveZip", "Save to output/upscaled_1980 + ZIP", (3250, 250), (400, 320),
         [sock("images", "IMAGE"), w("folder", "STRING"), w("format", "COMBO"), w("quality", "INT"), w("suffix", "STRING"),
          w("zip_name", "STRING"), w("overwrite", "BOOLEAN"), sock("filenames", "STRING", True)],
         [out("download_url", "STRING"), out("saved_files", "STRING")],
         ["upscaled_1980", "jpg", 95, "_1980", "", True], OURS),
    node("PA", "PreviewAny", "ZIP link (kept in the queue history)", (3750, 250), (400, 120),
         [sock("source", "*")], [out("STRING", "STRING")], [], CORE),
    node(1, "Note", "How to use", (80, -420), (1100, 340), [], [], [NOTE], {}, color="#432", bgcolor="#653"),
]
nodes[-1]["properties"] = {"text": NOTE}
by_id = {n["id"]: n for n in nodes}
links = []


def link(src, src_slot, dst, dst_name):
    s, d = by_id[src], by_id[dst]
    idx = next(i for i, inp in enumerate(d["inputs"]) if inp["name"] == dst_name)
    lid = len(links) + 1
    typ = s["outputs"][src_slot]["type"]
    links.append([lid, src, src_slot, dst, idx, typ])
    s["outputs"][src_slot]["links"].append(lid)
    d["inputs"][idx]["link"] = lid


link("T", 0, "SP", "min_short_side")
link("T", 0, "F", "values.c")
link("T", 0, "U", "shorter_edge")
link("L", 0, "SP", "images")
link("L", 0, "M", "images")
link("L", 1, "O", "filenames")
link("SP", 0, "S", "image")
link("SP", 0, "TP", "image")
link("SP", 1, "M", "index")
link("S", 0, "F", "values.a")
link("S", 1, "F", "values.b")
link("F", 0, "TP", "upscale_factor")
link("TP", 0, "U", "images")
link("U", 0, "M", "replacements")
link("M", 0, "O", "images")
link("O", 0, "PA", "source")

for n in nodes:
    for o in n["outputs"]:
        if not o["links"]:
            o["links"] = None
for i, n in enumerate(nodes):
    n["order"] = i

wf = {"id": str(uuid.uuid4()), "revision": 0, "last_node_id": 1, "last_link_id": len(links), "nodes": nodes,
      "links": links, "groups": [], "config": {}, "extra": {"ds": {"scale": 0.6, "offset": [40, 380]}}, "version": 0.4}
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Upscale to 1980 (folder + ZIP).json")
with open(dst, "w", encoding="utf-8") as fh:
    json.dump(wf, fh, ensure_ascii=False, indent=1)
print("written", dst, "nodes", len(nodes), "links", len(links))
