#!/usr/bin/env python3
"""Generates "Upscale to 1980 (folder + ZIP).json" — the example graph for this pack.

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
    "КАК ПОЛЬЗОВАТЬСЯ\n"
    "1. На первой ноде нажать «📁 Upload folder…» и выбрать папку с фото на своём компьютере "
    "(или перетащить папку прямо на ноду). Фото уедут на сервер в input/<имя папки>/, папка выберется сама.\n"
    "2. Число 1980 — целевая КОРОТКАЯ сторона. Нода «✂️ Split by Short Side» отправляет в Topaz ТОЛЬКО фото "
    "с короткой стороной < 1980 — за остальные FAL не платит вообще, даже если папка смешанная "
    "(обычный Switch так не умеет: в списковом режиме он включил бы Topaz для всех).\n"
    "3. Run. Результат: output/upscaled_1980/<имя файла>_1980.jpg — все фото пересохраняются в JPEG q95, sRGB, "
    "без EXIF. На последней ноде — галерея всех кадров (стрелки/миниатюры) и кнопка «⬇ Download ZIP» "
    "со всеми файлами прогона; та же ссылка видна в «Preview as Text» и остаётся в истории.\n"
    "МОДЕЛЬ: gen · Wonder 3.5 — генеративная, восстанавливает детали на мелких и пожатых исходниках. "
    "Старая precision · Standard V2 на таком материале дорисовывает JPEG-блоки: сетка «в рогожку» на "
    "стенах и восковая кожа. face_enhancement по умолчанию ВЫКЛ — на мелких лицах он и делает воск; "
    "включай, только если лицо крупным планом. Для уже чистых больших фото лучше precision · High Fidelity V3, "
    "для рендеров — precision · CGI, для скринов с текстом — Text Refine.\n"
    "Фактор Topaz считается автоматически (кратно 0.5, от 1x до 4x — предел нового API), затем короткая сторона "
    "подгоняется ровно в 1980 по Lanczos. Если исходник совсем мелкий (короткая сторона < 495 px), 4x не хватает "
    "и остаток добирается ресайзом. EXIF-поворот учитывается, HEIC (iPhone) читается, Display P3 → sRGB.\n"
    "Цена: $0.01 за мегапиксель результата — фото 1980×2640 ≈ $0.07."
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
    node("T", "PrimitiveInt", "Целевая короткая сторона (px)", (80, 80), (270, 82),
         [w("value", "INT")], [out("INT", "INT")], [1980, "fixed"], CORE),
    node("L", "FolderIOLoadImages", "Папка с фото — 📁 Upload folder… или перетащить папку сюда", (80, 250), (440, 200),
         [w("folder", "COMBO"), w("sort_by", "COMBO"), w("start_index", "INT"), w("max_images", "INT")],
         [out("images", "IMAGE", True), out("filenames", "STRING", True), out("count", "INT")],
         ["photos", "name", 0, 0], OURS),
    node("SP", "FolderIOSplitByShortSide", "Кому нужен апскейл? (короткая сторона < цели)", (620, 250), (380, 120),
         [sock("images", "IMAGE"), w("min_short_side", "INT")],
         [out("images_below", "IMAGE", True), out("index_below", "INT", True), out("count_below", "INT"), out("count_ok", "INT")],
         [1980], OURS),
    node("S", "GetImageSize", "Размер исходника", (1100, 80), (140, 66),
         [sock("image", "IMAGE")], [out("width", "INT"), out("height", "INT"), out("batch_size", "INT")], [], CORE),
    node("F", "ComfyMathExpression", "Фактор Topaz (кратно 0.5, 1x–4x)", (1340, 80), (400, 200),
         [sock("values.a", "FLOAT,INT,BOOLEAN", label="a"), sock("values.b", "FLOAT,INT,BOOLEAN", True, "b"),
          sock("values.c", "FLOAT,INT,BOOLEAN", True, "c"), sock("values.d", "FLOAT,INT,BOOLEAN", True, "d"),
          w("expression", "STRING")],
         [out("FLOAT", "FLOAT"), out("INT", "INT"), {"localized_name": "BOOL", "name": "BOOL", "type": "BOOLEAN", "links": []}],
         ["max(1, min(4, ceil(c / min(a, b) * 2) / 2))"], CORE),
    node("TP", "FalTopazUpscale2026", "Topaz Wonder 3.5 (FAL, $0.01/MP — только для images_below)", (1840, 80), (470, 480),
         [sock("image", "IMAGE"), w("model", "COMBO"), w("upscale_factor", "FLOAT"), w("face_enhancement", "BOOLEAN"),
          w("face_strength", "FLOAT"), w("face_creativity", "FLOAT"), w("enhancement_strength", "COMBO"),
          w("subject_detection", "COMBO"), w("creativity", "INT"), w("texture", "INT"), w("detail", "FLOAT"),
          w("denoise", "FLOAT"), w("sharpen", "FLOAT"), w("fix_compression", "FLOAT"), w("strength", "FLOAT"),
          w("prompt", "STRING"), w("autoprompt", "COMBO"), w("color_preservation", "COMBO")],
         [out("image", "IMAGE")],
         ["gen · Wonder 3.5 (лучший универсал)", 2, False, 0.8, 0.0, "auto", "auto", 0, 0,
          -1.0, -1.0, -1.0, -1.0, -1.0, "", "auto", "auto"], FAL),
    node("U", "ResizeImagesByShorterEdge", "Подгон короткой стороны ровно в цель", (2410, 80), (310, 58),
         [sock("images", "IMAGE"), w("shorter_edge", "INT")], [out("images", "IMAGE")], [512], CORE),
    node("M", "FolderIOMergeSubset", "Вернуть апскейленные на свои места", (2820, 250), (330, 90),
         [sock("images", "IMAGE"), sock("index", "INT"), sock("replacements", "IMAGE", True)],
         [out("images", "IMAGE", True)], [], OURS),
    node("O", "FolderIOSaveZip", "Сохранить в output/upscaled_1980 + ZIP", (3250, 250), (400, 320),
         [sock("images", "IMAGE"), w("folder", "STRING"), w("format", "COMBO"), w("quality", "INT"), w("suffix", "STRING"),
          w("zip_name", "STRING"), w("overwrite", "BOOLEAN"), sock("filenames", "STRING", True)],
         [out("download_url", "STRING"), out("saved_files", "STRING")],
         ["upscaled_1980", "jpg", 95, "_1980", "", True], OURS),
    node("PA", "PreviewAny", "Ссылка на ZIP (остаётся в истории)", (3750, 250), (400, 120),
         [sock("source", "*")], [out("STRING", "STRING")], [], CORE),
    node(1, "Note", "Инструкция", (80, -420), (1100, 340), [], [], [NOTE], {}, color="#432", bgcolor="#653"),
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
