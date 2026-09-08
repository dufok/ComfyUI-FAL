"""Functional smoke test for the folder-IO nodes — runs INSIDE the ComfyUI image, no server, no FAL key.

    docker run --rm -e PYTHONDONTWRITEBYTECODE=1 \
      -v /path/to/ComfyUI-FAL:/app/custom_nodes/ComfyUI-FAL:ro \
      --entrypoint sh vfx-comfyui:cpu -c \
      "pip install -q pillow-heif; python /app/custom_nodes/ComfyUI-FAL/tests/smoke_in_container.py"
"""
import importlib
import json
import os
import shutil
import sys
import zipfile

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/custom_nodes")
os.chdir("/app")

import folder_paths  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = "/tmp/folderio-smoke"
shutil.rmtree(ROOT, ignore_errors=True)
IN, OUT = os.path.join(ROOT, "input"), os.path.join(ROOT, "output")
os.makedirs(os.path.join(IN, "smoke"), exist_ok=True)
os.makedirs(OUT, exist_ok=True)
folder_paths.set_input_directory(IN)
folder_paths.set_output_directory(OUT)

pack = importlib.import_module("ComfyUI-FAL")
mod = sys.modules["ComfyUI-FAL.folderio_nodes"]
Load = pack.NODE_CLASS_MAPPINGS["FolderIOLoadImages"]
Save = pack.NODE_CLASS_MAPPINGS["FolderIOSaveZip"]
print("HEIF_OK =", mod.HEIF_OK)

# --- fixtures: 3 plain jpgs (natural-sort names), 1 EXIF-rotated jpg, 1 heic, 1 junk, 1 dotfile
d = os.path.join(IN, "smoke")
for name, size in (("photo_10.jpg", (64, 48)), ("photo_2.jpg", (64, 48)), ("photo_1.jpg", (64, 48))):
    Image.new("RGB", size, (200, 100, 50)).save(os.path.join(d, name), quality=90)
ex = Image.Exif()
ex[0x0112] = 6  # rotate 90 CW on display -> loader must return 48x64 (H=64, W=48)
Image.new("RGB", (64, 48), (10, 200, 10)).save(os.path.join(d, "rotated.jpg"), exif=ex.tobytes(), quality=90)
if mod.HEIF_OK:
    Image.new("RGB", (32, 24), (10, 10, 200)).save(os.path.join(d, "iphone.heic"), format="HEIF", quality=80)
open(os.path.join(d, "notes.txt"), "w").write("junk")
open(os.path.join(d, ".DS_Store"), "wb").write(b"\0")
open(os.path.join(d, "empty.jpg"), "wb").close()

# --- loader
assert "smoke" in Load.INPUT_TYPES()["required"]["folder"][0]
assert Load.VALIDATE_INPUTS("smoke") is True
assert isinstance(Load.VALIDATE_INPUTS("nope"), str), "missing folder must fail validation"
assert isinstance(Load.VALIDATE_INPUTS("../"), str) and isinstance(Load.VALIDATE_INPUTS(""), str)
h1 = Load.IS_CHANGED("smoke", "name", 0, 0)
imgs, stems, n = Load().load("smoke", "name", 0, 0)
print("loaded", n, stems, [tuple(t.shape) for t in imgs])
expect = ["iphone", "photo_1", "photo_2", "photo_10", "rotated"] if mod.HEIF_OK else ["photo_1", "photo_2", "photo_10", "rotated"]
assert stems == expect, stems
rot = imgs[stems.index("rotated")]
assert tuple(rot.shape) == (1, 64, 48, 3), f"EXIF orientation not applied: {tuple(rot.shape)}"
assert all(t.dtype.is_floating_point and 0.0 <= float(t.min()) and float(t.max()) <= 1.0 for t in imgs)
imgs2, stems2, n2 = Load().load("smoke", "name", 1, 2)
assert stems2 == expect[1:3], stems2
Image.new("RGB", (8, 8)).save(os.path.join(d, "photo_3.jpg"))
assert Load.IS_CHANGED("smoke", "name", 0, 0) != h1, "IS_CHANGED must track folder content"

# --- saver: jpg + suffix + zip, names from loader
r = Save().save(imgs, ["upscaled_1980"], ["jpg"], [95], ["_1980"], [""], [True],
                filenames=stems, prompt=[{}], extra_pnginfo=[None])
z = r["ui"]["zip"][0]
print("zip", z, "\nurl", r["result"][0])
assert len(r["ui"]["images"]) == n and r["ui"]["images"][0]["subfolder"] == "upscaled_1980"
names = zipfile.ZipFile(os.path.join(OUT, "upscaled_1980", z["filename"])).namelist()
assert names == [s + "_1980.jpg" for s in stems], names
assert z["url"].startswith("/view?filename=upscaled_1980.zip&subfolder=upscaled_1980&type=output")
with Image.open(os.path.join(OUT, "upscaled_1980", "rotated_1980.jpg")) as im:
    assert im.size == (48, 64), im.size

# --- saver: png with metadata, nested folder, no overwrite -> (1) suffix, numbering without filenames
r2 = Save().save(imgs[:1], ["upscaled_1980/nested"], ["png"], [95], [""], ["bundle"], [False],
                 prompt=[{"1": {"class_type": "X"}}], extra_pnginfo=[{"workflow": {"a": 1}}])
r3 = Save().save(imgs[:1], ["upscaled_1980/nested"], ["png"], [95], [""], ["bundle"], [False])
assert r2["ui"]["images"][0]["filename"] == "photo_00001.png"
assert r3["ui"]["images"][0]["filename"] == "photo_00001 (1).png", r3["ui"]["images"]
with Image.open(os.path.join(OUT, "upscaled_1980/nested/photo_00001.png")) as im:
    assert json.loads(im.text["workflow"]) == {"a": 1}
assert r2["ui"]["zip"][0]["url"] == "/view?filename=bundle.zip&subfolder=upscaled_1980/nested&type=output"

# --- saver: path traversal / odd names are neutralised, never escape output/
r4 = Save().save(imgs[:1], ["../../etc/../evil"], ["webp"], [80], ["/x"], ["../z"], [True], filenames=["../../passwd"])
print("traversal ->", r4["ui"]["images"], r4["ui"]["zip"][0]["filename"])
assert set(os.listdir(OUT)) == {"upscaled_1980", "etc"}, os.listdir(OUT)  # ".." dropped, stays under output/
assert os.path.isdir(os.path.join(OUT, "etc", "evil"))
assert r4["ui"]["images"][0]["filename"] == "passwdx.webp"

# --- saver: batched tensor input (B=2) + per-item names
import torch  # noqa: E402
b2 = torch.cat([imgs[0], imgs[0]], dim=0)
r5 = Save().save([b2, imgs[1]], ["batchy"], ["jpg"], [80], [""], [""], [True], filenames=["a", "b"])
assert [i["filename"] for i in r5["ui"]["images"]] == ["a_01.jpg", "a_02.jpg", "b.jpg"], r5["ui"]["images"]

# --- saver: nothing to save
try:
    Save().save([], ["x"], ["jpg"], [95], [""], [""], [True])
    raise SystemExit("expected ValueError for empty input")
except ValueError as e:
    print("empty ->", e)

# --- _safe_stem hardening: interior '..' (rejected by /view) collapses, length is bounded
assert mod._safe_stem("a..b") == "a.b" and mod._safe_stem("..x..") == "x" and mod._safe_stem("v1...final") == "v1.final"
long = mod._safe_stem("\u041b" * 300)
assert long and len(long.encode("utf-8")) <= mod.MAX_STEM_BYTES

# --- overwrite=False also protects the previous ZIP; no .part file left behind
r6 = Save().save(imgs[:1], ["twice"], ["jpg"], [90], [""], ["bundle"], [False], filenames=["k"])
r7 = Save().save(imgs[:1], ["twice"], ["jpg"], [90], [""], ["bundle"], [False], filenames=["k"])
assert r6["ui"]["zip"][0]["filename"] == "bundle.zip" and r7["ui"]["zip"][0]["filename"] == "bundle (1).zip"
assert zipfile.ZipFile(os.path.join(OUT, "twice", "bundle.zip")).namelist() == ["k.jpg"]
assert zipfile.ZipFile(os.path.join(OUT, "twice", "bundle (1).zip")).namelist() == ["k (1).jpg"]
assert r7["ui"]["zip"][0]["url"].startswith("/view?filename=bundle%20%281%29.zip")
assert not any(f.endswith(".part") for f in os.listdir(os.path.join(OUT, "twice")))

# --- split / merge (per-photo gating): iphone is 24x32, the rest 48x64 / 64x48
Split = pack.NODE_CLASS_MAPPINGS["FolderIOSplitByShortSide"]
Merge = pack.NODE_CLASS_MAPPINGS["FolderIOMergeSubset"]
below, idx, nb, nok = Split().split(imgs, [40])
assert idx == [0] and nb == 1 and nok == 4 and tuple(below[0].shape) == (1, 24, 32, 3), (idx, nb, nok)
m = Merge()
assert m.check_lazy_status(imgs, idx, (None,)) == ["replacements"]   # needs the upscaled list
assert m.check_lazy_status(imgs, [], (None,)) == []                   # nothing to do -> upscaler never runs
assert m.check_lazy_status(imgs, idx, [below[0]]) == []               # already evaluated
big = torch.zeros((1, 100, 100, 3))
(merged,) = m.merge(imgs, idx, [big])
assert len(merged) == 5 and tuple(merged[0].shape) == (1, 100, 100, 3) and merged[1] is imgs[1]
(passthru,) = m.merge(imgs, [], (None,))
assert len(passthru) == 5 and all(a is b for a, b in zip(passthru, imgs))
try:
    m.merge(imgs, [0, 1], [big])
    raise SystemExit("expected a count-mismatch error")
except ValueError as e:
    print("mismatch ->", e)
below2, idx2, nb2, nok2 = Split().split(imgs, [10])
assert below2 == [] and idx2 == [] and nb2 == 0 and nok2 == 5

# --- ICC: an embedded sRGB profile passes through with pixels intact
from PIL import ImageCms  # noqa: E402
srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
Image.new("RGB", (16, 16), (120, 60, 30)).save(os.path.join(d, "icc.png"), icc_profile=srgb)
imgs3, stems3, _ = Load().load("smoke", "name", 0, 0)
px = [round(v * 255) for v in imgs3[stems3.index("icc")][0, 0, 0].tolist()]
assert px == [120, 60, 30], px

# --- HEIC present but pillow-heif "missing": run still succeeds and the warning reaches the UI
mod.HEIF_OK = False
r8 = Load().load("smoke", "name", 0, 0)
assert isinstance(r8, dict) and "iphone.heic" in r8["ui"]["folderio_warning"][0], r8.get("ui")
assert "iphone" not in r8["result"][1]
mod.HEIF_OK = True

print("SMOKE OK")
