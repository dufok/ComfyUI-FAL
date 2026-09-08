"""End-to-end check against a RUNNING ComfyUI (run inside the container, it has Pillow):

    docker exec Stepn-Tool python /app/custom_nodes/ComfyUI-FAL/tests/e2e_api.py [http://127.0.0.1:8188]

Uploads a small folder through /upload/image, queues Load Images (folder) -> Save Images + ZIP
(no paid nodes), waits, then checks the history outputs, the ZIP served by /view and the
EXIF-rotated frame's dimensions. Leaves input/folderio_e2e and output/folderio_e2e_out behind.
"""
import io
import json
import sys
import time
import urllib.request
import uuid
import zipfile

from PIL import Image

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8188").rstrip("/")
FOLDER = "folderio_e2e"
OUT = FOLDER + "_out"


def get(path):
    return urllib.request.urlopen(BASE + path, timeout=60)


def post(path, data, ctype):
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": ctype})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def multipart(fields, filename, data):
    b = uuid.uuid4().hex
    body = io.BytesIO()
    for k, v in fields.items():
        body.write(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    body.write(f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="{filename}"\r\n'
               f"Content-Type: image/jpeg\r\n\r\n".encode())
    body.write(data)
    body.write(f"\r\n--{b}--\r\n".encode())
    return body.getvalue(), f"multipart/form-data; boundary={b}"


# 1. upload three images, one with EXIF orientation 6 (must come back rotated: 80x120)
for name, orient in (("a_1.jpg", None), ("a_2.jpg", None), ("rot.jpg", 6)):
    img = Image.new("RGB", (120, 80), (30, 120, 220))
    kw = {}
    if orient:
        ex = Image.Exif()
        ex[0x0112] = orient
        kw["exif"] = ex.tobytes()
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, **kw)
    body, ct = multipart({"subfolder": FOLDER, "type": "input", "overwrite": "true"}, name, buf.getvalue())
    r = post("/upload/image", body, ct)
    assert r["subfolder"] == FOLDER and r["name"] == name, r
print("uploaded 3 files ->", f"input/{FOLDER}")

# 2. nodes registered, fresh folder visible in the combo
info = json.loads(get("/object_info").read())
assert "FolderIOLoadImages" in info and "FolderIOSaveZip" in info, "nodes missing from /object_info"
assert FOLDER in info["FolderIOLoadImages"]["input"]["required"]["folder"][0]

# 3. queue loader -> saver
prompt = {
    "1": {"class_type": "FolderIOLoadImages",
          "inputs": {"folder": FOLDER, "sort_by": "name", "start_index": 0, "max_images": 0}},
    "2": {"class_type": "FolderIOSaveZip",
          "inputs": {"images": ["1", 0], "filenames": ["1", 1], "folder": OUT, "format": "jpg",
                     "quality": 92, "suffix": "_x", "zip_name": "", "overwrite": True}},
}
r = post("/prompt", json.dumps({"prompt": prompt, "client_id": "folderio-e2e"}).encode(), "application/json")
assert not r.get("node_errors"), r
pid = r["prompt_id"]
for _ in range(180):
    time.sleep(1)
    hist = json.loads(get(f"/history/{pid}").read())
    if pid in hist:
        break
else:
    raise SystemExit("timeout waiting for the prompt")
item = hist[pid]
assert item["status"]["status_str"] == "success", item["status"]
out = item["outputs"]["2"]
assert [i["filename"] for i in out["images"]] == ["a_1_x.jpg", "a_2_x.jpg", "rot_x.jpg"], out
z = out["zip"][0]
print("history outputs ok, zip:", z)

# 4. ZIP via /view
resp = get(z["url"])
ctype = resp.headers.get("Content-Type", "")
data = resp.read()
assert ctype.startswith("application/zip"), ctype
assert zipfile.ZipFile(io.BytesIO(data)).namelist() == ["a_1_x.jpg", "a_2_x.jpg", "rot_x.jpg"]
assert len(data) == z["bytes"], (len(data), z["bytes"])

# 5. EXIF orientation applied on the way in
with Image.open(io.BytesIO(get(f"/view?filename=rot_x.jpg&subfolder={OUT}&type=output").read())) as im:
    assert im.size == (80, 120), im.size

# 6. the extension is served
js = get("/extensions/ComfyUI-FAL/folderio.js").read().decode()
assert "Comfy.FolderIO" in js


def run_graph(prompt):
    r = post("/prompt", json.dumps({"prompt": prompt, "client_id": "folderio-e2e"}).encode(), "application/json")
    assert not r.get("node_errors"), r
    pid = r["prompt_id"]
    for _ in range(180):
        time.sleep(1)
        hist = json.loads(get(f"/history/{pid}").read())
        if pid in hist:
            item = hist[pid]
            assert item["status"]["status_str"] == "success", item["status"]
            return item
    raise SystemExit("timeout waiting for the prompt")


def sizes(folder_out):
    hist_sizes = {}
    for name in ("a_1_x.jpg", "a_2_x.jpg", "rot_x.jpg", "big_x.jpg"):
        try:
            with Image.open(io.BytesIO(get(f"/view?filename={name}&subfolder={folder_out}&type=output").read())) as im:
                hist_sizes[name] = im.size
        except Exception:  # noqa: BLE001
            pass
    return hist_sizes


def gated_graph(min_short_side, folder_out):
    # Split -> (ImageScaleBy x2 stands in for the paid upscaler) -> Merge (lazy) -> Save
    return {
        "1": {"class_type": "FolderIOLoadImages",
              "inputs": {"folder": FOLDER, "sort_by": "name", "start_index": 0, "max_images": 0}},
        "2": {"class_type": "FolderIOSplitByShortSide", "inputs": {"images": ["1", 0], "min_short_side": min_short_side}},
        "3": {"class_type": "ImageScaleBy", "inputs": {"image": ["2", 0], "upscale_method": "bilinear", "scale_by": 2.0}},
        "4": {"class_type": "FolderIOMergeSubset", "inputs": {"images": ["1", 0], "index": ["2", 1], "replacements": ["3", 0]}},
        "5": {"class_type": "FolderIOSaveZip",
              "inputs": {"images": ["4", 0], "filenames": ["1", 1], "folder": folder_out, "format": "jpg",
                         "quality": 92, "suffix": "_x", "zip_name": "", "overwrite": True}},
    }


# 7. nothing below the target: the merge must NOT pull the upscaler (an empty list would crash ImageScaleBy)
run_graph(gated_graph(50, OUT + "_lazy"))
got = sizes(OUT + "_lazy")
assert got == {"a_1_x.jpg": (120, 80), "a_2_x.jpg": (120, 80), "rot_x.jpg": (80, 120)}, got
print("lazy branch skipped ok:", got)

# 8. mixed folder: only the small ones go through the upscaler, the big one stays, order is kept
img = Image.new("RGB", (300, 200), (200, 40, 40))
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=85)
body, ct = multipart({"subfolder": FOLDER, "type": "input", "overwrite": "true"}, "big.jpg", buf.getvalue())
post("/upload/image", body, ct)
item = run_graph(gated_graph(100, OUT + "_mixed"))
names = [i["filename"] for i in item["outputs"]["5"]["images"]]
assert names == ["a_1_x.jpg", "a_2_x.jpg", "big_x.jpg", "rot_x.jpg"], names
got = sizes(OUT + "_mixed")
assert got == {"a_1_x.jpg": (240, 160), "a_2_x.jpg": (240, 160), "rot_x.jpg": (160, 240), "big_x.jpg": (300, 200)}, got
print("mixed folder gated ok:", got)

print("E2E OK —", BASE, z["url"])
