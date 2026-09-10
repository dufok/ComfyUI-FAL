"""Functional smoke test for the video nodes — runs INSIDE the ComfyUI image, no server, no FAL call.

    docker exec Stepn-Tool python /app/custom_nodes/ComfyUI-FAL/tests/test_video_smoke.py

Covers everything that does not need FAL: the 4n+1 length rule, the h264 even-dimensions guard,
encoding a depth batch and decoding chosen frames back out of it, the 16-bit PNG path of the
frame-sequence loader, cutting a flight to 4n+1 around the camera frames, and the failure modes
that must raise rather than pass a black clip downstream.
"""
import importlib
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/custom_nodes")
os.chdir("/app")

import folder_paths  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = "/tmp/falvideo-smoke"
shutil.rmtree(ROOT, ignore_errors=True)
IN, OUT = os.path.join(ROOT, "input"), os.path.join(ROOT, "output")
os.makedirs(os.path.join(IN, "depth"), exist_ok=True)
os.makedirs(OUT, exist_ok=True)
folder_paths.set_input_directory(IN)
folder_paths.set_output_directory(OUT)
os.environ.setdefault("FAL_KEY", "smoke-test-not-a-real-key")

pack = importlib.import_module("ComfyUI-FAL")
video = sys.modules["ComfyUI-FAL.fal_video"]
Vace = pack.NODE_CLASS_MAPPINGS["FalWanVaceDepth"]
Frames = pack.NODE_CLASS_MAPPINGS["FalVideoFrames"]
FromUrl = pack.NODE_CLASS_MAPPINGS["FalVideoFromUrl"]
Sequence = pack.NODE_CLASS_MAPPINGS["FolderIOLoadSequence"]


def raises(fn, needle):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        assert needle in str(e), f"expected {needle!r} in {e!r}"
        return str(e)
    raise AssertionError(f"expected a failure mentioning {needle!r}, got none")


# --- the 4n+1 rule: 61 is legal, 60 is not, and the error names both neighbours
assert video.check_frame_count(61) == 61
assert video.check_frame_count(17) == 17
msg = raises(lambda: video.check_frame_count(60), "4n+1")
assert "57" in msg and "61" in msg, msg

# --- h264 cannot encode odd dimensions; say so before PyAV does
raises(lambda: video.check_even_dims(1281, 720), "even dimensions")
video.check_even_dims(1280, 720)

# --- and its bounds: VACE takes 17..241 frames
raises(lambda: video.check_frame_count(13), "at least 17")
raises(lambda: video.check_frame_count(245), "at most 241")

# --- encode a depth batch, then take frames back out of it
N, H, W = 9, 64, 96
ramp = np.linspace(0.0, 1.0, N, dtype=np.float32)
batch = torch.from_numpy(np.stack([np.full((H, W, 3), v, dtype=np.float32) for v in ramp]))
mp4 = os.path.join(ROOT, "depth.mp4")
assert video.images_to_mp4(batch, 16, mp4) == N
frames, fps, w, h, seconds = video.probe(mp4)
assert (frames, w, h) == (N, W, H), (frames, w, h)
assert abs(fps - 16.0) < 0.01, fps

decoded = video.frames_from_file(mp4, [0, 4, 8])
assert tuple(decoded.shape) == (3, H, W, 3), decoded.shape
got = [float(decoded[i].mean()) for i in range(3)]
assert all(abs(a - b) < 0.05 for a, b in zip(got, [0.0, 0.5, 1.0])), got   # h264 is lossy, not blind
assert got[0] < got[1] < got[2], got                                       # frame order preserved

# --- values above 1.0 are clamped, not wrapped (the gokayfem uint8 cast is what this avoids)
hot = torch.full((2, H, W, 3), 3.0)
video.images_to_mp4(hot, 16, os.path.join(ROOT, "hot.mp4"))
assert float(video.frames_from_file(os.path.join(ROOT, "hot.mp4"), [0]).mean()) > 0.9

# --- Pick Frames on a real VIDEO, including counting from the end
vid = video.VideoFromFile(mp4)
images, count, info = Frames().run(vid, 0, 3, 4)
assert count == N and tuple(images.shape) == (3, H, W, 3), (count, images.shape)
last, _, _ = Frames().run(vid, -1, 1, 1)
assert abs(float(last.mean()) - 1.0) < 0.05, float(last.mean())
raises(lambda: Frames().run(vid, 7, 3, 1), "asked for frame 9 of a 9-frame video")
cams, _, _ = Frames().run(vid, -1, 1, 1, frames="0, 4, 8")                # overrides start/count/stride
assert tuple(cams.shape) == (3, H, W, 3)
assert all(abs(float(cams[i].mean()) - v) < 0.05 for i, v in enumerate((0.0, 0.5, 1.0)))
tail, _, _ = Frames().run(vid, 0, 1, 1, frames="-1")
assert abs(float(tail.mean()) - 1.0) < 0.05
fallback, _, _ = Frames().run(vid, -1, 1, 1, frames="")                   # empty list: widgets rule
assert abs(float(fallback.mean()) - 1.0) < 0.05
raises(lambda: Frames().run(vid, 0, 1, 1, frames="0, 9"), "asked for frame 9")

# --- URL → file: an error page must not be kept or handed on as a video
def _html(url, prefix):
    name = f"{prefix}_fake.mp4"
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(b"<!doctype html><title>403</title>")
    return name, "/view?filename=" + name, 0.0

real_save, video.save_file = video.save_file, _html
try:
    raises(lambda: FromUrl().run("https://example.invalid/x.mp4", "fal_video"), "did not return a video")
    assert not os.path.exists(os.path.join(OUT, "fal_video_fake.mp4")), "the junk file was left behind"
finally:
    video.save_file = real_save
raises(lambda: FromUrl().run("ftp://example.invalid/x.mp4", "fal_video"), "not an http(s) url")

# --- VACE: ambiguous or missing depth input fails before anything is uploaded
raises(lambda: Vace().run("a hall"), "exactly one")
raises(lambda: Vace().run("a hall", depth_images=batch, depth_video=vid), "exactly one")
raises(lambda: Vace().run("   ", depth_images=batch), "prompt is required")
raises(lambda: Vace().run("a hall", depth_images=batch), "at least 17")        # 9 frames, no upload
twenty = torch.zeros((20, H, W, 3))
msg = raises(lambda: Vace().run("a hall", depth_images=twenty), "4n+1")
assert "17 or 21" in msg and "camera frames" in msg, msg

# --- frame sequence loader: natural order, one batch, 16-bit read at full precision
d = os.path.join(IN, "depth")
for i, v in ((1, 0), (2, 32768), (10, 65535)):
    Image.fromarray(np.full((H, W), v, dtype=np.uint16)).save(os.path.join(d, f"depth_{i}.png"))
seq, positions, n, _ = Sequence().load("depth", "")
assert n == 3 and tuple(seq.shape) == (3, H, W, 3), (n, seq.shape)
levels = [float(seq[i].mean()) for i in range(3)]
assert levels[0] < 0.01 and abs(levels[1] - 0.5) < 0.01 and levels[2] > 0.99, levels  # depth_2 < depth_10
assert positions == ""                                                                  # no cameras listed

# an 8-bit frame among 16-bit ones is a normalisation jump mid-flight — warn, don't fail
Image.new("RGB", (W, H), (128, 128, 128)).save(os.path.join(d, "depth_11.png"))
mixed = Sequence().load("depth", "")
assert isinstance(mixed, dict) and "8-bit" in mixed["ui"]["folderio_warning"][0], mixed.get("ui")
assert mixed["result"][2] == 4

# a differently sized frame cannot be batched — name the file that broke it
Image.new("RGB", (W + 2, H), (0, 0, 0)).save(os.path.join(d, "depth_12.png"))
raises(lambda: Sequence().load("depth", ""), "depth_12.png")

# --- a flight: Blender frames 1..64, each frame's grey level = its own frame number
seqmod = sys.modules["ComfyUI-FAL.folderio_nodes"]
fl = os.path.join(IN, "flight")
os.makedirs(fl, exist_ok=True)
for n in range(1, 65):
    Image.new("RGB", (8, 8), (n, n, n)).save(os.path.join(fl, f"{n:04d}.png"))

def frame_ids(batch):
    return [round(float(batch[i, 0, 0, 0]) * 255) for i in range(batch.shape[0])]

# lead-in 1..5 before the first camera is cut; 6..64 is 59 frames -> 57, two dropped between cameras
imgs, positions, count, info = Sequence().load("flight", "6, 30, 64")
ids = frame_ids(imgs)
assert count == 57 == len(ids) and (count - 1) % 4 == 0, count
assert ids[0] == 6 and ids[-1] == 64 and 5 not in ids, (ids[:3], ids[-3:])
cam = [int(p) for p in positions.split(",")]
assert [ids[p] for p in cam] == [6, 30, 64], (positions, [ids[p] for p in cam])
assert "cut 5 before the first camera" in info and "dropped" in info, info
assert ids == sorted(ids) and len(set(ids)) == len(ids), "frames out of order or duplicated"

# cameras on the very first and last frame: 64 -> 61, three dropped, none of them a camera
_, positions, count, _ = Sequence().load("flight", "1 64")
assert count == 61 and positions == "0, 60", (count, positions)

# already 4n+1 between the cameras: nothing dropped
imgs, positions, count, info = Sequence().load("flight", "4, 64")          # 61 frames
assert count == 61 and "dropped none" in info, info

# too short, a missing camera, one camera, and garbage all say what to do
raises(lambda: Sequence().load("flight", "6, 20"), "Render more frames")
raises(lambda: Sequence().load("flight", "6, 99"), "camera frame 99 is not in the folder")
raises(lambda: Sequence().load("flight", "6"), "at least two camera frames")
raises(lambda: Sequence().load("flight", "six, 30"), "frame numbers")

# the same frame rendered twice under two names is refused, not silently merged
Image.new("RGB", (8, 8), (6, 6, 6)).save(os.path.join(fl, "copy_0006.png"))
raises(lambda: Sequence().load("flight", "6, 64"), "frame 6 appears twice")
os.unlink(os.path.join(fl, "copy_0006.png"))

# the planner itself: longer than VACE's 241 still keeps every camera and lands on 4n+1
keep, cam_pos, dropped = seqmod.plan_flight(list(range(1, 301)), [1, 150, 300])
assert len(keep) == 241 and [keep[p] + 1 for p in cam_pos] == [1, 150, 300] and len(dropped) == 59
assert seqmod.frame_number("depth_v2_0006.png") == 6 and seqmod.frame_number("0006 (1).png") == 6

# the whole sequence -> a video, the way the graph does it (without the odd-sized and 8-bit extras)
for extra in ("depth_11.png", "depth_12.png"):
    os.unlink(os.path.join(d, extra))
seq3, _, _, _ = Sequence().load("depth", "")
assert video.images_to_mp4(seq3, 16, os.path.join(ROOT, "seq.mp4")) == 3

print("VIDEO SMOKE OK")
