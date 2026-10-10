"""Offline check of FalMiniMaxH3ThreeDToVideo — the arguments it sends and the guards that fire
before anything is uploaded. No API calls, no key, no ComfyUI, no torch:

    python tests/test_h3_3d_args.py

Everything fal_video imports is stubbed, so this checks the node's own logic only. The encoding
path (VideoComponents -> h264) is the same core ComfyUI code test_video_smoke.py exercises.
"""
import importlib
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

calls, uploads, saved = [], [], []


def stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


class Frames(np.ndarray):
    """Enough of a torch tensor for the node: .shape and .clamp()."""
    def clamp(self, lo, hi):
        return np.clip(self, lo, hi).view(Frames)


def frames(n, h=64, w=96):
    return np.zeros((n, h, w, 3), dtype=np.float32).view(Frames)


class FakeVideo:
    def __init__(self, seconds, path="/nonexistent/clip.mp4"):
        self.seconds, self.path = seconds, path

    def get_duration(self):
        return self.seconds

    def get_stream_source(self):
        return self.path

    def save_to(self, path):
        open(path, "wb").close()


class FakeComponents:
    def __init__(self, images, audio, frame_rate):
        self.images, self.frame_rate = images, frame_rate


class FakeFromComponents:
    def __init__(self, c):
        self.c = c

    def save_to(self, path):
        with open(path, "wb") as f:
            f.write(b"\0" * 1000)


stub("torch")
stub("folder_paths", get_output_directory=lambda: "/tmp")
stub("fal_client", upload_file=lambda p: uploads.append(p) or f"https://stub/{len(uploads)}.mp4")
stub("av")
stub("comfy_api")
stub("comfy_api.input_impl", VideoFromComponents=FakeFromComponents,
     VideoFromFile=lambda p: ("VIDEO", p))
stub("comfy_api.util", VideoComponents=FakeComponents)

pkg = stub("falpkg")
pkg.__path__ = [ROOT]
stub("falpkg.folderio_nodes", VACE_MAX_FRAMES=241, VACE_MIN_FRAMES=17)
stub("falpkg.fal_common",
     subscribe=lambda ep, args: calls.append((ep, args)) or {"video": {"url": "https://fal/out.mp4"}},
     require_key=lambda: None,
     upload_image=lambda img: f"https://stub/ref{img.shape[1]}.png",
     upload_image_frames=lambda img: [],
     public_download_url=lambda f: f"/view?filename={f}",
     save_file=lambda url, prefix: saved.append((url, prefix)) or (f"{prefix}_out.mp4", "/view", 1.0),
     file_url=lambda node: node.get("url") if isinstance(node, dict) else None)
mod = importlib.import_module("falpkg.fal_video")
Node = mod.FalMiniMaxH3ThreeDToVideo
mod.probe = lambda path: None


def run(**kw):
    del calls[:]
    out = Node().run(**kw)
    return calls[0], out


def raises(fn, needle):
    try:
        fn()
    except RuntimeError as e:
        assert needle in str(e), f"expected {needle!r} in {e!r}"
        return
    raise AssertionError(f"expected a failure mentioning {needle!r}")


assert "FalMiniMaxH3ThreeDToVideo" in mod.NODE_CLASS_MAPPINGS
assert "FalMiniMaxH3ThreeDToVideo" in mod.NODE_DISPLAY_NAME_MAPPINGS
spec = Node.INPUT_TYPES()
assert spec["required"]["resolution"][0] == ["480P", "768P", "1080P"]
assert [k for k in spec["optional"] if k.startswith("ref_image_")] == [f"ref_image_{i}" for i in range(1, 7)]

# a VIDEO, no references: FAL is allowed to generate its own; nothing optional is invented
(ep, a), out = run(video=FakeVideo(6.0))
assert ep == "minimax/h3-max/3d-to-video", ep
assert a == {"video_url": "https://stub/1.mp4", "resolution": "768P",
             "enable_safety_checker": True, "max_generated_reference_images": 2}, a
assert out[0] == ("VIDEO", "/tmp/h3_3d_out.mp4") and out[1] == "h3_3d_out.mp4"
assert "$0.48" in out[3], out[3]                        # 6 s x $0.08

# references (different sizes, a gap in the sockets) -> uploaded in socket order, no generation cap
(_, a), out = run(video=FakeVideo(10.0), resolution="1080P", prompt="  the grey box is a seaplane ",
                  ref_image_1=frames(1, 100), ref_image_4=frames(1, 200), enable_safety_checker=False)
assert a["reference_image_urls"] == ["https://stub/ref100.png", "https://stub/ref200.png"], a
assert "max_generated_reference_images" not in a, a
assert a["prompt"] == "the grey box is a seaplane" and a["resolution"] == "1080P"
assert a["enable_safety_checker"] is False
assert "$1.60" in out[3] and "2 reference(s)" in out[3], out[3]

# a short clip is still billed 5 s, and the estimate says so
(_, a), out = run(video=FakeVideo(2.0), resolution="480P")
assert "$0.25" in out[3], out[3]                        # 5 s minimum x $0.05

# an IMAGE sequence is encoded at the given fps; 240 frames @ 24 = 10 s is fine
del uploads[:]
(_, a), out = run(frames=frames(240), fps=24.0)
assert len(uploads) == 1 and a["video_url"].startswith("https://stub/")
assert "10.00 s" in out[3], out[3]

# guards: all of them fire before a single byte is uploaded
del uploads[:]
raises(lambda: Node().run(video=FakeVideo(16.0)), "at most 15 s")
raises(lambda: Node().run(frames=frames(400), fps=24.0), "at most 15 s")      # 16.7 s
raises(lambda: Node().run(), "exactly one")
raises(lambda: Node().run(video=FakeVideo(5.0), frames=frames(24)), "exactly one")
raises(lambda: Node().run(frames=frames(24, 64, 97), fps=24.0), "even dimensions")
raises(lambda: Node().run(video=FakeVideo(5.0), resolution="4K"), "resolution must be")
assert uploads == [], uploads

print("test_h3_3d_args: all checks passed")
