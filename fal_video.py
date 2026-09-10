"""
FAL video nodes (category: FAL/Video) — the orbit half of the consistent-viewpoints pipeline.

  * FalWanVaceDepth  -> fal-ai/wan-vace-14b/depth  (depth sequence + reference first frame
                        -> temporally coherent photoreal orbit, $0.08/s at 720p)
  * FalVideoFromUrl  -> download a video URL into output/ and hand back a real VIDEO
  * FalVideoFrames   -> pull individual frames out of a VIDEO without decoding all of it

Why the pipeline needs this at all: FAL's image editors do not understand space. Only a 3D
reconstruction or a video model conditioned on depth does. A depth sequence rendered in
Blender plus one look-developed first frame buys temporal coherence, and temporal coherence
between neighbouring frames *is* consistency between viewpoints.

Why not ComfyUI-VideoHelperSuite: ComfyUI v0.23 already ships the VIDEO type with
Create Video / Get Video Components / Load Video / Save Video, and the image carries PyAV
with an h264 encoder. The only genuinely missing pieces were this endpoint, a download step
and cheap frame extraction — three nodes here instead of a fourth pinned pack in the
Dockerfile. The folder of depth frames is loaded by FolderIOLoadSequence (image/folder).

Why not the co-installed gokayfem pack: its Wan VACE nodes point at other models
(wan-vace-apps/video-edit, wan-22-vace-fun-a14b) and return a bare URL string — nothing on
disk. Worse, on any exception `handle_video_generation_error` returns the *string*
"Error: Unable to generate video." as a successful result, so a graph built on it finishes
green with a sentence where the video should be. Everything here raises instead.
"""
import os
import tempfile
import urllib.request
from fractions import Fraction

import numpy as np
import torch

import fal_client
import folder_paths

from .fal_common import (
    require_key,
    upload_image,
    upload_image_frames,
    public_download_url,
    save_file,
    file_url,
)

try:  # the stable shim; `latest` is the implementation it re-exports
    from comfy_api.input_impl import VideoFromComponents, VideoFromFile
    from comfy_api.util import VideoComponents
except ImportError:  # pragma: no cover — only if the shim is ever dropped
    from comfy_api.latest._input_impl import VideoFromComponents, VideoFromFile
    from comfy_api.latest._util import VideoComponents

try:  # ComfyUI requires PyAV, but never let a probe take the pack down
    import av
except ImportError:  # pragma: no cover
    av = None


ENDPOINT = "fal-ai/wan-vace-14b/depth"
NATIVE_FPS = 16          # Wan thinks in frames; it emits them at 16 fps regardless of the widget
PRICE_720P = 0.08        # $ per second of 720p output, i.e. per 16 generated frames
MAX_DEG_PER_FRAME = 6.0  # past ~6° neighbouring frames stop correlating and the object drifts


# --------------------------------------------------------------------------- guards

def check_frame_count(n):
    """Wan's temporal VAE packs 4 frames per latent plus one incompressible anchor, so the
    length has to be 4n+1 (…49, 57, 61, 65, 81…). Ask for 60 and the tail is either cut or
    padded with duplicates — an under-rotation or a freeze in exactly the frame you needed."""
    n = int(n)
    if n % 4 != 1:
        down = n - ((n - 1) % 4)
        raise RuntimeError(
            f"num_frames must be 4n+1 (…49, 57, 61, 65, 81…), got {n}. "
            f"Use {down} or {down + 4}.")
    return n


def check_angular_step(arc_degrees, frames):
    """A full 360° over 61 frames is 5.9°/frame — the working limit. Print, never raise:
    the arc is the user's, we only know what they typed."""
    if not arc_degrees or frames < 2:
        return ""
    step = float(arc_degrees) / (frames - 1)
    note = f"{arc_degrees:g}° over {frames} frames = {step:.2f}°/frame"
    if step > MAX_DEG_PER_FRAME:
        print(f"[FAL] warning: {note} — above {MAX_DEG_PER_FRAME}°/frame neighbouring frames "
              f"stop correlating and the object drifts. Shorten the arc or add frames.")
    return note


def check_even_dims(width, height):
    """h264/yuv420p cannot encode odd dimensions, and PyAV's error for it is unreadable."""
    if width % 2 or height % 2:
        raise RuntimeError(
            f"h264 needs even dimensions, got {width}x{height} — render the depth pass at an "
            f"even size (or crop one pixel) before building the video.")


# --------------------------------------------------------------------------- video helpers

def images_to_mp4(images, fps, path):
    """IMAGE batch [N,H,W,3] -> h264 mp4 at `path`. Values are clamped to 0..1 on the way in.

    Clamping is core ComfyUI's (VideoFromComponents), and it is the difference between this
    and the gokayfem helper, which casts to uint8 without clipping — Blender depth above 1.0
    wraps around there and turns into noise. Clip on the Blender side anyway; this is a net,
    not a plan."""
    if images is None or images.shape[0] == 0:
        raise RuntimeError("no depth frames — connect a frame sequence (or a VIDEO) first")
    n, h, w = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    check_even_dims(w, h)
    lo, hi = float(images.min()), float(images.max())
    if hi > 1.001 or lo < -0.001:
        print(f"[FAL] warning: depth range is {lo:.3f}..{hi:.3f}, outside 0..1 — everything "
              f"outside gets clamped. Clip the depth pass in the Blender/Comfyder export.")
    components = VideoComponents(images=images, audio=None,
                                 frame_rate=Fraction(round(float(fps) * 1000), 1000))
    VideoFromComponents(components).save_to(path)
    print(f"[FAL] encoded {n} depth frames {w}x{h} @ {fps} fps -> {os.path.getsize(path) / 1e6:.1f} MB")
    return n


def video_to_upload_url(video):
    """VIDEO -> uploaded FAL URL. Uses the file on disk when there is one, no re-encode."""
    src = video.get_stream_source()
    if isinstance(src, str) and os.path.isfile(src):
        return fal_client.upload_file(src)
    fd, path = tempfile.mkstemp(suffix=".mp4", prefix="fal_vid_")
    os.close(fd)
    try:
        video.save_to(path)
        return fal_client.upload_file(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def probe(path):
    """(frames, fps, width, height, seconds) read from container metadata — no decoding."""
    if av is None or not os.path.isfile(path):
        return None
    try:
        with av.open(path) as c:
            s = next((s for s in c.streams if s.type == "video"), None)
            if s is None:
                return None
            fps = float(s.average_rate or s.base_rate or 0) or 0.0
            frames = int(s.frames or 0)
            seconds = float(c.duration / 1e6) if c.duration else (frames / fps if fps else 0.0)
            if not frames and fps and seconds:
                frames = round(seconds * fps)
            return frames, fps, int(s.codec_context.width), int(s.codec_context.height), seconds
    except Exception as e:  # noqa: BLE001 — a probe failure must not fail a finished job
        print(f"[FAL] warning: could not probe {path}: {e}")
        return None


def describe(path):
    p = probe(path)
    if not p:
        return ""
    frames, fps, w, h, seconds = p
    return f"{frames} frames, {fps:.2f} fps, {w}x{h}, {seconds:.2f} s"


def frames_from_file(path, wanted):
    """Decode only the frames in `wanted` (a sorted list of indices) -> [K,H,W,3] tensor.

    Sequential decode with an early exit: seeking to an exact frame index in h264 is a lie
    (you land on the previous keyframe), and these clips are 241 frames at most.
    """
    if av is None:
        raise RuntimeError("PyAV is not available in this ComfyUI environment — cannot decode frames")
    remaining = list(wanted)
    out, i = [], 0
    with av.open(path) as c:
        stream = next((s for s in c.streams if s.type == "video"), None)
        if stream is None:
            raise RuntimeError(f"{os.path.basename(path)} has no video stream")
        stream.thread_type = "AUTO"
        for frame in c.decode(stream):
            while remaining and remaining[0] == i:
                arr = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
                out.append(torch.from_numpy(arr))
                remaining.pop(0)
            i += 1
            if not remaining:
                break
    if remaining:
        raise RuntimeError(
            f"frame {remaining[0]} is past the end of the video ({i} frames decoded)")
    return torch.stack(out, 0)


# --------------------------------------------------------------------------- nodes

class FalWanVaceDepth:
    """fal-ai/wan-vace-14b/depth — a depth sequence plus a first frame become a coherent orbit.

    The depth video carries every bit of geometry, so the camera move is yours, from 3D. The
    first frame carries the look. Nothing is invented between the two, which is the whole
    point: a flight generated "from imagination" drifts, one driven by rendered depth does not.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "What the footage is. Materials, light, lens — "
                                                 "never geometry: geometry comes from the depth."}),
            },
            "optional": {
                "depth_images": ("IMAGE", {"tooltip": "Depth sequence as an IMAGE batch (near = light). "
                                                      "Encoded to h264 here. Use this or depth_video."}),
                "depth_video": ("VIDEO", {"tooltip": "Depth sequence already muxed as a video. "
                                                     "Use this or depth_images, not both."}),
                "first_frame": ("IMAGE", {"tooltip": "The look-developed reference frame — frame 1 of "
                                                     "the orbit after Banana. This is where the "
                                                     "material and atmosphere come from."}),
                "last_frame": ("IMAGE",),
                "ref_images": ("IMAGE", {"tooltip": "Appearance references (batch). Nothing geometric "
                                                    "is taken from them."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "num_frames": ("INT", {"default": 61, "min": 17, "max": 241, "step": 4,
                                       "tooltip": "Must be 4n+1 (…49, 57, 61, 65, 81…) — the temporal "
                                                  "VAE packs 4 frames per latent plus one anchor. "
                                                  "360°/61 = 5.9°/frame, the working limit."}),
                "match_input_num_frames": ("BOOLEAN", {"default": False,
                                                       "tooltip": "Take the length from the depth input "
                                                                  "instead of the widget (still 4n+1)."}),
                "resolution": (["720p", "580p", "480p", "360p", "240p", "auto"], {"default": "720p"}),
                "aspect_ratio": (["auto", "16:9", "1:1", "9:16"], {"default": "auto"}),
                "frames_per_second": ("INT", {"default": 16, "min": 5, "max": 30,
                                              "tooltip": "Playback rate of the returned file. Wan always "
                                                         "generates num_frames at 16 fps — this does not "
                                                         "change how much is generated or what it costs."}),
                "arc_degrees": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 1.0,
                                          "tooltip": "How far the camera travels in the depth pass. "
                                                     "0 = don't check. Only used to warn when the "
                                                     "angular step per frame is too big."}),
                "preprocess": ("BOOLEAN", {"default": False,
                                           "tooltip": "Leave OFF. On, FAL runs a depth estimator over the "
                                                      "input — but the input already IS depth."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 60}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "sampler": (["unipc", "dpm++", "euler"], {"default": "unipc"}),
                "acceleration": (["regular", "none"], {"default": "regular"}),
                "video_quality": (["high", "maximum", "medium", "low"], {"default": "high"}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": False,
                                                        "tooltip": "Rewrites your prompt on FAL's side. "
                                                                   "Off — the prompt here is a contract."}),
                "enable_safety_checker": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "download_url", "info")
    OUTPUT_TOOLTIPS = (
        "The orbit. Wire into Save Video, or into FAL Video — Pick Frames to take a viewpoint out of it.",
        "File name inside ComfyUI's output/ directory.",
        "Direct download link (needs COMFYUI_PUBLIC_URL to be absolute).",
        "Endpoint, length, angular step, price estimate and what actually came back.",
    )
    FUNCTION = "run"
    CATEGORY = "FAL/Video"
    DESCRIPTION = ("Wan VACE 14B depth-to-video: a rendered depth sequence plus a look-developed "
                   "first frame become a temporally coherent photoreal orbit. ~$0.08 per second "
                   "of 720p, counted at 16 fps.")

    def run(self, prompt, depth_images=None, depth_video=None, first_frame=None, last_frame=None,
            ref_images=None, negative_prompt="", num_frames=61, match_input_num_frames=False,
            resolution="720p", aspect_ratio="auto", frames_per_second=16, arc_degrees=0.0,
            preprocess=False, seed=0, guidance_scale=5.0, num_inference_steps=30, shift=5.0,
            sampler="unipc", acceleration="regular", video_quality="high",
            enable_prompt_expansion=False, enable_safety_checker=False):
        require_key()
        if not prompt.strip():
            raise RuntimeError("prompt is required — describe materials, light and lens, not geometry")
        if (depth_images is None) == (depth_video is None):
            raise RuntimeError("connect exactly one of depth_images (IMAGE batch) or depth_video (VIDEO)")

        # --- the depth pass -> a URL on FAL
        tmp = None
        try:
            if depth_images is not None:
                fd, tmp = tempfile.mkstemp(suffix=".mp4", prefix="fal_depth_")
                os.close(fd)
                input_frames = images_to_mp4(depth_images, frames_per_second, tmp)
                video_url = fal_client.upload_file(tmp)
            else:
                input_frames = int(depth_video.get_frame_count())
                video_url = video_to_upload_url(depth_video)
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

        frames = check_frame_count(input_frames if match_input_num_frames else num_frames)
        if match_input_num_frames and frames != int(num_frames):
            print(f"[FAL] num_frames taken from the depth input: {frames}")
        step_note = check_angular_step(arc_degrees, frames)

        args = {
            "prompt": prompt.strip(),
            "video_url": video_url,
            "num_frames": frames,
            "frames_per_second": int(frames_per_second),
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "preprocess": bool(preprocess),
            "guidance_scale": float(guidance_scale),
            "num_inference_steps": int(num_inference_steps),
            "shift": float(shift),
            "sampler": sampler,
            "acceleration": acceleration,
            "video_quality": video_quality,
            "enable_prompt_expansion": bool(enable_prompt_expansion),
            "enable_safety_checker": bool(enable_safety_checker),
        }
        if negative_prompt.strip():
            args["negative_prompt"] = negative_prompt.strip()
        if seed:
            args["seed"] = int(seed)
        if first_frame is not None:
            args["first_frame_url"] = upload_image(first_frame)
        if last_frame is not None:
            args["last_frame_url"] = upload_image(last_frame)
        if ref_images is not None:
            args["ref_image_urls"] = upload_image_frames(ref_images)

        estimate = frames / NATIVE_FPS * PRICE_720P
        printable = dict(args, video_url=f"<depth {input_frames} frames>")
        print(f"[FAL] {ENDPOINT} <- {printable}")
        print(f"[FAL] ~${estimate:.2f} at the 720p rate ({frames} frames / {NATIVE_FPS} fps)"
              + (f" — {step_note}" if step_note else ""))

        result = fal_client.subscribe(ENDPOINT, arguments=args, with_logs=False)
        url = file_url(result.get("video") if isinstance(result, dict) else None)
        if not url:
            raise RuntimeError(f"no video url in the FAL response: {result}")

        fname, download_url, size_mb = save_file(url, "vace")
        path = os.path.join(folder_paths.get_output_directory(), fname)
        got = describe(path)
        info = (f"{ENDPOINT} | asked {frames} frames @ {resolution} ≈ ${estimate:.2f}"
                + (f" | {step_note}" if step_note else "")
                + f" | got {got or f'{size_mb:.1f} MB'} -> {fname}  ⬇ {download_url}")
        print(f"[FAL] DONE {info}")
        return (VideoFromFile(path), fname, download_url, info)


class FalVideoFromUrl:
    """A video URL -> a file in output/ and a real VIDEO.

    The other FAL pack's video nodes hand back a bare URL string and write nothing to disk;
    this is the missing step between such a node and anything that wants a video. It also
    refuses to pretend: an HTML error page or a truncated download raises here instead of
    becoming a black clip three nodes later.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": False,
                                   "tooltip": "https:// link to an mp4/webm. Wire a video_url output "
                                              "of any FAL node straight in."}),
                "filename_prefix": ("STRING", {"default": "fal_video"}),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_file", "download_url", "info")
    FUNCTION = "run"
    CATEGORY = "FAL/Video"
    DESCRIPTION = "Download a video URL into output/ and turn it into a VIDEO the graph can use."

    # A wrong URL is worth retrying, but an unchanged one should not re-download every run.
    @classmethod
    def IS_CHANGED(cls, url, filename_prefix):
        return f"{url}\0{filename_prefix}"

    def run(self, url, filename_prefix):
        url = (url or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            raise RuntimeError(f"not an http(s) url: {url!r}")
        prefix = "".join(c for c in (filename_prefix or "fal_video") if c.isalnum() or c in "-_") or "fal_video"

        fname, download_url, size_mb = save_file(url, prefix)
        path = os.path.join(folder_paths.get_output_directory(), fname)
        head = b""
        with open(path, "rb") as f:
            head = f.read(16)
        is_video = head[4:8] == b"ftyp" or head[:4] == b"\x1aE\xdf\xa3" or head[:4] == b"RIFF"
        if not is_video:
            os.unlink(path)
            snippet = head.decode("utf-8", "replace")
            raise RuntimeError(
                f"{url} did not return a video (first bytes: {snippet!r}) — an error page or an "
                f"expired link, most likely. Nothing was kept.")
        got = describe(path)
        info = f"{url} -> {fname} ({got or f'{size_mb:.1f} MB'})  ⬇ {download_url}"
        print(f"[FAL] {info}")
        return (VideoFromFile(path), fname, download_url, info)


class FalVideoFrames:
    """Take specific frames out of a VIDEO.

    Get Video Components decodes the whole clip into RAM — 81 frames of 720p is ~0.9 GB, and
    the point of an orbit is usually a handful of viewpoints out of it. This decodes only the
    frames asked for and stops there.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "start_index": ("INT", {"default": 0, "min": -240, "max": 240,
                                        "tooltip": "0-based. Negative counts from the end (-1 = last)."}),
                "count": ("INT", {"default": 1, "min": 1, "max": 241,
                                  "tooltip": "How many frames to take. Each 720p frame is ~11 MB in RAM."}),
                "stride": ("INT", {"default": 1, "min": 1, "max": 240,
                                   "tooltip": "Step between them. 61 frames over 360° with stride 10 "
                                              "gives a viewpoint roughly every 59°."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("images", "frame_count", "info")
    OUTPUT_TOOLTIPS = ("The picked frames, as a batch.", "Total frames in the video.", "What was taken.")
    FUNCTION = "run"
    CATEGORY = "FAL/Video"
    DESCRIPTION = ("Pick frames out of a video without decoding all of it — one viewpoint of the "
                   "orbit becomes IMAGE 3 for the final 2K pass.")

    def run(self, video, start_index, count, stride):
        src = video.get_stream_source()
        tmp = None
        try:
            if not isinstance(src, str):  # in-memory video: PyAV wants a file to seek in
                fd, tmp = tempfile.mkstemp(suffix=".mp4", prefix="fal_frames_")
                os.close(fd)
                video.save_to(tmp)
                path = tmp
            else:
                path = src
            total = int(video.get_frame_count() or 0) or (probe(path) or [0])[0]
            start = int(start_index)
            if start < 0:
                if not total:
                    raise RuntimeError("cannot count from the end: the frame count is unknown")
                start += total
            if start < 0:
                raise RuntimeError(f"start_index {start_index} is before the first frame")
            wanted = [start + i * int(stride) for i in range(int(count))]
            if total and wanted[-1] >= total:
                raise RuntimeError(
                    f"asked for frame {wanted[-1]} of a {total}-frame video "
                    f"(start_index={start_index}, count={count}, stride={stride})")
            images = frames_from_file(path, wanted)
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        info = f"frames {wanted} of {total or '?'}  ({images.shape[2]}x{images.shape[1]})"
        print(f"[FAL] {info}")
        return (images, total, info)


NODE_CLASS_MAPPINGS = {
    "FalWanVaceDepth": FalWanVaceDepth,
    "FalVideoFromUrl": FalVideoFromUrl,
    "FalVideoFrames": FalVideoFrames,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FalWanVaceDepth": "FAL Video — Wan VACE 14B depth → orbit ($0.08/s @720p)",
    "FalVideoFromUrl": "FAL Video — URL → file",
    "FalVideoFrames": "FAL Video — Pick Frames",
}
