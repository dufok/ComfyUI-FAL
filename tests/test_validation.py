"""Offline check of the validation / failure-translation layer in fal_common.

No network, no FAL_KEY, no cost. Runs in the container next to the other tests:

    docker exec Stepn-Tool python /app/custom_nodes/ComfyUI-FAL/tests/test_validation.py

and on any machine without ComfyUI's dependencies: whichever of the imports fal_common
needs (torch, numpy, PIL, fal_client, folder_paths) is missing gets stubbed in a temp dir.
The first line of output says which. In the container nothing is stubbed, so the real
fal_client exception classes are the ones being caught.
"""
import importlib.util
import io
import json
import os
import sys
import tempfile
import time

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_fal_common():
    """Import fal_common, stubbing only the modules this interpreter lacks.

    Inside the container everything real is present (with /app on the path for
    folder_paths), so the real fal_client and its real exception classes are exercised.
    """
    if os.path.isdir("/app") and "/app" not in sys.path:
        sys.path.insert(0, "/app")
    missing = []
    for mod in ("numpy", "torch", "PIL", "fal_client", "folder_paths"):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        stubs = tempfile.mkdtemp(prefix="fal_stubs_")
        generic = "def __getattr__(n): return lambda *a, **k: None\n"
        for mod in missing:
            if mod == "PIL":
                os.mkdir(os.path.join(stubs, "PIL"))
                for name in ("__init__", "Image"):
                    with open(os.path.join(stubs, "PIL", name + ".py"), "w") as f:
                        f.write(generic)
            elif mod == "fal_client":
                with open(os.path.join(stubs, "fal_client.py"), "w") as f:
                    f.write(
                        "class FalClientError(Exception): pass\n"
                        "class FalClientHTTPError(FalClientError):\n"
                        "    def __init__(self, message, status_code, response_headers, response, error_type=None):\n"
                        "        super().__init__(message)\n"
                        "        self.message, self.status_code = message, status_code\n"
                        "        self.response_headers, self.response = response_headers, response\n"
                        "        self.error_type = error_type\n"
                        "class FalClientTimeoutError(FalClientError): pass\n"
                        "def subscribe(*a, **k): raise AssertionError('no network in this test')\n"
                        "def upload_file(*a, **k): return 'https://stub/upload.png'\n")
            else:
                with open(os.path.join(stubs, mod + ".py"), "w") as f:
                    f.write(generic)
        sys.path.insert(0, stubs)
    print(f"stubbed: {', '.join(missing) if missing else 'nothing — real dependencies'}")
    spec = importlib.util.spec_from_file_location("fal_common", os.path.join(PACK, "fal_common.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fal_common"] = mod
    spec.loader.exec_module(mod)
    return mod


fc = _load_fal_common()
import fal_client

PASSED = FAILED = 0


def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}  {detail}")


def err(fn):
    """Run fn, return the exception text, or None when it did not raise."""
    try:
        fn()
        return None
    except Exception as e:  # noqa: BLE001 — the message is what is under test
        return str(e)


print("\n[1] empty media is caught before the paid call")
check("empty image_url", "image_url is empty" in (err(lambda: fc.validate_arguments("e", {"image_url": ""})) or ""))
check("None mask_url", "mask_url is empty" in (err(lambda: fc.validate_arguments("e", {"mask_url": None})) or ""))
check("empty image_urls", "empty list" in (err(lambda: fc.validate_arguments("e", {"image_urls": []})) or ""))
check("blank entry in a list",
      "empty entries at [1]" in (err(lambda: fc.validate_arguments("e", {"image_urls": ["http://a", ""]})) or ""))
check("normal args pass", err(lambda: fc.validate_arguments("e", {"image_url": "http://a", "prompt": "x"})) is None)
check("an empty prompt is not media", err(lambda: fc.validate_arguments("e", {"prompt": "", "seed": 0})) is None)

print("\n[2] schema checks, against a generated cache")
tmp = tempfile.mkdtemp(prefix="fal_schema_")
cache = {"endpoints": {"m": {"required": ["prompt"], "fields": {
    "image_urls": {"max_items": 2, "min_items": 1},
    "aspect_ratio": {"enum": ["16:9", "1:1"]}}}}}
path = os.path.join(tmp, "fal_schema.json")
with open(path, "w") as f:
    json.dump(cache, f)
fc.SCHEMA_CACHE, fc._SCHEMA = path, None

check("over maxItems", "at most 2" in (err(lambda: fc.validate_arguments("m", {"prompt": "p", "image_urls": ["a", "b", "c"]})) or ""))
check("missing required", "missing required 'prompt'" in (err(lambda: fc.validate_arguments("m", {"image_urls": ["a"]})) or ""))
check("value outside the enum", "not allowed" in (err(lambda: fc.validate_arguments("m", {"prompt": "p", "aspect_ratio": "21:9"})) or ""))
check("inside the bounds", err(lambda: fc.validate_arguments("m", {"prompt": "p", "image_urls": ["a", "b"], "aspect_ratio": "1:1"})) is None)
check("uncached endpoint is left alone", err(lambda: fc.validate_arguments("other", {"anything": "x"})) is None)

print("\n[3] the escape hatch")
os.environ["FAL_SKIP_VALIDATION"] = "1"
check("FAL_SKIP_VALIDATION sends anyway", err(lambda: fc.validate_arguments("m", {"image_urls": []})) is None)
del os.environ["FAL_SKIP_VALIDATION"]

print("\n[4] the three refusals read differently")
def http_error(message="", status_code=None, response_headers=None, error_type=None):
    """fal_client 1.0's FalClientHTTPError is a dataclass with `response` required."""
    return fal_client.FalClientHTTPError(message=message, status_code=status_code,
                                         response_headers=response_headers or {},
                                         response=None, error_type=error_type)


def msg(**kw):
    return fc._describe_http_error("ep", http_error(**kw))

check("nsfw via error_type", "content filter" in msg(message="x", status_code=400, error_type="NSFW_CONTENT"))
check("policy via body", "content filter" in msg(message="content_policy_violation", status_code=400))
check("policy via header", "content filter" in msg(message="x", status_code=400, response_headers={"x-fal-error-type": "moderation"}))
check("422 names the next step", "fal_registry.py schema ep" in msg(message="bad field", status_code=422))
check("file too large is not a field problem", "size limit" in msg(
    message="[{'loc': ['body', 'image_url'], 'msg': 'File size exceeds the maximum allowed size of 5242880 bytes.', 'type': 'file_too_large'}]",
    status_code=422))
check("429 says wait", "rate limited" in msg(message="slow down", status_code=429))
check("401 points at FAL_KEY", "rejected the key" in msg(message="nope", status_code=401))
check("5xx says resubmit", "safe to resubmit" in msg(message="boom", status_code=503))
check("unknown shape stays readable", "FAL error" in msg(message="weird", status_code=None))
check("content beats the status code", "content filter" in msg(message="nsfw detected", status_code=500))

print("\n[5] a filtered frame never passes as a render")
e = err(lambda: fc.check_content_filter({"has_nsfw_concepts": [True, True]}, "ep"))
check("every frame flagged raises", e is not None and "flagged every image (2/2)" in e)
check("nothing flagged is silent", fc.check_content_filter({"has_nsfw_concepts": [False, False]}, "ep") is None)
check("some flagged warns only", fc.check_content_filter({"has_nsfw_concepts": [True, False]}, "ep") is None)
check("no flag at all is silent", fc.check_content_filter({"images": [{"url": "u"}]}, "ep") is None)
check("a non-dict result is tolerated", fc.check_content_filter("nope", "ep") is None)

print("\n[6] subscribe(): the one door")
os.environ.setdefault("FAL_KEY", "test-key")
check("validates before calling out", "not sent" in (err(lambda: fc.subscribe("m", {"image_urls": []})) or ""))

fal_client.subscribe = lambda *a, **k: (_ for _ in ()).throw(
    http_error("nsfw", 400, None, "NSFW"))
check("http failure is translated", "content filter" in (err(lambda: fc.subscribe("m", {"prompt": "p"})) or ""))

fal_client.subscribe = lambda *a, **k: (_ for _ in ()).throw(fal_client.FalClientTimeoutError("too long"))
check("a timeout warns about the bill", "billable" in (err(lambda: fc.subscribe("m", {"prompt": "p"})) or ""))

fal_client.subscribe = lambda *a, **k: {"ok": True}
check("success passes straight through", fc.subscribe("m", {"prompt": "p"}) == {"ok": True})

print("\n[7] upload fitting — only where an endpoint documents a limit")
spec = importlib.util.spec_from_file_location("fal_registry", os.path.join(PACK, "fal_registry.py"))
reg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reg)
known, _ = reg._endpoints_in_pack()
stray = [e for e in fc.UPLOAD_FIT if e not in known]
check("every UPLOAD_FIT key is an endpoint the pack calls", not stray, str(stray))

fc._FIT_NOTES["https://x/1.png"] = "fitted A"
got = fc.fit_notes({"image_url": "https://x/1.png", "refs": ["https://x/2.png"], "prompt": "p"})
check("fit notes reach the node, once", got == ["fitted A"] and "https://x/1.png" not in fc._FIT_NOTES)

import PIL
import numpy as _np
if not (isinstance(getattr(PIL, "__version__", None), str) and isinstance(getattr(_np, "__version__", None), str)):
    print("  skip  the resizing itself needs real Pillow + numpy — run this file in the container")
else:
    from PIL import Image as PImage

    def size_of(data):
        return PImage.open(io.BytesIO(data)).size

    rec = fc.UPLOAD_FIT["fal-ai/recraft/vectorize"]
    plain = PImage.new("RGB", (1024, 768), (200, 30, 30))
    data, note = fc.fit_png(plain, rec, "rec")
    check("inside the limits: untouched", note == "" and size_of(data) == (1024, 768))

    data, note = fc.fit_png(PImage.new("RGB", (6000, 1500), (10, 10, 10)), rec, "rec")
    check("longest side capped under 4096", max(size_of(data)) <= 4095 and "fitted 6000x1500" in note)

    data, note = fc.fit_png(PImage.new("RGB", (4500, 4500), (90, 90, 90)), rec, "rec")
    w, h = size_of(data)
    check("pixel count capped under 16 MP", w * h < 16_000_000 and max(w, h) <= 4095)

    rng = _np.random.default_rng(0)
    noise = PImage.fromarray(rng.integers(0, 256, (2400, 2400, 3), dtype=_np.uint8))
    t = time.time()
    data, note = fc.fit_png(noise, rec, "rec")
    w, h = size_of(data)
    check(f"a frame PNG cannot compress gets under 5 MB ({len(data) / 1e6:.1f} MB at {w}x{h}, "
          f"{time.time() - t:.1f}s)", len(data) <= rec["max_bytes"] and w < 2400)

    e = err(lambda: fc.fit_png(PImage.new("RGB", (200, 300)), rec, "rec"))
    check("under the minimum side it refuses, never upscales", e is not None and "more than 256" in e)

    hi3d = fc.UPLOAD_FIT["hitem3d/hi3d/v3.0/image-to-3d"]
    check("a byte-only limit leaves a light frame alone", fc.fit_png(plain, hi3d, "hi3d")[1] == "")

print(f"\n{PASSED} passed, {FAILED} failed")
sys.exit(1 if FAILED else 0)
