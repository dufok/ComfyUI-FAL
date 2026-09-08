"""Offline check of FalTopazUpscale2026's routing and per-model gating — no API calls, no key.

    python tests/test_topaz_args.py
"""
import importlib
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(ROOT))

calls = []
fake_common = types.ModuleType("fal_common")
fake_common.run_image = lambda endpoint, args: calls.append((endpoint, args)) or "IMG"
fake_common.upload_image = lambda image: "https://example/x.png"
pkg = types.ModuleType("falpkg")
pkg.__path__ = [ROOT]
sys.modules["falpkg"] = pkg
sys.modules["falpkg.fal_common"] = fake_common
mod = importlib.import_module("falpkg.fal_topaz")
Node = mod.FalTopazUpscale2026


def call(**kw):
    del calls[:]
    kw.setdefault("image", None)
    kw.setdefault("upscale_factor", 2.0)
    kw.setdefault("face_enhancement", False)
    Node().run(**kw)
    return calls[0]


# Wonder 3.5 -> generative; enhancement_strength kept, subject_detection dropped (Wonder 3 only)
ep, a = call(model="gen · Wonder 3.5 (лучший универсал)", enhancement_strength="high",
             subject_detection="Foreground", fix_compression=0.5)
assert ep == mod.GENERATIVE, ep
assert a["model"] == "Wonder 3.5" and a["enhancement_strength"] == "high"
assert "subject_detection" not in a and "fix_compression" not in a, a
assert a["face_enhancement"] is False and a["output_format"] == "png"

# Redefine keeps the whole creative set
ep, a = call(model="gen · Redefine (prompt)", prompt=" a sunny street ", creativity=4, texture=3,
             sharpen=0.3, denoise=0.2, autoprompt="off")
assert ep == mod.GENERATIVE and a["model"] == "Redefine"
assert a["prompt"] == "a sunny street" and a["creativity"] == 4 and a["texture"] == 3
assert a["sharpen"] == 0.3 and a["denoise"] == 0.2 and a["autoprompt"] is False

# precision keeps fix_compression; CGI does not
ep, a = call(model="precision · Low Resolution V2", fix_compression=0.6, denoise=0.4, subject_detection="All")
assert ep == mod.PRECISION and a["model"] == "Low Resolution V2"
assert a["fix_compression"] == 0.6 and a["denoise"] == 0.4 and a["subject_detection"] == "All"
_, a = call(model="precision · CGI (renders)", fix_compression=0.6, sharpen=0.2)
assert "fix_compression" not in a and a["sharpen"] == 0.2, a

# face pass on -> strength/creativity ride along
_, a = call(model="gen · Wonder 3.5 (лучший универсал)", face_enhancement=True, face_strength=0.5, face_creativity=0.2)
assert a["face_enhancement"] is True and a["face_enhancement_strength"] == 0.5
assert a["face_enhancement_creativity"] == 0.2

# Bloom 2 -> creative endpoint, no face fields at all
ep, a = call(model="creative · Bloom 2", creativity=7, color_preservation="on", face_enhancement=True)
assert ep == mod.CREATIVE and a["model"] == "Bloom 2" and a["creativity"] == 7
assert a["color_preservation"] is True and not any(k.startswith("face") for k in a), a

# transparent -> image only, no factor, no model field
ep, a = call(model="alpha · Transparent (keeps alpha)", upscale_factor=3.0)
assert ep == mod.TRANSPARENT and set(a) == {"image_url", "output_format"}, a

# factor passes through, and Redefine's creativity range is enforced
_, a = call(model="precision · Standard V2", upscale_factor=4.0)
assert a["upscale_factor"] == 4.0
try:
    call(model="gen · Redefine (prompt)", creativity=9)
    raise SystemExit("expected a creativity range error")
except ValueError as e:
    assert "1–6" in str(e), e

# every registered model routes to a real endpoint and names a real API model
for label, (endpoint, api_model, allowed) in mod.MODELS.items():
    ep, a = call(model=label)
    assert ep == endpoint, label
    assert (a.get("model") == api_model) if api_model else ("model" not in a), (label, a)

print(f"TOPAZ ARGS OK — {len(mod.MODELS)} models routed")
