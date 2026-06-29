#!/usr/bin/env python3
"""
FAL catalog registry — the "list" + new-model checker for the ComfyUI-FAL pack.

The whole FAL catalog (1400+ models) is reachable with the normal FAL_KEY; per-model
parameter schemas are public. This script caches the catalog and lets you search it,
list favorites, diff for new models, and dump a model's input schema (the basis for
auto-generating nodes later).

Run where FAL_KEY is set — easiest inside the ComfyUI container:
    docker exec <cid> python /app/custom_nodes/ComfyUI-FAL/fal_registry.py <cmd>

Commands:
    fetch                 paginate the whole catalog -> fal_catalog.json (+ category counts)
    search <term>         case-insensitive match on endpoint_id / name / category / description
    favorites             list models you marked is_favorited on fal.ai
    category <name>       list models in a category (e.g. image-to-3d)
    diff                  compare live FAL vs cached fal_catalog.json -> added / removed
    schema <endpoint_id>  print the Input parameters (name, type, default) from queue-OpenAPI
"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

CATALOG_API = "https://api.fal.ai/v1/models"
SCHEMA_API = "https://fal.ai/api/openapi/queue/openapi.json"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fal_catalog.json")


def _key():
    k = os.environ.get("FAL_KEY", "").strip()
    if not k:
        sys.exit("FAL_KEY not set in the environment.")
    return k


def _get(url, auth=True, timeout=30):
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    if auth:
        req.add_header("Authorization", f"Key {_key()}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:300]}


# --------------------------------------------------------------------------- catalog

def fetch_all():
    """Cursor-paginate the full model catalog -> list of {endpoint_id, metadata}."""
    models, cursor, pages = [], None, 0
    while True:
        url = f"{CATALOG_API}?page_size=100" + (f"&cursor={urllib.parse.quote(cursor)}" if cursor else "")
        st, j = _get(url)
        if st != 200:
            sys.exit(f"catalog fetch failed ({st}): {j}")
        models.extend(j.get("models", []))
        pages += 1
        cursor = j.get("next_cursor")
        if not j.get("has_more") or not cursor:
            break
        if pages > 100:  # safety
            break
    return models


def save(models):
    with open(CACHE, "w") as f:
        json.dump({"models": models}, f, ensure_ascii=False, indent=0)


def load():
    if not os.path.isfile(CACHE):
        sys.exit(f"no cache at {CACHE} — run `fetch` first.")
    with open(CACHE) as f:
        return json.load(f).get("models", [])


def _counts(models):
    cats = {}
    for m in models:
        c = m.get("metadata", {}).get("category", "?")
        cats[c] = cats.get(c, 0) + 1
    return dict(sorted(cats.items(), key=lambda kv: -kv[1]))


def _fmt(m):
    md = m.get("metadata", {})
    fav = "★" if md.get("is_favorited") else " "
    return f"  {fav} {m['endpoint_id']:55} [{md.get('category','?')}]  {md.get('display_name','')}"


# --------------------------------------------------------------------------- commands

def cmd_fetch(_):
    models = fetch_all()
    save(models)
    print(f"cached {len(models)} models -> {CACHE}")
    favs = [m for m in models if m.get("metadata", {}).get("is_favorited")]
    print(f"favorites: {len(favs)}")
    for c, n in _counts(models).items():
        print(f"  {n:5}  {c}")


def cmd_search(args):
    term = (args[0] if args else "").lower()
    if not term:
        sys.exit("usage: search <term>")
    hits = [
        m for m in load()
        if term in json.dumps(m.get("metadata", {}), ensure_ascii=False).lower()
        or term in m["endpoint_id"].lower()
    ]
    print(f"{len(hits)} match '{term}':")
    for m in hits[:80]:
        print(_fmt(m))
    if len(hits) > 80:
        print(f"  … +{len(hits) - 80} more")


def cmd_favorites(_):
    favs = [m for m in load() if m.get("metadata", {}).get("is_favorited")]
    print(f"{len(favs)} favorites:")
    for m in favs:
        print(_fmt(m))


def cmd_category(args):
    cat = (args[0] if args else "").lower()
    hits = [m for m in load() if m.get("metadata", {}).get("category", "").lower() == cat]
    print(f"{len(hits)} in category '{cat}':")
    for m in hits:
        print(_fmt(m))


def cmd_diff(_):
    old_ids = {m["endpoint_id"] for m in load()}
    live = fetch_all()
    live_ids = {m["endpoint_id"] for m in live}
    added = sorted(live_ids - old_ids)
    removed = sorted(old_ids - live_ids)
    print(f"NEW on FAL ({len(added)}):")
    for eid in added:
        print("  +", eid)
    print(f"GONE ({len(removed)}):")
    for eid in removed:
        print("  -", eid)
    print("(run `fetch` to update the cache once reviewed)")


def cmd_schema(args):
    eid = args[0] if args else ""
    if not eid:
        sys.exit("usage: schema <endpoint_id>")
    st, j = _get(f"{SCHEMA_API}?endpoint_id={urllib.parse.quote(eid, safe='')}", auth=False)
    if st != 200:
        sys.exit(f"schema fetch failed ({st}): {j}")
    schemas = j.get("components", {}).get("schemas", {})
    # find the Input model (named "*Input" or referenced by the submit path body)
    inp = next((v for k, v in schemas.items() if k.lower().endswith("input")), None)
    out = next((v for k, v in schemas.items() if k.lower().endswith("output")), None)
    if not inp:
        sys.exit(f"no Input schema found; components: {list(schemas)}")
    print(f"INPUT for {eid}:")
    req = set(inp.get("required", []))
    for name, spec in inp.get("properties", {}).items():
        t = spec.get("type") or ("enum" if "enum" in spec else (spec.get("anyOf") and "anyOf") or "?")
        extra = []
        if name in req:
            extra.append("REQUIRED")
        if "default" in spec:
            extra.append(f"default={spec['default']!r}")
        if "enum" in spec:
            extra.append(f"enum={spec['enum']}")
        print(f"  {name:28} {str(t):10} {' '.join(extra)}")
    if out:
        print("OUTPUT properties:", list(out.get("properties", {})))


_CMDS = {
    "fetch": cmd_fetch, "search": cmd_search, "favorites": cmd_favorites,
    "category": cmd_category, "diff": cmd_diff, "schema": cmd_schema,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in _CMDS:
        sys.exit(__doc__)
    _CMDS[sys.argv[1]](sys.argv[2:])
