"""
What FAL actually charged — per call, per run, per day — behind the cost badge.

FAL puts the billed quantity on every result: the `x-fal-billable-units` response header,
already counted in the endpoint's priced unit, so units x unit price is the real cost of the
call. fal_client drops that header, so once a result is in, the same result URL is fetched a
second time (free) just to read it. Unit prices come from FAL's pricing API, which an ordinary
key may read. Verified 2026-09-17 on post-processing/grain: 1 unit x $0.001 = $0.001, and a
request refused with 422 carried 0 units — FAL does not bill validation refusals.

The run is ComfyUI's prompt, read from its executing context, so no node needs to change.
"Today" is kept in a small file next to this module: both workspaces bind-mount this folder,
so they share one day total, and it survives a container restart. The day turns over at
midnight in FAL_COST_TZ (an IANA zone, e.g. America/Argentina/Buenos_Aires), else in the
container's own local time.

Nothing in here may fail a job. Every step that can go wrong degrades to "cost unknown".
"""
import datetime
import json
import os
import sys
import threading
import urllib.parse
import urllib.request

PRICING_API = "https://api.fal.ai/v1/models/pricing"
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fal_spend.json")

_prices = {}                     # endpoint -> (unit_price, unit, currency), or None if unpriced
_run = {"prompt_id": None, "total": 0.0, "calls": 0}
_lock = threading.Lock()

try:                  # POSIX: both workspaces write one day file, so it is flock-ed across processes
    import fcntl
except ImportError:   # Windows has no fcntl; one ComfyUI process per machine, _lock is enough there
    fcntl = None


def _flock(f, exclusive):
    if fcntl is not None:
        fcntl.flock(f, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)


# --------------------------------------------------------------------------- pricing

def _fetch_price(endpoint):
    req = urllib.request.Request(f"{PRICING_API}?endpoint_id={urllib.parse.quote(endpoint, safe='/')}")
    req.add_header("Authorization", f"Key {os.environ.get('FAL_KEY', '').strip()}")
    with urllib.request.urlopen(req, timeout=15) as r:
        prices = json.load(r).get("prices") or []
    for p in prices:
        if p.get("endpoint_id") == endpoint:
            return float(p["unit_price"]), p.get("unit", "units"), p.get("currency", "USD")
    return None


def unit_price(endpoint):
    """(unit_price, unit, currency) for an endpoint, cached for the life of the process.

    A lookup that fails is not cached, so a network blip costs one unknown price, not a
    blind badge until the next restart.
    """
    if endpoint in _prices:
        return _prices[endpoint]
    try:
        price = _fetch_price(endpoint)
    except Exception as e:  # noqa: BLE001
        print(f"[FAL] cost: no price for {endpoint} ({e})")
        return None
    _prices[endpoint] = price
    return price


def billable_units(handle):
    """The units FAL billed for this request, read off its result response. None if absent."""
    try:
        r = handle.client.get(handle.response_url, timeout=30)
        raw = r.headers.get("x-fal-billable-units")
        return float(raw) if raw is not None else None
    except Exception:  # noqa: BLE001
        return None


def call_cost(endpoint, handle, result):
    """(cost or None, how it was worked out) for one finished call."""
    # The OpenRouter router reports its own cost in the body. That figure is exact and
    # already in dollars, so it wins over any unit arithmetic.
    usage = result.get("usage") if isinstance(result, dict) else None
    if isinstance(usage, dict) and usage.get("cost") is not None:
        return float(usage["cost"]), "reported by the model"
    units = billable_units(handle)
    if units is None:
        return None, "FAL sent no billable units"
    if units == 0:
        return 0.0, "0 units billed"
    price = unit_price(endpoint)
    if price is None:
        return None, f"{units:g} units, price unknown"
    up, unit, _ = price
    return units * up, f"{units:g} {unit} x ${up:g}"


# --------------------------------------------------------------------------- run and day

def _context():
    """(prompt_id, node_id) of whatever ComfyUI is executing, or (None, None) outside it."""
    try:
        from comfy_execution.utils import get_executing_context
        ctx = get_executing_context()
        if ctx is not None:
            return ctx.prompt_id, ctx.node_id
    except Exception:  # noqa: BLE001
        pass
    return None, None


def _today():
    tz = os.environ.get("FAL_COST_TZ", "").strip()
    if tz:
        try:
            from zoneinfo import ZoneInfo
            return datetime.datetime.now(ZoneInfo(tz)).date().isoformat()
        except Exception:  # noqa: BLE001
            pass
    return datetime.date.today().isoformat()


def _read_day(f):
    f.seek(0)
    try:
        state = json.loads(f.read() or "{}")
    except ValueError:
        state = {}
    if state.get("date") != _today():
        state = {"date": _today(), "total": 0.0, "calls": 0}
    return state


def _add_to_day(amount):
    """Add to today's total under an exclusive lock — both workspaces write this one file."""
    with open(STATE, "a+") as f:
        _flock(f, True)
        state = _read_day(f)
        state["total"] = round(state["total"] + amount, 6)
        state["calls"] += 1
        f.seek(0)
        f.truncate()
        f.write(json.dumps(state))
        f.flush()
    return state


def snapshot():
    """What the badge shows on page load, before any call of this session."""
    day = {"date": _today(), "total": 0.0, "calls": 0}
    try:
        with open(STATE, "a+") as f:
            _flock(f, False)
            day = _read_day(f)
    except OSError:
        pass
    return {"run": _run["total"], "today": day["total"], "date": day["date"], "last": ""}


# --------------------------------------------------------------------------- accounting

def _money(v):
    """Three decimals under a dollar, as the badge shows it: $0.001 must not print as $0.00."""
    return f"${v:.3f}" if v < 1 else f"${v:.2f}"


def _send(payload):
    """Push to every open ComfyUI tab of this workspace. Silent outside ComfyUI."""
    try:
        from server import PromptServer
        PromptServer.instance.send_sync("fal.cost", payload)
    except Exception:  # noqa: BLE001
        pass


def _account(endpoint, cost, how, failed=False, ctx=None):
    prompt_id, node_id = ctx or _context()
    with _lock:
        if prompt_id != _run["prompt_id"]:
            _run.update(prompt_id=prompt_id, total=0.0, calls=0)
        _run["total"] = round(_run["total"] + (cost or 0.0), 6)
        _run["calls"] += 1
        run_total = _run["total"]
    try:
        day = _add_to_day(cost or 0.0)
    except OSError as e:
        # the run total still reaches the badge; only the day total goes blank
        print(f"[FAL] cost: today's total not saved ({e})")
        day = {"date": _today(), "total": None}
    shown = "cost unknown" if cost is None else f"${cost:.4f}"
    status = " (failed, but billed)" if failed else ""
    last = f"{endpoint}: {shown}{status} — {how}"
    today = "?" if day["total"] is None else _money(day["total"])
    print(f"[FAL] {last} | run {_money(run_total)} · today {today}")
    _send({"run": run_total, "today": day["total"], "date": day["date"], "last": last,
           "endpoint": endpoint, "cost": cost, "prompt_id": prompt_id, "node": node_id})


def record(endpoint, handle, result):
    """Account one finished call. Never raises."""
    try:
        cost, how = call_cost(endpoint, handle, result)
        _account(endpoint, cost, how)
    except Exception as e:  # noqa: BLE001
        print(f"[FAL] cost: could not account {endpoint} ({e})")


def record_failure(endpoint, handle):
    """A call that failed after FAL queued it: count it only if FAL says it billed units."""
    try:
        units = billable_units(handle)
        if not units:
            return
        price = unit_price(endpoint)
        cost = units * price[0] if price else None
        _account(endpoint, cost, f"{units:g} units billed on a failed request", failed=True)
    except Exception as e:  # noqa: BLE001
        print(f"[FAL] cost: could not account the failed {endpoint} ({e})")


# --------------------------------------------------------------------------- the other pack

# gokayfem's ComfyUI-fal-API (v2) prices a call as "list price x 1 run". That is an estimate:
# a per-second video model is billed for the seconds it rendered. Every call it makes passes
# one function with the endpoint and the request id, which is all FAL needs to tell us what it
# billed, so that function is wrapped and its calls land on the same badge as ours. No fork,
# and if upstream renames the function the wrap is simply not installed.
OTHER_PACK_API = ".nodes.utils.api"
_seen = set()                    # request ids already counted (Fal Collect can fetch one twice)


def _record_other(api_mod, endpoint, request_id, ctx):
    """Off the calling thread: one extra GET must not hold up the other pack's node."""
    try:
        handle = api_mod.FalConfig().get_client().get_handle(endpoint, request_id)
        cost, how = call_cost(endpoint, handle, None)
        _account(endpoint, cost, how, ctx=ctx)
    except Exception as e:  # noqa: BLE001
        print(f"[FAL] cost: could not account {endpoint} ({e})")


def watch_other_pack():
    """Count ComfyUI-fal-API's calls as well. True if the wrap is in place. Never raises."""
    api_mod = next((m for name, m in list(sys.modules.items())
                    if name.endswith(OTHER_PACK_API)
                    and hasattr(m, "_record_ledger_entry") and hasattr(m, "FalConfig")), None)
    if api_mod is None:
        return False
    original = api_mod._record_ledger_entry
    if getattr(original, "_fal_cost_watched", False):
        return True

    def _record_ledger_entry(endpoint, request_id, *args, **kwargs):
        try:
            # free=True is a result re-fetched by request id: nothing new was billed
            free = kwargs.get("free", args[2] if len(args) > 2 else False)
            with _lock:
                fresh = bool(request_id) and not free and request_id not in _seen
                if fresh:
                    _seen.add(request_id)
            if fresh:
                threading.Thread(target=_record_other, daemon=True,
                                 args=(api_mod, endpoint, request_id, _context())).start()
        except BaseException:  # noqa: BLE001 — their call must go through whatever happens here
            pass
        return original(endpoint, request_id, *args, **kwargs)

    _record_ledger_entry._fal_cost_watched = True
    api_mod._record_ledger_entry = _record_ledger_entry
    return True


def install_watch():
    """Wrap the other pack once every pack has loaded (same deferral as fal_retag)."""
    from server import PromptServer

    async def _watch(_app):
        try:
            if watch_other_pack():
                print("[FAL] cost: ComfyUI-fal-API calls are counted on the badge too")
        except BaseException:  # noqa: BLE001 — raising here stops the server booting
            pass

    PromptServer.instance.app.on_startup.append(_watch)


def install_routes():
    """GET /fal/cost (and /api/fal/cost): the badge's state on page load."""
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/fal/cost")
    async def _fal_cost(_request):
        return web.json_response(snapshot())
