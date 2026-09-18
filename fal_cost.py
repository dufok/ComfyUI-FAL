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
import fcntl
import json
import os
import threading
import urllib.parse
import urllib.request

PRICING_API = "https://api.fal.ai/v1/models/pricing"
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fal_spend.json")

_prices = {}                     # endpoint -> (unit_price, unit, currency), or None if unpriced
_run = {"prompt_id": None, "total": 0.0, "calls": 0}
_lock = threading.Lock()


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
        fcntl.flock(f, fcntl.LOCK_EX)
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
            fcntl.flock(f, fcntl.LOCK_SH)
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


def _account(endpoint, cost, how, failed=False):
    prompt_id, node_id = _context()
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


def install_routes():
    """GET /fal/cost (and /api/fal/cost): the badge's state on page load."""
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/fal/cost")
    async def _fal_cost(_request):
        return web.json_response(snapshot())
