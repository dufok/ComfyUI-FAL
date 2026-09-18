// ComfyUI-FAL — cost badge.
//
// A small badge in the corner: what FAL charged for this run and for today.
// The server is the source of truth — after every FAL call it sends a `fal.cost` event
// with the run and day totals it keeps (fal_cost.py), so the badge only displays. On page
// load it asks GET /fal/cost for today's total so far; a new run starts it at $0 for the run.
// Hover shows the last call: endpoint, cost, and how the cost was worked out.
//
// Placement: the bottom-left corner is shared. The sidebar sits there (widening while one of
// its tabs is open), and LiteGraph draws the canvas info (T / I / N / V) just right of it. So
// the badge goes past both, and follows the sidebar as tabs open and close.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const money = (v) => (v == null ? "—" : `$${v < 1 ? v.toFixed(3) : v.toFixed(2)}`);

let badge = null;
const PAST_CANVAS_INFO = 140; // px right of the sidebar, clear of LiteGraph's T/I/N/V lines

function leftEdge() {
  let x = 0;
  for (const el of document.querySelectorAll(".side-tool-bar-container, .side-bar-panel")) {
    const r = el.getBoundingClientRect();
    // only panels docked on the left count; the sidebar can be moved to the right in settings
    if (r.width > 0 && r.height > 0 && r.left < innerWidth / 2) x = Math.max(x, r.right);
  }
  return x;
}

function place() {
  if (badge) badge.style.left = `${leftEdge() + PAST_CANVAS_INFO}px`;
}

function ensureBadge() {
  if (badge) return badge;
  badge = document.createElement("div");
  badge.id = "fal-cost-badge";
  Object.assign(badge.style, {
    position: "fixed",
    left: `${PAST_CANVAS_INFO}px`,
    bottom: "12px",
    transition: "left 0.15s ease-out",
    zIndex: "1000",
    padding: "3px 9px",
    borderRadius: "6px",
    font: "12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace",
    background: "rgba(24, 24, 27, 0.88)",
    color: "#e4e4e7",
    border: "1px solid rgba(255, 255, 255, 0.12)",
    userSelect: "none",
    pointerEvents: "auto",
  });
  document.body.appendChild(badge);
  place();
  setInterval(place, 500); // sidebar tabs open and close without any event to listen to
  window.addEventListener("resize", place);
  return badge;
}

function render(state) {
  const b = ensureBadge();
  b.textContent = `FAL  ${money(state.run)} run · ${money(state.today)} today`;
  b.title = state.last || "No FAL call yet in this session";
}

app.registerExtension({
  name: "ComfyUI-FAL.cost",
  async setup() {
    const state = { run: 0, today: null, last: "" };
    try {
      const res = await api.fetchApi("/fal/cost");
      if (res.ok) Object.assign(state, await res.json());
    } catch (e) {
      console.warn("[FAL cost] could not read today's total", e);
    }
    render(state);

    api.addEventListener("execution_start", () => {
      state.run = 0;
      render(state);
    });
    api.addEventListener("fal.cost", ({ detail }) => {
      Object.assign(state, detail);
      render(state);
    });
  },
});
