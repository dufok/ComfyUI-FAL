// ComfyUI-FAL — cost badge.
//
// A small badge in the corner: what FAL charged for this run and for today.
// The server is the source of truth — after every FAL call it sends a `fal.cost` event
// with the run and day totals it keeps (fal_cost.py), so the badge only displays. On page
// load it asks GET /fal/cost for today's total so far; a new run starts it at $0 for the run.
// Hover shows the last call: endpoint, cost, and how the cost was worked out.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const money = (v) => (v == null ? "—" : `$${v < 1 ? v.toFixed(3) : v.toFixed(2)}`);

let badge = null;

function ensureBadge() {
  if (badge) return badge;
  badge = document.createElement("div");
  badge.id = "fal-cost-badge";
  Object.assign(badge.style, {
    position: "fixed",
    left: "12px",
    bottom: "12px",
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
