"""Build self-contained static HTML dashboard from actual ECCNR measurement JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="color-scheme" content="light dark">
<title>CORTEX-ECCNR Experimental Dashboard</title>
<style>
  :root {
    --bg-light: #f4f6fb; --bg-dark: #0f172a;
    --surf-light: #ffffff; --surf-dark: #1e293b;
    --surf2-light: #e8eef7; --surf2-dark: #0f172a;
    --text-light: #0f172a; --text-dark: #e2e8f0;
    --muted-light: #475569; --muted-dark: #94a3b8;
    --border-light: #cbd5e1; --border-dark: #334155;
    --accent-light: #0369a1; --accent-dark: #38bdf8;
    --bg: var(--bg-light); --surf: var(--surf-light); --surf2: var(--surf2-light);
    --text: var(--text-light); --muted: var(--muted-light); --border: var(--border-light); --accent: var(--accent-light);
    @media (prefers-color-scheme: dark) {
      --bg: var(--bg-dark); --surf: var(--surf-dark); --surf2: var(--surf2-dark);
      --text: var(--text-dark); --muted: var(--muted-dark); --border: var(--border-dark); --accent: var(--accent-dark);
    }
    @supports (color: light-dark(white, black)) {
      --bg: light-dark(var(--bg-light), var(--bg-dark));
      --surf: light-dark(var(--surf-light), var(--surf-dark));
      --surf2: light-dark(var(--surf2-light), var(--surf2-dark));
      --text: light-dark(var(--text-light), var(--text-dark));
      --muted: light-dark(var(--muted-light), var(--muted-dark));
      --border: light-dark(var(--border-light), var(--border-dark));
      --accent: light-dark(var(--accent-light), var(--accent-dark));
    }
    color-scheme: light dark;
  }
  body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 24px; }
  h1, h2 { margin-top: 0; }
  .visually-hidden { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-block-end: 24px; }
  .card { background: var(--surf); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
  .metric { font-size: 1.55rem; font-weight: 700; color: var(--accent); }
  .table-wrapper { overflow: auto; max-inline-size: 100%; container-type: inline-size; border: 1px solid var(--border); border-radius: 10px; background: var(--surf); }
  table { width: 100%; min-inline-size: 680px; border-collapse: separate; border-spacing: 0; }
  caption { text-align: start; padding: 10px 12px; font-weight: 600; background: var(--surf2); border-block-end: 1px solid var(--border); }
  thead { position: sticky; inset-block-start: 0; z-index: 2; background: var(--surf2); }
  th, td { padding: 10px 12px; border-block-end: 1px solid var(--border); text-align: start; font-size: 0.9rem; white-space: nowrap; }
  th:has(> button) { padding: 0; }
  thead th > button { width: 100%; min-block-size: 36px; padding: 10px 12px; background: transparent; border: none; color: inherit; font: inherit; font-weight: 600; cursor: pointer; display: flex; justify-content: space-between; gap: 6px; }
  tbody th { position: sticky; inset-inline-start: 0; z-index: 1; background: var(--surf); }
  tbody tr { view-transition-name: match-element; }
  @media (prefers-reduced-motion: reduce) {
    ::view-transition-group(*), ::view-transition-old(*), ::view-transition-new(*) { animation: none; }
  }
  .sort-icon::after { content: "↕"; opacity: 0.45; }
  th[aria-sort="ascending"] .sort-icon::after { content: "↑"; opacity: 1; }
  th[aria-sort="descending"] .sort-icon::after { content: "↓"; opacity: 1; }
  code { background: var(--surf2); padding: 2px 6px; border-radius: 4px; color: var(--accent); }
  @container (width < 600px) {
    table { min-inline-size: 0; }
    table, thead, tbody, tr, th, td { display: block; white-space: normal; }
    thead { display: none; }
    tr { margin-block-end: 12px; border: 1px solid var(--border); }
    tbody th { position: sticky; inset-block-start: 0; background: var(--surf2); }
    td:nth-child(2)::before { content: "N: "; font-weight: 600; }
    td:nth-child(3)::before { content: "Mean Reward: "; font-weight: 600; }
    td:nth-child(4)::before { content: "Success Rate: "; font-weight: 600; }
    td:nth-child(5)::before { content: "Pred Error: "; font-weight: 600; }
    td:nth-child(6)::before { content: "Env Steps: "; font-weight: 600; }
    td:nth-child(7)::before { content: "Energy Proxy: "; font-weight: 600; }
    td:nth-child(8)::before { content: "Wall (s): "; font-weight: 600; }
  }
</style>
</head>
<body>
<h1>CORTEX-ECCNR: Evolutionary Connectome Consolidation &amp; Neural Replay</h1>
<p>Dataset: <code id="ds-id"></code> | Provenance: <code id="prov-tier"></code> | Topology Hash: <code id="topo-hash"></code></p>

<div class="grid">
  <div class="card"><div>Subgraph Nodes</div><div class="metric" id="m-nodes">-</div></div>
  <div class="card"><div>Induced Directed Edges</div><div class="metric" id="m-edges">-</div></div>
  <div class="card"><div>Successive Handoff vs Single</div><div class="metric" id="m-handoff">-</div></div>
  <div class="card"><div>Total Wall Time</div><div class="metric" id="m-time">-</div></div>
</div>

<p id="sort-status" role="status" class="visually-hidden"></p>
<div tabindex="0" role="region" aria-labelledby="dash-caption" class="table-wrapper">
  <table id="dash-table">
    <caption id="dash-caption">Matched-Budget Experimental Matrix (Sealed Evaluation — Click Headers to Sort)</caption>
    <thead>
      <tr>
        <th scope="col" data-type="text"><button type="button" commandfor="dash-table" command="--sort">Condition <span class="sort-icon" aria-hidden="true"></span></button></th>
        <th scope="col" data-type="number">N</th>
        <th scope="col" data-type="number"><button type="button" commandfor="dash-table" command="--sort">Mean Reward (±95% CI) <span class="sort-icon" aria-hidden="true"></span></button></th>
        <th scope="col" data-type="number"><button type="button" commandfor="dash-table" command="--sort">Success Rate (±95% CI) <span class="sort-icon" aria-hidden="true"></span></button></th>
        <th scope="col" data-type="number"><button type="button" commandfor="dash-table" command="--sort">Pred Error <span class="sort-icon" aria-hidden="true"></span></button></th>
        <th scope="col" data-type="number">Env Steps</th>
        <th scope="col" data-type="number"><button type="button" commandfor="dash-table" command="--sort">Energy Proxy <span class="sort-icon" aria-hidden="true"></span></button></th>
        <th scope="col" data-type="number"><button type="button" commandfor="dash-table" command="--sort">Wall (s) <span class="sort-icon" aria-hidden="true"></span></button></th>
      </tr>
    </thead>
    <tbody id="tbl-body"></tbody>
  </table>
</div>

<script>
const DATA = __DATA_JSON__;
document.getElementById('ds-id').textContent = DATA.dataset_id;
document.getElementById('prov-tier').textContent = DATA.provenance_tier;
document.getElementById('topo-hash').textContent = DATA.topology_hash;
document.getElementById('m-nodes').textContent = DATA.n_nodes;
document.getElementById('m-edges').textContent = DATA.n_edges;
const sa = DATA.successive_action_experiment || {};
document.getElementById('m-handoff').textContent = (sa.delta >= 0 ? '+' : '') + (sa.delta || 0).toFixed(4);
document.getElementById('m-time').textContent = (DATA.total_wall_seconds || 0) + 's';

const tbody = document.getElementById('tbl-body');
(DATA.summary || []).forEach(r => {
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <th scope="row" data-value="${r.condition}">${r.condition}</th>
    <td data-value="${r.n_replicates}">${r.n_replicates}</td>
    <td data-value="${r.mean_reward}">${r.mean_reward.toFixed(4)} ± ${r.ci95_reward.toFixed(4)}</td>
    <td data-value="${r.mean_success}">${(r.mean_success * 100).toFixed(1)}% ± ${(r.ci95_success * 100).toFixed(1)}%</td>
    <td data-value="${r.mean_pred_err}">${r.mean_pred_err.toFixed(4)}</td>
    <td data-value="${r.mean_env_steps}">${Math.round(r.mean_env_steps)}</td>
    <td data-value="${r.mean_energy}">${r.mean_energy.toFixed(3)}</td>
    <td data-value="${r.mean_wall_s}">${r.mean_wall_s.toFixed(2)}s</td>
  `;
  tbody.appendChild(tr);
});

const table = document.getElementById('dash-table');
const statusEl = document.getElementById('sort-status');
const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });

function reorder(container, orderedNodes) {
  const move = Object.hasOwn(Element.prototype, 'moveBefore')
    ? (node) => container.moveBefore(node, null)
    : (node) => container.append(node);
  for (const node of orderedNodes) move(node);
}

function sortBy(button) {
  const th = button.closest('th');
  const dir = th.getAttribute('aria-sort') === 'ascending' ? 'descending' : 'ascending';
  for (const h of table.tHead.rows[0].cells) h.removeAttribute('aria-sort');
  th.setAttribute('aria-sort', dir);
  const idx = th.cellIndex, type = th.dataset.type, factor = dir === 'ascending' ? 1 : -1;
  const keyed = [...tbody.rows].map(row => {
    const raw = row.cells[idx].dataset.value ?? row.cells[idx].textContent.trim();
    return { row, val: type === 'number' ? Number(raw) : raw };
  });
  keyed.sort((a, b) => factor * (type === 'number' ? a.val - b.val : collator.compare(a.val, b.val)));
  const ordered = keyed.map(k => k.row);
  const update = () => reorder(tbody, ordered);
  if (table.startViewTransition) table.startViewTransition(update);
  else if (document.startViewTransition) document.startViewTransition(update);
  else update();
  statusEl.textContent = `Sorted by ${button.textContent.replace(/[↕↑↓]/g, '').trim()}, ${dir}, ${ordered.length} rows`;
}

table.addEventListener('command', (e) => { if (e.command === '--sort') sortBy(e.source); });
if (!('commandForElement' in HTMLButtonElement.prototype)) {
  for (const b of table.tHead.querySelectorAll('button[command="--sort"]')) {
    b.addEventListener('click', () => sortBy(b));
  }
}
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("results/bptt_primary_report.json"))
    parser.add_argument("--out", type=Path, default=Path("dashboard/index.html"))
    args = parser.parse_args()
    data = json.loads(args.report.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(data)), encoding="utf-8")
    print(f"Dashboard written to {args.out}")


if __name__ == "__main__":
    main()
