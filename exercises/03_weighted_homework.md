# Exercise 3 — Weighted Costs (Homework)

**Goal:** Handle **weighted terrain** on `maps/03_weighted_grass.json` (25×25, 4-connected).  
Cell types: `0 = road (cost 1)`, `1 = wall (BLOCK)`, `2 = grass (cost 3)`.  
Produce the **minimum-cost** path start → goal.

---

## What you’re given
- JSON adds `weights` mapping, e.g. `{ "0": 1, "1": "BLOCK", "2": 3 }`.
- Viewer is the same; it will visualize your explored nodes and final path.

---

## What you must implement (conceptually)
- **Either** Dijkstra **or** A\* with a weight-aware cost.
- For each valid neighbor `v` of `u`:
  - Edge cost = **weight of the destination cell** `cost(v)` (road=1, grass=3).
  - Tentative `g_alt = g[u] + cost(v)`.
  - If `g_alt < g[v]`, update `g[v]`, set `parent[v] = u`, push/update in PQ.
- If using **A\***:
  - Your heuristic must be **admissible** under weights.  
  - Safe choice: **0** (reduces to Dijkstra).  
  - Slightly informed (still admissible): **Manhattan × min_cell_cost** → `h = (|dx| + |dy|) * 1` (since min cost is 1 on road).

Return the **ordered path** start → goal and the **total path cost**.

---

## Rules & constraints
- 4-connected moves only.
- Walls (`1`) are **impassable**.
- Costs are **cell-based** (entering a cell adds its weight).
- If you use a heuristic, it must **never overestimate** the remaining cost.

---

## Acceptance checks
- Valid path (no walls, 4-connected).
- Report **total cost** (sum of entered cell weights excluding the start cell; or define it clearly and be consistent).
- Your path should avoid grass when it’s cheaper overall to detour to a road “bridge”.
- Handles “no path” gracefully.

---

<details>
<summary><strong>Hints (open if stuck)</strong></summary>

- Define cost convention **explicitly** (common: cost is the weight of the **destination** cell). Keep it consistent.
- For A\*: `h = Manhattan * min_weight` is admissible. Here `min_weight = 1`, so `h = Manhattan`.
- If your A\* solution looks sub-optimal vs Dijkstra, your `h` may be **too large** (overestimating).
- Always track **parent pointers** for path reconstruction.
</details>

<details>
<summary><strong>Common mistakes</strong></summary>

- Forgetting to use the **weights** when relaxing neighbors (still using +1).
- Charging the start cell’s weight twice (be explicit in your convention).
- Using **Manhattan** directly when your **min cell cost > 1** (not our case here, but remember the rule).
- Treating walls as high cost instead of **BLOCK** — they must be **impassable**.
</details>

---

## What to submit
- **Total cost** of your path (numeric).
- **Path length** (steps) for context.
- A screenshot of your final path.
- (Optional) Node expansions for Dijkstra vs A\* under weights.
