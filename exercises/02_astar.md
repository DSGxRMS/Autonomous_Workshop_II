# Exercise 2 — A* on Map-2 

**Goal:** Implement **A\*** on `maps/02_small_astar.json` (20×20, 4-connected, uniform cost = 1) and produce the optimal path **start to goal**, showing fewer expansions than Dijkstra.

---

## What you’re given
- Same JSON schema as Map-1:
  - `width`, `height`
  - `start: [col, row]`, `goal: [col, row]`
  - `move: 4`
  - `cells`: `0 = free`, `1 = wall`
- Viewer shows: frontier (open), visited (closed), final path.

---

## What you must implement (conceptually)
1. Maintain **g** (cost from start), **h** (heuristic), **f = g + h**.
2. Use a **priority queue by lowest f** (tie-break rule of your choice; be consistent).
3. Pop node `u`. If `u == goal`, **stop** (first pop of goal is optimal if h is admissible & consistent).
4. For each 4-connected **free** neighbor `v`:
   - Tentative `g_alt = g[u] + 1`
   - If `g_alt < g[v]`, set `g[v] = g_alt`, `parent[v] = u`, and push `v` with `f = g[v] + h(v)`.
5. Keep a **closed** set to avoid re-expanding finalized nodes.

Return the **ordered path** start → goal.

---

## Heuristic to use
- **Manhattan**: `h = |dx| + |dy|`  
  Works for 4-connected, unit step cost. It’s admissible and consistent here.

---

## Rules & constraints
- Movement: **up, down, left, right only**.
- Each move costs **1**.
- Do **not** step on walls (`1`).
- Heuristic must **not overestimate** the true cost.

---

## Acceptance checks
- Valid path (start → goal, 4-connected steps, no walls).
- **Optimality** preserved (Manhattan is admissible here).
- **Efficiency**: report node expansions (open + closed) and compare to Dijkstra (should be smaller).

---

<details>
<summary><strong>Hints (open if stuck)</strong></summary>

- A simple tie-break: when `f` ties, prefer **lower h** (more goal-directed), or prefer **higher g** (deeper paths). Pick one and keep it consistent.
- If you see A\* exploring “like Dijkstra,” your `h` might be all zeros or computed incorrectly.
- Stop when you **pop** the goal, not when you first **push** it.
- You can ignore stale PQ entries by checking if the popped `g` matches your current `g[u]`.
</details>

<details>
<summary><strong>Common mistakes</strong></summary>

- Using **Euclidean** or a scaled heuristic that **overestimates** → breaks optimality.
- Mixing (row, col) vs (col, row) → weird jumps.
- Allowing diagonals by accident (not allowed here).
- Not marking **closed** → repeated expansions.
</details>

---

