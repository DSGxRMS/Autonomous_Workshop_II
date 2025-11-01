# Exercise 1 — Dijkstra on Map-1 (Intro)

**Goal:** Implement Dijkstra’s shortest path on `maps/01_intro_dijkstra.json` and produce the optimal path from **start to goal**.

---

## What you’re given
- A fixed JSON map format:
  - `width`, `height`
  - `start: [col, row]`
  - `goal: [col, row]`
  - `move: 4` (only up/down/left/right)
  - `cells`: 2D array (`0 = free`, `1 = wall`)
- A viewer (we’ll run it in the session) that:
  - Draws the grid, start (S), goal (G)
  - Shows visited nodes (closed), frontier (open)
  - Shows the final path when you return it

---

## What you must implement (conceptually)
1. Maintain a **cost map `g`**: distance from start to each cell (init to ∞ except start = 0).
2. Use a **priority queue** keyed by `g` (lowest first).
3. Pop the current node `u`; if `u == goal`, **stop** (first pop of goal is optimal).
4. For each 4-connected **free** neighbor `v` of `u`:
   - Tentative cost `alt = g[u] + 1`
   - If `alt < g[v]`, update `g[v] = alt` and set **parent[v] = u**, then push `v`.
5. Keep a **visited/closed** set so you don’t re-expand nodes you’ve already finalized.
6. If the queue empties without reaching goal → **No path** (shouldn’t happen on Map-1).

Return the **path** as an **ordered list of grid cells** from start → goal using your parent links.

---

## Rules & constraints
- Movement: **only** up, down, left, right.
- Cost: **each move costs 1** (uniform).
- You **must not** step on walls (`1`).
- If multiple optimal paths exist, any one is fine.

---

## Acceptance checks
- The path:
  - Starts at `start`, ends at `goal`
  - Moves only 1 cell at a time in 4-connected steps
  - Never crosses a wall
- **Optimality**: path length equals the reference length (revealed after submission).
- **Stability**: handles “no path” maps by returning **No path** cleanly.

---

<details>
<summary><strong>Hints (open if stuck)</strong></summary>

- Duplicates in the priority queue are fine: on pop, ignore entries whose `g` is stale.
- Use simple 2D arrays or dicts for `g`, `parent`, and `visited`.
- Stay consistent with **(col, row)** indexing; don’t mix with (row, col).
- Stop as soon as you **pop** the goal from the queue (that guarantees optimality here).
</details>

<details>
<summary><strong>Common mistakes</strong></summary>

- Not marking nodes **closed** → repeated expansions / stalls.
- Forgetting to store **parent pointers** → can’t reconstruct the path.
- Accidentally allowing **diagonals** (not allowed in this exercise).
- Index mix-ups causing “teleporting” or invalid neighbors.
</details>

---
