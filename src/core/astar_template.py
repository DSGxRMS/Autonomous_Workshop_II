#!/usr/bin/env python3
"""
A* (student template) — one expansion per step() for animation.

Implements the Algorithm API expected by the viewer:
- init(grid) - reset() - step() -> StepResult

Heuristic:
- Manhattan for 4-connected grids.
- Scaled by the minimum traversable cell cost if weights exist (keeps h admissible).

Tie-breaking in the PQ (already wired, students don't edit):
- (f, h, -g, seq, cell): lower f, then lower h, then deeper g, then FIFO by seq.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import heapq
from math import inf

from src.core.types import StepResult, Grid

Cell = Tuple[int, int]  # (col, row)


@dataclass
class AStarAlgo:
    name: str = "A*"

    # Internal state
    grid: Optional[Grid] = None
    open_pq: List[Tuple[int, int, int, int, Cell]] = field(default_factory=list)  # (f, h, -g, seq, cell)
    open_set: set = field(default_factory=set)         # for overlay
    closed_set: set = field(default_factory=set)
    g: Dict[Cell, int] = field(default_factory=dict)
    parent: Dict[Cell, Cell] = field(default_factory=dict)
    popped_count: int = 0
    done: bool = False
    no_path: bool = False
    goal_cell: Optional[Cell] = None
    seq: int = 0  # monotonic counter for PQ stability

    # -------------------- lifecycle --------------------

    def init(self, grid: Grid) -> None:
        """Initialize on a given grid."""
        self.grid = grid
        self.reset()

    def reset(self) -> None:
        """Clear all state and seed with the start node."""
        if self.grid is None:
            return
        self.open_pq.clear()
        self.open_set.clear()
        self.closed_set.clear()
        self.g.clear()
        self.parent.clear()
        self.popped_count = 0
        self.done = False
        self.no_path = False
        self.goal_cell = self.grid.goal
        self.seq = 0

        s = self.grid.start
        self.g[s] = 0
        f0 = self._h(s)
        heapq.heappush(self.open_pq, (f0, self._h(s), -self.g[s], self._bump(), s))
        self.open_set.add(s)

    # -------------------- helpers (kept implemented) --------------------

    def _bump(self) -> int:
        self.seq += 1
        return self.seq

    def _neighbors4(self, c: Cell) -> List[Cell]:
        """Return valid 4-connected neighbors for cell c."""
        x, y = c

        ######################
        ######################
        # FILL CODE HERE (NEIGHBOR_CANDIDATES)
        candidates: List[Cell] = [
            # (x + 1, y),
            # (x - 1, y),
            # (x, y + 1),
            # (x, y - 1),
        ]
        ######################
        ######################

        out: List[Cell] = []
        for n in candidates:

            ######################
            ######################
            # FILL CODE HERE (NEIGHBOR_FILTER)
            # if self.grid.in_bounds(n) and not self.grid.is_block(n):
            #     out.append(n)
            ######################
            ######################

            pass  # harmless until students add the two lines above
        return out

    def _min_traversable_cost(self) -> int:
        """Minimum positive cost among traversable cells. Defaults to 1."""
        if not self.grid or not self.grid.weights:
            return 1
        best = inf
        for _, v in self.grid.weights.items():
            if v == "BLOCK":
                continue
            try:
                c = int(v)
                if c > 0:
                    best = min(best, c)
            except Exception:
                pass
        return 1 if best is inf else int(best)

    def _h(self, c: Cell) -> int:
        """Admissible heuristic for 4-connected grid: Manhattan * min_cell_cost."""
        if self.grid is None:
            return 0
        (x, y) = c
        (gx, gy) = self.grid.goal
        dx = abs(gx - x)
        dy = abs(gy - y)
        return (dx + dy) * self._min_traversable_cost()

    def _reconstruct_path(self, end: Cell) -> List[Cell]:
        path: List[Cell] = []
        cur = end
        while cur in self.parent or cur == self.grid.start:
            path.append(cur)
            if cur == self.grid.start:
                break

            ######################
            ######################
            # FILL CODE HERE (BACKTRACK_STEP)
            # cur = self.parent[cur]
            ######################
            ######################

            break  # keeps the file runnable until students add 1 line above
        path.reverse()
        return path

    # -------------------- main stepping logic --------------------

    def step(self) -> StepResult:
        """
        Run ONE A* expansion step:
          - Pop the lowest-f node.
          - If goal, reconstruct and finish.
          - Else relax neighbors with edge cost = weight(dest) if provided, else 1.
        """
        if self.grid is None:
            return StepResult(status="idle", metrics={"algo": self.name})

        if self.done:
            path = self._reconstruct_path(self.goal_cell) if self.goal_cell else None
            return StepResult(
                status="done",
                path=path,
                metrics=self._metrics(path_len=len(path) if path else 0),
            )

        if self.no_path:
            return StepResult(status="no_path", metrics={"algo": self.name})

        if not self.open_pq:
            self.no_path = True
            return StepResult(status="no_path", metrics={"algo": self.name})

        # Pop best (f, h, -g, seq, u)
        f_u, h_u, neg_g_u, _, u = heapq.heappop(self.open_pq)
        g_u = -neg_g_u

        # Ignore stale pops
        if g_u != self.g.get(u, inf):
            return StepResult(status="running", current=u, metrics=self._metrics())

        # Finalize u
        self.popped_count += 1
        if u in self.open_set:
            self.open_set.remove(u)
        self.closed_set.add(u)

        ######################
        ######################
        # FILL CODE HERE (STOP_CONDITION)
        # if u == self.goal_cell:
        #     self.done = True
        #     path = self._reconstruct_path(u)
        #     return StepResult(
        #         status="done",
        #         closed=[u],
        #         current=u,
        #         path=path,
        #         metrics=self._metrics(path_len=len(path)),
        #     )
        ######################
        ######################

        # Relax neighbors
        opened_now: List[Cell] = []
        for v in self._neighbors4(u):
            try:
                step_cost = self.grid.cost_of(v)  # weight(dest) if provided
            except Exception:
                step_cost = 1

            ######################
            ######################
            # FILL CODE HERE (RELAXATION_VALUE)
            # alt = self.g[u] + step_cost
            ######################
            ######################

            ######################
            ######################
            # FILL CODE HERE (UPDATE_AND_PUSH)
            # if alt < self.g.get(v, inf):
            #     self.g[v] = alt
            #     self.parent[v] = u
            #     f_v = self.g[v] + self._h(v)
            #     heapq.heappush(self.open_pq, (f_v, self._h(v), -self.g[v], self._bump(), v))
            #     if v not in self.closed_set and v not in self.open_set:
            #         self.open_set.add(v)
            #         opened_now.append(v)
            ######################
            ######################

            pass

        return StepResult(
            status="running",
            opened=opened_now,
            closed=[u],
            current=u,
            metrics=self._metrics(),
        )

    # -------------------- metrics --------------------

    def _metrics(self, path_len: int = 0) -> dict:
        return {
            "algo": self.name,
            "popped": self.popped_count,
            "open_size": len(self.open_set),
            "closed_count": len(self.closed_set),
            "path_len": path_len,
            "total_cost": None,  # you can fill this with g[goal] in a weighted map if you want
        }
