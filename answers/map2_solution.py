# src/core/astar_solution.py
#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import heapq
from math import inf

from src.app.viewer import StepResult, Grid

Cell = Tuple[int, int]  # (col, row)

@dataclass
class AStarAlgo:
    name: str = "A*"

    grid: Optional[Grid] = None
    open_pq: List[Tuple[int, int, int, int, Cell]] = field(default_factory=list)  # (f, h, -g, seq, cell)
    open_set: set = field(default_factory=set)
    closed_set: set = field(default_factory=set)
    g: Dict[Cell, int] = field(default_factory=dict)
    parent: Dict[Cell, Cell] = field(default_factory=dict)
    popped_count: int = 0
    done: bool = False
    no_path: bool = False
    goal_cell: Optional[Cell] = None
    seq: int = 0

    def _bump(self) -> int:
        self.seq += 1
        return self.seq

    def init(self, grid: Grid) -> None:
        self.grid = grid
        self.reset()

    def reset(self) -> None:
        if self.grid is None:
            return
        self.open_pq.clear(); self.open_set.clear(); self.closed_set.clear()
        self.g.clear(); self.parent.clear()
        self.popped_count = 0; self.done = False; self.no_path = False
        self.goal_cell = self.grid.goal; self.seq = 0

        s = self.grid.start
        self.g[s] = 0
        h0 = self._h(s)
        heapq.heappush(self.open_pq, (h0, h0, -0, self._bump(), s))
        self.open_set.add(s)

    def _neighbors4(self, c: Cell) -> List[Cell]:
        x, y = c
        cand = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
        out: List[Cell] = []
        for n in cand:
            if self.grid.in_bounds(n) and not self.grid.is_block(n):
                out.append(n)
        return out

    def _h(self, c: Cell) -> int:
        (x, y) = c
        (gx, gy) = self.grid.goal
        return abs(gx - x) + abs(gy - y)  # Manhattan for 4-connected grid

    def _reconstruct_path(self, end: Cell) -> List[Cell]:
        path: List[Cell] = []
        cur = end
        while cur in self.parent or cur == self.grid.start:
            path.append(cur)
            if cur == self.grid.start:
                break
            cur = self.parent[cur]
        path.reverse()
        return path

    def step(self) -> StepResult:
        if self.grid is None:
            return StepResult(status="idle", metrics={"algo": self.name})

        if self.done:
            path = self._reconstruct_path(self.goal_cell) if self.goal_cell else None
            return StepResult(status="done", path=path,
                             metrics=self._metrics(path_len=len(path) if path else 0))

        if self.no_path:
            return StepResult(status="no_path", metrics={"algo": self.name})

        if not self.open_pq:
            self.no_path = True
            return StepResult(status="no_path", metrics={"algo": self.name})

        f_u, h_u, neg_g_u, _, u = heapq.heappop(self.open_pq)
        g_u = -neg_g_u
        if g_u != self.g.get(u, inf):
            return StepResult(status="running", current=u, metrics=self._metrics())

        self.popped_count += 1
        if u in self.open_set: self.open_set.remove(u)
        self.closed_set.add(u)

        if u == self.goal_cell:
            self.done = True
            path = self._reconstruct_path(u)
            return StepResult(status="done", closed=[u], current=u, path=path,
                              metrics=self._metrics(path_len=len(path)))

        opened_now: List[Cell] = []
        for v in self._neighbors4(u):
            step_cost = 1
            alt = self.g[u] + step_cost
            if alt < self.g.get(v, inf):
                self.g[v] = alt
                self.parent[v] = u
                f_v = self.g[v] + self._h(v)
                heapq.heappush(self.open_pq, (f_v, self._h(v), -self.g[v], self._bump(), v))
                if v not in self.closed_set and v not in self.open_set:
                    self.open_set.add(v)
                    opened_now.append(v)

        return StepResult(status="running", opened=opened_now, closed=[u], current=u,
                          metrics=self._metrics())

    def _metrics(self, path_len: int = 0) -> dict:
        return {
            "algo": self.name,
            "popped": self.popped_count,
            "open_size": len(self.open_set),
            "closed_count": len(self.closed_set),
            "path_len": path_len,
            "total_cost": None,
        }
