# src/core/dijkstra_template.py
#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import heapq

from src.core.types import StepResult, Grid


Cell = Tuple[int, int]  # (col, row)

@dataclass
class DijkstraAlgo:
    name: str = "Dijkstra"

    grid: Optional[Grid] = None
    open_pq: List[Tuple[int, Cell]] = field(default_factory=list)   # (g, cell)
    open_set: set = field(default_factory=set)
    closed_set: set = field(default_factory=set)
    g: Dict[Cell, int] = field(default_factory=dict)
    parent: Dict[Cell, Cell] = field(default_factory=dict)
    popped_count: int = 0
    done: bool = False
    no_path: bool = False
    goal_cell: Optional[Cell] = None

    def init(self, grid: Grid) -> None:
        self.grid = grid
        self.reset()

    def reset(self) -> None:
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

        s = self.grid.start
        self.g[s] = 0
        heapq.heappush(self.open_pq, (0, s))
        self.open_set.add(s)

    def _neighbors4(self, c: Cell) -> List[Cell]:
        x, y = c

        ######################
        ######################
        # FILL CODE HERE for (NEIGHBOR_CANDIDATES)
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
            # FILL CODE HERE for (NEIGHBOR_FILTER)
            # if self.grid.in_bounds(n) and not self.grid.is_block(n):
            #     out.append(n)
            ######################
            ######################

            pass  # keep 'pass' so file runs before they fill; harmless once they add code
        return out

    def _reconstruct_path(self, end: Cell) -> List[Cell]:
        path: List[Cell] = []
        cur = end
        while True:
            path.append(cur)
            if cur == self.grid.start:
                break

            ######################
            ######################
            # FILL CODE HERE for (BACKTRACK_STEP)
            # cur = self.parent[cur]
            ######################
            ######################

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

        g_u, u = heapq.heappop(self.open_pq)
        if g_u != self.g.get(u, float("inf")):
            return StepResult(status="running", current=u, metrics=self._metrics())

        self.popped_count += 1
        if u in self.open_set:
            self.open_set.remove(u)
        self.closed_set.add(u)

        ######################
        ######################
        # FILL CODE HERE for (STOP_CONDITION)
        # if u == self.goal_cell:
        #     self.done = True
        #     path = self._reconstruct_path(u)
        #     return StepResult(status="done", closed=[u], current=u, path=path,
        #                       metrics=self._metrics(path_len=len(path)))
        ######################
        ######################

        opened_now: List[Cell] = []
        for v in self._neighbors4(u):

            ######################
            ######################
            # FILL CODE HERE for (RELAXATION_VALUE)
            # alt = self.g[u] + 1
            ######################
            ######################

            ######################
            ######################
            # FILL CODE HERE for (UPDATE_AND_PUSH)
            # if alt < self.g.get(v, float("inf")):
            #     self.g[v] = alt
            #     self.parent[v] = u
            #     heapq.heappush(self.open_pq, (self.g[v], v))
            #     if v not in self.closed_set and v not in self.open_set:
            #         self.open_set.add(v)
            #         opened_now.append(v)
            ######################
            ######################

            pass

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


'''
What you’ll explain (pseudo, concise)

Neighbor candidates: for (x,y), the four 4-connected moves.

Neighbor filter: keep only “in bounds” and “not a wall”.

Stop condition: “stop when you pop the goal” (not when you first push).

Relaxation value: alt = g[u] + 1 (uniform cost).

Update & push: if better, update g and parent, push to PQ, mark frontier.

Backtrack step: cur = parent[cur] until start; reverse list.

This version forces a bit more reasoning (they decide all three core places: neighbor generation, stopping, and relaxation/update), but it’s still doable for freshers in-session.
'''