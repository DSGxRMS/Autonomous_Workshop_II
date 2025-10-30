# src/core/types.py
#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

Cell = Tuple[int, int]  # (col, row)

@dataclass
class Grid:
    width: int
    height: int
    cells: List[List[int]]             # [row][col]
    start: Cell
    goal: Cell
    move: int = 4
    weights: Dict[str, Any] = field(default_factory=dict)

    def in_bounds(self, c: Cell) -> bool:
        x, y = c
        return 0 <= x < self.width and 0 <= y < self.height

    def is_block(self, c: Cell) -> bool:
        x, y = c
        v = self.cells[y][x]
        w = self.weights.get(str(v), 1)
        return w == "BLOCK" or v == 1

    def cost_of(self, c: Cell) -> int:
        x, y = c
        v = self.cells[y][x]
        w = self.weights.get(str(v), 1)
        if w == "BLOCK" or v == 1:
            raise ValueError("Asked cost of a BLOCK cell")
        return int(w)

@dataclass
class StepResult:
    status: str                   # "idle" | "running" | "done" | "no_path"
    opened: List[Cell] = field(default_factory=list)
    closed: List[Cell] = field(default_factory=list)
    current: Optional[Cell] = None
    path: Optional[List[Cell]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
