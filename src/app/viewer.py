# src/app/viewer.py
#!/usr/bin/env python3
"""
Minimal grid viewer for the workshop.

- Loads maps/{01_intro_dijkstra.json, 02_small_astar.json, 03_weighted_grass.json}
- Renders cells, start/goal, (open/closed/path) overlays
- Controls:
    [1]/[2]/[3]     -> switch map
    [D]/[A]         -> select algorithm (Dijkstra / A*)
    [R]             -> reset
    [SPACE]         -> run/pause
    [N]             -> single step
    [+]/[-]         -> speed up / slow down (steps/sec)
    [ESC] or [Q]    -> quit
"""
import sys, json, time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import pygame
from src.core.dijkstra_student import DijkstraAlgo
from src.core.astar_student import AStarAlgo  # <-- new

# ---------- Config ----------
MAP_DIR = Path(__file__).resolve().parents[2] / "maps"
MAP_FILES = {
    "01_intro_dijkstra": MAP_DIR / "01_intro_dijkstra.json",
    "02_small_astar":    MAP_DIR / "02_small_astar.json",
    "03_weighted_grass": MAP_DIR / "03_weighted_grass.json",
}
PANEL_W = 260
GRID_MARGIN = 16
CELL_SIZE_DEFAULT = 24   # auto-tuned to fit screen height if needed
FONT_NAME = None         # default pygame font

# Colors
WHITE       = (255,255,255)
BLACK       = (  0,  0,  0)
GRAY_20     = (220,220,220)
GRAY_60     = (140,140,140)
BLUE        = ( 70,130,180)
RED         = (220, 50, 47)
GREEN       = ( 46,139, 87)
YELLOW      = (255,215,  0)
GRASS_GREEN = (144, 238, 144)  # LightGreen for cell value 2

# ---------- Data types ----------
Cell = Tuple[int,int]  # (col,row)

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
        x,y = c
        return 0 <= x < self.width and 0 <= y < self.height

    def is_block(self, c: Cell) -> bool:
        x,y = c
        v = self.cells[y][x]
        # treat explicit weights BLOCK or literal 1-walls as blocked
        w = self.weights.get(str(v), 1)
        return w == "BLOCK" or v == 1

    def cost_of(self, c: Cell) -> int:
        x,y = c
        v = self.cells[y][x]
        w = self.weights.get(str(v), 1)
        if w == "BLOCK" or v == 1:
            raise ValueError("Asked cost of a BLOCK cell")
        return int(w)

# ---------- Algorithm hook types ----------
@dataclass
class StepResult:
    status: str                   # "idle" | "running" | "done" | "no_path"
    opened: List[Cell] = field(default_factory=list)
    closed: List[Cell] = field(default_factory=list)
    current: Optional[Cell] = None
    path: Optional[List[Cell]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

class NoAlgo:
    """Placeholder for algorithms not yet implemented."""
    def __init__(self, name="(no algorithm)"):
        self.name = name
        self._grid: Optional[Grid] = None
        self._steps = 0

    def init(self, grid: Grid) -> None:
        self._grid = grid
        self._steps = 0

    def reset(self) -> None:
        self._steps = 0

    def step(self) -> StepResult:
        self._steps += 1
        return StepResult(
            status="idle",
            metrics={
                "algo": self.name,
                "popped": 0,
                "open_size": 0,
                "closed_count": 0,
                "path_len": 0,
                "total_cost": None,
                "steps": self._steps,
            },
        )

# ---------- Loader ----------
def load_map(path: Path) -> Grid:
    with open(path, "r") as f:
        data = json.load(f)
    width  = int(data["width"])
    height = int(data["height"])
    start  = tuple(data["start"])  # [col,row]
    goal   = tuple(data["goal"])
    move   = int(data.get("move", 4))
    cells  = data["cells"]
    weights = data.get("weights", {})
    # sanity checks
    assert len(cells) == height and all(len(r) == width for r in cells), "cells size mismatch"
    sx, sy = start
    gx, gy = goal
    assert 0 <= sx < width and 0 <= sy < height, "start out of bounds"
    assert 0 <= gx < width and 0 <= gy < height, "goal out of bounds"
    return Grid(width, height, cells, start, goal, move, weights)

# ---------- Viewer ----------
class Viewer:
    def __init__(self, grid: Grid):
        pygame.init()
        self.grid = grid
        self.cell_size = self._auto_cell_size(grid)
        self.font_small = pygame.font.Font(FONT_NAME, 14)
        self.font = pygame.font.Font(FONT_NAME, 18)
        self.font_big = pygame.font.Font(FONT_NAME, 22)

        grid_px_w = GRID_MARGIN*2 + grid.width * self.cell_size
        grid_px_h = GRID_MARGIN*2 + grid.height* self.cell_size
        self.canvas_rect = pygame.Rect(0, 0, grid_px_w, grid_px_h)
        win_w = grid_px_w + PANEL_W
        win_h = grid_px_h
        self.screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption("Pathfinding Workshop Viewer")

        # overlays to be filled by algorithms
        self.open_set: set[Cell] = set()
        self.closed_set: set[Cell] = set()
        self.path: List[Cell] = []

        # state
        self.running = False
        self.clock = pygame.time.Clock()
        self.steps_per_sec = 8
        self.state = "Idle"
        self.selected_map_key = self._infer_map_key()
        self.selected_algo = "Dijkstra"   # default label

        # default algo (Dijkstra) for Map-1
        self.algo = DijkstraAlgo(name="Dijkstra")
        self.algo.init(self.grid)
        self._last_metrics = {
            "algo": self.selected_algo,
            "popped": 0,
            "open_size": 0,
            "closed_count": 0,
            "path_len": 0,
            "total_cost": None,
            "steps": 0,
        }

    def _infer_map_key(self) -> str:
        for k,p in MAP_FILES.items():
            try:
                g = load_map(p)
            except Exception:
                continue
            if (g.width, g.height) == (self.grid.width, self.grid.height) and g.start == self.grid.start and g.goal == self.grid.goal:
                return k
        return "custom"

    def _auto_cell_size(self, grid: Grid) -> int:
        # fit height ~ 720 if possible
        target_h = 720 - GRID_MARGIN*2
        size = max(10, min(CELL_SIZE_DEFAULT, target_h // grid.height))
        return size

    # --------- UI Loop ----------
    def run(self):
        while True:
            self._handle_events()
            if self.running:
                self._tick_algorithm()
            self._draw()
            self.clock.tick(60)  # UI refresh FPS

    def _tick_algorithm(self):
        # throttle by steps/sec
        t0 = time.time()
        step_interval = 1.0 / max(1, self.steps_per_sec)
        if not hasattr(self, "_last_step_t"):
            self._last_step_t = 0.0
        if t0 - self._last_step_t >= step_interval:
            self._last_step_t = t0
            self._do_step()

    def _do_step(self):
        res = self.algo.step()
        # merge overlays
        for c in res.opened: self.open_set.add(c)
        for c in res.closed: self.closed_set.add(c)
        if res.path is not None: self.path = res.path
        # update state
        if res.status == "done":
            self.state = "Done"
            self.running = False
        elif res.status == "no_path":
            self.state = "No path"
            self.running = False
        elif res.status in ("running","idle"):
            self.state = "Running" if self.running else "Idle"
        # store last metrics for panel
        if res.metrics:
            self._last_metrics = res.metrics

    # --------- Event handling ----------
    def _handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit(); sys.exit(0)
                elif e.key == pygame.K_SPACE:
                    # toggle run/pause
                    if self.state in ("Done","No path"):
                        pass
                    else:
                        self.running = not self.running
                        self.state = "Running" if self.running else "Paused"
                elif e.key == pygame.K_r:
                    self._reset()
                elif e.key == pygame.K_n:
                    self._do_step()
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self.steps_per_sec = min(60, self.steps_per_sec + 1)
                elif e.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                    self.steps_per_sec = max(1, self.steps_per_sec - 1)
                elif e.key == pygame.K_1:
                    self._switch_map("01_intro_dijkstra")
                elif e.key == pygame.K_2:
                    self._switch_map("02_small_astar")
                elif e.key == pygame.K_3:
                    self._switch_map("03_weighted_grass")
                elif e.key == pygame.K_d:
                    self._switch_algo("Dijkstra")
                elif e.key == pygame.K_a:
                    self._switch_algo("A*")

    def _switch_map(self, key: str):
        if key not in MAP_FILES: return
        try:
            self.grid = load_map(MAP_FILES[key])
            # re-init viewer cleanly to recompute sizes & reset state
            self.__init__(self.grid)
            self.selected_map_key = key
        except Exception as ex:
            print(f"Failed to load map {key}: {ex}")

    def _switch_algo(self, label: str):
        self.selected_algo = label
        if label == "Dijkstra":
            self.algo = DijkstraAlgo(name="Dijkstra")
        elif label == "A*":
            self.algo = AStarAlgo(name="A*")
        else:
            self.algo = NoAlgo(name=label)
        self.algo.init(self.grid)
        self._reset_overlays()

    def _reset_overlays(self):
        self.open_set.clear()
        self.closed_set.clear()
        self.path = []
        self._last_metrics = {
            "algo": self.selected_algo,
            "popped": 0,
            "open_size": 0,
            "closed_count": 0,
            "path_len": 0,
            "total_cost": None,
            "steps": 0,
        }

    def _reset(self):
        self.running = False
        self.state = "Idle"
        self.algo.reset()
        self._reset_overlays()

    # --------- Rendering ----------
    def _draw(self):
        self.screen.fill(GRAY_20)
        self._draw_grid()
        self._draw_panel()
        pygame.display.flip()

    def _draw_grid(self):
        cs = self.cell_size
        # base cells
        for row in range(self.grid.height):
            for col in range(self.grid.width):
                v = self.grid.cells[row][col]
                rect = pygame.Rect(GRID_MARGIN + col*cs, GRID_MARGIN + row*cs, cs, cs)

                # base color by cell value
                if str(v) in self.grid.weights and self.grid.weights[str(v)] == "BLOCK":
                    color = BLACK
                elif v == 1:
                    color = BLACK
                elif v == 2:
                    color = GRASS_GREEN
                else:
                    color = WHITE

                pygame.draw.rect(self.screen, color, rect)
                # thin grid line
                pygame.draw.rect(self.screen, GRAY_60, rect, 1)

        # overlays: closed then open
        for (col,row) in self.closed_set:
            rect = pygame.Rect(GRID_MARGIN + col*cs, GRID_MARGIN + row*cs, cs, cs)
            s = pygame.Surface((cs, cs), pygame.SRCALPHA)
            s.fill((120,120,120,120))  # gray overlay
            self.screen.blit(s, rect.topleft)

        for (col,row) in self.open_set:
            rect = pygame.Rect(GRID_MARGIN + col*cs, GRID_MARGIN + row*cs, cs, cs)
            s = pygame.Surface((cs, cs), pygame.SRCALPHA)
            s.fill((173,216,230,140))  # blue-ish overlay
            self.screen.blit(s, rect.topleft)

        # path polyline
        if len(self.path) >= 2:
            pts = []
            for (col,row) in self.path:
                cx0 = GRID_MARGIN + col*cs + cs//2
                cy0 = GRID_MARGIN + row*cs + cs//2
                pts.append((cx0, cy0))
            pygame.draw.lines(self.screen, GREEN, False, pts, 4)

        # start / goal
        self._draw_badge(self.grid.start, "S", BLUE)
        self._draw_badge(self.grid.goal,  "G", RED)

    def _draw_badge(self, cell: Cell, label: str, color: Tuple[int,int,int]):
        cs = self.cell_size
        col,row = cell
        cx = GRID_MARGIN + col*cs + cs//2
        cy = GRID_MARGIN + row*cs + cs//2
        pygame.draw.circle(self.screen, color, (cx,cy), max(10, cs//2 - 2))
        txt = self.font_small.render(label, True, (255,255,255))
        self.screen.blit(txt, txt.get_rect(center=(cx,cy)))

    def _draw_panel(self):
        # right panel background
        grid_px_w = GRID_MARGIN*2 + self.grid.width * self.cell_size
        panel_rect = pygame.Rect(grid_px_w, 0, PANEL_W, GRID_MARGIN*2 + self.grid.height*self.cell_size)
        pygame.draw.rect(self.screen, (245,245,245), panel_rect)

        x0 = panel_rect.x + 14
        y = panel_rect.y + 16

        def line(text, big=False, color=(0,0,0)):
            nonlocal y
            f = self.font_big if big else self.font
            surf = f.render(text, True, color)
            self.screen.blit(surf, (x0, y))
            y += surf.get_height() + 8

        line("Pathfinding Viewer", big=True, color=BLUE)
        line(f"Map: {self.selected_map_key}")
        line(f"Algo: {self.selected_algo}")
        line(f"State: {self.state}")
        line(f"Speed: {self.steps_per_sec} steps/s")

        y += 8
        line("Metrics", big=False, color=YELLOW)
        m = getattr(self, "_last_metrics", {
            "popped":0,"open_size":0,"closed_count":0,"path_len":0,"total_cost":None,"steps":0,"algo":self.selected_algo
        })
        line(f"Popped: {m.get('popped',0)}")
        line(f"Open size: {m.get('open_size',0)}")
        line(f"Closed: {m.get('closed_count',0)}")
        line(f"Path len: {m.get('path_len',0)}")
        tc = m.get("total_cost", None)
        if tc is not None:
            line(f"Total cost: {tc}")

        y += 8
        line("Controls", big=False, color=YELLOW)
        controls = [
            "[1/2/3] Switch Map",
            "[D/A]   Choose Algo",
            "[R]     Reset",
            "[SPACE] Run / Pause",
            "[N]     Step once",
            "[+/-]   Speed",
            "[Q/ESC] Quit",
        ]
        for c in controls:
            line(c)

# ---------- main ----------
def main():
    # default map: 01
    try:
        grid = load_map(MAP_FILES["01_intro_dijkstra"])
    except Exception as ex:
        print(f"Failed to load default map: {ex}")
        sys.exit(1)
    Viewer(grid).run()

if __name__ == "__main__":
    main()
