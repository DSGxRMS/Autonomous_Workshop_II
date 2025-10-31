# src/app/viewer.py
#!/usr/bin/env python3
"""
Pathfinding Workshop Viewer — Race Theme

- Maps: 01_intro_dijkstra.json, 02_small_astar.json, 03_weighted_grass.json
- Keyboard:
    [1]/[2]/[3]  -> switch map
    [D]/[A]      -> select algorithm (Dijkstra / A*)
    [SPACE]      -> run/pause
    [N]          -> single step
    [R]          -> reset
    [+]/[-]      -> steps/sec
    [Q]/[ESC]    -> quit

Mode switch:
- ENV: WORKSHOP_MODE=student|instructor
- CLI: --mode=student|instructor

Assets: put these in repo_root/assets/
  asphalt.png
  grass.png
  tire_stack.png
  f1.png
  checkered_flag.png
"""

# --- bootstrap import path so `from src...` works when run as a script ---
import sys, os, json, time
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# -------------------------------------------------------------------------

from typing import List, Tuple, Optional, Dict, Any
import pygame

# --- optional theme/skin plug-in ---
try:
    from src.app import theme_skin as THEME  # same folder, optional
except Exception:
    THEME = None

# -------------------- Assets (you provide these files) --------------------
ASSETS_DIR      = Path(__file__).resolve().parents[2] / "assets"
ASPHALT_IMG     = ASSETS_DIR / "asphalt.png"
GRASS_IMG       = ASSETS_DIR / "grass.png"
TIRESTACK_IMG   = ASSETS_DIR / "tire_stack.png"
CAR_IMG         = ASSETS_DIR / "f1.png"
FLAG_IMG        = ASSETS_DIR / "checkered_flag.png"

# -------------------- Shared types --------------------
from src.core.types import Grid, StepResult, Cell

# ---------- Mode resolution & dynamic imports ----------
def resolve_mode() -> str:
    mode = os.getenv("WORKSHOP_MODE", "student").lower()
    for arg in sys.argv:
        if arg.startswith("--mode="):
            mode = arg.split("=", 1)[1].lower()
    return "instructor" if mode in ("instructor", "solution", "answers") else "student"

MODE = resolve_mode()

if MODE == "instructor":
    # answers/ implementations
    from src.core.answers.dijkstra_solution import DijkstraAlgo as DijkstraImpl
    from src.core.answers.astar_solution import AStarAlgo as AStarImpl
    try:
        from src.core.answers.dijkstra_weighted_solution import DijkstraAlgo as DijkstraWeightedImpl
    except Exception:
        DijkstraWeightedImpl = None
else:
    # student templates
    from src.core.dijkstra_template import DijkstraAlgo as DijkstraImpl
    from src.core.astar_template import AStarAlgo as AStarImpl
    DijkstraWeightedImpl = None

# ---------- Config ----------
MAP_DIR = Path(__file__).resolve().parents[2] / "maps"
MAP_FILES = {
    "01_intro_dijkstra": MAP_DIR / "01_intro_dijkstra.json",
    "02_small_astar":    MAP_DIR / "02_small_astar.json",
    "03_weighted_grass": MAP_DIR / "03_weighted_grass.json",
}
PANEL_W = 280
GRID_MARGIN = 16
CELL_SIZE_DEFAULT = 24
FONT_NAME = None  # default pygame font

# Fallback colors (used if an asset is missing)
WHITE       = (255,255,255)
BLACK       = (  0,  0,  0)
GRAY_20     = (220,220,220)
GRAY_60     = (140,140,140)
BLUE        = ( 70,130,180)
RED         = (220, 50, 47)
GREEN       = ( 46,139, 87)
YELLOW      = (255,215,  0)
GRASS_GREEN = (144, 238, 144)
ASPHALT_GRAY= (200,200,200)
BG_DARK     = (30,30,36)

# ---------- Asset loader / scaler ----------
class _Assets:
    """
    Loads and caches images; provides scaled surfaces per cell size.
    Falls back to None when file missing (viewer draws solid color fallback).
    """
    def __init__(self):
        self._raw: Dict[str, Optional[pygame.Surface]] = {}
        self._scaled_cache: Dict[Tuple[str, int], pygame.Surface] = {}

    def _load(self, key: str, path: Path):
        if key in self._raw:
            return
        if path.exists():
            img = pygame.image.load(str(path))
            # convert_alpha only if display surface exists
            if pygame.display.get_surface():
                img = img.convert_alpha()
            self._raw[key] = img
        else:
            self._raw[key] = None  # fallback sentinel

    def prepare(self):
        # call AFTER display.set_mode
        self._load("asphalt", ASPHALT_IMG)
        self._load("grass",   GRASS_IMG)
        self._load("tire",    TIRESTACK_IMG)
        self._load("car",     CAR_IMG)
        self._load("flag",    FLAG_IMG)

    def get(self, key: str, cell_size: int) -> Optional[pygame.Surface]:
        base = self._raw.get(key, None)
        if base is None:
            return None
        cache_key = (key, cell_size)
        if cache_key in self._scaled_cache:
            return self._scaled_cache[cache_key]
        target = max(1, int(cell_size))
        if pygame.display.get_surface():
            surf = pygame.transform.smoothscale(base, (target, target))
        else:
            surf = pygame.transform.scale(base, (target, target))
        self._scaled_cache[cache_key] = surf
        return surf

    def get_centered(self, key: str, cell_size: int, scale_factor: float = 0.85) -> Optional[pygame.Surface]:
        base = self._raw.get(key, None)
        if base is None:
            return None
        target = max(1, int(cell_size * scale_factor))
        cache_key = (f"{key}_icon", target)
        if cache_key in self._scaled_cache:
            return self._scaled_cache[cache_key]
        if pygame.display.get_surface():
            surf = pygame.transform.smoothscale(base, (target, target))
        else:
            surf = pygame.transform.scale(base, (target, target))
        self._scaled_cache[cache_key] = surf
        return surf

ASSETS = _Assets()

# ---------- Algorithm placeholder ----------
class NoAlgo:
    def __init__(self, name="(no algorithm)"):
        self.name = name
        self._grid: Optional[Grid] = None
        self._steps = 0
    def init(self, grid: Grid) -> None:
        self._grid = grid; self._steps = 0
    def reset(self) -> None:
        self._steps = 0
    def step(self) -> StepResult:
        self._steps += 1
        return StepResult(status="idle", metrics={"algo": self.name, "steps": self._steps})

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
    assert len(cells) == height and all(len(r) == width for r in cells), "cells size mismatch"
    sx, sy = start; gx, gy = goal
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

        # expose constants so theme_skin can read them off `self`
        self.GRID_MARGIN = GRID_MARGIN
        self.PANEL_W     = PANEL_W
        self.MODE        = MODE

        # initial window (resizable)
        # start with a reasonable size based on default cell size
        grid_px_w = GRID_MARGIN*2 + grid.width * self.cell_size
        grid_px_h = GRID_MARGIN*2 + grid.height* self.cell_size
        win_w = grid_px_w + PANEL_W
        win_h = grid_px_h
        self.screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
        pygame.display.set_caption("Pathfinding Workshop Viewer — Race Theme")

        # compute initial layout (center + scale for current window)
        self._layout(win_w, win_h)

        # IMPORTANT: load assets AFTER the display exists
        ASSETS.prepare()
        if THEME and hasattr(THEME, "prepare"):
            THEME.prepare(ASSETS)  # let the skin cache/adjust anything it needs

        self.open_set: set[Cell] = set()
        self.closed_set: set[Cell] = set()
        self.path: List[Cell] = []

        self.running = False
        self.clock = pygame.time.Clock()
        self.steps_per_sec = 8
        self.state = "Idle"
        self.selected_map_key = self._infer_map_key()
        self.selected_algo = "Dijkstra"

        self.algo = self._make_algo("Dijkstra")
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

    # ---------- layout: scale + center ----------
    def _layout(self, win_w: int, win_h: int):
        """
        Compute an integer cell_size that fits the window and center the grid.
        Also prepares canvas_rect and a top-left draw origin.
        """
        # Leave space for side panel on the right
        avail_w = max(1, win_w - PANEL_W - 2 * GRID_MARGIN)
        avail_h = max(1, win_h - 2 * GRID_MARGIN)

        # Choose a uniform cell size that fits both width and height
        cs_by_w = avail_w // self.grid.width
        cs_by_h = avail_h // self.grid.height
        self.cell_size = int(max(8, min(cs_by_w, cs_by_h))) or 8

        grid_draw_w = self.grid.width  * self.cell_size
        grid_draw_h = self.grid.height * self.cell_size

        # total grid plate including its inner margin
        grid_plate_w = grid_draw_w + 2 * GRID_MARGIN
        grid_plate_h = grid_draw_h + 2 * GRID_MARGIN

        # Center the grid plate in the remaining area (left area)
        left_area_w = win_w - PANEL_W
        left_area_h = win_h
        left_x = max(0, (left_area_w - grid_plate_w) // 2)
        top_y  = max(0, (left_area_h - grid_plate_h) // 2)

        # canvas_rect bounds the grid plate (used by themes)
        self.canvas_rect = pygame.Rect(left_x, top_y, grid_plate_w, grid_plate_h)

        # origin where tile (0,0) is drawn
        self._grid_origin = (self.canvas_rect.x + GRID_MARGIN,
                             self.canvas_rect.y + GRID_MARGIN)

    def _infer_map_key(self) -> str:
        for k,p in MAP_FILES.items():
            try: g = load_map(p)
            except Exception: continue
            if (g.width, g.height) == (self.grid.width, self.grid.height) and g.start == self.grid.start and g.goal == self.grid.goal:
                return k
        return "custom"

    def _auto_cell_size(self, grid: Grid) -> int:
        target_h = 720 - GRID_MARGIN*2
        size = max(14, min(CELL_SIZE_DEFAULT, target_h // grid.height))
        return size

    def run(self):
        while True:
            self._handle_events()
            if self.running:
                self._tick_algorithm()
            self._draw()
            self.clock.tick(60)

    def _tick_algorithm(self):
        t0 = time.time()
        step_interval = 1.0 / max(1, self.steps_per_sec)
        if not hasattr(self, "_last_step_t"):
            self._last_step_t = 0.0
        if t0 - self._last_step_t >= step_interval:
            self._last_step_t = t0
            self._do_step()

    def _do_step(self):
        res = self.algo.step()
        for c in res.opened: self.open_set.add(c)
        for c in res.closed: self.closed_set.add(c)
        if res.path is not None: self.path = res.path
        if res.status == "done":
            self.state = "Done"; self.running = False
        elif res.status == "no_path":
            self.state = "No path"; self.running = False
        elif res.status in ("running","idle"):
            self.state = "Running" if self.running else "Idle"
        if res.metrics:
            self._last_metrics = res.metrics

    def _handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit(); sys.exit(0)
                elif e.key == pygame.K_SPACE:
                    if self.state not in ("Done","No path"):
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
            elif e.type == pygame.VIDEORESIZE:
                # keep window resizable and recompute layout
                self.screen = pygame.display.set_mode((e.w, e.h), pygame.RESIZABLE)
                self._layout(e.w, e.h)

    def _make_algo(self, label: str):
        if label == "Dijkstra":
            if self._current_map_is_weighted() and MODE == "instructor" and DijkstraWeightedImpl:
                return DijkstraWeightedImpl(name="Dijkstra (weighted)")
            return DijkstraImpl(name="Dijkstra")
        elif label == "A*":
            return AStarImpl(name="A*")
        else:
            return NoAlgo(name=label)

    def _current_map_is_weighted(self) -> bool:
        return self.selected_map_key == "03_weighted_grass"

    def _switch_map(self, key: str):
        if key not in MAP_FILES: return
        try:
            self.grid = load_map(MAP_FILES[key])
            self.selected_map_key = key
            pygame.display.set_caption(f"Pathfinding Workshop Viewer — {key}")
            # keep scale/center for the new grid size
            self._reset_overlays()
            self.algo = self._make_algo(self.selected_algo)
            self.algo.init(self.grid)
            self._layout(*self.screen.get_size())
            self.running = False; self.state = "Idle"
        except Exception as ex:
            print(f"Failed to load map {key}: {ex}")

    def _switch_algo(self, label: str):
        self.selected_algo = label
        self.algo = self._make_algo(label)
        self.algo.init(self.grid)
        self._reset_overlays()
        # layout unchanged but we recompute to keep canvas_rect consistent
        self._layout(*self.screen.get_size())

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

    # ---------- drawing ----------
    def _draw(self):
        if THEME and hasattr(THEME, "draw"):
            THEME.draw(self, self.screen, ASSETS)  # skin owns visuals
        else:
            self.screen.fill(BG_DARK)
            self._draw_grid()
            self._draw_panel()
        pygame.display.flip()

    def _draw_grid(self):
        cs = self.cell_size
        ox, oy = self._grid_origin  # centered origin

        # prefetch scaled textures
        asphalt = ASSETS.get("asphalt", cs)
        grass   = ASSETS.get("grass",   cs)
        tire    = ASSETS.get("tire",    cs)
        car     = ASSETS.get_centered("car",  cs, scale_factor=0.85)
        flag    = ASSETS.get_centered("flag", cs, scale_factor=0.85)

        # tiles
        for row in range(self.grid.height):
            for col in range(self.grid.width):
                v = self.grid.cells[row][col]
                rect = pygame.Rect(ox + col*cs, oy + row*cs, cs, cs)

                # choose texture/fallback
                is_block = (str(v) in self.grid.weights and self.grid.weights[str(v)] == "BLOCK") or v == 1
                if is_block:
                    if tire:
                        self.screen.blit(tire, rect.topleft)
                    else:
                        pygame.draw.rect(self.screen, BLACK, rect)
                elif v == 2:
                    if grass:
                        self.screen.blit(grass, rect.topleft)
                    else:
                        pygame.draw.rect(self.screen, GRASS_GREEN, rect)
                else:
                    if asphalt:
                        self.screen.blit(asphalt, rect.topleft)
                    else:
                        pygame.draw.rect(self.screen, ASPHALT_GRAY, rect)

                # cell outline for readability
                pygame.draw.rect(self.screen, GRAY_60, rect, 1)

        # overlays: closed then open
        for (col,row) in self.closed_set:
            rect = pygame.Rect(ox + col*cs, oy + row*cs, cs, cs)
            s = pygame.Surface((cs, cs), pygame.SRCALPHA)
            s.fill((120,120,120,110))
            self.screen.blit(s, rect.topleft)

        for (col,row) in self.open_set:
            rect = pygame.Rect(ox + col*cs, oy + row*cs, cs, cs)
            s = pygame.Surface((cs, cs), pygame.SRCALPHA)
            s.fill((0,180,255,110))
            self.screen.blit(s, rect.topleft)

        # path polyline
        if len(self.path) >= 2:
            pts = []
            for (col,row) in self.path:
                cx0 = ox + col*cs + cs//2
                cy0 = oy + row*cs + cs//2
                pts.append((cx0, cy0))
            pygame.draw.lines(self.screen, GREEN, False, pts, 5)

        # start / goal icons
        self._draw_badge_icon(self.grid.start, car,  BLUE)
        self._draw_badge_icon(self.grid.goal,  flag, RED)

    def _draw_badge_icon(self, cell: Cell, icon: Optional[pygame.Surface], fallback_color: Tuple[int,int,int]):
        cs = self.cell_size
        ox, oy = self._grid_origin
        col,row = cell
        cx = ox + col*cs + cs//2
        cy = oy + row*cs + cs//2
        if icon is not None:
            rect = icon.get_rect(center=(cx, cy))
            self.screen.blit(icon, rect)
        else:
            # fallback: colored circle + letter
            pygame.draw.circle(self.screen, fallback_color, (cx,cy), max(10, cs//2 - 2))
            lbl = "S" if (col,row) == self.grid.start else "G"
            txt = self.font_small.render(lbl, True, (255,255,255))
            self.screen.blit(txt, txt.get_rect(center=(cx,cy)))

    def _draw_panel(self):
        # Panel pinned to the right edge and full height of window
        win_w, win_h = self.screen.get_size()
        panel_rect = pygame.Rect(win_w - PANEL_W, 0, PANEL_W, win_h)
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
        line(f"Mode: {MODE}")
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
        for c in [
            "[1/2/3] Switch Map",
            "[D/A]   Choose Algo",
            "[R]     Reset",
            "[SPACE] Run / Pause",
            "[N]     Step once",
            "[+/-]   Speed",
            "[Q/ESC] Quit",
            "Resize window: grid centers & scales",
        ]:
            line(c)

# ---------- main ----------
def main():
    try:
        grid = load_map(MAP_FILES["01_intro_dijkstra"])
    except Exception as ex:
        print(f"Failed to load default map: {ex}")
        sys.exit(1)
    Viewer(grid).run()

if __name__ == "__main__":
    main()
