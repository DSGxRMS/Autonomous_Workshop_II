# src/app/viewer.py
#!/usr/bin/env python3
"""
Pathfinding Workshop Viewer
"""

import sys, os, json, time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import pygame

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from src.app import theme_skin as THEME
except Exception:
    THEME = None

ASSETS_DIR      = Path(__file__).resolve().parents[2] / "assets"
ASPHALT_IMG     = ASSETS_DIR / "asphalt.png"
GRASS_IMG       = ASSETS_DIR / "grass.png"
TIRESTACK_IMG   = ASSETS_DIR / "tire_stack.png"
CAR_IMG         = ASSETS_DIR / "f1.png"
FLAG_IMG        = ASSETS_DIR / "checkered_flag.png"
LOGO_IMG        = ASSETS_DIR / "logo.png"
BACKDROP_IMG    = ASSETS_DIR / "backdrop.jpg"
WIN_MP3         = ASSETS_DIR / "win.mp3"

from src.core.types import Grid, StepResult, Cell

def resolve_mode() -> str:
    mode = os.getenv("WORKSHOP_MODE", "student").lower()
    for arg in sys.argv:
        if arg.startswith("--mode="):
            mode = arg.split("=", 1)[1].lower()
    return "instructor" if mode in ("instructor", "solution", "answers") else "student"

MODE = resolve_mode()

if MODE == "instructor":
    from src.core.answers.dijkstra_solution import DijkstraAlgo as DijkstraImpl
    from src.core.answers.astar_solution import AStarAlgo as AStarImpl
    try:
        from src.core.answers.dijkstra_weighted_solution import DijkstraAlgo as DijkstraWeightedImpl
    except Exception:
        DijkstraWeightedImpl = None
else:
    from src.core.dijkstra_template import DijkstraAlgo as DijkstraImpl
    from src.core.astar_template import AStarAlgo as AStarImpl
    DijkstraWeightedImpl = None  # student mode never uses weighted variant

MAP_DIR = Path(__file__).resolve().parents[2] / "maps"
MAP_FILES = {
    "01_intro_dijkstra": MAP_DIR / "01_intro_dijkstra.json",
    "02_small_astar":    MAP_DIR / "02_small_astar.json",
    "03_weighted_grass": MAP_DIR / "03_weighted_grass.json",
}
PANEL_W = 600
GRID_MARGIN = 16
CELL_SIZE_DEFAULT = 24
FONT_NAME = None

WHITE       = (255,255,255)
BLACK       = (0,0,0)
BLUE        = (70,130,180)
RED         = (220,50,47)
GREEN       = (46,139,87)
GRASS_GREEN = (144,238,144)
ASPHALT_GRAY= (200,200,200)
NEON_CYAN_A = (0,150,255,110)
NEON_MAG_A  = (255,0,120,90)
NEON_MINT   = (0,255,200)
CARD_BG     = (24,28,36,220)
CARD_HI     = (255,255,255,18)
TEXT_LIGHT  = (230,235,240)
ACCENT_GOLD = (255,210,0)

class _Assets:
    def __init__(self):
        self._raw: Dict[str, Optional[pygame.Surface]] = {}
        self._scaled_cache: Dict[Tuple[str, int], pygame.Surface] = {}

    def _load(self, key: str, path: Path):
        if key in self._raw:
            return
        if path.exists():
            img = pygame.image.load(str(path))
            if pygame.display.get_surface():
                img = img.convert_alpha()
            self._raw[key] = img
        else:
            self._raw[key] = None

    def prepare(self):
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
        surf = pygame.transform.smoothscale(base, (target, target)) if pygame.display.get_surface() \
               else pygame.transform.scale(base, (target, target))
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
        surf = pygame.transform.smoothscale(base, (target, target)) if pygame.display.get_surface() \
               else pygame.transform.scale(base, (target, target))
        self._scaled_cache[cache_key] = surf
        return surf

ASSETS = _Assets()

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

def load_map(path: Path) -> Grid:
    with open(path, "r") as f:
        data = json.load(f)
    width  = int(data["width"])
    height = int(data["height"])
    start  = tuple(data["start"])
    goal   = tuple(data["goal"])
    move   = int(data.get("move", 4))
    cells  = data["cells"]
    weights = data.get("weights", {})
    assert len(cells) == height and all(len(r) == width for r in cells)
    sx, sy = start; gx, gy = goal
    assert 0 <= sx < width and 0 <= sy < height
    assert 0 <= gx < width and 0 <= gy < height
    return Grid(width, height, cells, start, goal, move, weights)

class UIButton:
    def __init__(self, label: str, rect: pygame.Rect, callback, *, togglable: bool = False):
        self.label = label
        self.rect = rect
        self.callback = callback
        self.hover = False
        self.togglable = togglable
        self.active = False

    def set_active(self, value: bool):
        self.active = bool(value)

    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        base = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        bg_idle   = (36, 40, 48, 220)
        bg_hover  = (46, 50, 60, 230)
        bg_active = (58, 86, 160, 235)
        border_active = (120, 170, 255, 255)

        bg = bg_active if (self.active and self.togglable) else (bg_hover if self.hover else bg_idle)
        pygame.draw.rect(base, bg, base.get_rect(), border_radius=10)
        hi = pygame.Surface((self.rect.width, 18), pygame.SRCALPHA)
        pygame.draw.rect(hi, (255,255,255,20), hi.get_rect(), border_radius=10)
        base.blit(hi, (0,0))
        screen.blit(base, self.rect.topleft)
        if self.active and self.togglable:
            pygame.draw.rect(screen, border_active, self.rect, width=2, border_radius=10)
        text = font.render(self.label, True, (235,238,242))
        screen.blit(text, text.get_rect(center=self.rect.center))

    def handle_mouse(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()

class Viewer:
    def __init__(self, grid: Grid):
        pygame.init()
        self.grid = grid
        self.cell_size = self._auto_cell_size(grid)
        self.font_small = pygame.font.Font(FONT_NAME, 14)
        self.font = pygame.font.Font(FONT_NAME, 18)
        self.font_big = pygame.font.Font(FONT_NAME, 22)

        self.GRID_MARGIN = GRID_MARGIN
        self.PANEL_W     = PANEL_W
        self.MODE        = MODE

        grid_px_w = GRID_MARGIN*2 + grid.width * self.cell_size
        grid_px_h = GRID_MARGIN*2 + grid.height* self.cell_size
        win_w = grid_px_w + max(PANEL_W, 360)
        win_h = max(grid_px_h, 560)
        self.screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)

        try:
            if LOGO_IMG.exists():
                icon_surf = pygame.image.load(str(LOGO_IMG))
                icon_surf = icon_surf.convert_alpha() if icon_surf.get_masks()[3] else icon_surf.convert()
                pygame.display.set_icon(icon_surf)
        except Exception:
            pass

        pygame.display.set_caption("Pathfinding Exercise!")

        self.open_set: set[Cell] = set()
        self.closed_set: set[Cell] = set()
        self.path: List[Cell] = []

        self.running = False
        self.clock = pygame.time.Clock()
        self.steps_per_sec = 8
        self.state = "Idle"
        self.selected_map_key = self._infer_map_key()
        self.selected_algo = "Dijkstra"

        self._last_metrics = {
            "algo": self.selected_algo,
            "popped": 0,
            "open_size": 0,
            "closed_count": 0,
            "path_len": 0,
            "total_cost": None,
            "steps": 0,
        }

        self._buttons: list[UIButton] = []
        self._backdrop_raw = None
        self._backdrop_scaled_by_h = {}

        self._layout(win_w, win_h)

        ASSETS.prepare()
        if THEME and hasattr(THEME, "prepare"):
            THEME.prepare(ASSETS)

        self._min_w  = 640
        self._min_h  = 360

        self.algo = self._make_algo(self.selected_algo)
        self.algo.init(self.grid)

        self._refresh_active_states()

        self.run_time_total = 0.0
        self.run_time_start = None

        self._win_sound = None
        self._win_sound_played = False
        try:
            if WIN_MP3.exists():
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                self._win_sound = pygame.mixer.Sound(str(WIN_MP3))
        except Exception:
            self._win_sound = None

    def _layout(self, win_w: int, win_h: int):
        avail_w = max(1, win_w - self.PANEL_W - 2 * GRID_MARGIN)
        avail_h = max(1, win_h - 2 * GRID_MARGIN)

        cs_by_w = avail_w // self.grid.width
        cs_by_h = avail_h // self.grid.height
        self.cell_size = int(max(8, min(cs_by_w, cs_by_h))) or 8

        grid_draw_w = self.grid.width  * self.cell_size
        grid_draw_h = self.grid.height * self.cell_size

        grid_plate_w = grid_draw_w + 2 * GRID_MARGIN
        grid_plate_h = grid_draw_h + 2 * GRID_MARGIN

        left_x = max(0, (win_w - (grid_plate_w + self.PANEL_W)) // 2)
        left_x = min(left_x, max(0, win_w - self.PANEL_W - grid_plate_w))
        top_y  = max(0, (win_h - grid_plate_h) // 2)

        self.canvas_rect = pygame.Rect(left_x, top_y, grid_plate_w, grid_plate_h)
        self._grid_origin = (self.canvas_rect.x + GRID_MARGIN,
                             self.canvas_rect.y + GRID_MARGIN)

        self._right_band = pygame.Rect(self.canvas_rect.right, 0,
                                       max(self.PANEL_W, win_w - self.canvas_rect.right),
                                       win_h)
        self._build_buttons()

    def _infer_map_key(self) -> str:
        for k,p in MAP_FILES.items():
            try: g = load_map(p)
            except Exception: continue
            if (g.width, g.height) == (self.grid.width, self.grid.height) and g.start == self.grid.start and g.goal == self.grid.goal:
                return k
        return "custom"

    def _auto_cell_size(self, grid: Grid) -> int:
        target_h = 720 - GRID_MARGIN*2
        return max(14, min(CELL_SIZE_DEFAULT, target_h // grid.height))

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
            if self.run_time_start is not None:
                self.run_time_total += time.time() - self.run_time_start
            self.run_time_start = None
            self.state = "Done"; self.running = False

            if (self.selected_map_key == "01_intro_dijkstra" and
                self._win_sound is not None and
                not self._win_sound_played and MODE == "student"):
                try:
                    self._win_sound.play()
                    self._win_sound_played = True
                except Exception:
                    pass

        elif res.status == "no_path":
            self.state = "No path"; self.running = False
        elif res.status in ("running","idle"):
            self.state = "Running" if self.running else "Idle"

        if res.metrics:
            self._last_metrics = res.metrics

    def _apply_aspect_resize(self, req_w: int, req_h: int):
        min_cell = 8
        min_grid_w = GRID_MARGIN*2 + self.grid.width * min_cell
        min_grid_h = GRID_MARGIN*2 + self.grid.height * min_cell
        min_w = max(480, min_grid_w + self.PANEL_W)
        min_h = max(360, min_grid_h)

        new_w = max(min_w, req_w)
        new_h = max(min_h, req_h)

        self.screen = pygame.display.set_mode((new_w, new_h), pygame.RESIZABLE)
        self._layout(new_w, new_h)

    def _handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit(); sys.exit(0)
                elif e.key == pygame.K_SPACE:
                    if self.state not in ("Done","No path"):
                        self._toggle_run()
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
                self._apply_aspect_resize(e.w, e.h)
            elif e.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
                for b in self._buttons:
                    b.handle_mouse(e)

    def get_run_time_str(self) -> str:
        t = self.run_time_total
        if self.run_time_start is not None:
            t += (time.time() - self.run_time_start)
        ms  = int((t - int(t)) * 1000)
        sec = int(t) % 60
        mins= (int(t) // 60) % 60
        hrs = int(t) // 3600
        if hrs > 0:
            return f"{hrs:02d}:{mins:02d}:{sec:02d}.{ms:03d}"
        return f"{mins:02d}:{sec:02d}.{ms:03d}"

    def _make_algo(self, label: str):
        if label == "Dijkstra":
            # STUDENT: always use template (no weighted check)
            if MODE == "student":
                return DijkstraImpl(name="Dijkstra")
            # INSTRUCTOR: keep weighted variant for Map 3 if available
            if self._current_map_is_weighted() and DijkstraWeightedImpl:
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
            pygame.display.set_caption(f"Pathfinding — {key}")
            self._reset_overlays()
            self.algo = self._make_algo(self.selected_algo)
            self.algo.init(self.grid)
            self._layout(*self.screen.get_size())
            self.running = False; self.state = "Idle"
            self.run_time_total = 0.0
            self.run_time_start = None
            self._win_sound_played = False
            self._refresh_active_states()
        except Exception as ex:
            print(f"Failed to load map {key}: {ex}")

    def _switch_algo(self, label: str):
        self.selected_algo = label
        self.algo = self._make_algo(label)
        self.algo.init(self.grid)
        self._reset_overlays()
        self._layout(*self.screen.get_size())
        self.run_time_total = 0.0
        self.run_time_start = None
        self.state = "Idle"
        self._win_sound_played = False
        self._refresh_active_states()

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
        self.run_time_total = 0.0
        self.run_time_start = None
        self._win_sound_played = False
        self._refresh_active_states()

    def _draw(self):
        if THEME and hasattr(THEME, "draw"):
            THEME.draw(self, self.screen, ASSETS)
        else:
            self._draw_backdrop()
            self._draw_grid()
        self._draw_metrics_and_buttons()
        pygame.display.flip()

    def _draw_backdrop(self):
        w, h = self.screen.get_size()
        if self._backdrop_raw is None:
            if BACKDROP_IMG.exists():
                try:
                    img = pygame.image.load(str(BACKDROP_IMG))
                    self._backdrop_raw = img.convert() if not img.get_masks()[3] else img.convert_alpha()
                except Exception:
                    self._backdrop_raw = None
            else:
                self._backdrop_raw = None

        if self._backdrop_raw is None:
            top = (24, 26, 32); bot = (36, 40, 48)
            for y in range(h):
                t = y / max(1, h-1)
                c = (
                    int(top[0] + (bot[0]-top[0]) * t),
                    int(top[1] + (bot[1]-top[1]) * t),
                    int(top[2] + (bot[2]-top[2]) * t),
                )
                pygame.draw.line(self.screen, c, (0, y), (w, y))
            return

        key = h
        if key not in self._backdrop_scaled_by_h:
            bw, bh = self._backdrop_raw.get_width(), self._backdrop_raw.get_height()
            scale = max(w / bw, h / bh)
            tw, th = int(bw * scale), int(bh * scale)
            scaled = pygame.transform.smoothscale(self._backdrop_raw, (tw, th))
            self._backdrop_scaled_by_h[key] = scaled

        img = self._backdrop_scaled_by_h[key]
        x = (w - img.get_width()) // 2
        y = (h - img.get_height()) // 2
        self.screen.blit(img, (x, y))

    def _draw_grid(self):
        cs = self.cell_size
        ox, oy = self._grid_origin

        asphalt = ASSETS.get("asphalt", cs)
        grass   = ASSETS.get("grass",   cs)
        tire    = ASSETS.get("tire",    cs)
        car     = ASSETS.get_centered("car",  cs, scale_factor=0.85)
        flag    = ASSETS.get_centered("flag", cs, scale_factor=0.85)

        for row in range(self.grid.height):
            for col in range(self.grid.width):
                v = self.grid.cells[row][col]
                rect = pygame.Rect(ox + col*cs, oy + row*cs, cs, cs)

                is_block = (str(v) in self.grid.weights and self.grid.weights[str(v)] == "BLOCK") or v == 1
                if is_block:
                    if tire: self.screen.blit(tire, rect.topleft)
                    else:    pygame.draw.rect(self.screen, BLACK, rect)
                elif v == 2:
                    if grass: self.screen.blit(grass, rect.topleft)
                    else:     pygame.draw.rect(self.screen, GRASS_GREEN, rect)
                else:
                    if asphalt: self.screen.blit(asphalt, rect.topleft)
                    else:       pygame.draw.rect(self.screen, ASPHALT_GRAY, rect)

                pygame.draw.rect(self.screen, BLACK, rect, 1)

        for (col,row) in self.closed_set:
            rect = pygame.Rect(ox + col*cs, oy + row*cs, cs, cs)
            s = pygame.Surface((cs, cs), pygame.SRCALPHA); s.fill(NEON_MAG_A)
            self.screen.blit(s, rect.topleft)

        for (col,row) in self.open_set:
            rect = pygame.Rect(ox + col*cs, oy + row*cs, cs, cs)
            s = pygame.Surface((cs, cs), pygame.SRCALPHA); s.fill(NEON_CYAN_A)
            self.screen.blit(s, rect.topleft)

        if len(self.path) >= 2:
            pts = []
            for (col,row) in self.path:
                cx0 = ox + col*cs + cs//2
                cy0 = oy + row*cs + cs//2
                pts.append((cx0, cy0))
            glow = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            pygame.draw.lines(glow, (0, 255, 220, 60), False, pts, 7)
            self.screen.blit(glow, (0,0), special_flags=pygame.BLEND_ADD)
            pygame.draw.lines(self.screen, NEON_MINT, False, pts, 5)

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
            pygame.draw.circle(self.screen, fallback_color, (cx,cy), max(10, cs//2 - 2))
            txt = self.font_small.render("S" if (col,row) == self.grid.start else "G", True, WHITE)
            self.screen.blit(txt, txt.get_rect(center=(cx,cy)))

    def _build_buttons(self):
        self._buttons.clear()
        rb = self._right_band
        x = rb.x + 16
        y = rb.y + 210
        w = max(160, rb.width - 32)
        h = 38
        gap = 10

        def add(label, cb, *, togglable=False, store_as: str | None = None):
            nonlocal y
            rect = pygame.Rect(x, y, w, h)
            btn = UIButton(label, rect, cb, togglable=togglable)
            self._buttons.append(btn)
            if store_as: setattr(self, store_as, btn)
            y += h + gap

        add("Run / Pause", self._toggle_run, togglable=True, store_as="btn_run")
        add("Step Once", self._do_step)
        add("Reset", self._reset)

        minus_rect = pygame.Rect(x, y, (w-8)//2, h)
        plus_rect  = pygame.Rect(x + (w-8)//2 + 8, y, (w-8)//2, h)
        self.btn_speed_minus = UIButton("Speed −", minus_rect, lambda: self._bump_speed(-1))
        self.btn_speed_plus  = UIButton("Speed +", plus_rect,  lambda: self._bump_speed(+1))
        self._buttons.append(self.btn_speed_minus)
        self._buttons.append(self.btn_speed_plus)
        y += h + gap

        add("Algo: Dijkstra", lambda: self._switch_algo("Dijkstra"), togglable=True, store_as="btn_algo_d")
        add("Algo: A*",       lambda: self._switch_algo("A*"),       togglable=True, store_as="btn_algo_a")

        add("Map 1: Intro Dijkstra", lambda: self._switch_map("01_intro_dijkstra"), togglable=True, store_as="btn_map1")
        add("Map 2: Small A*",       lambda: self._switch_map("02_small_astar"),    togglable=True, store_as="btn_map2")
        add("Map 3: Weighted",       lambda: self._switch_map("03_weighted_grass"), togglable=True, store_as="btn_map3")

        self._refresh_active_states()

    def _refresh_active_states(self):
        if hasattr(self, "btn_run"):   self.btn_run.set_active(self.running)
        if hasattr(self, "btn_algo_d"): self.btn_algo_d.set_active(self.selected_algo == "Dijkstra")
        if hasattr(self, "btn_algo_a"): self.btn_algo_a.set_active(self.selected_algo == "A*")
        if hasattr(self, "btn_map1"):   self.btn_map1.set_active(self.selected_map_key == "01_intro_dijkstra")
        if hasattr(self, "btn_map2"):   self.btn_map2.set_active(self.selected_map_key == "02_small_astar")
        if hasattr(self, "btn_map3"):   self.btn_map3.set_active(self.selected_map_key == "03_weighted_grass")

    def _toggle_run(self):
        if self.state in ("Done", "No path"):
            return
        if not self.running:
            self.run_time_start = time.time()
            self.running = True
            self.state = "Running"
        else:
            if self.run_time_start is not None:
                self.run_time_total += time.time() - self.run_time_start
            self.run_time_start = None
            self.running = False
            self.state = "Paused"
        self._refresh_active_states()

    def _bump_speed(self, dv: int):
        self.steps_per_sec = int(max(1, min(60, self.steps_per_sec + dv)))

    def _draw_metrics_and_buttons(self):
        rb = self._right_band
        card_h = 190
        card = pygame.Surface((rb.width - 20, card_h), pygame.SRCALPHA)
        pygame.draw.rect(card, CARD_BG, card.get_rect(), border_radius=14)
        hi = pygame.Surface((card.get_width(), 24), pygame.SRCALPHA)
        pygame.draw.rect(hi, CARD_HI, hi.get_rect(), border_radius=14)
        card.blit(hi, (0,0))
        self.screen.blit(card, (rb.x + 10, rb.y + 10))

        x0 = rb.x + 24
        y0 = rb.y + 18

        def line(text, big=False, color=TEXT_LIGHT):
            nonlocal y0
            f = self.font_big if big else self.font
            surf = f.render(text, True, color)
            self.screen.blit(surf, (x0, y0))
            y0 += surf.get_height() + 6

        line("Metrics", big=True, color=ACCENT_GOLD)
        m = getattr(self, "_last_metrics", {})
        line(f"Popped: {m.get('popped', 0)}")
        line(f"Open: {m.get('open_size', 0)}")
        line(f"Closed: {m.get('closed_count', 0)}")
        line(f"Path Len: {m.get('path_len', 0)}")
        if m.get("total_cost", None) is not None:
            line(f"Total Cost: {m['total_cost']}")

        dash = "-" * 26
        line(dash)

        map_label = self._map_label_for_display()
        line(f"{map_label} active")
        line(f"Algo: {self.selected_algo}")
        line(f"Speed: {self.steps_per_sec} steps/s")

        for b in self._buttons:
            b.draw(self.screen, self.font)

    def _map_label_for_display(self) -> str:
        key = self.selected_map_key
        if key == "01_intro_dijkstra": return "Map 1"
        if key == "02_small_astar":    return "Map 2"
        if key == "03_weighted_grass": return "Map 3"
        return "Custom map"

def main():
    try:
        grid = load_map(MAP_FILES["01_intro_dijkstra"])
    except Exception as ex:
        print(f"Failed to load default map: {ex}")
        sys.exit(1)
    Viewer(grid).run()

if __name__ == "__main__":
    main()
