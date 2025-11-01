# src/app/theme_skin.py
"""
Neon F1 skin — premium look (visuals only; no logic)
- Backdrop: uses assets/backdrop.jpg if present; else a dark gradient
- Grid: textures, BLACK borders, neon overlays, pulsing mint path
- Start/Goal: clean icons (no halo)
- Right Panel: frosted glass underlay only (viewer draws buttons/metrics on top)
- Timer: shown in the right panel, below the Map buttons (reads viewer.get_run_time_str())

This file deliberately avoids drawing the viewer’s text/buttons,
so the viewer remains the source of truth for interactivity.
"""

from __future__ import annotations
import math, time
from pathlib import Path
from typing import Tuple, Optional
import pygame

# ---- palette ----
BLACK         = (0, 0, 0)
TEXT_LIGHT    = (230, 235, 240)
ACCENT_GOLD   = (255, 210, 0)
GREEN_NEON    = (0, 255, 200)
CYAN_NEON_A   = (0, 150, 255, 110)
MAGENTA_A     = (255,   0, 120,  90)
GRASS_GREEN   = (144, 238, 144)
ASPHALT_GRAY  = (200, 200, 200)

# panel colors
PANEL_FILL    = (18, 20, 28, 190)
PANEL_SHADOW  = (0, 0, 0, 140)

# timer pill colors (no outline now)
PILL_BG       = (24, 28, 36, 220)

ASSETS_DIR    = Path(__file__).resolve().parents[2] / "assets"
BACKDROP_IMG  = ASSETS_DIR / "backdrop.jpg"

# caches
_backdrop_raw: Optional[pygame.Surface] = None
_backdrop_scaled_by_h: dict[int, pygame.Surface] = {}

# ---------- helpers ----------
def _rounded_rect(surface: pygame.Surface, rect: pygame.Rect, color, radius=16, width=0):
    pygame.draw.rect(surface, color, rect, width=width, border_radius=radius)

def _glass_panel(screen: pygame.Surface, rect: pygame.Rect,
                 fill_rgba=PANEL_FILL, shadow_rgba=PANEL_SHADOW):
    if rect.width <= 0 or rect.height <= 0:
        return
    shadow = pygame.Surface((rect.width + 18, rect.height + 18), pygame.SRCALPHA)
    _rounded_rect(shadow, pygame.Rect(9, 9, rect.width, rect.height), shadow_rgba, radius=20)
    screen.blit(shadow, (rect.x - 9, rect.y - 9))
    card = pygame.Surface(rect.size, pygame.SRCALPHA)
    _rounded_rect(card, pygame.Rect(0, 0, rect.width, rect.height), fill_rgba, radius=20)
    # subtle top sheen
    hi = pygame.Surface((rect.width, max(18, rect.height // 12)), pygame.SRCALPHA)
    pygame.draw.rect(hi, (255,255,255,18), hi.get_rect(), border_radius=18)
    card.blit(hi, (0,0))
    screen.blit(card, rect.topleft)

def _load_backdrop():
    global _backdrop_raw
    if _backdrop_raw is not None:
        return
    if BACKDROP_IMG.exists():
        img = pygame.image.load(str(BACKDROP_IMG))
        _backdrop_raw = img.convert() if not img.get_masks()[3] else img.convert_alpha()
    else:
        _backdrop_raw = None

def _draw_backdrop(screen: pygame.Surface):
    """Backdrop: image if present; else gradient. Cached by height for speed."""
    _load_backdrop()
    w, h = screen.get_size()
    if _backdrop_raw is None:
        # gradient fallback
        top = (24, 26, 32); bot = (36, 40, 48)
        for y in range(h):
            t = y / max(1, h-1)
            c = (
                int(top[0] + (bot[0]-top[0]) * t),
                int(top[1] + (bot[1]-top[1]) * t),
                int(top[2] + (bot[2]-top[2]) * t),
            )
            pygame.draw.line(screen, c, (0, y), (w, y))
        return

    key = h
    if key not in _backdrop_scaled_by_h:
        bw, bh = _backdrop_raw.get_width(), _backdrop_raw.get_height()
        scale = max(w / bw, h / bh)
        tw, th = int(bw * scale), int(bh * scale)
        scaled = pygame.transform.smoothscale(_backdrop_raw, (tw, th))
        _backdrop_scaled_by_h[key] = scaled

    img = _backdrop_scaled_by_h[key]
    x = (w - img.get_width()) // 2
    y = (h - img.get_height()) // 2
    screen.blit(img, (x, y))

    # subtle vignette
    vignette = pygame.Surface((w, h), pygame.SRCALPHA)
    cx, cy = w // 2, h // 2
    maxr = int((w*w + h*h) ** 0.5 / 2)
    for r, a in ((maxr, 60), (int(maxr*0.75), 40), (int(maxr*0.5), 25)):
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.circle(s, (0,0,0,a), (cx, cy), r)
        vignette.blit(s, (0,0), special_flags=pygame.BLEND_RGBA_SUB)
    screen.blit(vignette, (0,0))

    # faint scanlines
    scan = pygame.Surface((w, h), pygame.SRCALPHA)
    for yy in range(0, h, 4):
        pygame.draw.line(scan, (0,0,0,18), (0, yy), (w, yy))
    screen.blit(scan, (0,0))

def _draw_grid(v, screen: pygame.Surface, assets):
    """F1-ish grid with BLACK borders, neon overlays & pulsing path (no start/goal halos)."""
    cs = v.cell_size
    ox, oy = v._grid_origin  # centered origin

    asphalt = assets.get("asphalt", cs)
    grass   = assets.get("grass",   cs)
    tire    = assets.get("tire",    cs)
    car     = assets.get_centered("car",  cs, 2.0)
    flag    = assets.get_centered("flag", cs, 1.20)

    # tiles
    for row in range(v.grid.height):
        for col in range(v.grid.width):
            val = v.grid.cells[row][col]
            rect = pygame.Rect(ox + col*cs, oy + row*cs, cs, cs)
            is_block = (str(val) in v.grid.weights and v.grid.weights[str(val)] == "BLOCK") or val == 1
            if is_block:
                if tire: screen.blit(tire, rect.topleft)
                else:    pygame.draw.rect(screen, BLACK, rect)
            elif val == 2:
                if grass: screen.blit(grass, rect.topleft)
                else:     pygame.draw.rect(screen, GRASS_GREEN, rect)
            else:
                if asphalt: screen.blit(asphalt, rect.topleft)
                else:       pygame.draw.rect(screen, ASPHALT_GRAY, rect)

            # black border for each tile
            pygame.draw.rect(screen, BLACK, rect, 1)

    # overlays: closed then open
    for (col,row) in v.closed_set:
        rect = pygame.Rect(ox + col*cs, oy + row*cs, cs, cs)
        s = pygame.Surface((cs, cs), pygame.SRCALPHA); s.fill(MAGENTA_A)
        screen.blit(s, rect.topleft)
    for (col,row) in v.open_set:
        rect = pygame.Rect(ox + col*cs, oy + row*cs, cs, cs)
        s = pygame.Surface((cs, cs), pygame.SRCALPHA); s.fill(CYAN_NEON_A)
        screen.blit(s, rect.topleft)

     # path: glow + pulse
    if len(v.path) >= 2:
        pts = []
        for (col, row) in v.path:
            cx = ox + col * cs + cs // 2
            cy = oy + row * cs + cs // 2
            pts.append((cx, cy))

        t = time.time()

        if getattr(v, "state", "") == "Done":
            # ---- FINISH EFFECT: pulse color white <-> green ----
            # k in [0,1] → blend(white -> green) and back
            k = 0.5 * (1.0 + math.sin(t * 8.0))
            white = (127, 255, 0)
            green = (144, 238, 144)  # mint-ish green; tweak if you want pure (0,255,0)
            col = (
                int(white[0] * (1 - k) + green[0] * k),
                int(white[1] * (1 - k) + green[1] * k),
                int(white[2] * (1 - k) + green[2] * k),
            )
            width = 6

            # soft outer glow
            glow = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            pygame.draw.lines(glow, (col[0], col[1], col[2], 70), False, pts, width + 2)
            screen.blit(glow, (0, 0), special_flags=pygame.BLEND_ADD)

            # core stroke
            pygame.draw.lines(screen, col, False, pts, width)

        else:
            # ---- RUNNING / IDLE: neon mint with width pulse (existing look) ----
            width = max(5, int(4 + 1.2 * abs(math.sin(t * 2.0))))
            glow = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            pygame.draw.lines(glow, (0, 255, 220, 60), False, pts, max(7, width + 2))
            screen.blit(glow, (0, 0), special_flags=pygame.BLEND_ADD)
            pygame.draw.lines(screen, GREEN_NEON, False, pts, width)

    # start / goal — clean icons, NO halo
    sx = ox + v.grid.start[0]*cs + cs//2
    sy = oy + v.grid.start[1]*cs + cs//2
    gx = ox + v.grid.goal[0]*cs  + cs//2
    gy = oy + v.grid.goal[1]*cs  + cs//2
    if car  is not None: screen.blit(car,  car.get_rect(center=(sx, sy)))
    if flag is not None: screen.blit(flag, flag.get_rect(center=(gx, gy)))

def _draw_timer_in_panel(v, screen: pygame.Surface):
    """Timer displayed inside the right panel, directly below the Map buttons."""
    if not hasattr(v, "get_run_time_str"):
        return
    rb = getattr(v, "_right_band", None)
    if not isinstance(rb, pygame.Rect) or rb.width <= 0:
        return

    # Anchor under Map 3 button if it exists; else place near the top of panel
    baseline_y = rb.y + 10
    if hasattr(v, "btn_map3") and isinstance(v.btn_map3.rect, pygame.Rect):
        baseline_y = v.btn_map3.rect.bottom + 16

    label = f"Run Time  {v.get_run_time_str()}"
    f = v.font
    surf = f.render(label, True, TEXT_LIGHT)

    pad_x, pad_y = 12, 6
    w = min(rb.width - 32, surf.get_width() + pad_x*2)
    h = surf.get_height() + pad_y*2
    x = rb.x + 16
    y = min(rb.bottom - h - 12, baseline_y)

    # pill (no stroke/outline)
    pill = pygame.Surface((w, h), pygame.SRCALPHA)
    _rounded_rect(pill, pill.get_rect(), PILL_BG, radius=12)
    screen.blit(pill, (x, y))
    screen.blit(surf, (x + pad_x, y + pad_y))

def prepare(assets) -> None:
    """Optional icon cleanup (remove white halos from car/flag icons)."""
    for key in ("car", "flag"):
        surf = assets._raw.get(key)
        if not surf: continue
        tmp = surf.convert() if not surf.get_masks()[3] else surf.convert_alpha()
        tmp.set_colorkey((255,255,255))
        assets._raw[key] = tmp

def draw(viewer, screen: pygame.Surface, assets) -> None:
    """
    Draw order:
      1) backdrop
      2) grid with neon overlays
      3) frosted right panel underlay
      4) timer pill inside panel (under Map buttons)
      (viewer draws text/buttons afterwards)
    """
    _draw_backdrop(screen)
    _draw_grid(viewer, screen, assets)

    # Right panel underlay that matches viewer._right_band
    rb = getattr(viewer, "_right_band", None)
    if isinstance(rb, pygame.Rect):
        _glass_panel(screen, rb, fill_rgba=PANEL_FILL, shadow_rgba=PANEL_SHADOW)

    # Timer inside the panel, placed under the Map buttons
    _draw_timer_in_panel(viewer, screen)
