# src/app/theme_skin.py
"""
Neon F1 skin — premium look (no logic changes)
- Cinematic backdrop with vignette + scanlines
- Bigger start/goal icons, no white halos
- Neon overlays (cyan=open, magenta=closed), pulsating mint path
- Glass panel with subtle shadow

Drop a dark wallpaper in: assets/backdrop.jpg
"""

from __future__ import annotations
import math, time
from pathlib import Path
from typing import Tuple, Optional
import pygame

# --- palette / alpha overlays ---
BLACK        = (0, 0, 0)
GRAY_60      = (140, 140, 140)
TEXT_LIGHT   = (230, 235, 240)
ACCENT_GOLD  = (255, 210, 0)
GREEN_NEON   = (0, 255, 200)
CYAN_NEON_A  = (0, 150, 255, 110)
MAGENTA_A    = (255, 0, 120, 90)
GRASS_GREEN  = (144, 238, 144)
ASPHALT_GRAY = (200, 200, 200)

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
BACKDROP_IMG = ASSETS_DIR / "backdrop.jpg"

# caches
_backdrop: Optional[pygame.Surface] = None
_backdrop_scaled: dict[int, pygame.Surface] = {}

def _rounded_rect(surface: pygame.Surface, rect: pygame.Rect, color, radius=14, width=0):
    pygame.draw.rect(surface, color, rect, width=width, border_radius=radius)

def _glass_panel(screen: pygame.Surface, rect: pygame.Rect,
                 fill_rgba=(24, 28, 36, 210), shadow_rgba=(0, 0, 0, 120)):
    shadow = pygame.Surface((rect.width + 18, rect.height + 18), pygame.SRCALPHA)
    _rounded_rect(shadow, pygame.Rect(9, 9, rect.width, rect.height), shadow_rgba, radius=18)
    screen.blit(shadow, (rect.x - 9, rect.y - 9))
    card = pygame.Surface(rect.size, pygame.SRCALPHA)
    _rounded_rect(card, pygame.Rect(0, 0, rect.width, rect.height), fill_rgba, radius=18)
    screen.blit(card, rect)

def _glow(screen: pygame.Surface, center: Tuple[int, int], color: Tuple[int, int, int], r: int):
    for a in (42, 28, 16, 8):
        s = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, a), (r * 2, r * 2), int(r * 1.6))
        screen.blit(s, (center[0] - r * 2, center[1] - r * 2), special_flags=pygame.BLEND_ADD)
    pygame.draw.circle(screen, color, center, r)

def _load_backdrop():
    global _backdrop
    if _backdrop is not None:
        return
    if BACKDROP_IMG.exists():
        img = pygame.image.load(str(BACKDROP_IMG))
        if pygame.display.get_surface(): img = img.convert()
        _backdrop = img
    else:
        _backdrop = None

def _draw_backdrop(screen: pygame.Surface):
    """Scale-and-fill backdrop with vignette + scanlines. Cached by height."""
    _load_backdrop()
    w, h = screen.get_size()
    if _backdrop is None:
        screen.fill(BLACK)
    else:
        key = h
        if key not in _backdrop_scaled:
            # scale to cover (letterbox-safe)
            bh = _backdrop.get_height()
            bw = _backdrop.get_width()
            scale = max(w / bw, h / bh)
            tw, th = int(bw * scale), int(bh * scale)
            if pygame.display.get_surface():
                scaled = pygame.transform.smoothscale(_backdrop, (tw, th))
            else:
                scaled = pygame.transform.scale(_backdrop, (tw, th))
            _backdrop_scaled[key] = scaled
        img = _backdrop_scaled[key]
        # center crop
        x = (w - img.get_width()) // 2
        y = (h - img.get_height()) // 2
        screen.blit(img, (x, y))

    # vignette (radial darkening)
    vignette = pygame.Surface((w, h), pygame.SRCALPHA)
    cx, cy = w // 2, h // 2
    maxr = int((w**2 + h**2) ** 0.5 / 2)
    for r, a in ((maxr, 60), (int(maxr*0.75), 40), (int(maxr*0.5), 25)):
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.circle(s, (0,0,0,a), (cx, cy), r)
        vignette.blit(s, (0,0), special_flags=pygame.BLEND_RGBA_SUB)
    screen.blit(vignette, (0,0))

    # scanlines
    scan = pygame.Surface((w, h), pygame.SRCALPHA)
    for y in range(0, h, 4):
        pygame.draw.line(scan, (0,0,0,20), (0, y), (w, y))
    screen.blit(scan, (0,0))

# ---- public hooks ----
def prepare(assets) -> None:
    """Optional pre-caching and icon cleanup (kill white halos)."""
    # Try to remove white halos by setting colorkey for near-white icons
    for key in ("car", "flag"):
        surf = assets._raw.get(key)
        if not surf: continue
        # If icon lacks alpha and is on white, treat white as transparent
        tmp = surf.convert() if not surf.get_masks()[3] else surf.convert_alpha()
        tmp.set_colorkey((255,255,255))
        assets._raw[key] = tmp

def draw(viewer, screen: pygame.Surface, assets) -> None:
    _draw_backdrop(screen)
    _draw_grid_neon(viewer, screen, assets)
    _draw_panel_neon(viewer, screen)

def _draw_grid_neon(v, screen, assets):
    cs = v.cell_size
    t = time.time()

    asphalt = assets.get("asphalt", cs)
    grass   = assets.get("grass", cs)
    tire    = assets.get("tire", cs)
    car     = assets.get_centered("car",  cs, 1.10)   # bigger icons
    flag    = assets.get_centered("flag", cs, 1.10)

    # board shadow plate
    grid_px_w = v.GRID_MARGIN*2 + v.grid.width * cs
    grid_px_h = v.GRID_MARGIN*2 + v.grid.height* cs
    plate = pygame.Surface((grid_px_w, grid_px_h), pygame.SRCALPHA)
    pygame.draw.rect(plate, (0,0,0,90), plate.get_rect(), border_radius=18)
    screen.blit(plate, (0,0))

    # tiles
    for row in range(v.grid.height):
        for col in range(v.grid.width):
            val = v.grid.cells[row][col]
            rect = pygame.Rect(v.GRID_MARGIN + col * cs, v.GRID_MARGIN + row * cs, cs, cs)
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

            pygame.draw.rect(screen, (255,255,255,35), rect, 1)  # thin bright stroke

    # overlays: closed then open
    for (col, row) in v.closed_set:
        rect = pygame.Rect(v.GRID_MARGIN + col * cs, v.GRID_MARGIN + row * cs, cs, cs)
        s = pygame.Surface((cs, cs), pygame.SRCALPHA); s.fill(MAGENTA_A)
        screen.blit(s, rect.topleft)
    for (col, row) in v.open_set:
        rect = pygame.Rect(v.GRID_MARGIN + col * cs, v.GRID_MARGIN + row * cs, cs, cs)
        s = pygame.Surface((cs, cs), pygame.SRCALPHA); s.fill(CYAN_NEON_A)
        screen.blit(s, rect.topleft)

    # neon path with pulse
    if len(v.path) >= 2:
        pts = []
        for (col, row) in v.path:
            cx = v.GRID_MARGIN + col * cs + cs // 2
            cy = v.GRID_MARGIN + row * cs + cs // 2
            pts.append((cx, cy))
        width = int(4 + 1.2 * abs(math.sin(t * 2.0)))
        # outer glow
        glow = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        pygame.draw.lines(glow, (0, 255, 220, 60), False, pts, max(6, width+2))
        screen.blit(glow, (0,0), special_flags=pygame.BLEND_ADD)
        # core line
        pygame.draw.lines(screen, GREEN_NEON, False, pts, max(4, width))

    # start/goal with glow (no base circle)
    _draw_badge_neon(v, screen, v.grid.start, car,  (80, 160, 255))
    _draw_badge_neon(v, screen, v.grid.goal,  flag, (255, 100, 120))

def _draw_badge_neon(v, screen, cell, icon: Optional[pygame.Surface], color):
    cs = v.cell_size
    col, row = cell
    cx = v.GRID_MARGIN + col * cs + cs // 2
    cy = v.GRID_MARGIN + row * cs + cs // 2
    r  = max(8, cs // 2)
    # glow only (no solid circle) → prevents white base look
    for a in (40, 24, 12):
        s = pygame.Surface((r*4, r*4), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, a), (r*2, r*2), int(r*1.4))
        screen.blit(s, (cx - r*2, cy - r*2), special_flags=pygame.BLEND_ADD)
    if icon is not None:
        rect = icon.get_rect(center=(cx, cy))
        screen.blit(icon, rect)

def _draw_panel_neon(v, screen):
    grid_px_w = v.GRID_MARGIN * 2 + v.grid.width * v.cell_size
    panel_rect = pygame.Rect(grid_px_w, 0, v.PANEL_W, v.GRID_MARGIN * 2 + v.grid.height * v.cell_size)
    _glass_panel(screen, panel_rect, fill_rgba=(18, 20, 28, 190), shadow_rgba=(0, 0, 0, 140))

    x0 = panel_rect.x + 16
    y  = panel_rect.y + 16

    def line(text: str, big=False, color=TEXT_LIGHT):
        nonlocal y
        f = v.font_big if big else v.font
        surf = f.render(text, True, color)
        screen.blit(surf, (x0, y))
        y += surf.get_height() + 8

    line("A* vs Dijkstra", big=True, color=ACCENT_GOLD)
    mode = getattr(v, "MODE", "student")
    line(f"Mode: {mode}")
    line(f"Map: {v.selected_map_key}")
    line(f"Algo: {v.selected_algo}")
    line(f"State: {v.state}")
    line(f"Speed: {v.steps_per_sec} steps/s")

    y += 6
    line("Metrics", color=ACCENT_GOLD)
    m = getattr(v, "_last_metrics", {})
    for k, label in (("popped","Popped"),("open_size","Open"),("closed_count","Closed"),
                     ("path_len","Path Len"),("total_cost","Total Cost")):
        val = m.get(k, None)
        if val is not None:
            surf = v.font.render(f"{label:>11}: {val}", True, TEXT_LIGHT)
            screen.blit(surf, (x0, y)); y += surf.get_height() + 4

    y += 8
    line("Controls", color=ACCENT_GOLD)
    for c in [
        "[1/2/3] Switch Map",
        "[D/A]   Dijkstra / A*",
        "[R]     Reset   [N] Step",
        "[SPACE] Run / Pause",
        "[+/-]   Speed   [Q] Quit",
        "[F]     Fullscreen (if enabled)",  # viewer tweak below
    ]:
        screen.blit(v.font.render(c, True, TEXT_LIGHT), (x0, y)); y += 22
