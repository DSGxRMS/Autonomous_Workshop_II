# Autonomous Workshop II 🏁

Hello! This repo is your playground to **see** and **build** classic pathfinding (Dijkstra & A*) on a race-track themed grid. You’ll run a viewer, switch maps, and fill tiny bits of code to make the algorithms work—step by step.

You don’t need heavy ML/robotics background. If you can run Python and edit a few lines, you’re set. 😀

---

## What’s inside

- **Interactive Viewer** — `src/app/viewer.py`  
  Visualizes the grid, start/finish, frontier (open/closed), and the evolving path.
- **Student Files (you’ll edit these)**  
  - `src/core/dijkstra_template.py` — a few short blanks to fill  
  - `src/core/astar_student.py` — a small skeleton with clearly labeled TODOs
- **Maps** — `/maps` JSONs describing start/goal, walls, and (sometimes) terrain costs
- **Assets** — `/assets` images for tiles and icons (asphalt, grass, tire stacks, car, flag)
- **Optional Visual Skin** — `src/app/theme_skin.py` (cosmetic only)

---

## Quick Start (5–10 minutes)

> Requires **Python 3.9+** and **pip**. We recommend python 3.13

### 1) Create & activate a virtual environment if you want

**Windows (PowerShell)**
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Minimal `requirements.txt`:
```
pygame>=2.6.1
```

---

## Run the viewer

**simply drag and drop the launcher.pyw file inside the terminal:**

If you see only a white/blank window, give it a second. If it persists, see **Troubleshooting**.

---

## Controls
**Hotkeys:**
- **[1] [2] [3]** — Switch map  
- **[D] / [A]** — Choose algorithm (Dijkstra / A*)  
- **[SPACE]** — Run / Pause  
- **[N]** — Step once  
- **[R]** — Reset  
- **[+]/[-]** — Steps per second  
- **[Q] / [ESC]** — Quit

The side panel shows **Mode**, **Map**, **Algo**, **State**, **Speed**, and **Metrics**.

---

## What you’ll do today

1. Run the launcher.  
2. Fill a few missing lines in **Dijkstra**.  
3. Compare behavior with **A***.  
4. Try the **weighted map** and see how cost influences the route.  


We’ll gradually reveal **answer passwords** during the workshop. When instructed, you’ll unlock the official solutions (see **./answers/**).

---

## Where to edit (and where not to)

### ✅ You will edit
- `src/core/dijkstra_template.py`  
  Fill the clearly marked blocks (neighbor gen, stop condition, relax & push, backtrack). ~5–10 lines total.
- `src/core/astar_student.py`  
  Small, focused edits if your mentor asks.

### ❌ Do not edit (read-only for you)
- `src/app/viewer.py` — app + visuals glue  
- `src/core/types.py` — shared types  
- `src/core/answers/*` — locked reference solutions  
- `/maps/*.json` — unless a mentor asks you to tweak a map

> If something “breaks,” revert only your last change to the student files. The viewer keeps logic vs. visuals separated so you can’t accidentally take down the whole app.

---

## Answers / Passwords 🔒

- Official solutions live in `src/core/answers/` and a password-protected archive provided by the autonomous team.
- During the session, we’ll provide **passwords / one-time codes** at checkpoints to:
  - Review a reference **Dijkstra** or **A*** solution
---

## Maps at a glance

Each map is a JSON like:
```json
{
  "width": 25,
  "height": 25,
  "start": [1, 1],
  "goal": [23, 23],
  "move": 4,
  "cells": [[0,0,1,...], ...],
  "weights": { "0": 1, "1": "BLOCK", "2": 3 }
}
```

- `0` = road (cost 1)  
- `1` = wall (BLOCK)  
- `2` = grass (example: cost 3)  
- `move: 4` → up/down/left/right only (no diagonals)

**If Dijkstra says “No path”** but you believe there is one, check:
- Start/goal aren’t on a wall  
- There’s an unbroken corridor of non-wall cells from start to goal  
- You actually filled the blanks in the template (see below)

---

## Your Dijkstra TODOs (recap)

In `src/core/dijkstra_template.py`, you’ll fill these spots (comments show the *actual* code to write):

- **Neighbor candidates**: four 4-connected moves  
- **Neighbor filter**: keep only in-bounds and not walls  
- **Stop condition**: stop when you *pop* the goal  
- **Relaxation**: uniform cost  
- **Update & push**: update `g`, `parent`, push to PQ, mark frontier  
- **Backtrack**: `cur = parent[cur]` until start, then reverse

That’s the entire algorithm.

---
## Troubleshooting

**1) Pygame window doesn’t show / is blank**  
- Wait a second; pygame logs in the terminal first.  
- Update graphics drivers (Windows) or run from a terminal (macOS/Linux).  
- Run with `python -u src/app/viewer.py` to see logs immediately.

**2) Dijkstra always says “No path”**  
- You probably missed a small blank in `dijkstra_template.py`:
  - Four neighbors (±x, ±y)
  - Filter in-bounds & not a wall
  - Stop when you *pop* the goal
  - Relax with `+1` and push/update PQ
  - Backtrack via `parent`

**3) Assets look plain or missing**  
- Ensure `/assets` contains:
  - `asphalt.png`, `grass.png`, `tire_stack.png`, `f1.png`, `checkered_flag.png`  
- Missing assets fall back to solid colors (still works).

**4) Python version mismatch**  
- Use the venv created above and install from `requirements.txt` only.

---

## One-page recap

- `pip install -r requirements.txt`  
- `python -u src/app/viewer.py`  
- Edit: `src/core/dijkstra_template.py`  
- Try maps [1] → [2] → [3] (weighted)  
- Use **Space** to watch expansions  
- Answers unlocked later via **passwords/codes** from mentors

Have fun—and may your heuristic be admissible and your lap times minimal. 🏎️💨
