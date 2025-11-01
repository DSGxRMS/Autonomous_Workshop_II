# Run_map.pyw
import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

# ---------------- Paths & import setup ----------------
REPO_ROOT = Path(__file__).resolve().parent
ASSETS = REPO_ROOT / "assets"
LOGO_PNG = ASSETS / "logo.png"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Force student mode
os.environ["WORKSHOP_MODE"] = "student"

# ---------------- Theming ----------------
BG_DARK       = "#181C24"
BG_CARD       = "#202634"
TXT_PRIMARY   = "#E6EBF0"

# Button colors (lighter than card)
BTN_BG        = "#2B3446"    # normal fill
BTN_BG_HOVER  = "#344058"    # hover fill
BTN_BORDER    = "#3F4B62"    # normal border
BTN_BORDER_HOVER = "#50607A" # hover border
BTN_SHADOW    = "#11151C"    # drop shadow

def center(win: tk.Tk | tk.Toplevel, w: int, h: int):
    win.update_idletasks()
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    x = max(0, (sw - w) // 2)
    y = max(0, (sh - h) // 3)
    win.geometry(f"{w}x{h}+{x}+{y}")

class RoundedButton(tk.Canvas):
    """Rounded button on Canvas with border + subtle shadow; hover & keyboard support."""
    def __init__(self, master, text, command=None, width=360, height=52,
                 radius=18, bg=BTN_BG, fg=TXT_PRIMARY, bg_hover=BTN_BG_HOVER,
                 border=BTN_BORDER, border_hover=BTN_BORDER_HOVER,  # kept arg for compat; unused for color swap
                 **kwargs):
        super().__init__(master, width=width, height=height,
                         highlightthickness=0, bd=0, bg=master["bg"],
                         takefocus=1, **kwargs)
        self._w_px   = width
        self._h_px   = height
        self._r      = radius
        self._bg     = bg
        self._bg_h   = bg_hover
        self._fg     = fg
        self._bdc    = border
        self._cmd    = command
        self._label  = text
        self._hover  = False

        self._draw()

        # Mouse
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

        # Keyboard (works when widget actually has focus, e.g., after click or via Tab)
        self.bind("<space>", self._on_click)   # Space to activate
        self.bind("<Return>", self._on_click)  # Enter to activate
        self.bind("<FocusIn>", lambda _e: self._draw())
        self.bind("<FocusOut>", lambda _e: self._draw())

    def _on_enter(self, _e):
        self._hover = True
        # DO NOT focus on hover (prevents focus ring from appearing on hover)
        self._draw()

    def _on_leave(self, _e):
        self._hover = False
        self._draw()

    def _rr(self, x1, y1, x2, y2, r, fill, outline="", ow=1):
        self.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90, fill=fill, outline=fill)
        self.create_arc(x2-2*r, y1, x2, y1+2*r, start=0, extent=90, fill=fill, outline=fill)
        self.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90, fill=fill, outline=fill)
        self.create_arc(x2-2*r, y2-2*r, x2, y2, start=270, extent=90, fill=fill, outline=fill)
        self.create_rectangle(x1+r, y1, x2-r, y2, fill=fill, outline=fill)
        self.create_rectangle(x1, y1+r, x2, y2-r, fill=fill, outline=fill)
        if outline:
            self.create_line(x1+r, y1, x2-r, y1, fill=outline, width=ow)
            self.create_line(x1+r, y2, x2-r, y2, fill=outline, width=ow)
            self.create_line(x1, y1+r, x1, y2-r, fill=outline, width=ow)
            self.create_line(x2, y1+r, x2, y2-r, fill=outline, width=ow)

    def _draw(self):
        self.delete("all")
        r = self._r
        pad = 2
        x1, y1 = pad, pad
        x2, y2 = self._w_px - pad, self._h_px - pad

        # Shadow
        self._rr(x1, y1+2, x2, y2+2, r, fill=BTN_SHADOW)

        # Fill (hover changes fill only)
        fill = self._bg_h if self._hover else self._bg
        self._rr(x1, y1, x2, y2, r, fill=fill)

        # Border — constant color (no hover change)
        self._rr(x1, y1, x2, y2, r, fill="", outline=self._bdc, ow=1)

        # Focus ring only when the widget really has focus (after click or Tab), not on hover
        if str(self.focus_get()) == str(self):
            self._rr(x1+2, y1+2, x2-2, y2-2, r-2, fill="", outline="#7BA7FF", ow=1)

        # Label
        self.create_text((self._w_px//2, self._h_px//2),
                         text=self._label, fill=self._fg,
                         font=("Segoe UI", 12, "bold"))

    def _on_click(self, _e=None):
        if callable(self._cmd):
            self._cmd()


def _load_logo(max_width: int):
    if not LOGO_PNG.exists():
        return None
    try:
        img = tk.PhotoImage(file=str(LOGO_PNG))
        w = img.width()
        if w > max_width:
            factor = max(1, int(round(w / max_width)))
            img = img.subsample(factor, factor)
        return img
    except Exception:
        return None

def start_viewer_and_close(root: tk.Tk):
    try:
        root.destroy()
        from src.app.viewer import main as viewer_main
        viewer_main()
    except Exception as e:
        messagebox.showerror("Launch error", f"Couldn't start the viewer:\n{e}")

def main():
    root = tk.Tk()
    root.title("Pathfinding Workshop — Launcher")
    root.configure(bg=BG_DARK)
    root.resizable(False, False)

    # Window icon from logo.png
    try:
        if LOGO_PNG.exists():
            icon_img = tk.PhotoImage(file=str(LOGO_PNG))
            root.iconphoto(True, icon_img)
            root._icon_img_ref = icon_img  # keep reference
    except Exception:
        pass

    WIDTH, HEIGHT = 560, 420
    center(root, WIDTH, HEIGHT)

    outer = tk.Frame(root, bg=BG_DARK)
    outer.pack(fill="both", expand=True, padx=18, pady=18)

    card = tk.Frame(outer, bg=BG_CARD, bd=0, highlightthickness=0)
    card.pack(fill="both", expand=True)

    inner = tk.Frame(card, bg=BG_CARD)
    inner.pack(fill="both", expand=True, padx=20, pady=16)

    tk.Label(inner, text="Before you start…", bg=BG_CARD, fg=TXT_PRIMARY,
             font=("Segoe UI", 14, "bold")).pack(anchor="w")

    tk.Label(
        inner,
        text=("We request you to use your laptop’s native speakers and set the "
              "system volume above 10% before starting the exercise."),
        bg=BG_CARD, fg=TXT_PRIMARY, justify="left", wraplength=WIDTH - 96,
        font=("Segoe UI", 11)
    ).pack(anchor="w", pady=(6, 10))

    # Single logo below the text
    mid_logo_img = _load_logo(max_width=220)
    if mid_logo_img:
        mid_logo = tk.Label(inner, image=mid_logo_img, bg=BG_CARD)
        mid_logo.pack(pady=(6, 12))
    else:
        mid_logo = None

    # Button (lighter fill + defined border, no sheen)
    btn = RoundedButton(
        inner,
        text="Take me to the track!",
        command=lambda: start_viewer_and_close(root),
        width=WIDTH - 96,
        height=56,
        radius=18
    )
    btn.pack(pady=(2, 6))
    btn.focus_set()  # immediate keyboard support

    # Enter key also starts
    root.bind("<Return>", lambda _e: start_viewer_and_close(root))

    # Keep references so images don't get GC'd
    root._mid_logo_img = mid_logo_img
    root._mid_logo = mid_logo

    root.mainloop()

if __name__ == "__main__":
    main()
