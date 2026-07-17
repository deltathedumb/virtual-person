"""A read-only 2D top-down preview window for a running simulation.

The renderer only consumes ``simulation.snapshot()``. It never calls world or
agent mutation methods itself, so it can be attached to or detached from a
simulation without changing its behavior. This mirrors the design note in
``README.md``: the world is symbolic and headless-first, and a renderer is a
consumer of ``snapshot()``, not a dependency of the reasoning or training
layers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .simulation import VirtualPersonSimulation

ROOM_LAYOUT: dict[str, dict[str, float]] = {
    "bedroom": {"x": 0, "y": 0, "w": 4, "h": 4},
    "bathroom": {"x": 4, "y": 0, "w": 3, "h": 4},
    "hallway": {"x": 0, "y": 4, "w": 7, "h": 2},
    "kitchen": {"x": 7, "y": 0, "w": 4, "h": 6},
    "living_room": {"x": 0, "y": 6, "w": 11, "h": 4},
}

_DRIVE_FIELDS = (
    ("hunger", "hunger", False),
    ("thirst", "thirst", False),
    ("fatigue", "fatigue", False),
    ("bladder", "bladder", False),
    ("hygiene", "hygiene", True),
    ("health", "health", True),
    ("social", "social need", False),
)

_PALETTE = {
    "bg": "#1a1d22",
    "panel": "#23272e",
    "line": "#383d46",
    "ink": "#e8e6e1",
    "ink_dim": "#8b909a",
    "accent": "#7fb1cf",
    "accent_soft": "#2c3f4b",
    "agent": "#e0a458",
    "good": "#4d7c53",
    "warn": "#c8853a",
    "bad": "#b3442f",
}


def _severity(pressure: float) -> str:
    if pressure >= 0.85:
        return "bad"
    if pressure >= 0.6:
        return "warn"
    return "good"


def draw_floor_plan(canvas: Any, simulation: VirtualPersonSimulation, cell_pixels: float) -> None:
    """Draw the current floor plan onto ``canvas`` from ``simulation``.

    Shared by :class:`Renderer` and any other Tkinter surface (such as
    ``runtime_ui.py``) that wants the same live floor-plan view embedded in
    its own window, so the drawing logic exists in exactly one place.
    """
    snap = simulation.snapshot()
    cell = cell_pixels
    agent_room = snap["world"]["room"]

    canvas.delete("all")
    for name, rect in ROOM_LAYOUT.items():
        x0, y0 = rect["x"] * cell, rect["y"] * cell
        x1, y1 = (rect["x"] + rect["w"]) * cell, (rect["y"] + rect["h"]) * cell
        canvas.create_rectangle(
            x0 + 2, y0 + 2, x1 - 2, y1 - 2,
            outline=_PALETTE["line"], width=2,
        )
        canvas.create_text(
            x0 + 10, y0 + 12, text=name.replace("_", " ").upper(),
            anchor="w", fill=_PALETTE["ink_dim"], font=("Consolas", 8),
        )

    objects = simulation.agent.world.objects
    room_counts: dict[str, int] = {}
    inventory = set(snap["agent"].get("inventory", []))
    for obj_id, obj in objects.items():
        if obj.container:
            continue
        rect = ROOM_LAYOUT.get(obj.room)
        if not rect:
            continue
        idx = room_counts.get(obj.room, 0)
        room_counts[obj.room] = idx + 1
        cols = max(1, int((rect["w"] - 0.6)))
        col = idx % cols
        row = idx // cols
        cx = (rect["x"] + 0.7 + col) * cell
        cy = (rect["y"] + 1.1 + row) * cell
        fill = _PALETTE["accent_soft"] if (obj.openable and obj.is_open) else _PALETTE["bg"]
        outline = _PALETTE["accent"] if (obj.openable and obj.is_open) else _PALETTE["ink_dim"]
        radius = 9
        canvas.create_oval(
            cx - radius, cy - radius, cx + radius, cy + radius,
            fill=fill, outline=outline, width=1.5,
        )
        label = obj_id + (" *" if obj_id in inventory else "")
        canvas.create_text(
            cx, cy + radius + 8, text=label,
            fill=_PALETTE["ink_dim"], font=("Consolas", 7),
        )

    rect = ROOM_LAYOUT.get(agent_room)
    if rect:
        ax = (rect["x"] + rect["w"] / 2) * cell
        ay = (rect["y"] + rect["h"] - 0.9) * cell
        canvas.create_oval(
            ax - 22, ay - 22, ax + 22, ay + 22,
            outline=_PALETTE["agent"], width=1,
        )
        canvas.create_oval(
            ax - 11, ay - 11, ax + 11, ay + 11,
            fill=_PALETTE["agent"], outline=_PALETTE["panel"], width=1.5,
        )


@dataclass
class RendererConfig:
    fps: float = 10.0
    cell_pixels: float = 56.0
    title: str = "Virtual Person - live preview"


class Renderer:
    """Opens a Tkinter window that redraws a simulation's snapshot on a timer.

    Usage::

        sim = VirtualPersonSimulation.default(seed=1)
        renderer = Renderer(sim, fps=10)
        renderer.start()          # opens the window, non-blocking
        ...
        renderer.stop()           # closes the window

    Or, to have the renderer also advance the simulation each frame::

        renderer = Renderer(sim, fps=10, step_seconds=60.0)
        renderer.run()             # blocks until the window is closed

    The renderer never mutates simulation or agent state itself beyond the
    optional ``step_seconds`` advance hook; all drawing reads only from
    ``simulation.snapshot()``.
    """

    def __init__(
        self,
        simulation: VirtualPersonSimulation,
        fps: float = 10.0,
        *,
        step_seconds: float | None = None,
        on_frame: Callable[[dict[str, Any]], None] | None = None,
        config: RendererConfig | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.simulation = simulation
        self.fps = fps
        self.step_seconds = step_seconds
        self.on_frame = on_frame
        self.config = config or RendererConfig(fps=fps)

        self._root = None
        self._canvas = None
        self._info_labels: dict[str, Any] = {}
        self._bars: dict[str, Any] = {}
        self._events_box = None
        self._after_id = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Build the window and begin the redraw timer. Non-blocking."""
        if self._running:
            return
        import tkinter as tk

        self._root = tk.Tk()
        self._root.title(self.config.title)
        self._root.configure(bg=_PALETTE["bg"])
        self._root.protocol("WM_DELETE_WINDOW", self.stop)

        plan_w = int(11 * self.config.cell_pixels)
        plan_h = int(10 * self.config.cell_pixels)

        self._canvas = tk.Canvas(
            self._root, width=plan_w, height=plan_h,
            bg=_PALETTE["bg"], highlightthickness=0,
        )
        self._canvas.grid(row=0, column=0, padx=12, pady=12, sticky="n")

        side = tk.Frame(self._root, bg=_PALETTE["bg"])
        side.grid(row=0, column=1, padx=(0, 12), pady=12, sticky="n")

        header = tk.Label(
            side, text="", justify="left", anchor="w",
            font=("Consolas", 10, "bold"), fg=_PALETTE["ink"], bg=_PALETTE["bg"],
        )
        header.pack(fill="x")
        self._info_labels["header"] = header

        meta = tk.Label(
            side, text="", justify="left", anchor="w",
            font=("Consolas", 9), fg=_PALETTE["ink_dim"], bg=_PALETTE["bg"],
        )
        meta.pack(fill="x", pady=(0, 8))
        self._info_labels["meta"] = meta

        drives_frame = tk.Frame(side, bg=_PALETTE["panel"])
        drives_frame.pack(fill="x", pady=(0, 8))
        for key, label, _ in _DRIVE_FIELDS:
            row = tk.Frame(drives_frame, bg=_PALETTE["panel"])
            row.pack(fill="x", padx=8, pady=2)
            name_label = tk.Label(
                row, text=label, width=11, anchor="w",
                font=("Consolas", 9), fg=_PALETTE["ink_dim"], bg=_PALETTE["panel"],
            )
            name_label.pack(side="left")
            bar_bg = tk.Canvas(
                row, width=90, height=8, bg=_PALETTE["accent_soft"],
                highlightthickness=0,
            )
            bar_bg.pack(side="left", padx=6)
            value_label = tk.Label(
                row, text="0.00", width=5, anchor="e",
                font=("Consolas", 9), fg=_PALETTE["ink"], bg=_PALETTE["panel"],
            )
            value_label.pack(side="left")
            self._bars[key] = (bar_bg, value_label)

        events_label = tk.Label(
            side, text="recent events", anchor="w",
            font=("Consolas", 8, "bold"), fg=_PALETTE["ink_dim"], bg=_PALETTE["bg"],
        )
        events_label.pack(fill="x", pady=(8, 2))

        events_box = tk.Text(
            side, width=42, height=10, bg=_PALETTE["panel"], fg=_PALETTE["ink"],
            font=("Consolas", 8), relief="flat", wrap="word",
        )
        events_box.pack(fill="both", expand=True)
        events_box.configure(state="disabled")
        self._events_box = events_box

        self._running = True
        self._schedule_frame()

    def _schedule_frame(self) -> None:
        if not self._running or self._root is None:
            return
        self._render_frame()
        delay_ms = max(1, int(1000.0 / self.fps))
        self._after_id = self._root.after(delay_ms, self._schedule_frame)

    def _render_frame(self) -> None:
        if self.step_seconds:
            self.simulation.step()
        snap = self.simulation.snapshot()
        self._draw_plan(snap)
        self._draw_sidebar(snap)
        if self.on_frame is not None:
            self.on_frame(snap)

    def _draw_plan(self, snap: dict[str, Any]) -> None:
        draw_floor_plan(self._canvas, self.simulation, self.config.cell_pixels)

    def _draw_sidebar(self, snap: dict[str, Any]) -> None:
        agent = snap["agent"]
        name = agent["self"]["name"]
        self._info_labels["header"].configure(
            text=f"{name}  —  {agent['activity']} ({agent['goal'] or 'no goal'})"
        )
        self._info_labels["meta"].configure(
            text=(
                f"day {snap['day']} · hour {snap['hour']:.1f} · "
                f"room {snap['world']['room']} · actions {agent['action_count']} · "
                f"failures {agent['failures']}"
            )
        )

        body = agent["body"]
        for key, _, inverted in _DRIVE_FIELDS:
            value = float(body[key])
            pressure = (1.0 - value) if inverted else value
            bar_bg, value_label = self._bars[key]
            bar_bg.delete("all")
            width = max(0.0, min(1.0, pressure)) * 90
            color = _PALETTE[_severity(pressure)]
            bar_bg.create_rectangle(0, 0, width, 8, fill=color, outline="")
            value_label.configure(text=f"{value:.2f}")

        events = snap.get("recent_events", [])
        box = self._events_box
        box.configure(state="normal")
        box.delete("1.0", "end")
        for event in reversed(events[-12:]):
            sign = "+" if event["reward"] > 0 else ""
            box.insert(
                "end",
                f"{event['time']:>8.0f}s  {event['message']}  ({sign}{event['reward']:.3f})\n",
            )
        box.configure(state="disabled")

    def run(self) -> None:
        """Start the window and block, pumping the Tk event loop until closed."""
        self.start()
        if self._root is not None:
            self._root.mainloop()

    def stop(self) -> None:
        """Cancel the redraw timer and destroy the window, if open."""
        self._running = False
        if self._after_id is not None and self._root is not None:
            try:
                self._root.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass
            self._root = None
        self._canvas = None
        self._events_box = None
        self._bars.clear()
        self._info_labels.clear()
