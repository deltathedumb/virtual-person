"""A GUI for running a trained checkpoint, watching the agent live, and
assigning/reviewing tasks through the task economy.

This is a runtime control panel, not a training tool: it loads an already
trained ``.pt`` checkpoint (see ``trainer_cli.py`` / ``trainer_ui.py`` for
training) and drives it against the live symbolic simulation via
``AutonomousSpikingAgent``. The floor-plan drawing is shared with
``renderer.py`` so both surfaces stay visually identical.

Safety note: this window can start/pause/step the agent and let a human
assign and score tasks. It never lets the model set its own task score,
touch its own balance, or bypass the world's action validation — those stay
exactly as implemented in ``economy.py`` and ``world.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from .env import VirtualPersonEnv
from .renderer import draw_floor_plan
from .spiking_mind import SpikingMind
from .spiking_runtime import AutonomousSpikingAgent

_PALETTE = {
    "bg": "#1a1d22",
    "panel": "#23272e",
    "line": "#383d46",
    "ink": "#e8e6e1",
    "ink_dim": "#8b909a",
    "accent": "#7fb1cf",
    "accent_soft": "#2c3f4b",
    "good": "#4d7c53",
    "warn": "#c8853a",
    "bad": "#b3442f",
}

_DRIVE_ROWS = (
    ("hunger", "hunger", False),
    ("thirst", "thirst", False),
    ("fatigue", "fatigue", False),
    ("bladder", "bladder", False),
    ("hygiene", "hygiene", True),
    ("health", "health", True),
    ("social", "social need", False),
)


def _severity(pressure: float) -> str:
    if pressure >= 0.85:
        return "bad"
    if pressure >= 0.6:
        return "warn"
    return "good"


class RuntimeApp:
    def __init__(self, root: Any) -> None:
        import tkinter as tk
        from tkinter import filedialog, ttk

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog

        self.root = root
        self.root.title("Virtual Person - runtime")
        self.root.configure(bg=_PALETTE["bg"])

        self.agent_runtime: AutonomousSpikingAgent | None = None
        self.running = False
        self.speed_steps_per_tick = 1
        self._after_id: str | None = None
        self.last_decision_message = ""
        self.last_drive_activity: dict[str, float] = {}

        self._build_layout()
        self._refresh_task_lists()

    # -- layout -----------------------------------------------------------

    def _build_layout(self) -> None:
        tk, ttk = self.tk, self.ttk

        top = tk.Frame(self.root, bg=_PALETTE["bg"])
        top.pack(fill="x", padx=10, pady=(10, 4))

        tk.Button(top, text="Load checkpoint...", command=self._on_load_checkpoint).pack(side="left")
        self.checkpoint_label = tk.Label(
            top, text="No checkpoint loaded", fg=_PALETTE["ink_dim"], bg=_PALETTE["bg"],
            font=("Consolas", 9),
        )
        self.checkpoint_label.pack(side="left", padx=10)

        controls = tk.Frame(self.root, bg=_PALETTE["bg"])
        controls.pack(fill="x", padx=10, pady=(0, 8))

        self.start_button = tk.Button(controls, text="Start", command=self._on_start, state="disabled")
        self.start_button.pack(side="left")
        self.pause_button = tk.Button(controls, text="Pause", command=self._on_pause, state="disabled")
        self.pause_button.pack(side="left", padx=(6, 0))
        self.step_button = tk.Button(controls, text="Step once", command=self._on_step, state="disabled")
        self.step_button.pack(side="left", padx=(6, 0))

        tk.Label(controls, text="  Speed (steps/tick):", fg=_PALETTE["ink_dim"], bg=_PALETTE["bg"]).pack(side="left")
        self.speed_var = tk.IntVar(value=1)
        tk.Spinbox(
            controls, from_=1, to=50, width=4, textvariable=self.speed_var,
            command=self._on_speed_change,
        ).pack(side="left", padx=(4, 0))

        self.deterministic_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            controls, text="Deterministic", variable=self.deterministic_var,
            fg=_PALETTE["ink"], bg=_PALETTE["bg"], selectcolor=_PALETTE["panel"],
        ).pack(side="left", padx=(12, 0))

        body = tk.Frame(self.root, bg=_PALETTE["bg"])
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # left: floor plan
        left = tk.Frame(body, bg=_PALETTE["panel"])
        left.pack(side="left", fill="both", expand=False, padx=(0, 8))
        plan_w = int(11 * 56)
        plan_h = int(10 * 56)
        self.canvas = tk.Canvas(left, width=plan_w, height=plan_h, bg=_PALETTE["bg"], highlightthickness=0)
        self.canvas.pack(padx=10, pady=10)

        # middle: decision + drives
        middle = tk.Frame(body, bg=_PALETTE["bg"], width=300)
        middle.pack(side="left", fill="y", padx=(0, 8))

        tk.Label(
            middle, text="LAST DECISION", font=("Consolas", 8, "bold"),
            fg=_PALETTE["ink_dim"], bg=_PALETTE["bg"], anchor="w",
        ).pack(fill="x")
        self.decision_text = tk.Text(
            middle, width=38, height=6, bg=_PALETTE["panel"], fg=_PALETTE["ink"],
            font=("Consolas", 9), relief="flat", wrap="word",
        )
        self.decision_text.pack(fill="x", pady=(2, 8))
        self.decision_text.configure(state="disabled")

        tk.Label(
            middle, text="DRIVES & BODY", font=("Consolas", 8, "bold"),
            fg=_PALETTE["ink_dim"], bg=_PALETTE["bg"], anchor="w",
        ).pack(fill="x")
        self.drive_frame = tk.Frame(middle, bg=_PALETTE["panel"])
        self.drive_frame.pack(fill="x", pady=(2, 8))
        self._drive_bars: dict[str, tuple[Any, Any]] = {}
        for key, label, _ in _DRIVE_ROWS:
            row = tk.Frame(self.drive_frame, bg=_PALETTE["panel"])
            row.pack(fill="x", padx=8, pady=2)
            tk.Label(
                row, text=label, width=11, anchor="w", font=("Consolas", 9),
                fg=_PALETTE["ink_dim"], bg=_PALETTE["panel"],
            ).pack(side="left")
            bar = tk.Canvas(row, width=90, height=8, bg=_PALETTE["accent_soft"], highlightthickness=0)
            bar.pack(side="left", padx=6)
            value_label = tk.Label(
                row, text="0.00", width=5, anchor="e", font=("Consolas", 9),
                fg=_PALETTE["ink"], bg=_PALETTE["panel"],
            )
            value_label.pack(side="left")
            self._drive_bars[key] = (bar, value_label)

        tk.Label(
            middle, text="NAMED DRIVE NEURONS", font=("Consolas", 8, "bold"),
            fg=_PALETTE["ink_dim"], bg=_PALETTE["bg"], anchor="w",
        ).pack(fill="x")
        self.neurons_text = tk.Text(
            middle, width=38, height=6, bg=_PALETTE["panel"], fg=_PALETTE["ink"],
            font=("Consolas", 9), relief="flat", wrap="word",
        )
        self.neurons_text.pack(fill="both", expand=True, pady=(2, 0))
        self.neurons_text.configure(state="disabled")

        # right: task economy panel
        right = tk.Frame(body, bg=_PALETTE["bg"], width=320)
        right.pack(side="left", fill="y")

        self.balance_label = tk.Label(
            right, text="Balance: 0.00", font=("Consolas", 10, "bold"),
            fg=_PALETTE["ink"], bg=_PALETTE["bg"], anchor="w",
        )
        self.balance_label.pack(fill="x")

        tk.Label(
            right, text="ASSIGN A NEW TASK", font=("Consolas", 8, "bold"),
            fg=_PALETTE["ink_dim"], bg=_PALETTE["bg"], anchor="w",
        ).pack(fill="x", pady=(8, 0))
        self.new_task_entry = tk.Entry(right, width=40, font=("Consolas", 9))
        self.new_task_entry.pack(fill="x", pady=(2, 2))
        tk.Button(right, text="Assign task", command=self._on_assign_task).pack(anchor="w")

        tk.Label(
            right, text="PENDING / IN PROGRESS", font=("Consolas", 8, "bold"),
            fg=_PALETTE["ink_dim"], bg=_PALETTE["bg"], anchor="w",
        ).pack(fill="x", pady=(10, 0))
        self.pending_list = tk.Listbox(right, height=5, font=("Consolas", 9), bg=_PALETTE["panel"], fg=_PALETTE["ink"])
        self.pending_list.pack(fill="x", pady=(2, 8))

        tk.Label(
            right, text="AWAITING YOUR REVIEW", font=("Consolas", 8, "bold"),
            fg=_PALETTE["ink_dim"], bg=_PALETTE["bg"], anchor="w",
        ).pack(fill="x")
        self.review_list = tk.Listbox(right, height=5, font=("Consolas", 9), bg=_PALETTE["panel"], fg=_PALETTE["ink"])
        self.review_list.pack(fill="x", pady=(2, 4))

        review_row = tk.Frame(right, bg=_PALETTE["bg"])
        review_row.pack(fill="x", pady=(0, 4))
        tk.Label(review_row, text="Score:", fg=_PALETTE["ink_dim"], bg=_PALETTE["bg"]).pack(side="left")
        self.score_var = tk.DoubleVar(value=0.8)
        tk.Scale(
            review_row, from_=0.0, to=1.0, resolution=0.05, orient="horizontal",
            variable=self.score_var, length=160, bg=_PALETTE["bg"], fg=_PALETTE["ink"],
            highlightthickness=0, troughcolor=_PALETTE["panel"],
        ).pack(side="left", padx=(4, 4))
        tk.Button(right, text="Submit review", command=self._on_review_selected).pack(anchor="w")

        tk.Label(
            right, text="RECENT HISTORY", font=("Consolas", 8, "bold"),
            fg=_PALETTE["ink_dim"], bg=_PALETTE["bg"], anchor="w",
        ).pack(fill="x", pady=(10, 0))
        self.history_list = tk.Listbox(right, height=6, font=("Consolas", 9), bg=_PALETTE["panel"], fg=_PALETTE["ink"])
        self.history_list.pack(fill="both", expand=True, pady=(2, 0))

        self.status_label = tk.Label(
            self.root, text="Load a checkpoint to begin.", fg=_PALETTE["ink_dim"],
            bg=_PALETTE["bg"], font=("Consolas", 9), anchor="w",
        )
        self.status_label.pack(fill="x", padx=10, pady=(0, 8))

    # -- checkpoint / runtime setup ----------------------------------------

    def _on_load_checkpoint(self) -> None:
        path = self.filedialog.askopenfilename(
            title="Load a trained checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            mind = SpikingMind.from_checkpoint(path, device="cpu")
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            self.status_label.configure(text=f"Failed to load checkpoint: {exc}")
            return

        environment = VirtualPersonEnv()
        self.agent_runtime = AutonomousSpikingAgent(mind, environment)
        self.checkpoint_label.configure(text=Path(path).name)
        self.start_button.configure(state="normal")
        self.step_button.configure(state="normal")
        self.status_label.configure(text="Checkpoint loaded. Ready to run.")
        self._render_all()

    # -- run controls -------------------------------------------------------

    def _on_start(self) -> None:
        if self.agent_runtime is None or self.running:
            return
        self.running = True
        self.start_button.configure(state="disabled")
        self.pause_button.configure(state="normal")
        self._tick()

    def _on_pause(self) -> None:
        self.running = False
        self.start_button.configure(state="normal")
        self.pause_button.configure(state="disabled")
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _on_step(self) -> None:
        if self.agent_runtime is None:
            return
        self._advance_one_step()
        self._render_all()

    def _on_speed_change(self) -> None:
        self.speed_steps_per_tick = max(1, int(self.speed_var.get()))

    def _tick(self) -> None:
        if not self.running or self.agent_runtime is None:
            return
        for _ in range(self.speed_steps_per_tick):
            terminated = self._advance_one_step()
            if terminated:
                self._on_pause()
                break
        self._render_all()
        if self.running:
            self._after_id = self.root.after(100, self._tick)

    def _advance_one_step(self) -> bool:
        assert self.agent_runtime is not None
        result = self.agent_runtime.step(deterministic=self.deterministic_var.get())
        self.last_decision_message = (
            f"{result.decision.action.kind.name} -> {result.decision.action.target}\n"
            f"{result.message}\n"
            f"confidence={result.decision.confidence:.2f}  "
            f"value={result.decision.value_estimate:.3f}  "
            f"reward={result.reward:.3f}"
        )
        self.last_drive_activity = dict(result.decision.drive_activity)
        return bool(result.terminated or result.truncated)

    # -- task economy -------------------------------------------------------

    @property
    def _economy(self):
        if self.agent_runtime is None:
            return None
        return self.agent_runtime.environment.sim.agent.world.economy

    def _on_assign_task(self) -> None:
        economy = self._economy
        if economy is None:
            self.status_label.configure(text="Load a checkpoint first.")
            return
        description = self.new_task_entry.get().strip()
        if not description:
            return
        sim_time = self.agent_runtime.environment.sim.sim_time
        economy.assign_task(description, sim_time=sim_time)
        self.new_task_entry.delete(0, "end")
        self._refresh_task_lists()

    def _on_review_selected(self) -> None:
        economy = self._economy
        if economy is None:
            return
        selection = self.review_list.curselection()
        if not selection:
            self.status_label.configure(text="Select a submitted task to review first.")
            return
        task_id = self._review_ids[selection[0]]
        score = float(self.score_var.get())
        sim_time = self.agent_runtime.environment.sim.sim_time
        review = economy.review_task(task_id, score=score, sim_time=sim_time)
        self.status_label.configure(
            text=f"Reviewed {task_id}: score={review.score:.2f}, payout={review.payout:.2f}"
        )
        self._refresh_task_lists()
        self._render_all()

    def _refresh_task_lists(self) -> None:
        self.pending_list.delete(0, "end")
        self.review_list.delete(0, "end")
        self.history_list.delete(0, "end")
        self._review_ids: list[str] = []

        economy = self._economy
        if economy is None:
            self.balance_label.configure(text="Balance: 0.00")
            return

        snap = economy.snapshot()
        self.balance_label.configure(text=f"Balance: {snap['balance']:.2f}")

        for task in snap["pending"]:
            self.pending_list.insert("end", f"{task['task_id']}: {task['description'][:40]}")
        for task in snap["awaiting_review"]:
            self.review_list.insert("end", f"{task['task_id']}: {task['description'][:40]}")
            self._review_ids.append(task["task_id"])
        for task in reversed(snap["history"]):
            review = task.get("review") or {}
            self.history_list.insert(
                "end",
                f"{task['task_id']}: score={review.get('score', 0):.2f} "
                f"pay={review.get('payout', 0):.2f} - {task['description'][:28]}",
            )

    # -- rendering ------------------------------------------------------------

    def _render_all(self) -> None:
        if self.agent_runtime is None:
            return
        sim = self.agent_runtime.environment.sim
        draw_floor_plan(self.canvas, sim, cell_pixels=56.0)
        self._render_decision()
        self._render_drives(sim)
        self._render_neurons()
        self._refresh_task_lists()

    def _render_decision(self) -> None:
        self.decision_text.configure(state="normal")
        self.decision_text.delete("1.0", "end")
        self.decision_text.insert("end", self.last_decision_message or "(no decision yet)")
        self.decision_text.configure(state="disabled")

    def _render_drives(self, sim) -> None:
        body = sim.agent.body.snapshot()
        for key, _, inverted in _DRIVE_ROWS:
            value = float(body[key])
            pressure = (1.0 - value) if inverted else value
            bar, value_label = self._drive_bars[key]
            bar.delete("all")
            width = max(0.0, min(1.0, pressure)) * 90
            color = _PALETTE[_severity(pressure)]
            bar.create_rectangle(0, 0, width, 8, fill=color, outline="")
            value_label.configure(text=f"{value:.2f}")

    def _render_neurons(self) -> None:
        self.neurons_text.configure(state="normal")
        self.neurons_text.delete("1.0", "end")
        if self.last_drive_activity:
            for name, rate in self.last_drive_activity.items():
                self.neurons_text.insert("end", f"{name}: {rate:.2f}\n")
        else:
            self.neurons_text.insert("end", "(no active drive neurons yet)")
        self.neurons_text.configure(state="disabled")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="virtual-person-runtime")
    parser.add_argument("--checkpoint", help="Optionally preload this checkpoint on launch.")
    args = parser.parse_args(argv)

    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise SystemExit(
            "Could not open the runtime window. Run this on a desktop session with "
            f"Tk support. Details: {exc}"
        )
    app = RuntimeApp(root)
    if args.checkpoint:
        try:
            mind = SpikingMind.from_checkpoint(args.checkpoint, device="cpu")
            app.agent_runtime = AutonomousSpikingAgent(mind, VirtualPersonEnv())
            app.checkpoint_label.configure(text=Path(args.checkpoint).name)
            app.start_button.configure(state="normal")
            app.step_button.configure(state="normal")
            app._render_all()
        except Exception as exc:  # noqa: BLE001
            app.status_label.configure(text=f"Failed to preload checkpoint: {exc}")

    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
