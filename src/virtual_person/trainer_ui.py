from __future__ import annotations

import argparse
import json
import queue
import threading
import traceback
from pathlib import Path
from typing import Any, Sequence

import torch

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
except ImportError as exc:  # pragma: no cover - depends on Python distribution
    raise RuntimeError(
        "Tkinter is required. On Windows, reinstall Python with Tcl/Tk enabled."
    ) from exc

from .bootstrap_data import generate_bootstrap_corpus
from .spike_tokenizer import ByteTokenizer
from .spike_training import (
    TrainConfig,
    build_dictionary_corpus,
    load_checkpoint,
    read_training_examples,
    train_model,
)
from .spiking import NodeLinkSpikeModel, SpikingModelConfig
from .trainer_support import (
    CURRICULUM_CATEGORIES,
    MODEL_PROFILES,
    SOURCE_CATEGORIES,
    CorpusSource,
    TrainerProject,
    create_starter_pack,
    create_workspace,
    detect_hardware,
    estimate_model,
    filter_sources_for_stage,
    format_bytes,
    recommended_profile,
    scan_sources,
)


GUIDES: dict[str, str] = {
    "Workspace": """
WHAT TO DO

1. Choose a workspace folder.
2. Click Create / Load Workspace.
3. Check the detected hardware.
4. Add the starter pack only to test the pipeline.

The workspace keeps raw data, generated corpora, checkpoints, logs, and exports
separate. Keep it on a drive with plenty of free space.

Do not begin a large run yet. First prove that a smoke-test model can read your
data, train, save, reload, and generate output.
""",
    "Data": """
WHAT TO FEED IT

English:
Natural conversations, stories, descriptions, explanations, corrections, and
longer passages. A dictionary alone will not teach fluent language.

Dictionary:
Definitions converted through the Dictionary Builder. Keep this a minority of
the corpus so the model does not speak only in definition-like sentences.

Procedures:
Cooking, cleaning, personal routines, computer use, scheduling, troubleshooting,
and ordinary young-adult action knowledge.

Safety/Judgment:
When to stop, inspect, ask, wait, request permission, or reject a dangerous plan.

Behavior:
Structured state, candidate actions, correct action index, drive features, and
expected value. This is what teaches hunger, thirst, boredom, and goals to affect
behavior.

Validate after every data change. Fix malformed JSONL before training.
""",
    "Model": """
HOW TO SIZE THE MODEL

Start with Architecture smoke test. It should finish quickly and only verifies
the system.

Next use CPU prototype or Small GPU experiment. Do not jump to a large model
until:
- the loss decreases,
- checkpoints reload,
- dedicated hunger/thirst neurons fire correctly,
- action examples are being learned,
- evaluation is better than random.

The current Python/PyTorch implementation unrolls sequence × ticks × layers.
Large settings become expensive very quickly. The estimate is approximate and
does not include every PyTorch allocation.
""",
    "Train": """
TRAIN IN THREE STAGES

Stage 1 — English and vocabulary:
Train on English and Dictionary sources. The model learns language structure and
word meanings.

Stage 2 — Practical knowledge and judgment:
Resume Stage 1's checkpoint. Add Procedures and Safety/Judgment.

Stage 3 — Autonomous behavior and drives:
Resume Stage 2's checkpoint. Add Behavior data so named hunger, thirst, fatigue,
bladder, and boredom neurons become connected to sensible actions.

Keep a separate checkpoint for each stage. Do not overwrite the only working
checkpoint. Cancel saves the current checkpoint at the next batch boundary.
""",
    "Evaluate": """
HOW TO TEST IT

1. Load a checkpoint.
2. Enter a prompt similar to its training data.
3. Set hunger, thirst, fatigue, and boredom.
4. Generate text and inspect the named drive neurons.

Early models may emit broken text. Compare checkpoints using the same prompts and
drive values. A useful improvement is consistent and repeatable, not one lucky
sample.

Also test action selection in the simulator before connecting physical hardware.
""",
}


class TrainerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Node-Link-Spike Training Studio")
        self.root.geometry("1280x820")
        self.root.minsize(1050, 700)

        self.hardware = detect_hardware()
        self.sources: list[CorpusSource] = []
        self.source_stats: dict[str, Any] = {}
        self.project: TrainerProject | None = None
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.cancel_event = threading.Event()
        self.training_thread: threading.Thread | None = None
        self.eval_model: NodeLinkSpikeModel | None = None
        self.eval_checkpoint_loaded: str | None = None

        self._create_variables()
        self._configure_style()
        self._build_ui()
        self._apply_profile(recommended_profile(self.hardware))
        self._update_hardware_text()
        self._update_guide()
        self._update_estimate()
        self.root.after(100, self._poll_events)

    def _create_variables(self) -> None:
        default_workspace = str((Path.cwd() / "training_workspace").resolve())
        self.workspace_var = tk.StringVar(value=default_workspace)
        self.category_var = tk.StringVar(value="English")
        self.bootstrap_episodes_var = tk.IntVar(value=50)

        self.profile_var = tk.StringVar(value="CPU prototype")
        self.hidden_var = tk.IntVar(value=96)
        self.layers_var = tk.IntVar(value=2)
        self.ticks_var = tk.IntVar(value=2)
        self.sequence_var = tk.IntVar(value=160)
        self.batch_var = tk.IntVar(value=2)
        self.epochs_var = tk.IntVar(value=3)
        self.learning_rate_var = tk.DoubleVar(value=3e-4)
        self.device_var = tk.StringVar(value=self.hardware.suggested_device)
        self.seed_var = tk.IntVar(value=0)

        self.stage_var = tk.StringVar(value="Stage 1 — English and vocabulary")
        self.output_checkpoint_var = tk.StringVar(value="")
        self.resume_enabled_var = tk.BooleanVar(value=False)
        self.resume_checkpoint_var = tk.StringVar(value="")

        self.progress_var = tk.DoubleVar(value=0.0)
        self.metric_loss_var = tk.StringVar(value="Loss: —")
        self.metric_language_var = tk.StringVar(value="Language: —")
        self.metric_action_var = tk.StringVar(value="Action: —")
        self.metric_value_var = tk.StringVar(value="Value: —")
        self.metric_spike_var = tk.StringVar(value="Spike rate: —")

        self.eval_checkpoint_var = tk.StringVar(value="")
        self.eval_hunger_var = tk.DoubleVar(value=0.20)
        self.eval_thirst_var = tk.DoubleVar(value=0.20)
        self.eval_fatigue_var = tk.DoubleVar(value=0.20)
        self.eval_boredom_var = tk.DoubleVar(value=0.20)
        self.eval_task_var = tk.BooleanVar(value=False)
        self.eval_tokens_var = tk.IntVar(value=100)

    def _configure_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Subheader.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("GuideTitle.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("StatusGood.TLabel", foreground="#187a32")
        style.configure("StatusWarn.TLabel", foreground="#9a6500")
        style.configure("StatusBad.TLabel", foreground="#a11d1d")

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, padding=(16, 12))
        header.pack(fill="x")
        ttk.Label(
            header,
            text="Node-Link-Spike Training Studio",
            style="Header.TLabel",
        ).pack(side="left")
        ttk.Label(
            header,
            text="Guided corpus → curriculum → checkpoint workflow",
        ).pack(side="left", padx=(18, 0), pady=(7, 0))

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        left = ttk.Frame(body)
        right = ttk.Frame(body, padding=(12, 8))
        body.add(left, weight=4)
        body.add(right, weight=1)

        self.notebook = ttk.Notebook(left)
        self.notebook.pack(fill="both", expand=True)
        self.tabs: dict[str, ttk.Frame] = {}
        for name in ("Workspace", "Data", "Model", "Train", "Evaluate"):
            frame = ttk.Frame(self.notebook, padding=14)
            self.notebook.add(frame, text=name)
            self.tabs[name] = frame
        self.notebook.bind("<<NotebookTabChanged>>", lambda _event: self._update_guide())

        self._build_workspace_tab()
        self._build_data_tab()
        self._build_model_tab()
        self._build_train_tab()
        self._build_evaluate_tab()

        ttk.Label(right, text="Guidance", style="GuideTitle.TLabel").pack(anchor="w")
        self.guide_text = scrolledtext.ScrolledText(
            right,
            wrap="word",
            height=25,
            font=("Segoe UI", 10),
            state="disabled",
        )
        self.guide_text.pack(fill="both", expand=True, pady=(8, 8))
        ttk.Label(right, text="Current readiness", style="Subheader.TLabel").pack(anchor="w")
        self.readiness_label = ttk.Label(
            right,
            text="Create or load a workspace.",
            wraplength=260,
            style="StatusWarn.TLabel",
        )
        self.readiness_label.pack(fill="x", pady=(5, 0))

        self.status_label = ttk.Label(
            self.root,
            text="Ready.",
            relief="sunken",
            anchor="w",
            padding=(8, 4),
        )
        self.status_label.pack(fill="x", side="bottom")

    def _build_workspace_tab(self) -> None:
        tab = self.tabs["Workspace"]
        ttk.Label(tab, text="1. Workspace and hardware", style="Subheader.TLabel").pack(anchor="w")

        row = ttk.Frame(tab)
        row.pack(fill="x", pady=(12, 6))
        ttk.Entry(row, textvariable=self.workspace_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse…", command=self._browse_workspace).pack(side="left", padx=6)
        ttk.Button(
            row,
            text="Create / Load Workspace",
            command=self._activate_workspace,
        ).pack(side="left")

        buttons = ttk.Frame(tab)
        buttons.pack(fill="x", pady=(6, 12))
        ttk.Button(
            buttons,
            text="Create Starter Pack",
            command=self._create_starter_pack,
        ).pack(side="left")
        ttk.Button(
            buttons,
            text="Open Training Guide",
            command=self._open_training_guide,
        ).pack(side="left", padx=6)

        ttk.Label(tab, text="Detected hardware", style="Subheader.TLabel").pack(anchor="w")
        self.hardware_text = scrolledtext.ScrolledText(
            tab,
            height=13,
            wrap="word",
            font=("Consolas", 10),
            state="disabled",
        )
        self.hardware_text.pack(fill="x", pady=(6, 12))

        ttk.Label(tab, text="First run checklist", style="Subheader.TLabel").pack(anchor="w")
        checklist = (
            "□ Create the workspace\n"
            "□ Add the starter pack\n"
            "□ Validate the data\n"
            "□ Use Architecture smoke test\n"
            "□ Train one epoch\n"
            "□ Load the checkpoint in Evaluate\n"
            "□ Only then add a real corpus"
        )
        ttk.Label(tab, text=checklist, justify="left").pack(anchor="w", pady=(6, 0))

    def _build_data_tab(self) -> None:
        tab = self.tabs["Data"]
        ttk.Label(tab, text="2. Build and validate the education corpus", style="Subheader.TLabel").pack(anchor="w")

        controls = ttk.Frame(tab)
        controls.pack(fill="x", pady=(10, 8))
        ttk.Label(controls, text="Category:").pack(side="left")
        ttk.Combobox(
            controls,
            textvariable=self.category_var,
            values=SOURCE_CATEGORIES,
            state="readonly",
            width=18,
        ).pack(side="left", padx=(5, 10))
        ttk.Button(controls, text="Add Files…", command=self._add_files).pack(side="left")
        ttk.Button(controls, text="Add Folder…", command=self._add_folder).pack(side="left", padx=5)
        ttk.Button(controls, text="Remove", command=self._remove_sources).pack(side="left")

        self.source_tree = ttk.Treeview(
            tab,
            columns=("category", "records", "characters", "status", "path"),
            show="headings",
            height=13,
        )
        self.source_tree.heading("category", text="Category")
        self.source_tree.heading("records", text="Records")
        self.source_tree.heading("characters", text="Characters")
        self.source_tree.heading("status", text="Status")
        self.source_tree.heading("path", text="Path")
        self.source_tree.column("category", width=130, anchor="w")
        self.source_tree.column("records", width=75, anchor="e")
        self.source_tree.column("characters", width=95, anchor="e")
        self.source_tree.column("status", width=90, anchor="center")
        self.source_tree.column("path", width=520, anchor="w")
        self.source_tree.pack(fill="both", expand=True)

        build = ttk.LabelFrame(tab, text="Corpus builders", padding=8)
        build.pack(fill="x", pady=10)
        ttk.Label(build, text="Bootstrap episodes:").pack(side="left")
        ttk.Spinbox(
            build,
            from_=1,
            to=100000,
            textvariable=self.bootstrap_episodes_var,
            width=8,
        ).pack(side="left", padx=5)
        ttk.Button(
            build,
            text="Generate Bootstrap Behavior",
            command=self._generate_bootstrap,
        ).pack(side="left")
        ttk.Button(
            build,
            text="Convert Dictionary…",
            command=self._convert_dictionary,
        ).pack(side="left", padx=6)
        ttk.Button(
            build,
            text="Validate All",
            command=self._validate_sources,
        ).pack(side="right")

        self.data_summary = ttk.Label(
            tab,
            text="No corpus has been validated.",
            justify="left",
            wraplength=900,
        )
        self.data_summary.pack(fill="x", pady=(0, 4))

    def _build_model_tab(self) -> None:
        tab = self.tabs["Model"]
        ttk.Label(tab, text="3. Select an architecture", style="Subheader.TLabel").pack(anchor="w")

        profile_row = ttk.Frame(tab)
        profile_row.pack(fill="x", pady=(10, 12))
        ttk.Label(profile_row, text="Profile:").pack(side="left")
        profile_box = ttk.Combobox(
            profile_row,
            textvariable=self.profile_var,
            values=tuple(MODEL_PROFILES),
            state="readonly",
            width=27,
        )
        profile_box.pack(side="left", padx=6)
        ttk.Button(
            profile_row,
            text="Apply Profile",
            command=lambda: self._apply_profile(self.profile_var.get()),
        ).pack(side="left")
        self.profile_description = ttk.Label(
            profile_row,
            text="",
            wraplength=520,
            justify="left",
        )
        self.profile_description.pack(side="left", padx=14)

        settings = ttk.LabelFrame(tab, text="Model and training dimensions", padding=10)
        settings.pack(fill="x")

        fields = [
            ("Hidden nodes", self.hidden_var),
            ("Spiking layers", self.layers_var),
            ("Ticks per token", self.ticks_var),
            ("Sequence bytes", self.sequence_var),
            ("Batch size", self.batch_var),
            ("Epochs", self.epochs_var),
            ("Learning rate", self.learning_rate_var),
            ("Seed", self.seed_var),
        ]
        for index, (label, variable) in enumerate(fields):
            row = index // 2
            column = (index % 2) * 2
            ttk.Label(settings, text=label + ":").grid(
                row=row,
                column=column,
                sticky="e",
                padx=(4, 6),
                pady=5,
            )
            ttk.Entry(settings, textvariable=variable, width=15).grid(
                row=row,
                column=column + 1,
                sticky="w",
                padx=(0, 24),
                pady=5,
            )

        ttk.Label(settings, text="Device:").grid(row=4, column=0, sticky="e", padx=(4, 6), pady=5)
        ttk.Combobox(
            settings,
            textvariable=self.device_var,
            values=("cpu", "cuda"),
            state="readonly",
            width=12,
        ).grid(row=4, column=1, sticky="w", pady=5)

        self.estimate_text = scrolledtext.ScrolledText(
            tab,
            height=13,
            wrap="word",
            font=("Consolas", 10),
            state="disabled",
        )
        self.estimate_text.pack(fill="x", pady=12)

        for variable in (
            self.hidden_var,
            self.layers_var,
            self.ticks_var,
            self.sequence_var,
            self.batch_var,
        ):
            variable.trace_add("write", lambda *_args: self._update_estimate())

    def _build_train_tab(self) -> None:
        tab = self.tabs["Train"]
        ttk.Label(tab, text="4. Curriculum training", style="Subheader.TLabel").pack(anchor="w")

        setup = ttk.LabelFrame(tab, text="Run configuration", padding=8)
        setup.pack(fill="x", pady=(10, 8))

        ttk.Label(setup, text="Curriculum stage:").grid(row=0, column=0, sticky="e", pady=4)
        ttk.Combobox(
            setup,
            textvariable=self.stage_var,
            values=tuple(CURRICULUM_CATEGORIES),
            state="readonly",
            width=39,
        ).grid(row=0, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(setup, text="Output checkpoint:").grid(row=1, column=0, sticky="e", pady=4)
        ttk.Entry(setup, textvariable=self.output_checkpoint_var, width=70).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=6,
            pady=4,
        )
        ttk.Button(
            setup,
            text="Browse…",
            command=self._browse_output_checkpoint,
        ).grid(row=1, column=2, pady=4)

        ttk.Checkbutton(
            setup,
            text="Continue from checkpoint",
            variable=self.resume_enabled_var,
        ).grid(row=2, column=0, sticky="e", pady=4)
        ttk.Entry(setup, textvariable=self.resume_checkpoint_var, width=70).grid(
            row=2,
            column=1,
            sticky="ew",
            padx=6,
            pady=4,
        )
        ttk.Button(
            setup,
            text="Browse…",
            command=self._browse_resume_checkpoint,
        ).grid(row=2, column=2, pady=4)
        setup.columnconfigure(1, weight=1)

        self.preflight_text = scrolledtext.ScrolledText(
            tab,
            height=8,
            wrap="word",
            font=("Consolas", 10),
            state="disabled",
        )
        self.preflight_text.pack(fill="x")
        ttk.Button(
            tab,
            text="Refresh Preflight",
            command=self._refresh_preflight,
        ).pack(anchor="e", pady=(4, 8))

        progress_row = ttk.Frame(tab)
        progress_row.pack(fill="x")
        self.progress = ttk.Progressbar(
            progress_row,
            variable=self.progress_var,
            maximum=100.0,
        )
        self.progress.pack(side="left", fill="x", expand=True)
        self.start_button = ttk.Button(
            progress_row,
            text="Start Training",
            command=self._start_training,
        )
        self.start_button.pack(side="left", padx=6)
        self.cancel_button = ttk.Button(
            progress_row,
            text="Cancel",
            command=self._cancel_training,
            state="disabled",
        )
        self.cancel_button.pack(side="left")

        metrics = ttk.Frame(tab)
        metrics.pack(fill="x", pady=6)
        for variable in (
            self.metric_loss_var,
            self.metric_language_var,
            self.metric_action_var,
            self.metric_value_var,
            self.metric_spike_var,
        ):
            ttk.Label(metrics, textvariable=variable).pack(side="left", padx=(0, 18))

        self.train_log = scrolledtext.ScrolledText(
            tab,
            height=17,
            wrap="word",
            font=("Consolas", 9),
            state="disabled",
        )
        self.train_log.pack(fill="both", expand=True)

    def _build_evaluate_tab(self) -> None:
        tab = self.tabs["Evaluate"]
        ttk.Label(tab, text="5. Inspect a checkpoint", style="Subheader.TLabel").pack(anchor="w")

        checkpoint = ttk.Frame(tab)
        checkpoint.pack(fill="x", pady=(10, 6))
        ttk.Entry(checkpoint, textvariable=self.eval_checkpoint_var).pack(
            side="left",
            fill="x",
            expand=True,
        )
        ttk.Button(checkpoint, text="Browse…", command=self._browse_eval_checkpoint).pack(
            side="left",
            padx=6,
        )
        ttk.Button(checkpoint, text="Load", command=self._load_eval_checkpoint).pack(side="left")

        prompt_frame = ttk.LabelFrame(tab, text="Prompt", padding=8)
        prompt_frame.pack(fill="x")
        self.eval_prompt = scrolledtext.ScrolledText(
            prompt_frame,
            height=7,
            wrap="word",
            font=("Segoe UI", 10),
        )
        self.eval_prompt.pack(fill="x")
        self.eval_prompt.insert(
            "1.0",
            "Mira is hungry and standing in a kitchen. She can see eggs, a pan, "
            "a stove, and a sink. She decides to",
        )

        drives = ttk.LabelFrame(tab, text="Interoceptive state", padding=8)
        drives.pack(fill="x", pady=8)
        sliders = [
            ("Hunger", self.eval_hunger_var),
            ("Thirst", self.eval_thirst_var),
            ("Fatigue", self.eval_fatigue_var),
            ("Boredom", self.eval_boredom_var),
        ]
        for row, (label, variable) in enumerate(sliders):
            ttk.Label(drives, text=label + ":").grid(row=row, column=0, sticky="e")
            ttk.Scale(
                drives,
                from_=0.0,
                to=1.0,
                variable=variable,
                orient="horizontal",
                length=350,
            ).grid(row=row, column=1, sticky="ew", padx=8)
            ttk.Label(drives, textvariable=variable, width=7).grid(row=row, column=2)
        ttk.Checkbutton(
            drives,
            text="Task pending",
            variable=self.eval_task_var,
        ).grid(row=0, column=3, padx=15)
        ttk.Label(drives, text="New tokens:").grid(row=1, column=3)
        ttk.Spinbox(
            drives,
            from_=1,
            to=1000,
            textvariable=self.eval_tokens_var,
            width=8,
        ).grid(row=1, column=4)
        drives.columnconfigure(1, weight=1)

        ttk.Button(
            tab,
            text="Generate and Inspect Neurons",
            command=self._evaluate,
        ).pack(anchor="e")

        self.eval_output = scrolledtext.ScrolledText(
            tab,
            height=18,
            wrap="word",
            font=("Consolas", 9),
            state="disabled",
        )
        self.eval_output.pack(fill="both", expand=True, pady=(6, 0))

    # ------------------------------------------------------------------
    # Workspace and persistence
    # ------------------------------------------------------------------
    def _browse_workspace(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.workspace_var.get() or str(Path.cwd()))
        if selected:
            self.workspace_var.set(selected)

    def _activate_workspace(self) -> None:
        try:
            layout = create_workspace(self.workspace_var.get())
            self.workspace_var.set(str(layout["root"]))
            project_path = layout["root"] / "trainer_project.json"
            if project_path.exists():
                self.project = TrainerProject.load(project_path)
                self.sources = list(self.project.sources)
                self._restore_project_settings()
            else:
                self.project = TrainerProject(workspace=str(layout["root"]))
                self.project.save()
            self.output_checkpoint_var.set(
                self.output_checkpoint_var.get()
                or str(layout["checkpoints"] / "stage1_spiking_mind.pt")
            )
            self._refresh_source_tree()
            self._set_status(f"Workspace ready: {layout['root']}")
            self._update_readiness()
        except Exception as exc:
            messagebox.showerror("Workspace error", str(exc))

    def _open_training_guide(self) -> None:
        guide = Path(__file__).resolve().parents[2] / "TRAINING_GUIDE.md"
        if not guide.exists():
            guide = Path.cwd() / "TRAINING_GUIDE.md"
        messagebox.showinfo(
            "Training guide",
            f"The full guide is included at:\n{guide}\n\n"
            "The Guidance panel also explains the active step.",
        )

    def _save_project(self) -> None:
        if self.project is None:
            return
        self.project.workspace = self.workspace_var.get()
        self.project.sources = list(self.sources)
        self.project.model = self._model_settings()
        self.project.training = {
            "stage": self.stage_var.get(),
            "output_checkpoint": self.output_checkpoint_var.get(),
            "resume_enabled": bool(self.resume_enabled_var.get()),
            "resume_checkpoint": self.resume_checkpoint_var.get(),
            "device": self.device_var.get(),
            "seed": int(self.seed_var.get()),
        }
        self.project.save()

    def _restore_project_settings(self) -> None:
        if self.project is None:
            return
        model = self.project.model
        mapping = {
            "hidden_size": self.hidden_var,
            "layers": self.layers_var,
            "ticks": self.ticks_var,
            "sequence_length": self.sequence_var,
            "batch_size": self.batch_var,
            "epochs": self.epochs_var,
            "learning_rate": self.learning_rate_var,
        }
        for key, variable in mapping.items():
            if key in model:
                variable.set(model[key])
        training = self.project.training
        if training.get("stage") in CURRICULUM_CATEGORIES:
            self.stage_var.set(training["stage"])
        if training.get("output_checkpoint"):
            self.output_checkpoint_var.set(training["output_checkpoint"])
        self.resume_enabled_var.set(bool(training.get("resume_enabled", False)))
        self.resume_checkpoint_var.set(str(training.get("resume_checkpoint", "")))
        if training.get("device") in {"cpu", "cuda"}:
            self.device_var.set(training["device"])
        if "seed" in training:
            self.seed_var.set(int(training["seed"]))

    def _create_starter_pack(self) -> None:
        if self.project is None:
            self._activate_workspace()
        try:
            additions = create_starter_pack(self.workspace_var.get())
            existing = {str(Path(source.path).resolve()) for source in self.sources}
            for source in additions:
                if str(Path(source.path).resolve()) not in existing:
                    self.sources.append(source)
            self._refresh_source_tree()
            self._save_project()
            self._set_status("Starter pack created. Validate it in the Data tab.")
            self.notebook.select(self.tabs["Data"])
        except Exception as exc:
            messagebox.showerror("Starter pack error", str(exc))

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Add corpus files",
            filetypes=[
                ("Training data", "*.txt *.jsonl"),
                ("Text", "*.txt"),
                ("JSON Lines", "*.jsonl"),
                ("All files", "*.*"),
            ],
        )
        self._append_sources(paths, self.category_var.get())

    def _add_folder(self) -> None:
        path = filedialog.askdirectory(title="Add corpus folder")
        if path:
            self._append_sources([path], self.category_var.get())

    def _append_sources(self, paths: Sequence[str], category: str) -> None:
        existing = {str(Path(source.path).expanduser().resolve()) for source in self.sources}
        for raw in paths:
            normalized = str(Path(raw).expanduser().resolve())
            if normalized not in existing:
                self.sources.append(CorpusSource(normalized, category))
                existing.add(normalized)
        self._refresh_source_tree()
        self._save_project()
        self._update_readiness()

    def _remove_sources(self) -> None:
        selected = set(self.source_tree.selection())
        if not selected:
            return
        self.sources = [
            source
            for index, source in enumerate(self.sources)
            if f"source-{index}" not in selected
        ]
        self.source_stats.clear()
        self._refresh_source_tree()
        self._save_project()
        self._update_readiness()

    def _refresh_source_tree(self) -> None:
        for item in self.source_tree.get_children():
            self.source_tree.delete(item)
        for index, source in enumerate(self.sources):
            stats = self.source_stats.get(source.path)
            if stats is None:
                records = "—"
                characters = "—"
                status = "Not scanned"
            else:
                records = f"{stats.records:,}"
                characters = f"{stats.characters:,}"
                if stats.missing_paths:
                    status = "Missing"
                elif stats.malformed_records:
                    status = f"{stats.malformed_records} bad"
                else:
                    status = "Valid"
            self.source_tree.insert(
                "",
                "end",
                iid=f"source-{index}",
                values=(source.category, records, characters, status, source.path),
            )

    def _validate_sources(self) -> None:
        if not self.sources:
            messagebox.showwarning("No data", "Add at least one source first.")
            return
        self._set_status("Validating corpus…")
        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _scan_worker(self) -> None:
        try:
            total, by_path, by_category = scan_sources(self.sources)
            self.events.put(("scan_done", (total, by_path, by_category)))
        except Exception:
            self.events.put(("error", traceback.format_exc()))

    def _generate_bootstrap(self) -> None:
        if self.project is None:
            self._activate_workspace()
        output = Path(self.workspace_var.get()) / "corpora" / "bootstrap_behavior.jsonl"
        episodes = max(1, int(self.bootstrap_episodes_var.get()))
        self._set_status(f"Generating {episodes} bootstrap episodes…")

        def worker() -> None:
            try:
                count = generate_bootstrap_corpus(output, episodes=episodes, seed=int(self.seed_var.get()))
                self.events.put(("builder_done", (str(output), "Behavior", count)))
            except Exception:
                self.events.put(("error", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    def _convert_dictionary(self) -> None:
        if self.project is None:
            self._activate_workspace()
        source = filedialog.askopenfilename(
            title="Select dictionary CSV or JSONL",
            filetypes=[("Dictionary", "*.csv *.jsonl"), ("All files", "*.*")],
        )
        if not source:
            return
        output = Path(self.workspace_var.get()) / "corpora" / f"{Path(source).stem}_dictionary.jsonl"
        self._set_status("Converting dictionary into contextual examples…")

        def worker() -> None:
            try:
                count = build_dictionary_corpus(source, output)
                self.events.put(("builder_done", (str(output), "Dictionary", count)))
            except Exception:
                self.events.put(("error", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def _apply_profile(self, name: str) -> None:
        profile = MODEL_PROFILES[name]
        self.profile_var.set(name)
        self.hidden_var.set(profile.hidden_size)
        self.layers_var.set(profile.layers)
        self.ticks_var.set(profile.ticks)
        self.sequence_var.set(profile.sequence_length)
        self.batch_var.set(profile.batch_size)
        self.epochs_var.set(profile.epochs)
        self.learning_rate_var.set(profile.learning_rate)
        if hasattr(self, "profile_description"):
            self.profile_description.configure(text=profile.description)
        self._update_estimate()

    def _model_settings(self) -> dict[str, Any]:
        return {
            "hidden_size": max(4, int(self.hidden_var.get())),
            "layers": max(1, int(self.layers_var.get())),
            "ticks": max(1, int(self.ticks_var.get())),
            "sequence_length": max(8, int(self.sequence_var.get())),
            "batch_size": max(1, int(self.batch_var.get())),
            "epochs": max(1, int(self.epochs_var.get())),
            "learning_rate": float(self.learning_rate_var.get()),
        }

    def _spiking_config(self) -> SpikingModelConfig:
        settings = self._model_settings()
        return SpikingModelConfig(
            hidden_size=settings["hidden_size"],
            layer_count=settings["layers"],
            ticks_per_token=settings["ticks"],
        )

    def _update_estimate(self) -> None:
        if not hasattr(self, "estimate_text"):
            return
        try:
            config = self._spiking_config()
            settings = self._model_settings()
            estimate = estimate_model(
                config,
                batch_size=settings["batch_size"],
                sequence_length=settings["sequence_length"],
            )
            available = (
                self.hardware.gpu_memory_bytes
                if self.device_var.get() == "cuda"
                else self.hardware.ram_bytes
            )
            ratio = (
                estimate.total_training_bytes_estimate / available
                if available
                else None
            )
            warning = ""
            if ratio is not None and ratio > 0.75:
                warning = (
                    "\nWARNING: The rough estimate exceeds 75% of detected memory. "
                    "Reduce batch, sequence length, layers, or hidden nodes."
                )
            text = (
                f"Parameters:             {estimate.parameters:,} "
                f"({estimate.parameter_millions:.3f} million)\n"
                f"FP32 parameters:        {format_bytes(estimate.parameter_bytes_fp32)}\n"
                f"FP32 weights+Adam+grad: {format_bytes(estimate.optimizer_bytes_fp32)}\n"
                f"Saved activations est.: {format_bytes(estimate.activation_bytes_estimate)}\n"
                f"Rough training total:   {format_bytes(estimate.total_training_bytes_estimate)}\n"
                f"Selected device:        {self.device_var.get()}\n"
                f"Detected memory:        {format_bytes(available)}\n"
                f"\nCost grows roughly with batch × sequence × ticks × layers."
                f"{warning}"
            )
        except Exception as exc:
            text = f"Enter valid numeric settings.\n\n{exc}"
        self._replace_text(self.estimate_text, text)

    def _update_hardware_text(self) -> None:
        hardware = self.hardware
        text = (
            f"Python:          {hardware.python}\n"
            f"Platform:        {hardware.platform}\n"
            f"PyTorch:         {hardware.torch_version}\n"
            f"CPU threads:     {hardware.cpu_count}\n"
            f"System RAM:      {format_bytes(hardware.ram_bytes)}\n"
            f"CUDA available:  {hardware.cuda_available}\n"
            f"CUDA devices:    {hardware.cuda_device_count}\n"
            f"GPU:             {hardware.gpu_name or 'none detected'}\n"
            f"GPU memory:      {format_bytes(hardware.gpu_memory_bytes)}\n"
            f"Suggested device:{hardware.suggested_device}\n"
            f"Suggested profile: {recommended_profile(hardware)}"
        )
        self._replace_text(self.hardware_text, text)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _browse_output_checkpoint(self) -> None:
        initial = self.output_checkpoint_var.get() or str(Path(self.workspace_var.get()) / "checkpoints")
        selected = filedialog.asksaveasfilename(
            title="Output checkpoint",
            initialdir=str(Path(initial).parent),
            initialfile=Path(initial).name,
            defaultextension=".pt",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
        )
        if selected:
            self.output_checkpoint_var.set(selected)

    def _browse_resume_checkpoint(self) -> None:
        selected = filedialog.askopenfilename(
            title="Checkpoint to continue",
            initialdir=str(Path(self.workspace_var.get()) / "checkpoints"),
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
        )
        if selected:
            self.resume_checkpoint_var.set(selected)
            self.resume_enabled_var.set(True)

    def _preflight(self) -> tuple[list[str], bool]:
        lines: list[str] = []
        okay = True
        workspace = Path(self.workspace_var.get())
        if workspace.exists():
            lines.append("[OK] Workspace exists.")
        else:
            lines.append("[FAIL] Create the workspace first.")
            okay = False

        stage_sources = filter_sources_for_stage(self.sources, self.stage_var.get())
        if stage_sources:
            lines.append(
                f"[OK] {len(stage_sources)} enabled source(s) are included in this stage."
            )
        else:
            lines.append("[FAIL] This curriculum stage has no matching data sources.")
            okay = False

        total, _by_path, _by_category = scan_sources(stage_sources)
        if total.records > 0:
            lines.append(
                f"[OK] {total.records:,} records and {total.characters:,} characters scanned."
            )
        else:
            lines.append("[FAIL] No readable training records were found.")
            okay = False
        if total.malformed_records:
            lines.append(
                f"[FAIL] {total.malformed_records} malformed JSONL/text record(s)."
            )
            okay = False
        if total.missing_paths:
            lines.append(f"[FAIL] {total.missing_paths} source path(s) are missing.")
            okay = False
        if total.records < 100:
            lines.append(
                "[WARN] Fewer than 100 records. This is suitable only for a pipeline test."
            )
        elif total.records < 10_000:
            lines.append(
                "[WARN] This is still a small research corpus, not adult-level language data."
            )

        if self.device_var.get() == "cuda" and not self.hardware.cuda_available:
            lines.append("[FAIL] CUDA was selected but PyTorch cannot access a CUDA device.")
            okay = False
        else:
            lines.append(f"[OK] Device selection: {self.device_var.get()}.")

        output = Path(self.output_checkpoint_var.get())
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            lines.append(f"[OK] Checkpoint can be written to {output}.")
        except Exception as exc:
            lines.append(f"[FAIL] Checkpoint path is not writable: {exc}")
            okay = False

        if self.resume_enabled_var.get():
            resume = Path(self.resume_checkpoint_var.get())
            if resume.is_file():
                lines.append(f"[OK] Starting weights will be loaded from {resume}.")
            else:
                lines.append("[FAIL] Continue-from-checkpoint is enabled, but the file is missing.")
                okay = False
        else:
            lines.append("[INFO] Training will initialize new random weights.")

        return lines, okay

    def _refresh_preflight(self) -> None:
        lines, okay = self._preflight()
        self._replace_text(self.preflight_text, "\n".join(lines))
        self.readiness_label.configure(
            text="Ready to train." if okay else "Resolve the failed preflight items.",
            style="StatusGood.TLabel" if okay else "StatusBad.TLabel",
        )

    def _start_training(self) -> None:
        if self.training_thread and self.training_thread.is_alive():
            return
        lines, okay = self._preflight()
        self._replace_text(self.preflight_text, "\n".join(lines))
        if not okay:
            messagebox.showerror("Preflight failed", "Resolve the failed items before training.")
            return

        self._save_project()
        self.cancel_event.clear()
        self.progress_var.set(0.0)
        self._replace_text(self.train_log, "")
        self._append_log("Preparing training data…")
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")

        settings = self._model_settings()
        stage_sources = filter_sources_for_stage(self.sources, self.stage_var.get())
        paths = [source.path for source in stage_sources]
        output = self.output_checkpoint_var.get()
        resume = self.resume_checkpoint_var.get() if self.resume_enabled_var.get() else None
        device = self.device_var.get()
        seed = int(self.seed_var.get())

        self.training_thread = threading.Thread(
            target=self._training_worker,
            args=(paths, output, resume, device, seed, settings),
            daemon=True,
        )
        self.training_thread.start()

    def _training_worker(
        self,
        paths: Sequence[str],
        output: str,
        resume: str | None,
        device: str,
        seed: int,
        settings: dict[str, Any],
    ) -> None:
        try:
            self.events.put(("log", f"Reading {len(paths)} source(s)…"))
            examples = read_training_examples(paths)
            self.events.put(("log", f"Loaded {len(examples):,} training records."))

            optimizer_state = None
            start_step = 0
            if resume:
                model, payload = load_checkpoint(resume, device=device)
                optimizer_state = payload.get("optimizer")
                start_step = int(payload.get("step", 0))
                self.events.put(
                    (
                        "log",
                        "Loaded checkpoint architecture: "
                        + json.dumps(model.architecture_summary()),
                    )
                )
            else:
                model = NodeLinkSpikeModel(
                    SpikingModelConfig(
                        hidden_size=settings["hidden_size"],
                        layer_count=settings["layers"],
                        ticks_per_token=settings["ticks"],
                    )
                )
                self.events.put(
                    (
                        "log",
                        "Initialized model: " + json.dumps(model.architecture_summary()),
                    )
                )

            config = TrainConfig(
                sequence_length=settings["sequence_length"],
                batch_size=settings["batch_size"],
                epochs=settings["epochs"],
                learning_rate=settings["learning_rate"],
                device=device,
                seed=seed,
                num_workers=0,
            )
            history = train_model(
                model,
                examples,
                config,
                checkpoint_path=output,
                progress_callback=lambda row: self.events.put(("progress", row)),
                stop_requested=self.cancel_event.is_set,
                optimizer_state=optimizer_state,
                start_step=start_step,
            )
            self.events.put(
                (
                    "training_done",
                    {
                        "history": history,
                        "checkpoint": output,
                        "cancelled": self.cancel_event.is_set(),
                    },
                )
            )
        except Exception:
            self.events.put(("error", traceback.format_exc()))

    def _cancel_training(self) -> None:
        if self.training_thread and self.training_thread.is_alive():
            self.cancel_event.set()
            self._append_log(
                "Cancellation requested. The trainer will save after the current batch."
            )
            self.cancel_button.configure(state="disabled")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _browse_eval_checkpoint(self) -> None:
        selected = filedialog.askopenfilename(
            title="Checkpoint to evaluate",
            initialdir=str(Path(self.workspace_var.get()) / "checkpoints"),
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
        )
        if selected:
            self.eval_checkpoint_var.set(selected)

    def _load_eval_checkpoint(self) -> None:
        path = self.eval_checkpoint_var.get()
        if not Path(path).is_file():
            messagebox.showerror("Checkpoint", "Select an existing .pt checkpoint.")
            return
        self._set_status("Loading checkpoint…")

        def worker() -> None:
            try:
                model, payload = load_checkpoint(path, device=self.device_var.get())
                model.eval()
                self.events.put(("eval_loaded", (path, model, payload)))
            except Exception:
                self.events.put(("error", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    def _evaluate(self) -> None:
        path = self.eval_checkpoint_var.get()
        if self.eval_model is None or self.eval_checkpoint_loaded != path:
            messagebox.showwarning("Load checkpoint", "Load the selected checkpoint first.")
            return
        prompt = self.eval_prompt.get("1.0", "end").strip()
        if not prompt:
            messagebox.showwarning("Prompt", "Enter a prompt.")
            return
        self._set_status("Generating…")

        hunger = float(self.eval_hunger_var.get())
        thirst = float(self.eval_thirst_var.get())
        fatigue = float(self.eval_fatigue_var.get())
        boredom = float(self.eval_boredom_var.get())
        task = bool(self.eval_task_var.get())
        new_tokens = max(1, int(self.eval_tokens_var.get()))
        model = self.eval_model
        device = next(model.parameters()).device

        def worker() -> None:
            try:
                tokenizer = ByteTokenizer()
                ids = tokenizer.encode(prompt, add_eos=False)
                tokens = torch.tensor([ids], dtype=torch.long, device=device)
                features = torch.tensor(
                    [[
                        hunger,
                        thirst,
                        fatigue,
                        0.10,
                        0.05,
                        0.0,
                        0.10,
                        boredom,
                        0.45,
                        0.10,
                        0.10,
                        0.50,
                        0.50,
                        0.50,
                        1.0 if task else 0.0,
                        1.0,
                    ]],
                    dtype=torch.float32,
                    device=device,
                )
                with torch.no_grad():
                    inspection = model(tokens, state_features=features)
                    generated = model.generate(
                        tokens,
                        state_features=features,
                        max_new_tokens=new_tokens,
                        temperature=0.8,
                        top_k=40,
                    )
                full_text = tokenizer.decode(generated[0].detach().cpu().tolist())
                continuation = full_text[len(prompt):]
                drive_report = model.drive_activity_report(
                    inspection.drive_spikes,
                    minimum_rate=1e-9,
                )
                result = (
                    "CONTINUATION\n"
                    "------------\n"
                    f"{continuation}\n\n"
                    "ACTIVE DEDICATED DRIVE NEURONS\n"
                    "------------------------------\n"
                    + (
                        "\n".join(f"{name}: {rate:.3f}" for name, rate in drive_report.items())
                        or "none"
                    )
                    + "\n\n"
                    "MODEL\n"
                    "-----\n"
                    + json.dumps(model.architecture_summary(), indent=2)
                )
                self.events.put(("eval_result", result))
            except Exception:
                self.events.put(("error", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Event and display helpers
    # ------------------------------------------------------------------
    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "scan_done":
                    total, by_path, by_category = payload
                    self.source_stats = by_path
                    self._refresh_source_tree()
                    category_text = ", ".join(
                        f"{category}: {stats.records:,}"
                        for category, stats in sorted(by_category.items())
                    )
                    self.data_summary.configure(
                        text=(
                            f"Files: {total.files:,} | Records: {total.records:,} | "
                            f"Characters: {total.characters:,} | "
                            f"Language: {total.language_records:,} | "
                            f"Action: {total.action_records:,} | "
                            f"Malformed: {total.malformed_records:,}\n"
                            f"By category: {category_text or 'none'}"
                        )
                    )
                    self._set_status("Corpus validation finished.")
                    self._update_readiness()
                elif kind == "builder_done":
                    path, category, count = payload
                    self._append_sources([path], category)
                    self._set_status(f"Built {count:,} records at {path}.")
                    self._validate_sources()
                elif kind == "log":
                    self._append_log(str(payload))
                elif kind == "progress":
                    row = payload
                    self.progress_var.set(float(row["progress"]) * 100.0)
                    self.metric_loss_var.set(f"Loss: {row['loss']:.4f}")
                    self.metric_language_var.set(f"Language: {row['language_loss']:.4f}")
                    self.metric_action_var.set(f"Action: {row['action_loss']:.4f}")
                    self.metric_value_var.set(f"Value: {row['value_loss']:.4f}")
                    self.metric_spike_var.set(f"Spike rate: {row['spike_rate']:.4f}")
                    self._append_log(
                        f"epoch {int(row['epoch'])}/{int(row['epochs'])} "
                        f"batch {int(row['batch'])}/{int(row['batches'])} "
                        f"loss={row['loss']:.5f} "
                        f"lang={row['language_loss']:.5f} "
                        f"action={row['action_loss']:.5f} "
                        f"value={row['value_loss']:.5f} "
                        f"spikes={row['spike_rate']:.5f}"
                    )
                elif kind == "training_done":
                    self.start_button.configure(state="normal")
                    self.cancel_button.configure(state="disabled")
                    checkpoint = payload["checkpoint"]
                    cancelled = payload["cancelled"]
                    self._append_log(
                        ("Training cancelled; checkpoint saved." if cancelled else "Training completed.")
                        + f"\nCheckpoint: {checkpoint}"
                    )
                    self.eval_checkpoint_var.set(checkpoint)
                    self.resume_checkpoint_var.set(checkpoint)
                    self.resume_enabled_var.set(True)
                    self._save_project()
                    self._set_status("Training run finished.")
                elif kind == "eval_loaded":
                    path, model, payload = payload
                    self.eval_model = model
                    self.eval_checkpoint_loaded = path
                    self._replace_text(
                        self.eval_output,
                        "Checkpoint loaded.\n\n"
                        + json.dumps(
                            {
                                "path": path,
                                "step": payload.get("step"),
                                "metadata": payload.get("metadata"),
                                "architecture": model.architecture_summary(),
                            },
                            indent=2,
                        ),
                    )
                    self._set_status("Checkpoint loaded for evaluation.")
                elif kind == "eval_result":
                    self._replace_text(self.eval_output, str(payload))
                    self._set_status("Evaluation completed.")
                elif kind == "error":
                    self.start_button.configure(state="normal")
                    self.cancel_button.configure(state="disabled")
                    self._append_log(str(payload))
                    self._set_status("An operation failed. See the log.")
                    messagebox.showerror(
                        "Operation failed",
                        "The operation failed. Full details were written to the log/output panel.",
                    )
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _current_tab_name(self) -> str:
        selected = self.notebook.select()
        return str(self.notebook.tab(selected, "text"))

    def _update_guide(self) -> None:
        name = self._current_tab_name()
        self._replace_text(self.guide_text, textwrap_dedent(GUIDES.get(name, "")))
        self._update_readiness()

    def _update_readiness(self) -> None:
        if self.project is None:
            text = "Create or load a workspace."
            style = "StatusWarn.TLabel"
        elif not self.sources:
            text = "Workspace ready. Add training sources."
            style = "StatusWarn.TLabel"
        elif not self.source_stats:
            text = "Sources added. Validate the corpus."
            style = "StatusWarn.TLabel"
        else:
            total_records = sum(stats.records for stats in self.source_stats.values())
            malformed = sum(stats.malformed_records for stats in self.source_stats.values())
            if malformed:
                text = f"Fix {malformed} malformed record(s)."
                style = "StatusBad.TLabel"
            else:
                text = f"Corpus validated: {total_records:,} records."
                style = "StatusGood.TLabel"
        self.readiness_label.configure(text=text, style=style)

    @staticmethod
    def _replace_text(widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        self.train_log.configure(state="normal")
        self.train_log.insert("end", text.rstrip() + "\n")
        self.train_log.see("end")
        self.train_log.configure(state="disabled")

    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=text)


def textwrap_dedent(value: str) -> str:
    # Local tiny helper avoids retaining indentation from the source file.
    lines = value.strip("\n").splitlines()
    nonblank = [line for line in lines if line.strip()]
    if not nonblank:
        return ""
    indent = min(len(line) - len(line.lstrip()) for line in nonblank)
    return "\n".join(line[indent:] if len(line) >= indent else line for line in lines)


def diagnose() -> dict[str, Any]:
    hardware = detect_hardware()
    profile = recommended_profile(hardware)
    tkinter_available = True
    tkinter_error = None
    try:
        interpreter = tk.Tcl()
        interpreter.eval("info patchlevel")
    except Exception as exc:  # pragma: no cover - platform specific
        tkinter_available = False
        tkinter_error = str(exc)
    return {
        "tkinter_available": tkinter_available,
        "tkinter_error": tkinter_error,
        "hardware": {
            "python": hardware.python,
            "platform": hardware.platform,
            "torch": hardware.torch_version,
            "cpu_count": hardware.cpu_count,
            "ram": format_bytes(hardware.ram_bytes),
            "cuda_available": hardware.cuda_available,
            "gpu": hardware.gpu_name,
            "gpu_memory": format_bytes(hardware.gpu_memory_bytes),
        },
        "recommended_profile": profile,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="virtual-person-trainer")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print hardware/Tk diagnostics without opening a window.",
    )
    args = parser.parse_args(argv)
    if args.diagnose:
        print(json.dumps(diagnose(), indent=2))
        return 0
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise SystemExit(
            "Could not open the trainer window. Run this on a desktop session with "
            f"Tk support. Details: {exc}"
        )
    TrainerApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
