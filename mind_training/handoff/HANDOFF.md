# Handoff — Virtual Person AI / Node-Link-Spike training

Written by a prior Claude Code session for a fresh agent picking this up on
different (faster) hardware. Read this before doing anything else. It
supersedes any assumptions from the README about what stage the project is
at — the README describes the general system; this file describes the
actual current state of *this* workspace as of the handoff.

## What this project is

`virtual-person-ai` (installed via `pip install -e .` from the repo root,
`c:\Users\M\Documents\Coding\vp`) is a Python/PyTorch scaffold for an
embodied simulated person: a symbolic apartment world, a virtual computer,
a hunger/thirst/fatigue/etc. drive system, and a custom recurrent spiking
neural architecture ("Node-Link-Spike") trained to do byte-level language
modeling, action-candidate selection, and value/reward prediction jointly.

Read `README.md`, `CLI_TRAINING_GUIDE.md`, `TRAINING_GUIDE.md` in the repo
root for the full designed system. This document is about what's actually
been done in this workspace, what's broken, and what to do next.

## Hardware this was built on (the reason for the handoff)

- Intel Core i5-3570K (2012-era), 4 cores / 4 threads, 12.8GB RAM
- No usable GPU (GeForce GT 710, 1GB VRAM — CUDA unavailable)
- All training so far has been **CPU-only** and has taken **30-45 minutes
  per full 3-stage curriculum run** even at the small "CPU prototype"
  profile (206K parameters). This is the entire reason for this handoff:
  the next iteration needs a much larger corpus and probably a larger
  model, and this hardware cannot do that in reasonable time.

## Current workspace layout

```
mind_training/                          <- the trainer workspace
  trainer_project.json                  <- registered sources, model/training settings
  raw/                                  <- hand-authored corpus source files
    english/                            <- conversations.txt, scenes_and_reasoning.txt, starter_english.txt
    procedures/                         <- cooking, hygiene, computer_use, apartment_object_use, starters
    safety/                             <- paired_good_bad.txt
    behavior/                           <- starter_behavior.jsonl, shop_and_tasks.jsonl
  corpora/                              <- generated corpus files
    bootstrap_behavior.jsonl            <- 445 simulator-generated Behavior records (see below)
    sample_dictionary_dictionary.jsonl  <- 19 dictionary-derived records
  checkpoints/
    stage1_language.pt / .run.json      <- CURRENT (post template-variety-fix) checkpoints
    stage2_practical.pt / .run.json
    stage3_autonomous.pt / .run.json
    pre_fix_backup/                     <- checkpoints from BEFORE the JSON-leakage fix (stale, kept per checkpoint policy)
    pre_template_fix_backup/            <- checkpoints from AFTER the JSON fix but BEFORE the phrasing-variety fix (also stale)
  logs/
    curriculum_summary.json             <- per-stage metrics from the most recent curriculum run
    stage*_training.jsonl               <- per-step training logs
  eval_suite.json                       <- the fixed 28-case evaluation suite (see below)
  run_eval_suite.py                     <- standalone script to run eval_suite.json against any checkpoint
  stage3_eval_report_v3.json            <- MOST RECENT eval suite run (post both fixes)
  stage3_eval_report.json / _v2.json    <- earlier eval runs, kept for comparison, describe the bugs below
  snapshot_viewer.html                  <- standalone 2D floor-plan viewer (paste a dump_snapshot.py JSON in)
  sample_snapshot.json                  <- example snapshot for the viewer
  vm/
    windows10.qcow2                     <- 40GB QEMU virtual disk, Windows 10 IN PROGRESS INSTALLING (see VM section)
```

Also relevant outside `mind_training/`:
```
src/virtual_person/
  bootstrap_data.py       <- generates the Behavior corpus (see "Two bugs fixed" below — READ THIS)
  economy.py               <- NEW: human-reviewed task economy (assign/submit/review tasks, basic job fallback)
  computer.py               <- extended with a Shop app (groceries/hygiene/movie, spends simulated balance)
  world.py                  <- extended with ApartmentWorld.economy, _purchase() handler
  renderer.py               <- Tkinter live 2D floor-plan preview window (Renderer class) + draw_floor_plan() shared fn
  runtime_ui.py             <- NEW: full Tkinter GUI to load a checkpoint, run/pause/step it, watch drives, assign/review tasks
  vp_package.py             <- NEW: .vp file format (zip: checkpoint + memory.sqlite3 + training-data manifest+hashes)
  vm_runtime.py             <- NEW: QEMU QMP-based VM sensors/actuators (SensorSuite/ActuatorSuite impl)
tests/
  test_economy.py, test_runtime_ui.py, test_vp_package.py, test_vm_runtime.py, test_renderer.py  <- all new, all passing
examples/
  dump_snapshot.py          <- dumps a simulation snapshot to JSON for snapshot_viewer.html
  live_preview.py           <- demo of the Renderer class
pyproject.toml              <- added `virtual-person-runtime` script entry (registered, needs `pip install -e .` to take effect after any fresh clone)
```

Full test suite: **57/57 passing** as of the last check (`python -m pytest -q`
from repo root). Run this first to confirm the environment still works
before doing anything else.

## Two corpus bugs found and fixed this session — READ BEFORE TOUCHING bootstrap_data.py

The Behavior corpus is generated by `generate_bootstrap_corpus()` in
`src/virtual_person/bootstrap_data.py`, which runs the actual simulator
with randomized drive states and records the planner's chosen action as
ground truth. This produced two successive bugs, both now fixed, but the
history matters for understanding the checkpoint backups and why a third
round of work is likely still needed:

1. **Raw JSON leakage** (original design flaw): `state_text` was
   `json.dumps(observation)`. The model over-learned literal JSON syntax
   and leaked `{"asleep":false,"bladder":0.65...}` fragments into every
   generation regardless of prompt. **Fixed** by rendering the observation
   as natural-language prose in `_describe_state()`/`_describe_candidate()`.
   Checkpoints from before this fix are in `pre_fix_backup/`.

2. **Template-repetition leakage** (introduced by fix #1, found immediately
   after): the natural-language rendering used exactly ONE fixed sentence
   template per slot ("Reachable adjoining rooms: X.", "go to the X.",
   etc.) across all ~445 records. The model then over-fit to THOSE fixed
   phrases instead of JSON — same failure mode, different surface form,
   leaking "Reachable adjoining room" fragments into unrelated prompts.
   **Fixed** by giving each slot 3-4 randomly-chosen alternate phrasings
   (see `_ROOM_TEMPLATES`, `_NEIGHBOR_TEMPLATES`, `_DRIVE_INTRO_TEMPLATES`,
   `_VISIBLE_TEMPLATES`, `_CANDIDATE_TEMPLATES` in `bootstrap_data.py`),
   selected via the simulation's own seeded `random.Random`, so generation
   is still deterministic per seed but varied across records. Checkpoints
   from before this second fix are in `pre_template_fix_backup/`.

### The result after both fixes, and the real remaining problem

Held-out metrics after both fixes look fine in isolation (Stage 3: language
loss 0.77, byte perplexity 2.15, action accuracy 95.5%, spike rate 0.186,
no NaN/instability anywhere) — see `logs/curriculum_summary.json` for the
full per-stage breakdown and `stage3_eval_report_v3.json` for the 28-case
qualitative eval.

**But actual generated text is still broken** — not leaking one fixed
phrase anymore, but producing ungrammatical, often mid-word-truncated
recombinations of fragments from the several templates ("bedrom",
"inoticeable", "hirst is noticeable physical stight"). Read the full
diagnosis at the end of the conversation this handoff comes from, but the
short version: **this is very likely no longer a corpus-formatting bug —
it's a capacity/data-scale ceiling.** A 206K-parameter, 2-layer, 96-hidden
model trained on ~700 total records (of which ~440 are short Behavior
records, still structurally similar to each other even with phrase
variety) does not have enough real natural running English, or enough
parameters, to learn actual grammar — it's stitching together sub-word
fragments across all sources rather than generalizing.

**Recommended next steps, in order of expected impact:**

1. **Add substantially more natural English** — thousands of records of
   real prose (conversations, stories, explanations, corrections), not
   dozens. The current `raw/english/` files (`conversations.txt`,
   `scenes_and_reasoning.txt`, `starter_english.txt`) total only ~75
   records / ~10,600 characters. This is the single highest-leverage fix:
   without a much larger and more dominant natural-English signal, the
   model will keep defaulting to whatever structural pattern is most
   frequent in the corpus, currently the templated Behavior text.
2. **Move up the scaling ladder** once corpus size actually justifies it —
   `"Small GPU experiment"` profile (192 hidden, 4 layers, 3 ticks, 256
   seq length) per `trainer_support.py`'s `MODEL_PROFILES`. Do this
   *after* growing the corpus, not instead of it — the project's own
   scaling policy (`TRAINING_GUIDE.md`) says not to scale the model before
   verifying corpus quality and throughput at the current size.
3. Do NOT keep patching `bootstrap_data.py`'s phrasing — two rounds of
   that have already been done and the returns are diminishing. The next
   lever is corpus volume/diversity, not more template variants.

## Current corpus inventory (as of this handoff)

```
Category         Files  Records  Characters
Behavior         3      459      219,138
Dictionary       1      19       1,842
English          3      75       10,600      <- BOTTLENECK, see above
Procedures       5      97       16,989
Safety/Judgment  2      42       5,909
Total            14     692      254,478
```

Run `virtual-person-trainer --workspace ./mind_training validate --strict`
to reconfirm this is still accurate (0 malformed, 0 missing paths last
checked).

## Non-training features built this session (all tested, all working)

These are independent of the corpus/model-quality problem above and don't
need to be redone:

### 1. Task economy (`src/virtual_person/economy.py`)
Human-reviewed task assignment: `assign_task(description)`,
`submit_task(task_id)`, `review_task(task_id, score)` — score 0.0-1.0 maps
linearly to payout (max 50.0) AND to the existing `CognitiveDrives`
meaningful-progress/failed-attempt signals (fulfillment/frustration).
`do_basic_job()` is a low-value, always-available fallback for when no
task has been assigned. Entirely human-authored scoring — the model can
never score its own work. Wired into `ApartmentWorld.economy`.

### 2. Shop app (`src/virtual_person/computer.py`, `world.py`)
The virtual computer's desktop got a "Shop" app. Catalog: groceries (8.0,
restocks fridge eggs+water), hygiene_kit (4.0, +0.35 hygiene), movie_rental
(5.0, leisure only). Purchases go through `ApartmentWorld._purchase()`,
which checks balance and applies real in-sim effects — the model can
request a purchase but cannot fabricate funds.
**Deliberately excluded per user direction earlier in the conversation**:
no substance/drug purchases, nothing framed as an addictive
high-reward-short-duration loop.

### 3. Runtime GUI (`src/virtual_person/runtime_ui.py`, script:
`virtual-person-runtime`)
Full control panel for an ALREADY TRAINED checkpoint (not a training tool
— training still happens via `trainer_cli.py`/`trainer_ui.py`). Load
checkpoint, Start/Pause/Step, live floor-plan (shares drawing code with
`renderer.py`'s `draw_floor_plan()`), decision readout, drive/body bars,
named drive-neuron report, and a full task-assign/review panel wired
directly to `TaskEconomy`.

### 4. `.vp` model package format (`src/virtual_person/vp_package.py`)
Zip archive bundling: `metadata.json`, `model.pt` (the checkpoint,
unmodified format), `memory.sqlite3` (snapshotted via SQLite's own
`backup()` API, not a raw file copy), `training_data/manifest.json`
(per-source category/records/characters/**sha256**). `embed_data=False`
keeps just the hashed manifest without copying source files in. CLI:
`virtual-person-trainer pack/unpack/inspect-vp`.

### 5. QEMU VM integration (`src/virtual_person/vm_runtime.py`)
`QmpClient` (raw QMP JSON protocol), `QemuVirtualMachine` (process
lifecycle via `qemu-system-x86_64`, QMP connect-with-retry), `QemuSensors`
(screen capture via `screendump`), `QemuActuators` (mouse move/click via
QMP `input-send-event` with a `usb-tablet` device for absolute
coordinates, keyboard via qcode translation). Implements the EXISTING
`SensorSuite`/`ActuatorSuite` ABCs from `real_world.py`, so it goes through
the same `RealWorldRuntime` safety gating (dry-run, forbidden-operations,
human-approval) as any other actuator — verified this directly, not just
unit-tested. All 7 tests in `test_vm_runtime.py` pass against real QEMU
processes (they skip gracefully if `qemu-system-x86_64`/`qemu-img` aren't
on PATH).

**IMPORTANT — VM STATE AT HANDOFF TIME:** A Windows 10 install is
IN PROGRESS in `mind_training/vm/windows10.qcow2` (40GB qcow2 disk), booted
from `Windows.iso` in the repo root (~8.45GB, real bootable Windows 10
ISO, already gitignored). The user was going through OOBE ("getting
ready...") interactively in a visible QEMU window
(`qemu-system-x86_64.exe`, PID was 20016, using ~4GB RAM) at the time of
this handoff — check whether that process is still running and whether
Windows finished installing. Once it reaches a working desktop:
1. Install Guest Additions equivalent isn't needed — the QMP approach
   doesn't depend on anything inside the guest.
2. Switch subsequent boots to NOT pass `-cdrom Windows.iso -boot d`
   (that was only for the installer) — just boot the qcow2 disk directly.
3. Hand the running VM to `QemuSensors`/`QemuActuators` per the pattern
   verified in this session's testing (see `test_vm_runtime.py` and the
   manual test transcript in conversation history).

## Fixed evaluation suite

`mind_training/eval_suite.json` — 28 fixed cases across English,
Dictionary, Procedures, Safety/Judgment, Computer use, each drive
(hunger/thirst/fatigue/bladder/boredom) at notice/urgent levels,
conflicting drives, waiting, asking, and a repetition probe. Run it with:

```powershell
python mind_training/run_eval_suite.py <checkpoint.pt> --out <report.json>
```

Always compare new checkpoints against this exact suite — never judge a
checkpoint from one generation sample (this is explicit guidance from the
project's own TRAINING_GUIDE.md).

## What NOT to do

- Don't re-run more phrasing-template patches on `bootstrap_data.py` — two
  rounds already done, diminishing returns, the real lever is corpus
  *volume* of natural English.
- Don't delete `pre_fix_backup/` or `pre_template_fix_backup/` checkpoints
  without explicit user permission (project's own checkpoint policy).
- Don't build a custom internet-facing network protocol for anything (this
  came up earlier in the session and was explicitly refused — use vetted
  tools like Tailscale/SSH if remote access is ever needed again).
- Don't add substance/drug-purchase mechanics to the economy/shop system
  (explicitly scoped out by the user).
- Don't jump to a large model/corpus config without measuring throughput
  first, per the project's own scaling policy in TRAINING_GUIDE.md.

## Immediate next actions for the new agent

1. `python -m pytest -q` from repo root — confirm 57/57 still pass on the
   new machine.
2. Check Windows VM state (`mind_training/vm/windows10.qcow2`, was
   mid-install at handoff).
3. Decide with the user whether to prioritize (a) growing the English
   corpus substantially before any more training, or (b) something else
   they've asked for since this handoff was written.
4. If training resumes: back up current checkpoints first (see the
   `pre_*_backup/` pattern already established), regenerate/validate
   corpus, retrain, then re-run `run_eval_suite.py` and actually read the
   generated text before trusting the loss numbers alone — this session's
   entire debugging arc was loss-numbers-look-fine but
   generated-text-is-broken, twice in a row.
