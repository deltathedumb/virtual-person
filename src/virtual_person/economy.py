"""A human-reviewed task economy.

The user assigns free-text tasks; the agent marks them submitted when it
believes they are done; the user reviews the actual result and assigns a
score. Payout and the resulting change in cognitive drives are both derived
from that human-authored score, never from anything the model computes for
itself. This mirrors the project's separation of concerns: success detection
and reward calculation stay outside the learned model.

When no task has been assigned, ``basic_job`` is available as a low-effort,
low-payout, low-fulfillment fallback so the agent is never without something
to do for pay.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from .drives import CognitiveDrives, clamp


class TaskStatus(Enum):
    PENDING = auto()
    SUBMITTED = auto()
    REVIEWED = auto()


@dataclass(slots=True)
class TaskReview:
    score: float
    payout: float
    note: str = ""
    reviewed_time: float = 0.0


@dataclass(slots=True)
class AssignedTask:
    task_id: str
    description: str
    created_time: float
    status: TaskStatus = TaskStatus.PENDING
    submitted_time: float | None = None
    submission_note: str = ""
    review: TaskReview | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status.name.lower(),
            "created_time": self.created_time,
            "submitted_time": self.submitted_time,
            "submission_note": self.submission_note,
            "review": (
                {
                    "score": self.review.score,
                    "payout": self.review.payout,
                    "note": self.review.note,
                    "reviewed_time": self.review.reviewed_time,
                }
                if self.review
                else None
            ),
        }


# Basic, non-meaningful fallback work: available any time nothing has been
# assigned. Low payout, low fulfillment by design, so it never competes with
# a genuinely reviewed task as the more rewarding option.
BASIC_JOB_PAYOUT = 2.0
BASIC_JOB_MEANINGFUL_PROGRESS = 0.05

# A reviewed task's payout scales from 0 at score 0.0 up to this at score 1.0.
MAX_TASK_PAYOUT = 50.0


class TaskEconomy:
    """Tracks a simulated balance and a queue of human-assigned, human-scored tasks."""

    def __init__(self, cognitive_drives: CognitiveDrives | None = None) -> None:
        self.balance: float = 0.0
        self.cognitive_drives = cognitive_drives or CognitiveDrives()
        self.tasks: dict[str, AssignedTask] = {}
        self.history: list[AssignedTask] = []
        self._id_counter = itertools.count(1)

    # -- user-facing API -------------------------------------------------

    def assign_task(self, description: str, sim_time: float = 0.0) -> AssignedTask:
        """Create a new task for the agent to attempt."""
        if not description.strip():
            raise ValueError("Task description must not be empty.")
        task_id = f"task-{next(self._id_counter)}"
        task = AssignedTask(task_id=task_id, description=description.strip(), created_time=sim_time)
        self.tasks[task_id] = task
        return task

    def pending_task(self) -> AssignedTask | None:
        """The oldest task still awaiting submission, if any."""
        candidates = [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]
        return min(candidates, key=lambda t: t.created_time) if candidates else None

    def awaiting_review(self) -> list[AssignedTask]:
        return [t for t in self.tasks.values() if t.status == TaskStatus.SUBMITTED]

    # -- agent-facing API -------------------------------------------------

    def submit_task(self, task_id: str, sim_time: float = 0.0, note: str = "") -> AssignedTask:
        """The agent marks a task as done and ready for human review."""
        task = self._require_task(task_id)
        if task.status != TaskStatus.PENDING:
            raise ValueError(f"Task {task_id} is not pending (status={task.status.name}).")
        task.status = TaskStatus.SUBMITTED
        task.submitted_time = sim_time
        task.submission_note = note
        return task

    def do_basic_job(self, sim_time: float = 0.0) -> float:
        """Non-meaningful fallback work: flat small payout, minimal fulfillment.

        Always available, regardless of any assigned tasks, so the agent is
        never blocked on human review to earn something.
        """
        self.balance += BASIC_JOB_PAYOUT
        self.cognitive_drives.update(
            0.0,
            meaningful_progress=BASIC_JOB_MEANINGFUL_PROGRESS,
        )
        return BASIC_JOB_PAYOUT

    # -- reviewer (human) API ---------------------------------------------

    def review_task(
        self,
        task_id: str,
        score: float,
        sim_time: float = 0.0,
        note: str = "",
    ) -> TaskReview:
        """A human reviews a submitted task's actual result and scores it.

        ``score`` is 0.0 (did not accomplish the task) to 1.0 (excellent).
        Payout scales with score; so does the fulfillment/frustration signal
        fed into the agent's cognitive drives. This is the only place money
        or the meaningful-progress signal enters the system for a task.
        """
        score = clamp(float(score))
        task = self._require_task(task_id)
        if task.status != TaskStatus.SUBMITTED:
            raise ValueError(f"Task {task_id} is not awaiting review (status={task.status.name}).")

        payout = round(MAX_TASK_PAYOUT * score, 2)
        review = TaskReview(score=score, payout=payout, note=note, reviewed_time=sim_time)
        task.review = review
        task.status = TaskStatus.REVIEWED

        self.balance += payout
        self.cognitive_drives.update(
            0.0,
            meaningful_progress=score,
            failed_attempt=1.0 - score,
        )

        self.history.append(task)
        del self.tasks[task_id]
        return review

    # -- introspection ------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "balance": self.balance,
            "pending": [t.as_dict() for t in self.tasks.values() if t.status == TaskStatus.PENDING],
            "awaiting_review": [t.as_dict() for t in self.tasks.values() if t.status == TaskStatus.SUBMITTED],
            "history": [t.as_dict() for t in self.history[-20:]],
        }

    def _require_task(self, task_id: str) -> AssignedTask:
        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"Unknown or already-reviewed task: {task_id}")
        return task
