"""Run every problem detector over a session.

A thin aggregator so future detector types — sub-agent opportunities,
no-closed-loop (vibe-fixing) — slot in here without touching ``api/app.py``
again: each is its own module with its own ``detect(session)``, and this just
collects whatever they find.
"""

from __future__ import annotations

from Final_app.analysis import plan_mode, task_forest
from Final_app.ir.models import Problem, Session


def detect_problems(session: Session) -> list[Problem]:
    problems: list[Problem] = []

    for detector in (plan_mode, task_forest):
        problem = detector.detect(session)
        if problem is not None:
            problems.append(problem)

    return problems
