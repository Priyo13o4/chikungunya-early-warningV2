"""Common path utilities for the v6 snapshot.

v6 lives under `.../chikungunya-early-warning/versions/Priyo-Version-Hist/v6/`.
Many experiments need to access shared repo-level resources (e.g. `data/`,
`stan_models/`, other `versions/` directories). These helpers locate the real
repo root reliably regardless of where v6 is nested.
"""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """Find the enclosing chikungunya-early-warning repository root.

    Strategy: walk upward from `start` until we find a directory that looks
    like the repo root (must contain `data/` and `versions/`).

    Returns `start` if no such parent exists.
    """
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "data").is_dir() and (candidate / "versions").is_dir():
            return candidate
    return start
