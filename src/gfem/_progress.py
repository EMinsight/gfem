"""Progress rows on the log display's console.

`gfem.log.enable` calls `start` with the display's rich console when progress
is asked for, and `disable` (or the next `enable`) calls `stop`. Both are
no-ops for now: no row is drawn yet.
"""

from __future__ import annotations

from typing import Any


def start(console: Any) -> None:
    """Show progress rows on ``console``, a rich ``Console`` on stderr."""
    del console


def stop() -> None:
    """Stop showing progress rows; a no-op when none are shown."""
