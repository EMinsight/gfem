"""A progress board that records the rows it is asked to draw."""


class RecordingBoard:
    """A board that records what it is asked to draw, as ``(event, ...)`` tuples."""

    def __init__(self):
        self.events = []
        self.rows = {}  # key -> [label, total, depth, completed, closed]

    def open(self, label, total, depth, *, spinner):
        key = len(self.rows)
        self.rows[key] = {
            "label": label,
            "total": total,
            "depth": depth,
            "spinner": spinner,
            "done": 0,
            "closed": False,
        }
        self.events.append(("open", label, total, depth, spinner))
        return key

    def advance(self, key, n, fields):
        self.rows[key]["done"] += n
        self.events.append(("advance", self.rows[key]["label"], n, dict(fields)))

    def update(self, key, completed, fields):
        if completed is not None:
            self.rows[key]["done"] = completed
        self.events.append(("update", self.rows[key]["label"], completed, dict(fields)))

    def phase(self, key, name):
        self.events.append(("phase", self.rows[key]["label"], name))

    def close(self, key):
        self.rows[key]["closed"] = True
        self.events.append(("close", self.rows[key]["label"]))

    def summary(self):
        """Each row opened: label, total, units done, closed."""
        return [
            (r["label"], r["total"], r["done"], r["closed"]) for r in self.rows.values()
        ]
