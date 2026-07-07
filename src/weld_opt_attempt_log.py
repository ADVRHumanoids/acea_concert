from pathlib import Path
from datetime import datetime

import numpy as np


class WeldOptAttemptLog:
    def __init__(self, path="/tmp/concert_weld_attempts.txt"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def write(self, attempt, event, **fields):
        with self.path.open("a") as file:
            file.write(
                f"[{_timestamp()}] attempt {int(attempt)}: {event}\n")
            for name, value in fields.items():
                if name == "pairs":
                    file.write("  pairs:\n")
                    for pair in value:
                        file.write(f"    - {pair}\n")
                else:
                    file.write(f"  {name}: {_format_value(value)}\n")
            file.write("\n")


def _timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _format_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)
