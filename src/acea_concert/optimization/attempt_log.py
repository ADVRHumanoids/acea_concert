from pathlib import Path


class WeldOptAttemptLog:
    def __init__(self, path="/tmp/concert_weld_attempts.txt"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def write(self, attempt, event, **fields):
        with self.path.open("a") as file:
            file.write(f"attempt {int(attempt)}: {event}\n")
            if "node" in fields:
                file.write(f"  node: {fields['node']}\n")
            if "pairs" in fields:
                file.write("  collisions:\n")
                for pair in fields["pairs"]:
                    file.write(f"    - {pair}\n")
            file.write("\n")
