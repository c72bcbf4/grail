import time


class Timer:
    def __init__(self, disable):
        self.disable = disable
        self.timers = {}

    def start(self, timer):
        if self.disable:
            return

        if timer in self.timers:
            raise ValueError(f"timer '{timer}' already started")

        self.timers[timer] = {"start": time.time()}

    def end(self, timer):
        if self.disable:
            return

        if timer not in self.timers:
            raise ValueError(f"timer '{timer}' not started yet")

        self.timers[timer]["end"] = time.time()

    def compute(self):
        durations = {
            f"duration_{n}": round((t["end"] - t["start"]) / 60, 2)
            for n, t in self.timers.items()
        }

        self.timers = {}
        return durations
