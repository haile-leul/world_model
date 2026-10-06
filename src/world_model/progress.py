"""Terminal progress bars, throttled log output, and stage announcements."""

import os
import sys
import time
from tqdm import tqdm


def enabled():
    return os.environ.get("WORLD_MODEL_PROGRESS", "1").lower() not in {"0", "false", "off"}


def stage(message):
    if enabled():
        tqdm.write(f"[world-model] {message}", file=sys.stderr)


class Progress:
    """Update only after completed work; context exit closes bars even on errors.

    Noninteractive output emits at most one progress line per 10 seconds, plus
    start/end. Shortened episodes show their actual step count, never fake 100%.
    """

    def __init__(self, total, description, unit="step"):
        self.total, self.description, self.unit = total, description, unit
        self.n = 0
        self.start = self.last = time.monotonic()
        self.active = enabled()
        self.terminal = self.active and sys.stderr.isatty()
        self.bar = None
        self.detail = ""

    def __enter__(self):
        if self.terminal:
            self.bar = tqdm(
                total=self.total,
                desc=self.description,
                unit=self.unit,
                file=sys.stderr,
                ascii=True,
                dynamic_ncols=True,
                mininterval=0.5,
                miniters=0,
                leave=True,
            )
        elif self.active:
            self._log("starting")
        return self

    def update(self, amount=1, **metrics):
        self.n += amount
        self.detail = ", ".join(f"{key}={value}" for key, value in metrics.items())
        if self.bar is not None:
            self.bar.set_postfix_str(self.detail, refresh=False)
            self.bar.update(amount)
        elif self.active and time.monotonic() - self.last >= 10:
            self._log("running")

    def _log(self, status):
        now = time.monotonic()
        elapsed = now - self.start
        rate = self.n / elapsed if elapsed > 0 else 0
        remaining = max(0, self.total - self.n)
        eta = f"{remaining / rate:.0f}s" if rate else "--"
        percent = 100 * self.n / self.total if self.total else 0
        stage(
            f"{self.description}: {self.n}/{self.total} {self.unit} "
            f"({percent:.0f}%), elapsed={elapsed:.1f}s, "
            f"{rate:.1f} {self.unit}/s, ETA={eta} [{status}] {self.detail}"
        )
        self.last = now

    def __exit__(self, exc_type, exc_value, traceback):
        status = (
            "interrupted" if exc_type else ("complete" if self.n == self.total else "ended early")
        )
        if self.bar is not None:
            self.bar.close()
        elif self.active:
            self._log(status)
        return False


def track(iterable, description, unit="item", total=None):
    """For loops whose iterations always run to completion (no early break)."""
    with Progress(len(iterable) if total is None else total, description, unit) as progress:
        for item in iterable:
            yield item
            progress.update()
