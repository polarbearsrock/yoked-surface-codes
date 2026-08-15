import sys
import threading
import time
from typing import Any, Iterable


class ThrottledProgressPrinter:
    """Prints output immediately while throttling transient progress updates."""

    def __init__(
        self,
        *,
        outs: Iterable[Any],
        print_progress: bool,
        min_progress_delay: float,
    ):
        self.outs = tuple(outs)
        self.print_progress = print_progress
        self.next_can_print_time = time.monotonic()
        self.latest_msg = ""
        self.latest_printed_msg = ""
        self.min_progress_delay = min_progress_delay
        self.is_worker_running = False
        self.lock = threading.Lock()

    def print_out(self, msg: str) -> None:
        with self.lock:
            for out in self.outs:
                print(msg, file=out, flush=True)

    def show_latest_progress(self, msg: str) -> None:
        if not self.print_progress:
            return
        with self.lock:
            if msg == self.latest_msg:
                return
            self.latest_msg = msg
            if not self.is_worker_running:
                delay = self._try_print_else_delay()
                if delay > 0:
                    self.is_worker_running = True
                    threading.Thread(target=self._print_worker, daemon=True).start()

    def flush(self) -> None:
        with self.lock:
            if self.latest_msg and self.latest_printed_msg != self.latest_msg:
                print(f"\033[31m{self.latest_msg}\033[0m", file=sys.stderr, flush=True)
                self.latest_printed_msg = self.latest_msg

    def _try_print_else_delay(self) -> float:
        now = time.monotonic()
        delay = self.next_can_print_time - now
        if delay <= 0:
            self.next_can_print_time = now + self.min_progress_delay
            self.is_worker_running = False
            if self.latest_msg and self.latest_msg != self.latest_printed_msg:
                print(f"\033[31m{self.latest_msg}\033[0m", file=sys.stderr, flush=True)
                self.latest_printed_msg = self.latest_msg
        return max(delay, 0)

    def _print_worker(self) -> None:
        while True:
            with self.lock:
                delay = self._try_print_else_delay()
            if delay == 0:
                return
            time.sleep(delay)
