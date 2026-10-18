from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Iterator

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


def tqdm(
    iterable: Iterable | None = None,
    *,
    total: int | None = None,
    desc: str = "",
    unit: str = "it",
    unit_scale: bool = False,
    dynamic_ncols: bool = False,
):
    if _tqdm is not None:
        return _tqdm(
            iterable,
            total=total,
            desc=desc,
            unit=unit,
            unit_scale=unit_scale,
            dynamic_ncols=dynamic_ncols,
        )
    return ConsoleProgress(iterable, total=total, desc=desc, unit=unit)


class ConsoleProgress:
    def __init__(
        self,
        iterable: Iterable | None = None,
        *,
        total: int | None = None,
        desc: str = "",
        unit: str = "it",
    ) -> None:
        self.iterable = iterable
        self.total = total if total is not None else self._get_length(iterable)
        self.desc = desc
        self.unit = unit
        self.current = 0
        self.started_at = time.monotonic()
        self.last_rendered_at = 0.0
        self.postfix: dict[str, object] = {}
        self.closed = False

    @staticmethod
    def _get_length(iterable: Iterable | None) -> int | None:
        try:
            return len(iterable) if iterable is not None else None
        except TypeError:
            return None

    def __iter__(self) -> Iterator:
        if self.iterable is None:
            raise TypeError("Progress iterable is not set")
        try:
            for item in self.iterable:
                yield item
                self.update()
        finally:
            self.close()

    def __enter__(self) -> ConsoleProgress:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def update(self, count: int = 1) -> None:
        self.current += count
        now = time.monotonic()
        if now - self.last_rendered_at >= 0.5 or self.total is not None and self.current >= self.total:
            self._render(now)

    def set_postfix(self, ordered_dict=None, refresh: bool = True, **kwargs) -> None:
        if ordered_dict:
            self.postfix.update(dict(ordered_dict))
        self.postfix.update(kwargs)
        if refresh:
            self._render(time.monotonic())

    def close(self) -> None:
        if self.closed:
            return
        self._render(time.monotonic())
        sys.stderr.write("\n")
        sys.stderr.flush()
        self.closed = True

    def _render(self, now: float) -> None:
        elapsed = max(now - self.started_at, 1e-9)
        rate = self.current / elapsed
        if self.total is None:
            status = f"{self.current} {self.unit}"
            eta = "ETA --:--"
        else:
            percent = 100.0 if self.total == 0 else min(self.current / self.total * 100, 100.0)
            status = f"{percent:6.2f}% {self.current}/{self.total} {self.unit}"
            remaining = max(self.total - self.current, 0)
            eta = f"ETA {format_seconds(remaining / rate) if rate else '--:--'}"
        extras = " ".join(f"{key}={value}" for key, value in self.postfix.items())
        message = f"\r{self.desc}: {status} [{rate:.2f} {self.unit}/s, {eta}]"
        if extras:
            message += f" {extras}"
        sys.stderr.write(message)
        sys.stderr.flush()
        self.last_rendered_at = now


def format_seconds(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"
