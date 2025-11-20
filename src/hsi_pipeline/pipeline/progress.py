"""Rich-based progress helper for the HSI pipeline."""

from __future__ import annotations

from typing import Dict, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.theme import Theme


class PipelineProgress:
    """Wraps Rich progress bars to present a friendly UI."""

    def __init__(self, console: Optional[Console] = None) -> None:
        custom_theme = Theme(
            {
                "title": "bold cyan",
                "accent": "bright_white",
                "path": "bright_black",
                "progress.data": "bold cyan",
            }
        )
        self.console = console or Console(theme=custom_theme)
        self.progress = Progress(
            SpinnerColumn(style="bold cyan", speed=0.8),
            TextColumn("[bold white]{task.description}", justify="left"),
            BarColumn(
                bar_width=None,
                complete_style="bold green",
                finished_style="bright_green",
                pulse_style="magenta",
                style="magenta",
            ),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=24,
            console=self.console,
            transient=False,
        )
        self._tasks: Dict[str, int] = {}

    def __enter__(self) -> "PipelineProgress":
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.progress.stop()

    def start_dataset(self, name: str, index: int, total: int) -> None:
        header = f"[title]Dataset {index}/{total}: [accent]{name}"
        self.console.rule(header, style="cyan")

    def log(self, message: str, style: str = "green") -> None:
        self.console.log(f"[{style}]{message}")

    def create_task(self, key: str, description: str, total: Optional[int]) -> None:
        total_value = None if total is None or total <= 0 else total
        if key in self._tasks:
            self.progress.remove_task(self._tasks[key])
        self._tasks[key] = self.progress.add_task(description, total=total_value)

    def advance(self, key: str, advance: float = 1.0) -> None:
        task_id = self._tasks.get(key)
        if task_id is not None:
            self.progress.advance(task_id, advance)

    def complete(self, key: str) -> None:
        task_id = self._tasks.get(key)
        if task_id is None:
            return
        total = self.progress.tasks[task_id].total
        if total is not None:
            self.progress.update(task_id, completed=total)
        else:
            self.progress.remove_task(task_id)
            del self._tasks[key]
