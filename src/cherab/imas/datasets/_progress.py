"""Progress bar implementation for pooch using rich."""

try:
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TaskID,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )
except ImportError as e:
    raise ImportError(
        "Missing optional dependency 'rich' required for cherab.imas.datasets module. "
        + "Please use pip or conda to install 'rich'."
    ) from e


class PoochRichProgress:
    def __init__(self, filename: str = ""):
        self.progress: Progress = Progress(
            TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(elapsed_when_finished=True),
        )
        self.task_id: TaskID = self.progress.add_task(description="", filename=filename, total=None)

        # Initialize variables
        self._finished_speed: float | None = 0.0

    @property
    def total(self) -> float | None:
        return self.progress.tasks[self.task_id].total

    @total.setter
    def total(self, value: int) -> None:
        self.progress.update(self.task_id, total=value)
        self.progress.start()

    def close(self) -> None:
        # Restore the finished speed before closing because `reset` is called at the end of
        # a download in pooch.
        self.progress.tasks[self.task_id].finished_speed = self._finished_speed
        self.progress.stop_task(self.task_id)
        self.progress.stop()

    def update(self, chunk_size: int) -> None:
        self.progress.update(self.task_id, advance=chunk_size)

    def reset(self) -> None:
        # Store the finished speed before resetting
        self._finished_speed = self.progress.tasks[self.task_id].speed

        # Reset the progress bar
        self.progress.update(self.task_id, completed=0)
