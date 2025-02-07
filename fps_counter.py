import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _Entry:
    timestamp: float
    recorded_at: float


class FPSCounterStat:
    def __init__(self, intervals: np.ndarray):
        self._intervals = intervals

    @property
    def min_interval(self) -> float:
        if len(self._intervals):
            return np.min(self._intervals)
        else:
            return np.nan

    @property
    def max_interval(self) -> float:
        if len(self._intervals):
            return np.max(self._intervals)
        else:
            return np.nan

    @property
    def mean_interval(self) -> float:
        if len(self._intervals):
            return np.mean(self._intervals)
        else:
            return np.nan

    @property
    def ci95_interval_min(self) -> float:
        if len(self._intervals):
            return self._intervals.mean() - self._intervals.std() * 2
        else:
            return np.nan

    @property
    def ci95_interval_max(self) -> float:
        if len(self._intervals):
            return self._intervals.mean() + self._intervals.std() * 2
        else:
            return np.nan

    @property
    def min_fps(self) -> float:
        if len(self._intervals):
            return 1 / self.max_interval
        else:
            return np.nan

    @property
    def max_fps(self) -> float:
        if len(self._intervals):
            return 1 / self.min_interval
        else:
            return np.nan

    @property
    def mean_fps(self) -> float:
        if len(self._intervals):
            return 1 / np.mean(self._intervals)
        else:
            return np.nan

    @property
    def ci95_fps_min(self) -> float:
        if len(self._intervals):
            return 1 / self.ci95_interval_max
        else:
            return np.nan

    @property
    def ci95_fps_max(self) -> float:
        if len(self._intervals):
            return 1 / self.ci95_interval_min
        else:
            return np.nan


class FPSCounter:
    def __init__(self, max_record_seconds: float = 1):
        self._max_record_seconds = max_record_seconds
        self._timestamps: deque[_Entry] = deque()

    def add(self, timestamp: float):
        now = time.time()
        entry = _Entry(timestamp=timestamp, recorded_at=now)
        self._timestamps.append(entry)
        while self._timestamps:
            if self._timestamps[0].recorded_at < now - self._max_record_seconds:
                self._timestamps.popleft()
            else:
                break

    def _to_timestamp_array(self) -> np.ndarray:
        return np.array([e.timestamp for e in self._timestamps])

    def get_stat(self) -> FPSCounterStat:
        timestamps = self._to_timestamp_array()
        intervals = np.diff(timestamps)
        return FPSCounterStat(intervals=intervals)
