from typing import Iterable

import cv2
import numpy as np

from app_logging import create_logger


class VideoReader:
    _logger = create_logger()

    def __init__(self, path: str):
        self._path = path
        self._cap: cv2.VideoCapture | None = None

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def _check_open(self) -> None:
        if not self.is_open():
            raise ValueError("Video reader is not open")

    def open(self) -> None:
        if self.is_open():
            raise ValueError("Video reader is open")

        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Failed to open video file: {self._path}")
        self._logger.info(f"Video reader opened: {self._path}")

    def close(self) -> None:
        self._check_open()

        self._logger.info(f"Closing video reader: {self._path}")
        self._cap.release()
        self._cap = None

    def get_frame_count(self) -> int:
        self._check_open()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_current_frame_count(self) -> int:
        self._check_open()
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

    def get_fps(self) -> float:
        self._check_open()
        return float(self._cap.get(cv2.CAP_PROP_FPS))

    def get_width(self) -> int:
        self._check_open()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_height(self) -> int:
        self._check_open()
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self) -> Iterable[np.ndarray]:
        self._check_open()
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame
