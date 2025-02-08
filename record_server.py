from multiprocessing import Queue, Process
from queue import Empty

import cv2
import numpy as np

from app_logging import create_logger


class VideoWriter:
    _logger = create_logger()

    def __init__(self, video_path, fps: float, width: int, height: int):
        self._video_path = video_path
        self._fps = fps
        self._size = width, height
        # noinspection PyUnresolvedReferences
        self._fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self._writer = None
        self._write_count = 0
        self.open()
        self._logger.info(
            f"VideoWriter created: {self._video_path}\nfps={fps}, size={width}x{height}"
        )

    def is_open(self) -> bool:
        return self._writer is not None

    def _check_open(self) -> None:
        if not self.is_open():
            raise RuntimeError("VideoWriter is not open")

    def open(self):
        self._writer = cv2.VideoWriter(str(self._video_path), self._fmt, self._fps, self._size)
        if not self._writer.isOpened():
            raise Exception(f"VideoWriter failed to open: {self._video_path}")

    def close(self):
        self._check_open()
        self._logger.info(f"Closing VideoWriter: {self._video_path}")
        self._writer.release()
        self._logger.info(f"VideoWriter closed: {self._video_path}")

    def write(self, frame):
        self._check_open()
        if not self._writer.isOpened():
            raise Exception("VideoWriter is not opened.")
        self._writer.write(frame)
        self._write_count += 1

    @property
    def write_count(self) -> int:
        self._check_open()
        return self._write_count


class RecordServer:
    _logger = create_logger()

    def __init__(self):
        self._q_in = Queue()  # (command_name, data)
        self._q_frames = Queue()  # np.ndarray
        self._p = Process(target=self._worker, args=(self._q_in, self._q_frames))
        self._p.start()

    @classmethod
    def _worker(cls, q_in: Queue, q_frames: Queue) -> None:
        cls._logger.info("Worker started")

        writer: VideoWriter | None = None

        while True:
            try:
                data_in = q_in.get(block=False)
            except Empty:
                pass
            else:
                command_name, data = data_in
                if command_name == "stop":
                    if writer is not None and writer.is_open():
                        writer.close()
                    break
                elif command_name == "record-begin":
                    if writer is not None and writer.is_open():
                        writer.close()
                    video_path, fps, width, height = data
                    assert isinstance(video_path, str), f"Invalid video_path: {video_path}"
                    assert isinstance(fps, (int, float)), f"Invalid fps: {fps}"
                    assert isinstance(width, int), f"Invalid width: {width}"
                    assert isinstance(height, int), f"Invalid height: {height}"
                    writer = VideoWriter(video_path, fps, width, height)
                else:
                    assert False, f"Unknown command: {command_name}"

            try:
                frame: np.ndarray = q_frames.get(block=True, timeout=0.01)
            except Empty:
                pass
            else:
                if isinstance(frame, np.ndarray):
                    if writer is not None and writer.is_open():
                        writer.write(frame)
                elif frame == "record-end":
                    if writer is not None and writer.is_open():
                        writer.close()
                        writer = None
                else:
                    assert False, f"Received data is not a numpy array or record-end: {frame}"

        cls._logger.info("Worker finished")

    def request_stop(self) -> None:
        self._q_in.put(("stop", None))

    def request_record_begin(self, video_path: str, fps: float, width: int, height: int) -> None:
        self._q_in.put(("record-begin", (video_path, fps, width, height)))

    def request_record_end(self) -> None:
        self._q_frames.put("record-end")

    def put_frames(self, frames: list[np.ndarray]) -> None:
        for frame in frames:
            self._q_frames.put(frame)

    def get_queue_count(self) -> int:
        return self._q_frames.qsize()

    def shutdown(self) -> None:
        # noinspection DuplicatedCode
        self._logger.info("Shutting down")
        self.request_stop()
        self._logger.info("Waiting for process")
        while True:
            self._p.join(timeout=1)
            if self._p.is_alive():
                self._p.terminate()
                self._logger.info("Force terminate")
            else:
                break
        self._q_in.close()
        self._q_frames.close()
        self._logger.info("Shutdown finished")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
