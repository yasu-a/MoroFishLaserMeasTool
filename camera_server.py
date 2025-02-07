import time
from dataclasses import dataclass
from multiprocessing import Queue, Process
from queue import Empty

import cv2
import numpy as np

from app_logging import create_logger


@dataclass(frozen=True)
class VideoSpec:
    width: int
    height: int
    fps: float

    @classmethod
    def from_video_capture(cls, cap: cv2.VideoCapture) -> "VideoSpec":
        return cls(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS),
        )


@dataclass(slots=True)
class CaptureResult:
    frame: np.ndarray
    timestamp: float


class VideoReader:
    _logger = create_logger()

    def __init__(self, dev_name: int, video_spec: VideoSpec):
        self._dev_name = dev_name
        self._video_spec = video_spec
        self._cap: cv2.VideoCapture | None = None
        self.open()

    def get_device_name(self) -> int:
        return self._dev_name

    def get_configured_spec(self) -> VideoSpec:
        return self._video_spec

    def get_actual_spec(self) -> VideoSpec:
        self._check_open()
        return VideoSpec.from_video_capture(self._cap)

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def _check_open(self) -> None:
        if not self.is_open():
            raise RuntimeError(f"Camera {self._dev_name} reader is not open")

    def open(self):
        # カメラを開く
        self._cap = cv2.VideoCapture(self._dev_name)
        if not self._cap.isOpened():
            self._cap = None

        # キャプチャ設定
        self._logger.info(f"Setting width={self.get_configured_spec().width}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.get_configured_spec().width)
        self._logger.info(f"Setting height={self.get_configured_spec().height}")
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.get_configured_spec().height)
        self._logger.info(f"Setting fps={self.get_configured_spec().fps}")
        self._cap.set(cv2.CAP_PROP_FPS, self.get_configured_spec().fps)
        if not self._cap.isOpened():
            self._cap = None

        # 設定を検証
        if self.get_configured_spec() != self.get_actual_spec():
            self._logger.error(
                f"Camera {self._dev_name!r} configuration mismatch\n"
                f"expected={self.get_configured_spec()!r}\n"
                f"actual={self.get_actual_spec()!r}"
            )
            self._cap = None

        if self._cap is not None:
            self._logger.info(f"Camera {self._dev_name!r} open")
        else:
            self._logger.error(f"Failed to open camera {self._dev_name}")

    def close(self) -> None:
        self._cap.release()
        self._cap = None

    def read(self) -> CaptureResult | None:
        self._check_open()
        flag, frame = self._cap.read()
        timestamp = time.time()
        if flag:
            return CaptureResult(frame=frame, timestamp=timestamp)
        else:
            return None


@dataclass(frozen=True)
class CameraInfo:
    is_available: bool
    dev_name: int | None
    configured_spec: VideoSpec | None
    actual_spec: VideoSpec | None


class CameraServer:
    _logger = create_logger()

    def __init__(self):
        self._configured_spec: VideoSpec | None = None
        self._q_in = Queue()  # item: (command name, data)
        self._q_out = Queue()  # item: CaptureResult
        self._q_frames = Queue()
        self._p = Process(target=self._worker, args=(self._q_in, self._q_out, self._q_frames))
        self._p.start()

    @classmethod
    def _worker(cls, q_in: Queue, q_out: Queue, q_frames: Queue) -> None:
        cls._logger.info("Worker started")

        reader: VideoReader | None = None

        while True:
            try:
                data_in = q_in.get(block=False)
            except Empty:
                pass
            else:
                command_name, data = data_in
                if command_name == "stop":
                    if reader is not None and reader.is_open():
                        reader.close()
                    break
                elif command_name == "open":
                    if reader is not None and reader.is_open():
                        reader.close()

                    dev_name, spec = data
                    assert isinstance(dev_name, int), f"Invalid dev_name: {dev_name}"
                    assert isinstance(spec, VideoSpec), f"Invalid video_spec: {spec}"
                    reader = VideoReader(dev_name, spec)
                elif command_name == "close":
                    if reader is not None and reader.is_open():
                        reader.close()
                        reader = None
                elif command_name == "get_camera_info":
                    if reader is not None and reader.is_open():
                        camera_info = CameraInfo(
                            is_available=True,
                            dev_name=reader.get_device_name(),
                            configured_spec=reader.get_configured_spec(),
                            actual_spec=reader.get_actual_spec(),
                        )
                    else:
                        camera_info = CameraInfo(
                            is_available=False,
                            dev_name=None,
                            configured_spec=None,
                            actual_spec=None,
                        )
                    q_out.put(camera_info)
                else:
                    assert False, f"Unknown command: {command_name}"

            if reader is not None and reader.is_open():
                capture_result = reader.read()
                if capture_result is not None:
                    q_frames.put(capture_result)

        cls._logger.info("Worker finished")

    def request_stop(self) -> None:
        self._q_in.put(("stop", None))

    def request_open(self, dev_name: int, spec: VideoSpec) -> None:
        self._q_in.put(("open", (dev_name, spec)))

    def request_close(self) -> None:
        self._q_in.put(("close", None))

    def request_camera_info(self) -> None:
        self._q_in.put(("get_camera_info", None))

    def get_requested_camera_info(self) -> CameraInfo | None:
        try:
            data = self._q_out.get(block=False)
        except Empty:
            return None
        else:
            if isinstance(data, CameraInfo):
                return data
            else:
                assert False, f"Received data is not a CameraInfo: {data}"

    def get_all_frames(self) -> list[CaptureResult]:
        n = max(1, self._q_frames.qsize())
        lst: list[CaptureResult] = []
        for _ in range(n):
            try:
                capture_result = self._q_frames.get(block=True, timeout=0.01)
            except Empty:
                break
            else:
                lst.append(capture_result)
        return lst

    def shutdown(self) -> None:
        self._logger.info("Shutting down")
        self.request_stop()
        self._logger.info("Waiting for process")
        while True:
            self._p.join(timeout=1)  # TODO: CameraServer join always timeouts
            if self._p.is_alive():
                self._p.terminate()
                self._logger.info("Force terminate")
            else:
                break
        self._q_in.close()
        self._q_out.close()
        self._q_frames.close()
        self._logger.info("Shutdown finished")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
