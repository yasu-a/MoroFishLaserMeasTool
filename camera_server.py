import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Queue, Process
from queue import Empty

import cv2
import numpy as np

from app_logging import create_logger
from model.camera_spec import CameraSpec


@dataclass(slots=True)
class CaptureResult:
    frame: np.ndarray
    timestamp: float


class CameraReader:
    _logger = create_logger()

    def __init__(self, dev_id: int, video_spec: CameraSpec):
        self._dev_id = dev_id
        self._video_spec = video_spec
        self._cap: cv2.VideoCapture | None = None
        self.open()

    def get_device_name(self) -> int:
        return self._dev_id

    def get_configured_spec(self) -> CameraSpec:
        return self._video_spec

    def get_actual_spec(self) -> CameraSpec:
        self._check_open()
        return CameraSpec.from_video_capture(self._cap)

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def _check_open(self) -> None:
        if not self.is_open():
            raise RuntimeError(f"Camera {self._dev_id} reader is not open")

    def open(self):
        self._logger.info(f"Opening camera {self._dev_id}")

        # カメラを開く
        self._cap = cv2.VideoCapture(self._dev_id)
        if not self._cap.isOpened():
            self._logger.error(f"Failed to create camera with OpenCV: {self._dev_id}")
            self._cap = None

        # キャプチャ設定
        if self._cap is not None:
            if self.get_configured_spec() != self.get_actual_spec():
                self._logger.info(f"Setting width={self.get_configured_spec().width}")
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.get_configured_spec().width)
                self._logger.info(f"Setting height={self.get_configured_spec().height}")
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.get_configured_spec().height)
                self._logger.info(f"Setting fps={self.get_configured_spec().fps}")
                self._cap.set(cv2.CAP_PROP_FPS, self.get_configured_spec().fps)
            if not self._cap.isOpened():
                self._logger.error(f"Failed to set camera {self._dev_id} configuration")
                self._cap = None

        # 設定を検証
        if self._cap is not None:
            if self.get_configured_spec() != self.get_actual_spec():
                self._logger.error(
                    f"Camera {self._dev_id!r} configuration mismatch\n"
                    f"expected={self.get_configured_spec()!r}\n"
                    f"actual={self.get_actual_spec()!r}"
                )
                # self._cap = None

        if self._cap is not None:
            self._logger.info(f"Camera {self._dev_id!r} open")
        else:
            self._logger.error(f"Failed to open camera {self._dev_id}")

    def close(self) -> None:
        if self.is_open():
            self._logger.info(f"Closing camera {self._dev_id!r}")
            self._cap.release()
            self._cap = None
            self._logger.info(f"Camera {self._dev_id!r} closed")

    def read(self) -> CaptureResult | None:
        self._check_open()
        timestamp = time.time()
        flag, frame = self._cap.read()
        if flag:
            return CaptureResult(frame=frame, timestamp=timestamp)
        else:
            return None


class CameraInfo(ABC):
    @property
    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError()

    @property
    @abstractmethod
    def dev_id(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def configured_spec(self) -> CameraSpec:
        raise NotImplementedError()

    @property
    @abstractmethod
    def actual_spec(self) -> CameraSpec:
        raise NotImplementedError()


class UnavailableCameraInfo(CameraInfo):
    @property
    def is_available(self) -> bool:
        return False

    @property
    def dev_id(self) -> int:
        raise ValueError("Camera info is not available")

    @property
    def configured_spec(self) -> CameraSpec:
        raise ValueError("Camera info is not available")

    @property
    def actual_spec(self) -> CameraSpec:
        raise ValueError("Camera info is not available")


class AvailableCameraInfo(CameraInfo):
    def __init__(self, dev_id: int, configured_spec: CameraSpec, actual_spec: CameraSpec):
        self._dev_id = dev_id
        self._configured_spec = configured_spec
        self._actual_spec = actual_spec

    @property
    def is_available(self) -> bool:
        return True

    @property
    def dev_id(self) -> int:
        return self._dev_id

    @property
    def configured_spec(self) -> CameraSpec:
        return self._configured_spec

    @property
    def actual_spec(self) -> CameraSpec:
        return self._actual_spec


class CameraServer:
    _logger = create_logger()

    def __init__(self):
        self._configured_spec: CameraSpec | None = None
        self._q_in = Queue(maxsize=128)  # item: (command name, data)
        self._q_out = Queue(maxsize=128)  # item: CaptureResult
        self._q_frames = Queue(maxsize=128)
        self._p = Process(target=self._worker, args=(self._q_in, self._q_out, self._q_frames))
        self._p.start()

    @classmethod
    def _worker(cls, q_in: Queue, q_out: Queue, q_frames: Queue) -> None:
        cls._logger.info("Worker started")

        reader: CameraReader | None = None

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

                    dev_id, spec = data
                    assert isinstance(dev_id, int), f"Invalid dev_id: {dev_id}"
                    assert isinstance(spec, CameraSpec), f"Invalid video_spec: {spec}"
                    reader = CameraReader(dev_id, spec)
                elif command_name == "close":
                    if reader is not None and reader.is_open():
                        reader.close()
                        reader = None
                elif command_name == "get_camera_info":
                    if reader is not None and reader.is_open():
                        camera_info = AvailableCameraInfo(
                            dev_id=reader.get_device_name(),
                            configured_spec=reader.get_configured_spec(),
                            actual_spec=reader.get_actual_spec(),
                        )
                    else:
                        camera_info = UnavailableCameraInfo()
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

    def request_open(self, dev_id: int, spec: CameraSpec) -> None:
        self._q_in.put(("open", (dev_id, spec)))

    def request_close(self) -> None:
        self._q_in.put(("close", None))

    def request_camera_info(self) -> None:
        self._q_in.put(("get_camera_info", None))

    def get_requested_camera_info(self) -> CameraInfo | None:  # None if still no data available
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
        # noinspection DuplicatedCode
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
