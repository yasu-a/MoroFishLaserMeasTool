import copy
import time

import cv2

import repo.global_config
from active_profile_names import ActiveProfileNames
from app_logging import create_logger
from camera_server import CameraServer, CaptureResult, CameraInfo, \
    UnavailableCameraInfo
from core.tk.app import Application
from core.tk.event import KeyEvent
from core.tk.key import Key
from fps_counter import FPSCounter
from record_server import RecordServer
from scene.main_menu import MainScene


class _CameraConfigObserver:
    @classmethod
    def _get_hash_of_camera_config(cls):
        global_config = repo.global_config.get()
        camera_dev_id = copy.deepcopy(global_config.camera_dev_id)
        camera_spec = copy.deepcopy(global_config.camera_spec)
        return hash((camera_dev_id, camera_spec))

    def __init__(self):
        self._hash = self._get_hash_of_camera_config()
        self._mtime = repo.global_config.get_mtime()

    def is_modified(self) -> bool:
        old_mtime = self._mtime
        current_mtime = repo.global_config.get_mtime()
        if old_mtime == current_mtime:
            return False
        else:
            self._mtime = current_mtime

        old_hash = self._hash
        current_hash = self._get_hash_of_camera_config()
        self._hash = current_hash
        return old_hash != current_hash


class MyApplication(Application):
    _logger = create_logger()

    def __init__(self, camera_server: CameraServer, record_server: RecordServer):
        super().__init__()

        self.camera_server = camera_server
        self.record_server = record_server

        self.camera_info: CameraInfo = UnavailableCameraInfo()
        self._camera_config_observer = _CameraConfigObserver()

        self.last_capture: CaptureResult | None = None
        self.fps_counter = FPSCounter()
        self.is_recording: bool = False
        self.last_recording_queue_count: int | None = None
        self.active_profile_names = ActiveProfileNames()

    def canvas_size_hint(self) -> tuple[int, int]:
        if self.last_capture is not None:
            return self.last_capture.frame.shape[1], self.last_capture.frame.shape[0]
        else:
            return 640, 480

    def key_event(self, event: KeyEvent) -> bool:
        if super().key_event(event):
            return True

        # FIXME: SCENEごとに管理
        if event.down:
            if event.key == Key.SPACE:
                self.is_recording = not self.is_recording
                if self.is_recording:
                    self.record_server.request_record_begin(
                        "./__out__.mp4",
                        fps=self.camera_info.actual_spec.fps,
                        width=self.camera_info.actual_spec.width,
                        height=self.camera_info.actual_spec.height,
                    )
                else:
                    self.record_server.request_record_end()
                    print("Recording stopped")
                return True

    def reflect_camera_config(self):
        global_config = repo.global_config.get()

        self.camera_server.request_open(
            dev_id=global_config.camera_dev_id,
            spec=global_config.camera_spec
        )
        self.camera_server.request_camera_info()

        while True:
            info = self.camera_server.get_requested_camera_info()
            if info is None:
                continue
            break
        self.camera_info = info

        if info.is_available:
            if info.actual_spec != info.configured_spec:
                self._logger.error(
                    f"Camera {global_config.camera_dev_id!r} configuration mismatch\n"
                    f"expected={global_config.camera_spec!r}\n"
                    f"actual={info.actual_spec!r}"
                )

    def _is_camera_config_modified(self) -> bool:
        return self._camera_config_observer.is_modified()

    def loop(self):
        cv2.namedWindow("win")
        self._mouse_handler.register_callback("win")

        self.reflect_camera_config()

        self.move_to(MainScene(self))

        while True:
            if self._is_camera_config_modified():
                self.reflect_camera_config()

            capture_results = self.camera_server.get_all_frames()
            for capture_result in capture_results:
                self.fps_counter.add(capture_result.timestamp)
            if self.is_recording:
                self.record_server.put_frames([
                    capture_result.frame
                    for capture_result in capture_results
                ])

            if capture_results:
                self.last_capture = capture_results[-1]
            elif self.last_capture is not None and self.last_capture.timestamp > time.time() - 1:
                pass
            else:
                self.last_capture = None

            if self.is_recording:
                self.last_recording_queue_count = self.record_server.get_queue_count()
            else:
                self.last_recording_queue_count = None

            if self._check_signal("quit"):
                break

            self.update()
            im_out = self.render()
            cv2.imshow("win", im_out)
            self.do_event()
