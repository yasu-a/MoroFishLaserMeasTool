import copy
import itertools
import time

import cv2

import repo.distortion
import repo.global_config
from app_logging import create_logger
from camera_server import CameraServer, CaptureResult, CameraInfo, \
    UnavailableCameraInfo
from core.tk.app import Application
from core.tk.event import KeyEvent
from core.tk.key import Key
from fps_counter import FPSCounter
from model.distortion import DistortionProfile
from record_server import RecordServer


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
        self.last_capture_undistort: CaptureResult | None = None
        self.fps_counter = FPSCounter()
        self.is_recording: bool = False
        self.last_recording_queue_count: int | None = None

        self._distortion_profile: DistortionProfile | None = None
        self._last_time_distortion_profile_get = None

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

    def get_distortion_profile(self) -> DistortionProfile:
        now = time.monotonic()
        if self._last_time_distortion_profile_get is None \
                or now - self._last_time_distortion_profile_get >= 1:  # TODO: efficient file system observation
            self._last_time_distortion_profile_get = now
            name = repo.global_config.get().active_profile_names.distortion_profile_name
            try:
                self._distortion_profile = repo.distortion.get(name)
            except FileNotFoundError:
                self._distortion_profile = None
        return self._distortion_profile

    def loop(self):
        cv2.namedWindow("win")
        self._mouse_handler.register_callback("win")

        self.reflect_camera_config()

        for loop_count in itertools.count():
            t_loop_start = time.monotonic()

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

                distortion_profile = self.get_distortion_profile()
                if distortion_profile is None:
                    self.last_capture_undistort = None
                else:
                    self.last_capture_undistort = CaptureResult(
                        frame=distortion_profile.params.undistort(self.last_capture.frame),
                        timestamp=self.last_capture.timestamp,
                    )
            elif self.last_capture is not None and self.last_capture.timestamp > time.time() - 1:
                pass
            else:
                self.last_capture = None
                self.last_capture_undistort = None

            if self.is_recording:
                self.last_recording_queue_count = self.record_server.get_queue_count()
            else:
                self.last_recording_queue_count = None

            if self._check_signal("quit"):
                break

            self.update()
            im_out = self.render()
            cv2.imshow("win", im_out)

            t_loop_end = time.monotonic()
            delay = int(((1 / self._params.rendering_fps) - (t_loop_end - t_loop_start)) * 1000)
            self.do_event(delay)

            if (loop_count - self._params.rendering_fps) % (self._params.rendering_fps * 30) == 0:
                from pprint import pformat
                self._logger.debug(pformat(self._params.char_printer.get_cache_stat()))
