import time

import cv2

from active_profile_names import ActiveProfileNames
from app_tk.app import Application
from app_tk.event import KeyEvent
from app_tk.key import Key
from camera_server import CameraServer, CameraInfo, CaptureResult, VideoSpec
from fps_counter import FPSCounter
from record_server import RecordServer
from scene_main import MainScene


class MyApplication(Application):
    def __init__(self, camera_server: CameraServer, record_server: RecordServer):
        super().__init__()
        self.camera_server = camera_server
        self.record_server = record_server
        self.camera_info: CameraInfo | None = None
        self.last_capture: CaptureResult | None = None
        self.fps_counter = FPSCounter()
        self.ui_color = (0, 180, 0)
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

    def loop(self):
        self.camera_server.request_open(0, VideoSpec(width=1920, height=1080, fps=60))
        self.camera_server.request_camera_info()

        cv2.namedWindow("win")
        self._mouse_handler.register_callback("win")

        while True:
            info = self.camera_server.get_requested_camera_info()
            if info is None:
                continue
            if info.actual_spec != info.configured_spec:
                print("INVALID CAMERA SPEC")
                print("CONFIGURED SPEC")
                print(info.configured_spec)
                print("ACTUAL SPEC")
                print(info.actual_spec)
            self.camera_info = info
            break

        self.move_to(MainScene(self))

        while True:
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
