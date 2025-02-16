import time
from datetime import datetime
from typing import cast

import repo.distortion
import repo.global_config
import repo.image
import repo.raw_image
import repo.video
from camera_server import CameraInfo, CaptureResult
from core.tk.component.button import ButtonComponent
from core.tk.component.component import Component
from core.tk.component.label import LabelComponent
from core.tk.component.separator import SeparatorComponent
from core.tk.component.spacer import SpacerComponent
from core.tk.component.toast import Toast
from core.tk.dialog import InputNameDialog
from core.tk.event import KeyEvent
from core.tk.global_state import get_app
from core.tk.key import Key
from fps_counter import FPSCounterStat
from model.distortion import DistortionProfile
from my_app import MyApplication
from scene.my_scene import MyScene
from util import video_service


class RecordingScene(MyScene):
    def __init__(self, distortion_profile: DistortionProfile | None):
        super().__init__()
        self._distortion_profile = distortion_profile
        self._is_recording_end_requested = False

    def load_event(self):
        self.add_component(LabelComponent(self, "Save Image", bold=True))
        self.add_component(SpacerComponent(self))
        self.add_component(LabelComponent(self, "", name="l-distortion-info"))
        self.add_component(ButtonComponent(self, "Record Video <SPACE>", name="b-record"))
        self.add_component(SeparatorComponent(self))
        self.add_component(LabelComponent(self, "", name="l-recording-info"))
        self.add_component(SeparatorComponent(self))
        self.add_component(ButtonComponent(self, "Back", name="b-back"))

        if self._distortion_profile is None:
            self.find_component(LabelComponent, "Distortion Profile").set_text(
                "Distortion correction: (NOT SET)"
            )
            get_app().make_toast(
                Toast(
                    self,
                    "error",
                    "Distortion correction profile is not set",
                )
            )
        else:
            self.find_component(LabelComponent, "l-distortion-info").set_text(
                f"Distortion correction: {self._distortion_profile.name}"
            )

    def key_event(self, event: KeyEvent) -> bool:
        if event.down:
            if event.key == Key.SPACE:
                self.find_component(ButtonComponent, "b-record").click()
                return True
        return super().key_event(event)

    def _start_recording(self) -> None:
        app: MyApplication = cast(MyApplication, get_app())
        try:
            app.start_recording(self._distortion_profile)
        except ValueError as e:
            get_app().make_toast(
                Toast(
                    self,
                    "error",
                    e.args[0],
                )
            )
            return

        get_app().make_toast(
            Toast(
                self,
                "info",
                f"Recording started",
            )
        )

    def _stop_recording(self) -> None:
        app: MyApplication = cast(MyApplication, get_app())
        app.stop_recording()
        self._is_recording_end_requested = True

    def _is_recording_finished(self) -> bool:
        if not self._is_recording_end_requested:
            return False
        app: MyApplication = cast(MyApplication, get_app())
        if app.is_recording_done():
            self._is_recording_end_requested = False
            return True

    def _show_name_input_dialog(self):
        def validator(name: str) -> str | None:
            if name == "":
                return "Name cannot be empty"
            if name.strip() != name:
                return "File name cannot contain leading or trailing spaces"
            return None

        def already_exist_checker(name: str) -> bool:
            return repo.video.exists(name)

        def callback(name: str | None):
            get_app().close_dialog()
            if name is None:
                get_app().make_toast(
                    Toast(
                        self,
                        "error",
                        f"Recording discarded",
                    )
                )
                return

            video_service.create_video_from_file(repo.video.get_temp_video_path(), name)
            get_app().make_toast(
                Toast(
                    self,
                    "info",
                    f"Recording saved: {name}",
                )
            )

        get_app().show_dialog(
            InputNameDialog(
                title="Input name of video",
                validator=validator,
                already_exist_checker=already_exist_checker,
                callback=callback,
            )
        )

    def update(self):
        # 録画が完全に終了したら保存ダイアログを表示する
        if self._is_recording_end_requested:
            if self._is_recording_finished():
                self._show_name_input_dialog()
            return

        # 情報UIを更新する
        app: MyApplication = cast(MyApplication, get_app())
        text = []
        # - Camera
        camera_info: CameraInfo = app.camera_info
        text.append("[CAMERA INFO]")
        if camera_info.is_available:
            if camera_info.actual_spec != camera_info.configured_spec:
                text.append("WARNING: CAMERA SPEC NOT SYNCHRONIZED")
            c, a = camera_info.configured_spec, camera_info.actual_spec
            text.append(f"        Width Height    FPS")
            text.append(f"Config {c.width:>6} {c.height:>6} {c.fps:>6.2f}")
            text.append(f"Actual {a.width:>6} {a.height:>6} {a.fps:>6.2f}")
        else:
            text.append("(NO CAMERA CONNECTED)")
        # - Capture State
        last_capture: CaptureResult | None = app.last_capture
        if last_capture is not None:
            text.append("[CAPTURE STATUS]")
            t_capture = datetime.fromtimestamp(last_capture.timestamp)
            t_now = datetime.fromtimestamp(time.time())
            text.append(f"{t_capture!s}")
            text.append(f"{t_now!s}")
            text.append(f"Time lag: {(t_now - t_capture).total_seconds():7.3f} sec.")
        # - FPS
        fps_stat: FPSCounterStat = app.fps_counter.get_stat()
        text.append("[FPS]")
        text.append(
            f"Min-Average-Max: {fps_stat.min_fps:.2f}-{fps_stat.max_fps:.2f}-{fps_stat.max_fps:.2f}"
        )
        # - Recording state
        text.append("[RECORDING STATUS]")
        if app.is_recording():
            text.append("Recording")
        if not app.is_recording_done():
            text.append("Writing to storage")
        if app.last_recording_queue_count is not None:
            text.append(f"Captured frames on queue: {app.last_recording_queue_count:3d}")
        self.find_component(LabelComponent, "l-recording-info").set_text("\n".join(text))

    def _on_button_triggered(self, sender: Component) -> None:
        if sender.get_name() == "b-record":
            app: MyApplication = cast(MyApplication, get_app())
            if app.is_recording_done():
                self._start_recording()
            elif app.is_recording():
                self._stop_recording()
            return
        if sender.get_name() == "b-back":
            get_app().move_back()
            return
        super()._on_button_triggered(sender)
