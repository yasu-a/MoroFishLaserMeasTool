import time
from datetime import datetime
from typing import cast

from active_profile_names import ActiveProfileNames
from camera_server import CaptureResult, CameraInfo
from core.tk.component.button import ButtonComponent
from core.tk.component.component import Component
from core.tk.component.label import LabelComponent
from core.tk.component.spacer import SpacerComponent
from core.tk.event import KeyEvent
from core.tk.global_state import get_app
from core.tk.key import Key
from core.tk.rendering import UIRenderingContext, Canvas
from fps_counter import FPSCounterStat
from my_app import MyApplication
from scene.camera_param import CameraParaSelectImageDelegate
from scene.distortion_correction import DistortionCorrectionScene
from scene.global_config import GlobalConfigScene
from scene.my_scene import MyScene
from scene.save_image import SaveImageScene
from scene.select_image_item import SelectImageItemScene
from scene.select_profile_menu import SelectProfileMenuScene


class MainScene(MyScene):
    def __init__(self):
        super().__init__(is_stationed=True)

        self._is_visible = True

    def load_event(self):
        self.add_component(LabelComponent(self, "Main Menu", bold=True))
        self.add_component(LabelComponent(self, "TAB to show/hide this menu"))

        self.add_component(SpacerComponent(self))

        self.add_component(LabelComponent(self, "Camera spec"))
        self.add_component(LabelComponent(self, "", name="l-camera-spec"))

        self.add_component(SpacerComponent(self))

        self.add_component(LabelComponent(self, "FPS"))
        self.add_component(LabelComponent(self, "", name="l-fps"))

        self.add_component(SpacerComponent(self))

        self.add_component(LabelComponent(self, "Frame Timestamp"))
        self.add_component(LabelComponent(self, "", name="l-timestamp"))

        self.add_component(SpacerComponent(self))

        self.add_component(LabelComponent(self, "", name="l-profile"))

        self.add_component(SpacerComponent(self))

        self.add_component(LabelComponent(self, "", name="l-record"))

        self.add_component(ButtonComponent(self, "Global Config", name="b-global-config"))
        self.add_component(ButtonComponent(self, "Select Profile", name="b-select-profile"))
        self.add_component(ButtonComponent(self, "Distortion Corrections", name="b-distortion"))
        self.add_component(ButtonComponent(self, "Camera Parameters", name="b-camera-param"))
        self.add_component(ButtonComponent(self, "Laser Parameters", name="b-laser-param"))
        self.add_component(ButtonComponent(self, "Laser Extraction", name="b-laser-ext"))
        self.add_component(ButtonComponent(self, "Save Screenshot", name="b-save-image"))
        self.add_component(SpacerComponent(self))
        self.add_component(ButtonComponent(self, "Exit", name="b-exit"))

    def update(self):
        app = cast(MyApplication, get_app())

        camera_info: CameraInfo = app.camera_info
        text = []
        if camera_info.is_available:
            if camera_info.actual_spec != camera_info.configured_spec:
                text.append("[!!!] WARNING: CONFIGURED CAMERA SPEC NOT SYNCHRONIZED [!!!]")
            text.append(f"Config: {camera_info.configured_spec}")
            text.append(f"Actual: {camera_info.actual_spec}")
        else:
            text.append("NO CAMERA CONNECTED")
        self.find_component(LabelComponent, "l-camera-spec").set_text("\n".join(text))

        fps_stat: FPSCounterStat = app.fps_counter.get_stat()
        text = [
            f"Min-Average-Max: {fps_stat.min_fps:.2f}-{fps_stat.max_fps:.2f}-{fps_stat.max_fps:.2f}"
        ]
        self.find_component(LabelComponent, "l-fps").set_text("\n".join(text))

        last_capture: CaptureResult | None = app.last_capture
        text = []
        if last_capture is not None:
            text.append(f"{datetime.fromtimestamp(last_capture.timestamp)!s}")
            text.append(f"{datetime.fromtimestamp(time.time())!s}")
        self.find_component(LabelComponent, "l-timestamp").set_text("\n".join(text))

        active_profile_names: ActiveProfileNames = app.active_profile_names
        text = [
            f"Distortion: {active_profile_names.distortion_profile_name or '(NONE)'}",
            f"Camera: {active_profile_names.camera_profile_name or '(NONE)'}",
            f"Laser: {active_profile_names.laser_profile_name or '(NONE)'}",
        ]
        self.find_component(LabelComponent, "l-profile").set_text("\n".join(text))

        is_recording: bool = app.is_recording
        last_recording_queue_count = app.last_recording_queue_count
        text = []
        if is_recording:
            text.append("RECORDING")
            if last_recording_queue_count is not None:
                text.append(f"Captured frames on queue {last_recording_queue_count:3d}")
        self.find_component(LabelComponent, "l-record").set_text("\n".join(text))

    def render_ui(self, canvas: Canvas, ctx: UIRenderingContext) -> UIRenderingContext:
        if not self._is_visible:
            return ctx
        return super().render_ui(canvas, ctx)

    def key_event(self, event: KeyEvent) -> bool:
        if super().key_event(event):
            return True
        if event.down:
            if event.key == Key.TAB:
                self._is_visible = not self._is_visible
                return True
        return False

    def _on_button_triggered(self, sender: Component) -> None:
        if isinstance(sender, ButtonComponent):
            if sender.get_name() == "b-global-config":
                get_app().move_to(GlobalConfigScene())
                pass
            if sender.get_name() == "b-select-profile":
                get_app().move_to(SelectProfileMenuScene())
                return
            if sender.get_name() == "b-distortion":
                get_app().move_to(DistortionCorrectionScene())
                return
            if sender.get_name() == "b-camera-param":
                get_app().move_to(
                    SelectImageItemScene(
                        CameraParaSelectImageDelegate()
                    )
                )
                pass
            if sender.get_name() == "b-laser-param":
                # Implement laser parameter scene
                pass
            if sender.get_name() == "b-laser-ext":
                # Implement laser extraction scene
                pass
            if sender.get_name() == "b-save-image":
                get_app().move_to(SaveImageScene())
                pass
            if sender.get_name() == "b-exit":
                get_app().send_signal("quit")
                return
