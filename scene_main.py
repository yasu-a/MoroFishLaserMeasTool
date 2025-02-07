import time
from datetime import datetime
from typing import TYPE_CHECKING

from active_profile_names import ActiveProfileNames
from app_tk.component.button import ButtonComponent
from app_tk.component.component import Component
from app_tk.component.label import LabelComponent
from app_tk.component.spacer import SpacerComponent
from app_tk.event import KeyEvent
from app_tk.key import Key
from app_tk.rendering import RenderingContext
from camera_server import CameraInfo, CaptureResult
from fps_counter import FPSCounterStat
from scene_base import MyScene
from scene_camera_param import CameraParamScene
from scene_distortion_correction import DistortionCorrectionScene
from scene_save_image import SaveImageScene
from scene_select_profile_menu import SelectProfileMenuScene

if TYPE_CHECKING:
    from app_tk.app import Application


class MainScene(MyScene):
    def __init__(self, app: "Application"):
        super().__init__(app, is_stationed=True)

        self._is_visible = True

    def load_event(self):
        self.add_component(LabelComponent, "Main Menu", bold=True)
        self.add_component(LabelComponent, "TAB to show/hide this menu")

        self.add_component(SpacerComponent)

        self.add_component(LabelComponent, "Camera spec:")
        self.add_component(LabelComponent, "", name="camera_spec")

        self.add_component(SpacerComponent)

        self.add_component(LabelComponent, "FPS:")
        self.add_component(LabelComponent, "", name="fps")

        self.add_component(SpacerComponent)

        self.add_component(LabelComponent, "Frame timestamp:")
        self.add_component(LabelComponent, "", name="timestamp")

        self.add_component(SpacerComponent)

        self.add_component(LabelComponent, "", name="profile")

        self.add_component(SpacerComponent)

        self.add_component(LabelComponent, "", name="record")

        self.add_component(ButtonComponent, "Select Profile", name="b-select-profile")
        self.add_component(ButtonComponent, "Distortion Corrections", name="b-distortion")
        self.add_component(ButtonComponent, "Camera Parameters", name="b-camera-param")
        self.add_component(ButtonComponent, "Laser Parameters", name="b-laser-param")
        self.add_component(ButtonComponent, "Laser Extraction", name="b-laser-ext")
        self.add_component(ButtonComponent, "Save Screenshot", name="b-save-image")
        self.add_component(SpacerComponent)
        self.add_component(ButtonComponent, "Exit", name="b-exit")

    def update(self):
        camera_info: CameraInfo = self.get_app().camera_info
        text = []
        if camera_info is not None:
            if camera_info.actual_spec != camera_info.configured_spec:
                text.append("WARNING: Camera settings are not synchronized")
            text.append(f"Config: {camera_info.configured_spec}")
            text.append(f"Actual: {camera_info.actual_spec}")
        self.find_component(LabelComponent, "camera_spec").set_text("\n".join(text))

        fps_stat: FPSCounterStat = self.get_app().fps_counter.get_stat()
        text = [
            f"Min: {fps_stat.min_fps:6.2f}",
            f"Ave: {fps_stat.mean_fps:6.2f}",
            f"Max: {fps_stat.max_fps:6.2f}",
        ]
        self.find_component(LabelComponent, "fps").set_text("\n".join(text))

        last_capture: CaptureResult | None = self.get_app().last_capture
        text = []
        if last_capture is not None:
            text.append(f"{datetime.fromtimestamp(last_capture.timestamp)!s}")
            text.append(f"{datetime.fromtimestamp(time.time())!s}")
        self.find_component(LabelComponent, "timestamp").set_text("\n".join(text))

        active_profile_names: ActiveProfileNames = self.get_app().active_profile_names
        text = [
            f"Distortion: {active_profile_names.distortion_profile_name or '(NONE)'}",
            f"Camera: {active_profile_names.camera_profile_name or '(NONE)'}",
            f"Laser: {active_profile_names.laser_profile_name or '(NONE)'}",
        ]
        self.find_component(LabelComponent, "profile").set_text("\n".join(text))

        is_recording: bool = self.get_app().is_recording
        last_recording_queue_count = self.get_app().last_recording_queue_count
        text = []
        if is_recording:
            text.append("RECORDING")
            if last_recording_queue_count is not None:
                text.append(f"Captured frames on queue {last_recording_queue_count:3d}")
        self.find_component(LabelComponent, "record").set_text("\n".join(text))

    def render_ui(self, rendering_ctx: RenderingContext) -> RenderingContext:
        if not self._is_visible:
            return rendering_ctx
        return super().render_ui(rendering_ctx)

    def key_event(self, event: KeyEvent) -> bool:
        if super().key_event(event):
            return True
        if event.down:
            if event.key == Key.TAB:
                self._is_visible = not self._is_visible
                return True
        return False

    def on_button_triggered(self, sender: Component) -> None:
        if isinstance(sender, ButtonComponent):
            if sender.get_name() == "b-select-profile":
                self.get_app().move_to(SelectProfileMenuScene(self.get_app()))
                return
            if sender.get_name() == "b-distortion":
                self.get_app().move_to(DistortionCorrectionScene(self.get_app()))
                return
            if sender.get_name() == "b-camera-param":
                self.get_app().move_to(CameraParamScene(self.get_app()))
                pass
            if sender.get_name() == "b-laser-param":
                # Implement laser parameter scene
                pass
            if sender.get_name() == "b-laser-ext":
                # Implement laser extraction scene
                pass
            if sender.get_name() == "b-save-image":
                self.get_app().move_to(SaveImageScene(self.get_app()))
                pass
            if sender.get_name() == "b-exit":
                self.get_app().send_signal("quit")
                return
