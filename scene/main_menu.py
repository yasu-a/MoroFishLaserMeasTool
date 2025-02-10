import time
from datetime import datetime
from typing import cast

import repo.camera_param
import repo.distortion
import repo.global_config
import repo.image
import repo.laser_detection
import repo.laser_param
from camera_server import CaptureResult, CameraInfo
from core.tk.component.button import ButtonComponent
from core.tk.component.component import Component
from core.tk.component.label import LabelComponent
from core.tk.component.separator import SeparatorComponent
from core.tk.component.spacer import SpacerComponent
from core.tk.component.toast import Toast
from core.tk.dialog import MessageDialog, SelectImageItemDialog, SelectItemDialog
from core.tk.event import KeyEvent
from core.tk.global_state import get_app
from core.tk.key import Key
from core.tk.rendering import UIRenderingContext
from fps_counter import FPSCounterStat
from model.active_profile_names import ActiveProfileNames
from my_app import MyApplication
from repo import open_in_explorer
from scene.camera_param import CameraParamScene
from scene.distortion_correction import DistortionCorrectionScene
from scene.global_config import GlobalConfigScene
from scene.my_scene import MyScene
from scene.screenshot import ScreenShotScene


class MainScene(MyScene):
    def __init__(self):
        super().__init__(is_stationed=True)

        self._is_visible = True

    def load_event(self):
        # title
        self.add_component(LabelComponent(self, "Main Menu", bold=True))
        self.add_component(LabelComponent(self, "TAB to show/hide this menu"))

        self.add_component(SeparatorComponent(self))

        # camera spec
        self.add_component(LabelComponent(self, "CAMERA SPEC"))
        self.add_component(LabelComponent(self, "", name="l-camera-spec"))

        self.add_component(SpacerComponent(self))

        # fps
        self.add_component(LabelComponent(self, "FPS"))
        self.add_component(LabelComponent(self, "", name="l-fps"))

        self.add_component(SpacerComponent(self))

        # frame timestamp
        self.add_component(LabelComponent(self, "FRAME TIMESTAMP"))
        self.add_component(LabelComponent(self, "", name="l-timestamp"))

        self.add_component(SeparatorComponent(self))

        # recording status
        self.add_component(LabelComponent(self, "RECORDING STAT"))
        self.add_component(LabelComponent(self, "", name="l-record"))

        self.add_component(SeparatorComponent(self))

        # active profiles
        self.add_component(LabelComponent(self, "ACTIVE PROFILES"))
        self.add_component(LabelComponent(self, "", name="l-profile"))
        self.add_component(ButtonComponent(self, "Edit", name="b-select-profile"))

        self.add_component(SeparatorComponent(self))

        # buttons
        self.add_component(ButtonComponent(self, "Global Config", name="b-global-config"))
        self.add_component(SpacerComponent(self))
        self.add_component(
            ButtonComponent(self, "Create Distortion Profile", name="b-distortion")
        )
        self.add_component(
            ButtonComponent(self, "Create Camera Parameter Profile", name="b-camera-param")
        )
        self.add_component(
            ButtonComponent(self, "Create Laser Parameter Profile", name="b-laser-param")
        )
        self.add_component(
            ButtonComponent(self, "Create Laser Extraction Profile", name="b-laser-ext")
        )
        self.add_component(SpacerComponent(self))
        self.add_component(ButtonComponent(self, "Screenshot", name="b-save-image"))
        self.add_component(SpacerComponent(self))
        self.add_component(
            ButtonComponent(self, "Open Data Folder in Explorer", name="b-open-data-folder")
        )
        self.add_component(SpacerComponent(self))
        self.add_component(ButtonComponent(self, "Exit", name="b-exit"))

    def update(self):
        app = cast(MyApplication, get_app())

        camera_info: CameraInfo = app.camera_info
        text = []
        if camera_info.is_available:
            if camera_info.actual_spec != camera_info.configured_spec:
                text.append("WARNING: CAMERA SPEC NOT SYNCHRONIZED")
            c, a = camera_info.configured_spec, camera_info.actual_spec
            text.append(f"        Width Height    FPS")
            text.append(f"Config {c.width:>6} {c.height:>6} {c.fps:>6.2f}")
            text.append(f"Actual {a.width:>6} {a.height:>6} {a.fps:>6.2f}")
        else:
            text.append("(NO CAMERA CONNECTED)")
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

        active_profile_names: ActiveProfileNames = repo.global_config.get().active_profile_names
        text = [
            f"Distortion: {active_profile_names.distortion_profile_name or '(NONE)'}",
            f"Camera: {active_profile_names.camera_param_profile_name or '(NONE)'}",
            f"Laser: {active_profile_names.laser_param_profile_name or '(NONE)'}",
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

    def render_ui(self, ctx: UIRenderingContext) -> UIRenderingContext:
        if not self._is_visible:
            return ctx
        return super().render_ui(ctx)

    def key_event(self, event: KeyEvent) -> bool:
        if super().key_event(event):
            return True
        if event.down:
            if event.key == Key.TAB:
                self._is_visible = not self._is_visible
                return True
        return False

    def _show_select_profile_dialog(self, profile_type: str) -> None:
        def callback(name: str | None) -> None:
            get_app().close_dialog()

            if name is None:
                return

            global_config = repo.global_config.get()
            if profile_type == "Distortion":
                global_config.active_profile_names.distortion_profile_name = name
            elif profile_type == "Camera Parameter":
                global_config.active_profile_names.camera_param_profile_name = name
            elif profile_type == "Laser Parameter":
                global_config.active_profile_names.laser_param_profile_name = name
            elif profile_type == "Laser Detection":
                global_config.active_profile_names.laser_detection_profile_name = name
            else:
                assert False, profile_type
            repo.global_config.put(global_config)

            get_app().make_toast(
                Toast(
                    self,
                    "info",
                    f"{profile_type} Profile Selected: {name}",
                )
            )

        if profile_type == "Distortion":
            items = repo.distortion.list_names()
        elif profile_type == "Camera Parameter":
            items = repo.camera_param.list_names()
        elif profile_type == "Laser Parameter":
            items = repo.laser_param.list_names()
        elif profile_type == "Laser Detection":
            items = repo.laser_detection.list_names()
        else:
            assert False, profile_type

        if items:
            get_app().show_dialog(
                SelectItemDialog(
                    title=f"Select {profile_type} Profile",
                    items=items,
                    callback=callback,
                )
            )
        else:
            get_app().show_dialog(
                MessageDialog(
                    is_error=True,
                    message=f"No {profile_type} found",
                    callback=lambda _: get_app().close_dialog(),
                )
            )

    def _show_profile_select_or_deselect_dialog(self, profile_type: str) -> None:
        def callback(item: str | None) -> None:
            get_app().close_dialog()

            if item is None:
                return

            if item.startswith("Deselect: "):
                global_config = repo.global_config.get()
                if profile_type == "Distortion":
                    global_config.active_profile_names.distortion_profile_name = None
                elif profile_type == "Camera Parameter":
                    global_config.active_profile_names.camera_param_profile_name = None
                elif profile_type == "Laser Parameter":
                    global_config.active_profile_names.laser_param_profile_name = None
                elif profile_type == "Laser Detection":
                    global_config.active_profile_names.laser_detection_profile_name = None
                else:
                    assert False, profile_type

                repo.global_config.put(global_config)

                get_app().make_toast(
                    Toast(
                        self,
                        "info",
                        f"{profile_type} Profile Deselected",
                    )
                )
            else:
                self._show_select_profile_dialog(profile_type)

        global_config = repo.global_config.get()
        if profile_type == "Distortion":
            current = global_config.active_profile_names.distortion_profile_name
        elif profile_type == "Camera Parameter":
            current = global_config.active_profile_names.camera_param_profile_name
        elif profile_type == "Laser Parameter":
            current = global_config.active_profile_names.laser_param_profile_name
        elif profile_type == "Laser Detection":
            current = global_config.active_profile_names.laser_detection_profile_name
        else:
            assert False, profile_type

        if current is None:
            self._show_select_profile_dialog(profile_type)
        else:
            get_app().show_dialog(
                MessageDialog(
                    message=f"What do you want to do for {profile_type}",
                    buttons=(
                        "Select another",
                        f"Deselect: '{current}'",
                    ),
                    callback=callback,
                )
            )

    def _show_select_profile_to_edit_dialog(self) -> None:
        def callback(item: str | None) -> None:
            get_app().close_dialog()

            if item is None:
                return

            self._show_profile_select_or_deselect_dialog(item)

        get_app().show_dialog(
            SelectItemDialog(
                title="What profile do you edit?",
                items=["Distortion", "Camera Parameter", "Laser Parameter", "Laser Detection"],
                callback=callback,
            )
        )

    def _on_button_triggered(self, sender: Component) -> None:
        if isinstance(sender, ButtonComponent):
            if sender.get_name() == "b-global-config":
                get_app().move_to(GlobalConfigScene())
                pass
            if sender.get_name() == "b-select-profile":
                self._show_select_profile_to_edit_dialog()
                return
            if sender.get_name() == "b-distortion":
                get_app().move_to(DistortionCorrectionScene())
                return
            if sender.get_name() == "b-camera-param":
                def callback(item: str | None) -> None:
                    get_app().close_dialog()
                    if item is not None:
                        image = repo.image.get(item)
                        get_app().move_to(
                            CameraParamScene(
                                image=image,
                            )
                        )

                get_app().show_dialog(
                    SelectImageItemDialog(
                        title="Select Image for Camera Calibration",
                        items=repo.image.list_names(),
                        callback=callback,
                        image_getter=lambda name: repo.image.get(name).data,
                    )
                )
                return
            if sender.get_name() == "b-laser-param":
                # Implement laser parameter scene
                pass
            if sender.get_name() == "b-laser-ext":
                # Implement laser extraction scene
                pass
            if sender.get_name() == "b-save-image":
                get_app().move_to(ScreenShotScene())
                pass
            if sender.get_name() == "b-open-data-folder":
                open_in_explorer()
                pass
            if sender.get_name() == "b-exit":
                def callback(button_name: str):
                    if button_name == "QUIT":
                        get_app().send_signal("quit")
                    get_app().close_dialog()

                get_app().show_dialog(
                    MessageDialog(
                        is_error=True,
                        message="Are you sure to quit application?",
                        buttons=("CANCEL", "QUIT"),
                        callback=callback,
                    )
                )
                return
