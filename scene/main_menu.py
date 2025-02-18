import time
from datetime import datetime
from typing import cast

import repo.camera_param
import repo.distortion
import repo.global_config
import repo.image
import repo.laser_detection
import repo.laser_param
import repo.video
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
from fps_counter import FPSCounterStat
from model.active_profile_names import ActiveProfileNames
from model.camera_param import CameraParamProfile
from model.distortion import DistortionProfile
from model.laser_detection import LaserDetectionProfile
from model.laser_param import LaserParamProfile
from my_app import MyApplication
from repo import open_in_explorer
from scene.camera_param import CameraParamScene
from scene.distortion import DistortionCorrectionScene
from scene.global_config import GlobalConfigScene
from scene.laser_detection import LaserDetectionScene
from scene.laser_param import LaserParamScene
from scene.meas import MeasScene
from scene.my_scene import MyScene
from scene.recording import RecordingScene
from scene.screenshot import ScreenShotScene
from scene.stitch_scene import StitchingScene


class MainScene(MyScene):
    def __init__(self):
        super().__init__(is_stationed=True)

    def load_event(self):
        # title
        self.add_component(LabelComponent(self, "Main Menu", bold=True))
        self.add_component(LabelComponent(self, "TAB to show or hide this UI"))

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
            ButtonComponent(self, "Create Laser Detection Profile", name="b-laser-detection")
        )
        self.add_component(SpacerComponent(self))
        self.add_component(ButtonComponent(self, "Screenshot", name="b-save-image"))
        self.add_component(ButtonComponent(self, "Measurement", name="b-meas"))
        self.add_component(SpacerComponent(self))
        self.add_component(ButtonComponent(self, "Recording", name="b-recording"))
        self.add_component(ButtonComponent(self, "Stitching", name="b-stitch"))
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
            f"Camera Param: {active_profile_names.camera_param_profile_name or '(NONE)'}",
            f"Laser Param: {active_profile_names.laser_param_profile_name or '(NONE)'}",
            f"Laser Detection: {active_profile_names.laser_detection_profile_name or '(NONE)'}",
        ]
        self.find_component(LabelComponent, "l-profile").set_text("\n".join(text))

    def key_event(self, event: KeyEvent) -> bool:
        return super().key_event(event)

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

    @classmethod
    def get_distortion_profile(cls) -> DistortionProfile:
        # TODO: move to MyApplication and replace similar procedures
        name = repo.global_config.get().active_profile_names.distortion_profile_name
        if name is None:
            raise ValueError("Distortion profile is not selected")
        try:
            return repo.distortion.get(name)
        except FileNotFoundError:
            raise ValueError(f"Distortion profile '{name}' not found")

    @classmethod
    def get_camera_param_profile(cls) -> CameraParamProfile:
        # TODO: move to MyApplication and replace similar procedures
        name = repo.global_config.get().active_profile_names.camera_param_profile_name
        if name is None:
            raise ValueError("Camera parameter profile is not selected")
        try:
            return repo.camera_param.get(name)
        except FileNotFoundError:
            raise ValueError(f"Camera parameter profile '{name}' not found")

    @classmethod
    def get_laser_param_profile(cls) -> LaserParamProfile:
        # TODO: move to MyApplication and replace similar procedures
        name = repo.global_config.get().active_profile_names.laser_param_profile_name
        if name is None:
            raise ValueError("Laser parameter profile is not selected")
        try:
            return repo.laser_param.get(name)
        except FileNotFoundError:
            raise ValueError(f"Laser parameter profile '{name}' not found")

    @classmethod
    def get_laser_detection_profile(cls) -> LaserDetectionProfile:
        # TODO: move to MyApplication and replace similar procedures
        name = repo.global_config.get().active_profile_names.laser_detection_profile_name
        if name is None:
            raise ValueError("Laser detection profile is not selected")
        try:
            return repo.laser_detection.get(name)
        except FileNotFoundError:
            raise ValueError(f"Laser detection profile '{name}' not found")

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
                try:
                    camera_param_profile = self.get_camera_param_profile()
                except ValueError as e:
                    get_app().make_toast(
                        Toast(
                            self,
                            "error",
                            e.args[0],
                        )
                    )
                    return
                else:
                    def callback(item: str | None) -> None:
                        get_app().close_dialog()
                        if item is not None:
                            image = repo.image.get(item)
                            get_app().move_to(
                                LaserParamScene(
                                    image=image,
                                    camera_param_profile=camera_param_profile,
                                )
                            )

                    get_app().show_dialog(
                        SelectImageItemDialog(
                            title="Select Image for Laser Calibration",
                            items=repo.image.list_names(),
                            callback=callback,
                            image_getter=lambda name: repo.image.get(name).data,
                        )
                    )
                    return
            if sender.get_name() == "b-laser-detection":
                def callback(item: str | None) -> None:
                    get_app().close_dialog()
                    if item is not None:
                        image = repo.image.get(item)
                        get_app().move_to(
                            LaserDetectionScene(
                                image=image,
                            )
                        )

                get_app().show_dialog(
                    SelectImageItemDialog(
                        title="Select Image for Laser Calibration",
                        items=repo.image.list_names(),
                        callback=callback,
                        image_getter=lambda name: repo.image.get(name).data,
                    )
                )
                return
            if sender.get_name() == "b-save-image":
                get_app().move_to(ScreenShotScene())
                return
            if sender.get_name() == "b-meas":
                try:
                    distortion_profile = self.get_distortion_profile()
                    camera_param_profile = self.get_camera_param_profile()
                    laser_param_profile = self.get_laser_param_profile()
                    laser_detection_profile = self.get_laser_detection_profile()
                except ValueError as e:
                    get_app().make_toast(
                        Toast(
                            self,
                            "error",
                            e.args[0],
                        )
                    )
                    return
                else:
                    get_app().move_to(
                        MeasScene(
                            distortion_profile=distortion_profile,
                            camera_param_profile=camera_param_profile,
                            laser_param_profile=laser_param_profile,
                            laser_detection_profile=laser_detection_profile,
                        )
                    )
                return
            if sender.get_name() == "b-recording":
                get_app().move_to(
                    RecordingScene(
                        distortion_profile=self.get_distortion_profile(),
                        roi_for_preview=repo.global_config.get().roi,
                    )
                )
                return
            if sender.get_name() == "b-stitching":
                def callback(name: str | None) -> None:
                    get_app().close_dialog()
                    if name is None:
                        return
                    video = repo.video.get(name)
                    get_app().move_to(
                        StitchingScene(
                            video=video,
                            camera_param_profile=self.get_camera_param_profile(),
                            laser_param_profile=self.get_laser_param_profile(),
                            laser_detection_profile=self.get_laser_detection_profile(),
                        )
                    )

                get_app().show_dialog(
                    SelectItemDialog(
                        title="Select Video for Stitching",
                        items=repo.video.list_names(),
                        callback=callback,
                    )
                )
            if sender.get_name() == "b-open-data-folder":
                open_in_explorer()
                return
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
