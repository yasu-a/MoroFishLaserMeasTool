from typing import cast

import numpy as np

import repo.distortion
import repo.global_config
import repo.image
import repo.raw_image
from core.tk.component.button import ButtonComponent
from core.tk.component.check_box import CheckBoxComponent
from core.tk.component.component import Component
from core.tk.component.label import LabelComponent
from core.tk.component.spacer import SpacerComponent
from core.tk.component.toast import Toast
from core.tk.dialog import InputNameDialog, SelectItemDialog
from core.tk.event import KeyEvent
from core.tk.global_state import get_app
from core.tk.key import Key
from model.distortion import DistortionProfile
from model.image import Image
from model.raw_image import RawImage
from my_app import MyApplication
from scene.my_scene import MyScene


class ScreenShotScene(MyScene):
    def __init__(self):
        super().__init__()

        distortion_profile_name: str | None \
            = repo.global_config.get().active_profile_names.distortion_profile_name

        self._distortion_profile: DistortionProfile | None = None
        if distortion_profile_name is not None:
            self._distortion_profile = repo.distortion.get(distortion_profile_name)

        self._last_image = None

    def load_event(self):
        self.add_component(LabelComponent(self, "Save Image", bold=True))
        self.add_component(SpacerComponent(self))
        self.add_component(ButtonComponent(self, "Take Screenshot <SPACE>", name="b-take"))
        self.add_component(SpacerComponent(self))
        self.add_component(LabelComponent(self, "Distortion Correction"))
        self.add_component(
            CheckBoxComponent(
                self,
                (
                    f"Use Profile: {self._distortion_profile.name}"
                    if self._distortion_profile is not None
                    else "Use Profile: (PROFILE NOT CONFIGURED)"
                ),
                True,
                name="cb-ud",
            )
        )
        self.add_component(
            ButtonComponent(self, "Apply Correction to Image", name="b-apply-correction")
        )
        self.add_component(SpacerComponent(self))
        self.add_component(ButtonComponent(self, "Back", name="b-back"))

    def _on_value_changed(self, sender: "Component") -> None:
        if sender.get_name() == "cb-ud":
            assert isinstance(sender, CheckBoxComponent)
            if sender.get_value() and self._distortion_profile is None:
                with self.disable_listener_context():
                    sender.set_value(False)
                get_app().make_toast(
                    Toast(
                        self,
                        "error",
                        "Distortion correction not available: profile not configured"
                    )
                )
        super()._on_value_changed(sender)

    def _update_picture(self, image: np.ndarray | None):
        cb_ud = self.find_component(CheckBoxComponent, "cb-ud")
        if cb_ud.get_value():  # undistortion enabled
            image = self._distortion_profile.params.undistort(image)

        self._last_image = image
        self.set_picture_in_picture(image)

    def key_event(self, event: KeyEvent) -> bool:
        if event.down:
            if event.key == Key.SPACE:
                self.find_component(ButtonComponent, "b-take").click()
                return True
        return super().key_event(event)

    def _on_button_triggered(self, sender: Component) -> None:
        if sender.get_name() == "b-take":
            undistortion_enabled = self.find_component(CheckBoxComponent, "cb-ud").get_value()
            last_capture = cast(MyApplication, get_app()).last_capture
            if last_capture is not None:
                self._update_picture(last_capture.frame)
                get_app().make_toast(
                    Toast(
                        self,
                        "info",
                        "Screenshot taken",
                    )
                )

                def validator(name: str) -> str | None:
                    if name == "":
                        return "Please enter a name"
                    return None

                def already_exist_checker(name: str) -> bool:
                    if undistortion_enabled:
                        return repo.image.exists(name)
                    else:
                        return repo.raw_image.exists(name)

                def callback(name: str | None) -> None:
                    get_app().close_dialog()
                    if name is None:
                        return
                    if undistortion_enabled:
                        repo.image.put(Image(name=name, data=self._last_image))
                    else:
                        repo.raw_image.put(RawImage(name=name, data=self._last_image))
                    get_app().make_toast(
                        Toast(
                            self,
                            "info",
                            f"Image saved: {name}",
                        )
                    )

                get_app().show_dialog(
                    InputNameDialog(
                        title="Enter a name for the image",
                        value="" if undistortion_enabled else "raw-",
                        validator=validator,
                        already_exist_checker=already_exist_checker,
                        callback=callback,
                    )
                )
                return
            else:
                get_app().make_toast(
                    Toast(
                        self,
                        "error",
                        "Failed to get screenshot: no camera available",
                    )
                )
                return
        if sender.get_name() == "b-apply-correction":
            def callback(item: str | None):
                get_app().close_dialog()
                if item is not None:
                    raw_image: RawImage = repo.raw_image.get(item)
                    data_undistorted = self._distortion_profile.params.undistort(raw_image.data)
                    self.set_picture_in_picture(data_undistorted)
                    get_app().make_toast(
                        Toast(
                            self,
                            "info",
                            f"Distortion of {raw_image.name!r} successfully corrected!",
                        )
                    )

                    def validator(name: str) -> str | None:
                        if name == "":
                            return "Please enter a name"
                        return None

                    def already_exist_checker(name: str) -> bool:
                        return repo.image.exists(name)

                    def callback(item: str | None):
                        get_app().close_dialog()
                        if item is not None:
                            image = Image(
                                name=item,
                                data=data_undistorted,
                            )
                            repo.image.put(image)
                            get_app().make_toast(
                                Toast(
                                    self,
                                    "info",
                                    f"Image saved: {item!r}",
                                )
                            )
                        else:
                            get_app().make_toast(
                                Toast(
                                    self,
                                    "error",
                                    "Canceled",
                                )
                            )

                    get_app().show_dialog(
                        InputNameDialog(
                            title="Enter a name for the corrected image",
                            value=item[4:] if item.startswith("raw-") else item,
                            validator=validator,
                            already_exist_checker=already_exist_checker,
                            callback=callback,
                        )
                    )

            get_app().show_dialog(
                SelectItemDialog(
                    title="Select Raw Image",
                    items=repo.raw_image.list_names(),
                    callback=callback,
                )
            )
            return
        if sender.get_name() == "b-back":
            get_app().move_back()
            return
        super()._on_button_triggered(sender)
