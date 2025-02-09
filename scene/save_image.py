from typing import cast

import numpy as np

import repo.image
from core.tk.component.button import ButtonComponent
from core.tk.component.component import Component
from core.tk.component.label import LabelComponent
from core.tk.component.spacer import SpacerComponent
from core.tk.component.toast import Toast
from core.tk.dialog import InputNameDialog
from core.tk.event import KeyEvent
from core.tk.global_state import get_app
from core.tk.key import Key
from model.image import Image
from my_app import MyApplication
from scene.my_scene import MyScene


class SaveImageScene(MyScene):
    def __init__(self):
        super().__init__()

        self._last_image = None

    def load_event(self):
        self.add_component(LabelComponent(self, "Save Image", bold=True))
        self.add_component(SpacerComponent(self))
        self.add_component(ButtonComponent(self, "Take Screenshot <SPACE>", name="b-take"))
        self.add_component(SpacerComponent(self))
        self.add_component(ButtonComponent(self, "Back", name="b-back"))

    def update_picture(self, image: np.ndarray | None):
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
            last_capture = cast(MyApplication, get_app()).last_capture
            if last_capture is not None:
                self.update_picture(last_capture.frame)
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
                    return repo.image.exists(name)

                def callback(name: str | None) -> None:
                    get_app().close_dialog()
                    if name is None:
                        return
                    repo.image.put(Image(name=name, data=self._last_image))
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
        if sender.get_name() == "b-back":
            get_app().move_back()
            return
        super()._on_button_triggered(sender)
