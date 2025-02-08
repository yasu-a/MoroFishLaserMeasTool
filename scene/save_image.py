from typing import cast

import numpy as np

import repo.image
from core.tk.component.button import ButtonComponent
from core.tk.component.component import Component
from core.tk.component.global_state import get_app
from core.tk.component.label import LabelComponent
from core.tk.component.line_edit import LineEditComponent
from core.tk.component.spacer import SpacerComponent
from core.tk.component.toast import Toast
from model import Image
from my_app import MyApplication
from scene.my_scene import MyScene


class SaveImageScene(MyScene):
    def __init__(self):
        super().__init__()

        self._last_image = None

    def load_event(self):
        self.add_component(LabelComponent(self, "Save Image", bold=True))
        self.add_component(SpacerComponent(self))
        self.add_component(ButtonComponent(self, "Take Screenshot", name="b-take"))
        self.add_component(SpacerComponent(self))
        self.add_component(LabelComponent(self, "Name:"))
        self.add_component(LineEditComponent(self, name="e-name"))
        self.add_component(LabelComponent(self, "", name="l-info"))
        self.add_component(ButtonComponent(self, "Save", name="b-save"))
        self.add_component(SpacerComponent(self))
        self.add_component(ButtonComponent(self, "Back", name="b-back"))

    def set_overwrite_state(self, state: bool):
        if state:
            # already exists
            self.find_component(ButtonComponent, "b-save").set_text("Overwrite and Save")
            self.find_component(LabelComponent, "l-info").set_text(
                f"Name already exists. Are you sure to overwrite it?"
            )
        else:
            self.find_component(ButtonComponent, "b-save").set_text("Save")
            self.find_component(LabelComponent, "l-info").set_text("")

    def _on_value_changed(self, sender: Component) -> None:
        if sender.get_name() == "e-name":
            name = self.find_component(LineEditComponent, "e-name").get_value()
            self.set_overwrite_state(repo.image.exists(name))
            return
        super()._on_value_changed(sender)

    def update_picture(self, image: np.ndarray | None):
        self._last_image = image
        self.set_picture_in_picture(image)

    def _on_button_triggered(self, sender: Component) -> None:
        if sender.get_name() == "b-take":
            last_capture = cast(MyApplication, get_app()).last_capture
            if last_capture is not None:
                self.update_picture(last_capture.frame)
                get_app().make_toast(
                    Toast(
                        self,
                        "info",
                        "Screenshot taken (not saved yet)",
                    )
                )
            return
        if sender.get_name() == "b-save":
            if self._last_image is not None:
                name = self.find_component(LineEditComponent, "e-name").get_value()
                if not name:
                    get_app().make_toast(
                        Toast(
                            self,
                            "error",
                            "Please enter a name",
                        )
                    )
                    return
                repo.image.put(Image(name=name, data=self._last_image))
                self.find_component(LineEditComponent, "e-name").set_value("")
                self.update_picture(None)
                get_app().make_toast(
                    Toast(
                        self,
                        "info",
                        f"Image saved: {name}",
                    )
                )
            else:
                get_app().make_toast(
                    Toast(
                        self,
                        "error",
                        "No screenshot taken",
                    )
                )

            return
        if sender.get_name() == "b-back":
            get_app().move_back()
