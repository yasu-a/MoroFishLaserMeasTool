from typing import TYPE_CHECKING

import cv2
import numpy as np

import repo.image
from core.tk.component.button import ButtonComponent
from core.tk.component.component import Component
from core.tk.component.label import LabelComponent
from core.tk.component.line_edit import LineEditComponent
from core.tk.component.spacer import SpacerComponent
from model import Image
from scene.my_scene import MyScene

if TYPE_CHECKING:
    from core.tk.app import Application


class SaveImageScene(MyScene):
    def __init__(self, app: "Application"):
        super().__init__(app)

        self._last_image = None

    def load_event(self):
        self.add_component(LabelComponent, "Save Image", bold=True)
        self.add_component(SpacerComponent)
        self.add_component(ButtonComponent, "Take Screenshot", name="b-take")
        self.add_component(SpacerComponent)
        self.add_component(LabelComponent, "Name:")
        self.add_component(LineEditComponent, name="e-name")
        self.add_component(LabelComponent, "", name="l-info")
        self.add_component(ButtonComponent, "Save", name="b-save")
        self.add_component(SpacerComponent)
        self.add_component(ButtonComponent, "Back", name="b-back")

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

    def on_value_changed(self, sender: Component) -> None:
        if sender.get_name() == "e-name":
            name = self.find_component(LineEditComponent, "e-name").get_value()
            self.set_overwrite_state(repo.image.exists(name))
            return
        super().on_value_changed(sender)

    def update_picture(self, image: np.ndarray | None):
        self._last_image = image
        self.set_picture_in_picture(image)

    def on_button_triggered(self, sender: Component) -> None:
        if sender.get_name() == "b-take":
            if self.get_app().last_capture is not None:
                self.update_picture(self.get_app().last_capture.frame)
                self.get_app().make_toast(
                    "info",
                    "Screenshot taken (not saved yet)"
                )
            return
        if sender.get_name() == "b-save":
            if self._last_image is not None:
                name = self.find_component(LineEditComponent, "e-name").get_value()
                if not name:
                    self.get_app().make_toast(
                        "error",
                        "Please enter a name"
                    )
                    return
                repo.image.put(Image(name=name, data=self._last_image))
                self.find_component(LineEditComponent, "e-name").set_value("")
                self.update_picture(None)
                self.get_app().make_toast(
                    "info",
                    f"Image saved: {name}"
                )
            else:
                self.get_app().make_toast(
                    "error",
                    "No screenshot taken"
                )

            return
        if sender.get_name() == "b-back":
            self.get_app().move_back()
