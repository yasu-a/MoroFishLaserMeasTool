from abc import ABC, abstractmethod

import cv2

from core.tk.app import Application
from core.tk.component.button import ButtonComponent
from core.tk.component.component import Component
from core.tk.component.label import LabelComponent
from core.tk.component.line_edit import LineEditComponent
from scene.my_scene import MyScene


class SaveProfileDelegate(ABC):
    @abstractmethod
    def check_exist(self, name: str) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def execute(self, name: str) \
            -> str | None:  # returns None if success, otherwise returns error message
        raise NotImplementedError()


class SaveProfileScene(MyScene):
    def __init__(self, app: "Application", delegator: SaveProfileDelegate):
        super().__init__(app)
        self._delegator = delegator

    def load_event(self):
        self.add_component(LabelComponent, "Save Profile", bold=True)
        self.add_component(LabelComponent, "Enter profile name:")
        self.add_component(
            LineEditComponent,
            name="e-name",
        )
        self.add_component(LabelComponent, "", name="l-info")
        self.add_component(ButtonComponent, "Save", name="b-save")
        self.add_component(ButtonComponent, "Cancel", name="b-cancel")

    CORNER_SUB_PIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def set_name_exists(self, state: bool):
        if state:
            self.find_component(LabelComponent, "l-info").set_text(
                f"NAME ALREADY EXISTS. DO YOU WANT TO OVERWRITE IT?"
            )
            self.find_component(ButtonComponent, "b-save").set_text(
                "Overwrite",
            )
        else:
            self.find_component(LabelComponent, "l-info").set_text(
                ""
            )
            self.find_component(ButtonComponent, "b-save").set_text(
                "Save"
            )

    def on_value_changed(self, sender: Component) -> None:
        if isinstance(sender, LineEditComponent):
            if sender.get_name() == "e-name":
                self.set_name_exists(self._delegator.check_exist(sender.get_value()))
                return
        super().on_value_changed(sender)

    def on_button_triggered(self, sender: Component) -> None:
        if isinstance(sender, ButtonComponent):
            if sender.get_name() == "b-save":
                name = self.find_component(LineEditComponent, "e-name").get_value().strip()
                if not name:
                    fail_message = "Name is empty"
                else:
                    fail_message = self._delegator.execute(name)
                if fail_message is None:
                    self.get_app().make_toast("info", "Profile saved")
                    self.get_app().move_back()
                else:
                    self.get_app().make_toast("error", f"Error: {fail_message}")
                return
            if sender.get_name() == "b-cancel":
                self.get_app().make_toast("error", f"Canceled")
                self.get_app().move_back()
                return
        super().on_button_triggered(sender)
