from typing import TYPE_CHECKING

import repo.distortion
from app_tk.app import Application
from app_tk.component.button import ButtonComponent
from app_tk.component.component import Component
from app_tk.component.label import LabelComponent
from app_tk.component.spacer import SpacerComponent
from scene_base import MyScene
from scene_select_item import SelectItemDelegate, SelectItemScene

if TYPE_CHECKING:
    from my_app import MyApplication


class DistortionSelectItemDelegate(SelectItemDelegate):
    def __init__(self, app: "MyApplication"):
        self._app = app

    def list_name(self) -> list[str]:
        return repo.distortion.list_names()

    def execute(self, name: str) -> str | None:
        self._app.active_profile_names.distortion_profile_name = name
        return None


class SelectProfileMenuScene(MyScene):
    def __init__(self, app: "Application"):
        super().__init__(app)

    def load_event(self):
        self.add_component(LabelComponent, "Profile Config", bold=True)
        self.add_component(SpacerComponent)
        self.add_component(ButtonComponent, "Distortion Profile", name="b-distortion")
        self.add_component(LabelComponent, "", name="l-distortion")
        self.add_component(SpacerComponent)
        self.add_component(ButtonComponent, "Camera Parameter Profile", name="b-camera-param")
        self.add_component(LabelComponent, "", name="l-camera-param")
        self.add_component(SpacerComponent)
        self.add_component(ButtonComponent, "Laser Parameter Profile", name="b-laser-param")
        self.add_component(LabelComponent, "", name="l-laser-param")
        self.add_component(SpacerComponent)
        self.add_component(ButtonComponent, "Back", name="b-back")

    def show_event(self):
        active_profile_names = self.get_app().active_profile_names

        if active_profile_names.distortion_profile_name is None:
            self.find_component(LabelComponent, "l-distortion").set_text(
                "(NO PROFILE SELECTED)",
            )
        else:
            self.find_component(LabelComponent, "l-distortion").set_text(
                active_profile_names.distortion_profile_name,
            )

        if active_profile_names.camera_profile_name is None:
            self.find_component(LabelComponent, "l-camera-param").set_text(
                "(NO PROFILE SELECTED)",
            )
        else:
            self.find_component(LabelComponent, "l-camera-param").set_text(
                active_profile_names.camera_profile_name,
            )

        if active_profile_names.laser_profile_name is None:
            self.find_component(LabelComponent, "l-laser-param").set_text(
                "(NO PROFILE SELECTED)",
            )
        else:
            self.find_component(LabelComponent, "l-laser-param").set_text(
                active_profile_names.laser_profile_name,
            )

    def on_button_triggered(self, sender: Component) -> None:
        if isinstance(sender, ButtonComponent):
            if sender.get_name() == "b-distortion":
                self.get_app().move_to(
                    SelectItemScene(
                        self.get_app(),
                        DistortionSelectItemDelegate(self.get_app()),
                    )
                )
                return
            if sender.get_name() == "b-camera-param":
                return
            if sender.get_name() == "b-laser-param":
                return
            if sender.get_name() == "b-back":
                self._app.go_back()
                return
