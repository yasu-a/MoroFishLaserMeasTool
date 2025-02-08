import numpy as np

import repo.image
from core.tk.app import ApplicationWindowSize
from scene.select_item import SelectItemScene


class SelectImageItemScene(SelectItemScene):
    def create_background(self, window_size: "ApplicationWindowSize") -> np.ndarray | None:
        return None

    def selection_change_event(self, name: str | None):
        super().selection_change_event(name)
        if name is not None:
            self.set_picture_in_picture(repo.image.get(name).data)
