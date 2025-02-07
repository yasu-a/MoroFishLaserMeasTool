import numpy as np

import repo.image
from scene.select_item import SelectItemScene


class SelectImageItemScene(SelectItemScene):
    def render_canvas(self) -> np.ndarray | None:
        return None

    def selection_change_event(self, name: str | None):
        super().selection_change_event(name)
        if name is not None:
            self.set_picture_in_picture(repo.image.get(name).data)
