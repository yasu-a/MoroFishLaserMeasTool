from dataclasses import dataclass

import numpy as np

from core.tk.style import ApplicationUIStyle


@dataclass(frozen=True)
class UIRenderingContext:
    style: ApplicationUIStyle
    font: int
    scale: float
    top: int
    left: int

    @property
    def font_height(self) -> int:
        return int(round(self.scale * 32, 0))

    @property
    def font_offset_y(self) -> int:
        return int(round(self.font_height * 0.8, 0))


class Canvas:
    def __init__(self, im: np.ndarray):
        self._im = im

    @property
    def im(self) -> np.ndarray:
        return self._im


@dataclass(frozen=True)
class RenderingResult:
    height: int
