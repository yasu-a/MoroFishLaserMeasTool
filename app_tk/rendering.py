from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RenderingContext:
    canvas: np.ndarray
    color: tuple[int, int, int]
    max_width: int
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

    def update_buffer_by_alpha_blend_mask(
            self,
            x: int,
            y: int,
            mask: np.ndarray,
            color: tuple[int, int, int],
    ) -> None:
        h, w = mask.shape[:2]
        idx = slice(y, y + h), slice(x, x + w)
        image_view = self.canvas[idx]
        image_view[mask > 0, :] \
            = (np.array(color)[None, None, :] * (mask > 0)[:, :, None])[mask > 0, :]


@dataclass(frozen=True)
class RenderingResult:
    height: int
