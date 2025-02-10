import contextlib
from dataclasses import dataclass
from typing import ContextManager

import cv2
import numpy as np

from core.tk.color import Color
from core.tk.font_renderer import CharPrinter
from core.tk.style import ApplicationUIStyle


@dataclass(slots=True)
class UIRenderingContext:
    style: ApplicationUIStyle
    fg_colors: dict[str, Color]  # mapping from sub-context (info, error, ...) to color
    bg_colors: dict[str, Color | None]
    edge_colors: dict[str, Color | None]
    char_printer: CharPrinter
    font_size: float
    top: int
    left: int
    max_width: int
    sub_context: str | None = None
    canvas_image: np.ndarray | None = None

    def __hash__(self):
        return hash(
            (
                id(self.style),
                tuple(sorted(self.fg_colors.items())),
                tuple(sorted(self.bg_colors.items())),
                tuple(sorted(self.edge_colors.items())),
                id(self.char_printer),
                self.font_size,
                self.top,
                self.left,
                self.max_width,
                self.sub_context,
            )
        )

    @contextlib.contextmanager
    def enter_sub_context(self, sub_context: str) -> ContextManager["UIRenderingContext"]:
        prev_sub_context = self.sub_context
        self.sub_context = sub_context
        yield self
        self.sub_context = prev_sub_context

    @property
    def fg_color(self) -> Color:
        try:
            return self.fg_colors[self.sub_context]
        except KeyError:
            raise ValueError("Sub-context not determined")

    @property
    def bg_color(self) -> Color | None:
        try:
            return self.bg_colors[self.sub_context]
        except KeyError:
            return None

    @property
    def edge_color(self) -> Color | None:
        try:
            return self.edge_colors[self.sub_context]
        except KeyError:
            return None

    @contextlib.contextmanager
    def offer_canvas_image(self, im: np.ndarray) -> ContextManager["UIRenderingContext"]:
        prev_canvas_image = self.canvas_image
        self.canvas_image = im
        yield self
        self.canvas_image = prev_canvas_image

    @property
    def canvas(self) -> "Canvas":
        return Canvas(self.canvas_image, self)  # TODO: cache me


class Canvas:
    def __init__(self, im: np.ndarray, ctx: UIRenderingContext):
        self._im = im
        self._ctx = ctx

    @property
    def width(self) -> int:
        return self._im.shape[1]

    @property
    def height(self) -> int:
        return self._im.shape[0]

    def text(
            self,
            *,
            text: str,
            pos: tuple[int, int],
            max_width: int,
            max_height: int = None,
            scale: float = 1.0,
            fg_color: Color,
            bg_color: Color = None,
            edge_color: Color = None,
            bold: bool = False,
    ) -> int:  # height
        # bg
        height = None
        if bg_color is not None:
            if max_height is None:
                text = text.replace("\n", " ")  # force single line mode
                height = self._ctx.char_printer.text_height(text, self._ctx.font_size * scale)
            else:
                height = max_height
            cv2.rectangle(
                self._im,
                pos,
                (pos[0] + max_width, pos[1] + height),
                bg_color,
                -1,
            )

        base_thickness = 4 if bold else 1

        # edge
        if edge_color is not None:
            edge_thickness = base_thickness + 10
            self._ctx.char_printer.put_text_multiline(
                im=self._im,
                pos=pos,
                text=text,
                size=self._ctx.font_size * scale,
                color=edge_color,
                thickness=edge_thickness,
                max_width=max_width,
                max_height=max_height,
            )

        # fg
        _, text_height = self._ctx.char_printer.put_text_multiline(
            im=self._im,
            pos=pos,
            text=text,
            size=self._ctx.font_size * scale,
            color=fg_color,
            thickness=base_thickness,
            max_width=max_width,
            max_height=max_height,
        )
        if height is None:
            height = text_height

        return height

    def rectangle(self, pos: tuple[int, int], size: tuple[int, int], color: Color):
        cv2.rectangle(
            self._im,
            pos,
            (pos[0] + size[0], pos[1] + size[1]),
            color,
            1,
            cv2.LINE_AA,
        )

    def paste(self, im: np.ndarray, pos: tuple[int, int]):
        im_warp = cv2.warpAffine(
            im,
            np.float32([
                [1, 0, pos[0]],
                [0, 1, pos[1]],
            ]),
            (self._im.shape[1], self._im.shape[0]),
        )
        mask_warp = cv2.warpAffine(
            np.full((im.shape[0], im.shape[1]), 255, np.uint8),
            np.float32([
                [1, 0, pos[0]],
                [0, 1, pos[1]],
            ]),
            (self._im.shape[1], self._im.shape[0]),
        )
        self._im[mask_warp == 255] = im_warp[mask_warp == 255]

    def fullscreen_fill(self, color: Color):
        cv2.rectangle(
            self._im,
            (0, 0),
            (self._im.shape[1], self._im.shape[0]),
            color,
            -1,
        )

    def fullscreen_rect(self, color: Color):
        cv2.rectangle(
            self._im,
            (0, 0),
            (self._im.shape[1], self._im.shape[0]),
            color,
            10,
            cv2.LINE_AA,
        )


@dataclass(frozen=True)
class RenderingResult:
    height: int
